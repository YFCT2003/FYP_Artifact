# -*- coding: UTF-8 -*-
"""
@Project：FYP
@File：train_sliding_attention.py
@Date：2026/4/9 17:36

TCR-pHLA Binding Prediction Training Script (single-GPU, AMP)

Data flow:
    pHLA (HLA_seq, Epitope) -> ESM(fine-tuned, frozen) -> remove_special_tokens -> phla_embed [B, L_K, d_esm]
    TCR  (CDR3beta)         -> ESM(pretrained, frozen)  -> remove_special_tokens -> tcr_embed  [B, L_Q, d_esm]
                                       |
                             SlidingCrossAttention (dual-stream)
                                       |
                            cat(tcr_pooled, phla_pooled) -> classifier -> logits [B, 2]
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import EsmTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, auc,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Architecture import PLMSlidingModel

from utils import (
    load_sequences, EarlyStopping, logger,
    save_checkpoint, load_checkpoint,
)
import logging
from logging import FileHandler

import matplotlib.pyplot as plt


# ==========================================
# 1. Dataset Definition
# ==========================================
class TCRpHLADataset(Dataset):
    """
    TCR-pHLA binding prediction dataset.
    Each sample contains a pHLA pair (HLA + Epitope) and a TCR CDR3beta sequence with a binary label.
    Returns unpadded token ids for dynamic padding in the collator.
    """
    def __init__(self, df, tokenizer, phla_max_length=50, tcr_max_length=35):
        self.epitope = df['antigen.epitope'].values
        self.hla_seq = df['mhc.seq'].values
        self.cdr3_beta = df['cdr3.beta'].values
        self.labels = df['label'].astype(int).values
        self.tokenizer = tokenizer
        self.phla_max_length = phla_max_length
        self.tcr_max_length = tcr_max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # pHLA pair encoding: [CLS] HLA_tokens [EOS] Epitope_tokens [EOS]
        phla_encoded = self.tokenizer(
            self.hla_seq[idx], self.epitope[idx],
            truncation="longest_first", max_length=self.phla_max_length
        )
        # TCR single encoding: [CLS] CDR3beta_tokens [EOS]
        tcr_encoded = self.tokenizer(
            self.cdr3_beta[idx],
            truncation=True, max_length=self.tcr_max_length
        )
        return {
            'phla_input_ids': phla_encoded['input_ids'],
            'phla_attention_mask': phla_encoded['attention_mask'],
            'tcr_input_ids': tcr_encoded['input_ids'],
            'tcr_attention_mask': tcr_encoded['attention_mask'],
            'label': self.labels[idx]
        }


# ==========================================
# 2. Dynamic Padding Collator
# ==========================================
class TCRpHLACollator:
    """Dynamically pad pHLA and TCR inputs separately within each batch."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        phla_inputs = [{
            'input_ids': item['phla_input_ids'],
            'attention_mask': item['phla_attention_mask']
        } for item in batch]
        tcr_inputs = [{
            'input_ids': item['tcr_input_ids'],
            'attention_mask': item['tcr_attention_mask']
        } for item in batch]
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)

        phla_batch = self.tokenizer.pad(phla_inputs, return_tensors='pt')
        tcr_batch = self.tokenizer.pad(tcr_inputs, return_tensors='pt')

        return {
            'phla_input_ids': phla_batch['input_ids'],
            'phla_attention_mask': phla_batch['attention_mask'],
            'tcr_input_ids': tcr_batch['input_ids'],
            'tcr_attention_mask': tcr_batch['attention_mask'],
            'labels': labels
        }


# ==========================================
# 3. Hard Split by columns (stratified)
# ==========================================
def hard_split_df(df, target_cols, val_ratio, random_state=42):
    """
    Hard split train/val by target_cols: all rows with the same composite key will not appear on both sides simultaneously.
    Supports a single column (str) or multiple columns (list); when using multiple columns, join with '|' to form a composite key.
    At the same time, maintain the positive-to-negative label ratio close to the global ratio.
    """
    rng = np.random.default_rng(random_state)
    cols = [target_cols] if isinstance(target_cols, str) else target_cols

    composite_key = df[cols[0]].astype(str)
    for col in cols[1:]:
        composite_key += '|' + df[col].astype(str)

    total_pos = int(df['label'].sum())
    total_neg = len(df) - total_pos
    target_val_pos = int(total_pos * val_ratio)
    target_val_neg = int(total_neg * val_ratio)

    groups = []
    for key, group in df.groupby(composite_key):
        pos = int(group['label'].sum())
        neg = len(group) - pos
        groups.append({'key': key, 'indices': group.index, 'pos': pos, 'neg': neg})

    rng.shuffle(groups)

    val_indices = []
    val_pos = 0
    val_neg = 0

    for g in groups:
        need_pos = val_pos < target_val_pos
        need_neg = val_neg < target_val_neg
        if need_pos or need_neg:
            val_indices.extend(g['indices'])
            val_pos += g['pos']
            val_neg += g['neg']

    val_df = df.loc[val_indices]
    train_df = df.drop(val_indices)

    col_str = '+'.join(cols)
    logger.info(
        f"Hard-split on [{col_str}]: {len(groups)} unique groups → "
        f"train={len(train_df)} (pos={int(train_df['label'].sum())}, "
        f"neg={len(train_df)-int(train_df['label'].sum())}), "
        f"val={len(val_df)} (pos={val_pos}, neg={val_neg})"
    )

    return train_df, val_df


# ==========================================
# 4. Metrics helper
# ==========================================
def compute_metrics(labels, probs, preds):
    """
    Compute all classification metrics from labels, predicted probabilities, and hard predictions.
    Returns a dict with: auroc, auprc, accuracy, recall, precision, f1.
    """
    n_unique = len(np.unique(labels))
    return {
        'auroc': roc_auc_score(labels, probs) if n_unique > 1 else 0.0,
        'auprc': average_precision_score(labels, probs) if n_unique > 1 else 0.0,
        'accuracy': accuracy_score(labels, preds),
        'recall': recall_score(labels, preds),
        'precision': precision_score(labels, preds),
        'f1': f1_score(labels, preds),
    }


# ==========================================
# 5. Focal Loss
# ==========================================
class WeightedFocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    alpha: weight for class 0 (negative). Class 1 gets 1-alpha.
    gamma: focusing parameter — higher values down-weight easy samples.
    Set gamma=0 to recover standard weighted cross-entropy.
    """
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('alpha', torch.tensor([alpha, 1.0 - alpha], dtype=torch.float32))

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        at = self.alpha.gather(0, targets.data.view(-1))
        return (at * (1 - pt) ** self.gamma * ce_loss).mean()


# ==========================================
# 6. Main Training Process
# ==========================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Folder Setup ----
    folder_name = (
        f"SlidingAttn_d{args.d_model}_h{args.n_heads}_iter{args.n_iter}"
        f"_lr_{args.lr}_bs_{args.batch_size_train}"
        f"_drop_{args.classifier_dropout}_sigma_{args.sigma}"
        f"_focal_gamma_{args.focal_gamma}"
        f"_wd_{args.weight_decay}"
    )
    experiment_dir = os.path.join(args.weights_root, folder_name)
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    curves_dir = os.path.join(experiment_dir, "curves")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(curves_dir, exist_ok=True)
    best_model_path = os.path.join(experiment_dir, "best_model.pt")
    history_csv_path = os.path.join(experiment_dir, "training_history.csv")

    # Write dedicated sliding.log
    log_path = os.path.join(experiment_dir, "sliding.log")
    fh = FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    logger.info(f"Logging to {log_path}")

    start_epoch = 0
    train_history = []
    val_history = []

    # ---- Load Tokenizer and Model ----
    if args.d_ff is None:
        args.d_ff = 4 * args.d_model

    tokenizer = EsmTokenizer.from_pretrained(args.phla_model_name)

    model = PLMSlidingModel(
        phla_model_path=args.phla_model_path,
        phla_model_name=args.phla_model_name,
        tcr_model_name=args.tcr_model_name,
        d_model=args.d_model,
        n_heads=args.n_heads,
        L_Q=args.L_Q,
        L_K=args.L_K,
        sigma=args.sigma,
        window_T=args.window_T,
        n_iter=args.n_iter,
        d_ff=args.d_ff,
        freeze_phla=args.freeze_phla,
        freeze_tcr=args.freeze_tcr,
        classifier_dropout=args.classifier_dropout,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable:,} / {total:,}")

    # ---- Data Loading ----
    df = load_sequences(args.data_path)
    train_df, val_df = hard_split_df(
        df, target_cols=args.hard_split_cols, val_ratio=args.val_split, random_state=args.seed
    )

    train_dataset = TCRpHLADataset(train_df, tokenizer, args.phla_max_length, args.tcr_max_length)
    val_dataset = TCRpHLADataset(val_df, tokenizer, args.phla_max_length, args.tcr_max_length)

    collator = TCRpHLACollator(tokenizer)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size_train, collate_fn=collator,
        num_workers=args.num_workers, shuffle=True, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size_val, collate_fn=collator,
        num_workers=args.num_workers, shuffle=False, pin_memory=True
    )

    # ---- Loss Function ----
    label_counts = train_df['label'].value_counts().sort_index()
    n_samples = len(train_df)
    pos_ratio = label_counts.iloc[1] / n_samples
    logger.info(f"Class distribution — Neg: {label_counts.iloc[0]}, Pos: {label_counts.iloc[1]} ({pos_ratio:.1%} pos)")

    if args.focal_gamma > 0:
        # alpha = minority class ratio, so minority gets higher weight 1-alpha
        alpha = pos_ratio if args.focal_alpha is None else args.focal_alpha
        loss_fn = WeightedFocalLoss(alpha=alpha, gamma=args.focal_gamma).to(device)
        logger.info(f"Using Focal Loss (alpha={alpha:.4f}, gamma={args.focal_gamma})")
    else:
        class_weights = torch.tensor([
            n_samples / (2 * label_counts.iloc[0]),
            n_samples / (2 * label_counts.iloc[1])
        ], dtype=torch.float).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        logger.info(f"Using CrossEntropyLoss with class weights: [{class_weights[0]:.4f}, {class_weights[1]:.4f}]")

    # ---- Optimizer (only trainable parameters), Scheduler, AMP ----
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler('cuda')

    actual_update_steps_per_epoch = max(1, len(train_dataloader) // args.grad_accumulation_steps)
    total_training_steps = args.epochs * actual_update_steps_per_epoch

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=total_training_steps
    )

    if args.resume_from and os.path.exists(args.resume_from):
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint_data = load_checkpoint(
            args.resume_from, model, optimizer=optimizer, scheduler=scheduler, device=device
        )
        start_epoch = checkpoint_data['epoch']
        train_history = checkpoint_data.get('train_history', [])
        val_history = checkpoint_data.get('val_history', [])
        logger.info(f"Resumed from Epoch {start_epoch}")

    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=best_model_path)

    # ---- CSV Logging Setup ----
    logger.info("================ Starting Sliding Attention Training ================")

    file_mode = 'a' if (args.resume_from and os.path.exists(history_csv_path)) else 'w'
    csv_file = open(history_csv_path, mode=file_mode, newline='')
    csv_writer = csv.writer(csv_file)
    if file_mode == 'w':
        csv_writer.writerow([
            'epoch',
            'train_loss', 'train_acc', 'train_auroc', 'train_auprc',
            'train_recall', 'train_precision', 'train_f1',
            'val_loss', 'val_acc', 'val_auroc', 'val_auprc',
            'val_recall', 'val_precision', 'val_f1',
        ])

    # ---- Training Loop ----
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_probs = []
        train_preds = []
        train_labels = []

        for step, batch in enumerate(train_dataloader):
            phla_input_ids = batch['phla_input_ids'].to(device)
            phla_attention_mask = batch['phla_attention_mask'].to(device)
            tcr_input_ids = batch['tcr_input_ids'].to(device)
            tcr_attention_mask = batch['tcr_attention_mask'].to(device)
            labels = batch['labels'].to(device)

            is_last_step = (step + 1) == len(train_dataloader)
            is_accumulating = ((step + 1) % args.grad_accumulation_steps != 0) and not is_last_step

            # Forward pass under AMP
            with torch.amp.autocast('cuda'):
                logits, _ = model(phla_input_ids, phla_attention_mask,
                                  tcr_input_ids, tcr_attention_mask)
                loss = loss_fn(logits, labels) / args.grad_accumulation_steps

            # Backward (accumulate gradients until sync point)
            scaler.scale(loss).backward()

            if not is_accumulating:
                # Gradient clipping + optimizer step
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)

                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Only step scheduler when optimizer was not skipped by GradScaler
                scale_after = scaler.get_scale()
                if scale_before <= scale_after:
                    scheduler.step()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                probs = torch.softmax(logits, dim=-1)[:, 1]
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)
                train_probs.append(probs.cpu())
                train_preds.append(preds.cpu())
                train_labels.append(labels.cpu())
            train_loss += loss.item() * args.grad_accumulation_steps

            if (step + 1) % args.log_interval == 0:
                step_acc = train_correct / train_total
                logger.info(
                    f"Epoch [{epoch + 1}/{args.epochs}], "
                    f"Step [{step + 1}/{len(train_dataloader)}], "
                    f"Loss: {loss.item() * args.grad_accumulation_steps:.4f}, Acc: {step_acc:.4f}"
                )

        # ---- Train metrics ----
        train_labels_np = torch.cat(train_labels).numpy()
        train_probs_np = torch.cat(train_probs).numpy()
        train_preds_np = torch.cat(train_preds).numpy()
        avg_train_loss = train_loss / len(train_dataloader)
        train_m = compute_metrics(train_labels_np, train_probs_np, train_preds_np)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_probs = []
        val_preds = []
        val_labels = []

        with torch.no_grad(), torch.amp.autocast('cuda'):
            for batch in val_dataloader:
                phla_input_ids = batch['phla_input_ids'].to(device)
                phla_attention_mask = batch['phla_attention_mask'].to(device)
                tcr_input_ids = batch['tcr_input_ids'].to(device)
                tcr_attention_mask = batch['tcr_attention_mask'].to(device)
                labels = batch['labels'].to(device)

                logits, _ = model(phla_input_ids, phla_attention_mask,
                                  tcr_input_ids, tcr_attention_mask)
                loss = loss_fn(logits, labels)

                val_loss += loss.item()
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                probs = torch.softmax(logits, dim=-1)[:, 1]
                val_probs.append(probs.cpu())
                val_preds.append(preds.cpu())
                val_labels.append(labels.cpu())

        avg_val_loss = val_loss / len(val_dataloader)
        val_labels_np = torch.cat(val_labels).numpy()
        val_probs_np = torch.cat(val_probs).numpy()
        val_preds_np = torch.cat(val_preds).numpy()
        val_m = compute_metrics(val_labels_np, val_probs_np, val_preds_np)

        # ---- Logging, Checkpointing, Early Stopping ----
        train_history.append({
            'epoch': epoch + 1, 'loss': avg_train_loss,
            'accuracy': train_m['accuracy'], 'auroc': train_m['auroc'], 'auprc': train_m['auprc'],
            'recall': train_m['recall'], 'precision': train_m['precision'], 'f1': train_m['f1'],
        })
        val_history.append({
            'epoch': epoch + 1, 'loss': avg_val_loss,
            'accuracy': val_m['accuracy'], 'auroc': val_m['auroc'], 'auprc': val_m['auprc'],
            'recall': val_m['recall'], 'precision': val_m['precision'], 'f1': val_m['f1'],
        })
        csv_writer.writerow([
            epoch + 1,
            avg_train_loss, train_m['accuracy'], train_m['auroc'], train_m['auprc'],
            train_m['recall'], train_m['precision'], train_m['f1'],
            avg_val_loss, val_m['accuracy'], val_m['auroc'], val_m['auprc'],
            val_m['recall'], val_m['precision'], val_m['f1'],
        ])
        csv_file.flush()

        logger.info(
            f"==== Epoch {epoch + 1} Summary ===="
            f" Train — Loss: {avg_train_loss:.4f} | Acc: {train_m['accuracy']:.4f}"
            f" | AUROC: {train_m['auroc']:.4f} | AUPRC: {train_m['auprc']:.4f}"
            f" || Val — Loss: {avg_val_loss:.4f} | Acc: {val_m['accuracy']:.4f}"
            f" | AUROC: {val_m['auroc']:.4f} | AUPRC: {val_m['auprc']:.4f}"
            f" | Recall: {val_m['recall']:.4f} | Precision: {val_m['precision']:.4f} | F1: {val_m['f1']:.4f}"
        )

        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            save_checkpoint(
                epoch + 1, model, optimizer, scheduler,
                train_history, val_history, args, checkpoint_path
            )

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            break

    csv_file.close()
    logger.info("================ Generating Plots ================")
    plot_training_curves(train_history, val_history, curves_dir)
    logger.info(f"Training completed. Best model saved to: {best_model_path}")


# ==========================================
# 7. Plotting
# ==========================================
def plot_training_curves(train_history, val_history, save_dir, prefix="sliding_attn"):
    """
    2x4 grid, 7 metric subplots, train (blue) + val (red) line plots overlaid.
    """
    # ---- Global plot parameters ----
    plt.rcParams.update({
        "font.family":          "Arial",
        "font.size":            11,
        "axes.titlesize":       20,
        "axes.labelsize":       18,
        "xtick.labelsize":      11,
        "ytick.labelsize":      11,
        "legend.fontsize":      10,
        "legend.title_fontsize":11,
        "axes.titlepad":        15,
        "axes.linewidth":       0.75,
        "lines.linewidth":      1.0,
        "patch.linewidth":      0.5,
        "xtick.major.width":    0.75,
        "ytick.major.width":    0.75,
        "xtick.major.size":     3.0,
        "ytick.major.size":     3.0,
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "legend.frameon":       False,
        "figure.dpi":           300,
        "savefig.dpi":          600,
        "savefig.bbox":         "tight",
        "savefig.format":       "svg",
    })

    os.makedirs(save_dir, exist_ok=True)
    epochs = [h['epoch'] for h in train_history]
    metrics = ['loss', 'accuracy', 'auroc', 'auprc', 'recall', 'precision', 'f1']

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.plot(epochs, [h[metric] for h in train_history],
                marker='o', label='Train', color='#4C72B0')
        ax.plot(epochs, [h[metric] for h in val_history],
                marker='o', label='Val', color='#C44E52')
        ax.set_title(metric.upper())
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Hide the 8th empty subplot
    axes[-1].set_visible(False)

    fig.suptitle('Training & Validation Metrics', fontsize=16)
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_curves.svg"))
    plt.close()

    logger.info(f"Plots saved to {save_dir}")


# ==========================================
# 8. Argument Parser
# ==========================================
def add_args_func(parser):
    # Basic Params
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data_path", type=str,
                        default=r"D:\Python\Python_project\FYP\data\df_clean_pos_9.csv",
                        help="CSV with columns: antigen.epitope, mhc.seq, cdr3.beta, label")
    parser.add_argument("--weights_root", type=str,
                        default=r"D:\Python\Python_project\FYP\weights",
                        help="Root directory for saving weights")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # ---- Encoder Params ----
    parser.add_argument("--phla_model_path", type=str, required=True,
                        help="Path to fine-tuned pHLA ESM weights (from train_mlm_pHLA.py)")
    parser.add_argument("--phla_model_name", type=str, default="facebook/esm2_t33_650M_UR50D",
                        help="ESM-2 model name for pHLA encoder")
    parser.add_argument("--tcr_model_name", type=str, default="facebook/esm2_t33_650M_UR50D",
                        help="ESM-2 model name for TCR encoder")
    parser.add_argument("--freeze_phla", action="store_true", default=True,
                        help="Freeze pHLA encoder (default: True)")
    parser.add_argument("--no_freeze_phla", action="store_false", dest="freeze_phla",
                        help="Unfreeze pHLA encoder")
    parser.add_argument("--freeze_tcr", action="store_true", default=True,
                        help="Freeze TCR encoder (default: True)")
    parser.add_argument("--no_freeze_tcr", action="store_false", dest="freeze_tcr",
                        help="Unfreeze TCR encoder")

    # ---- Sliding Cross-Attention Params ----
    parser.add_argument("--d_model", type=int, default=256,
                        help="Internal dimension of sliding attention")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--L_Q", type=int, default=23,
                        help="Max TCR residue length after removing special tokens (typical CDR3b: 10-20)")
    parser.add_argument("--L_K", type=int, default=43,
                        help="Max pHLA residue length after removing special tokens (34 HLA + 9 Epi)")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="Gaussian kernel bandwidth for spatial bias")
    parser.add_argument("--window_T", type=float, default=None,
                        help="Hard truncation threshold for spatial bias (None = disabled)")
    parser.add_argument("--n_iter", type=int, default=3,
                        help="Number of sliding iterations")
    parser.add_argument("--d_ff", type=int, default=None,
                        help="FFN hidden dimension in sliding attention (default: 4 × d_model)")
    parser.add_argument("--classifier_dropout", type=float, default=0.0,
                        help="Dropout after classifier hidden layer (default: 0.0, disabled)")

    # ---- Loss Function ----
    parser.add_argument("--focal_gamma", type=float, default=0.0,
                        help="Focal Loss gamma (default: 0 = disabled). Set 2.0 to enable, or 1.0 for mild focusing.")
    parser.add_argument("--focal_alpha", type=float, default=None,
                        help="Focal Loss alpha for class 0 (default: auto-compute from data). "
                             "Lower = more weight on positive class.")

    # ---- Training Hyperparameters ----
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--batch_size_train", type=int, default=64, help="Training batch size")
    parser.add_argument("--batch_size_val", type=int, default=128, help="Validation batch size")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation data split ratio")
    parser.add_argument("--hard_split_cols", type=str, nargs='+',
                        default=["antigen.epitope", "mhc.seq"],
                        help="Columns for hard split (space-separated). Composite key of these columns "
                             "stays together in train or val. Default: antigen.epitope mhc.seq")

    # ---- Tokenization ----
    parser.add_argument("--phla_max_length", type=int, default=50,
                        help="Max token length for pHLA pair encoding")
    parser.add_argument("--tcr_max_length", type=int, default=35,
                        help="Max token length for TCR CDR3beta encoding")

    # ---- Resource ----
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers")

    # ---- Logging & Saving ----
    parser.add_argument("--log_interval", type=int, default=100, help="Steps between logging")
    parser.add_argument("--save_interval", type=int, default=10, help="Epochs between saving checkpoints")

    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TCR-pHLA Binding Prediction with Sliding Cross-Attention")
    add_args_func(parser)
    args = parser.parse_args()
    main(args)
