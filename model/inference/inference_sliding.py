"""
@Project: FYP
@File: inference_sliding.py

Inference script for TCR-pHLA binding prediction with the Sliding Cross-Attention model.
Loads a trained PLMSlidingModel checkpoint and runs evaluation on test data.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import EsmTokenizer
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Architecture import PLMSlidingModel


class TCRpHLADataset(Dataset):
    """TCR-pHLA binding dataset for inference."""
    def __init__(self, df, tokenizer, phla_max_length=50, tcr_max_length=35):
        self.epitope = df['antigen.epitope'].values
        self.hla_seq = df['mhc.seq'].values
        self.cdr3_beta = df['cdr3.beta'].values
        self.labels = df['label'].astype(int).values if 'label' in df.columns else None
        self.tokenizer = tokenizer
        self.phla_max_length = phla_max_length
        self.tcr_max_length = tcr_max_length

    def __len__(self):
        return len(self.cdr3_beta)

    def __getitem__(self, idx):
        phla_encoded = self.tokenizer(
            self.hla_seq[idx], self.epitope[idx],
            truncation="longest_first", max_length=self.phla_max_length
        )
        tcr_encoded = self.tokenizer(
            self.cdr3_beta[idx],
            truncation=True, max_length=self.tcr_max_length
        )
        item = {
            'phla_input_ids': phla_encoded['input_ids'],
            'phla_attention_mask': phla_encoded['attention_mask'],
            'tcr_input_ids': tcr_encoded['input_ids'],
            'tcr_attention_mask': tcr_encoded['attention_mask'],
        }
        if self.labels is not None:
            item['label'] = self.labels[idx]
        return item


class TCRpHLACollator:
    """Dynamic padding collator for TCR-pHLA batches."""
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

        phla_batch = self.tokenizer.pad(phla_inputs, return_tensors='pt')
        tcr_batch = self.tokenizer.pad(tcr_inputs, return_tensors='pt')

        result = {
            'phla_input_ids': phla_batch['input_ids'],
            'phla_attention_mask': phla_batch['attention_mask'],
            'tcr_input_ids': tcr_batch['input_ids'],
            'tcr_attention_mask': tcr_batch['attention_mask'],
        }
        if 'label' in batch[0]:
            result['labels'] = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        return result


def load_model_checkpoint(model, checkpoint_path, device):
    """Load model weights from a checkpoint or state dict."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    return model


def compute_metrics(labels, probs, preds):
    """Compute classification metrics."""
    n_unique = len(np.unique(labels))
    return {
        'accuracy': accuracy_score(labels, preds),
        'auroc': roc_auc_score(labels, probs) if n_unique > 1 else float('nan'),
        'auprc': average_precision_score(labels, probs) if n_unique > 1 else float('nan'),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load Tokenizer ----
    tokenizer = EsmTokenizer.from_pretrained(args.tcr_model_name)

    # ---- Load Model ----
    print(f"Loading model from {args.checkpoint_path} ...")
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
        freeze_phla=True,
        freeze_tcr=True,
        classifier_dropout=0.0,
    ).to(device)
    model.eval()

    model = load_model_checkpoint(model, args.checkpoint_path, device)
    print("Model loaded successfully.")

    # ---- Load Data ----
    print(f"Loading data from {args.data_path} ...")
    df = pd.read_csv(args.data_path)

    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)
    print(f"Total samples: {len(df)}")

    dataset = TCRpHLADataset(df, tokenizer, args.phla_max_length, args.tcr_max_length)
    collator = TCRpHLACollator(tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collator,
        num_workers=args.num_workers, shuffle=False, pin_memory=True
    )

    # ---- Inference ----
    print("Running inference ...")
    all_logits = []
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for batch in dataloader:
            phla_input_ids = batch['phla_input_ids'].to(device)
            phla_attention_mask = batch['phla_attention_mask'].to(device)
            tcr_input_ids = batch['tcr_input_ids'].to(device)
            tcr_attention_mask = batch['tcr_attention_mask'].to(device)

            logits, _ = model(phla_input_ids, phla_attention_mask,
                              tcr_input_ids, tcr_attention_mask)

            all_logits.append(logits.cpu())
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs[:, 1].cpu())
            all_preds.append(logits.argmax(dim=-1).cpu())

            if 'labels' in batch:
                all_labels.append(batch['labels'].cpu())

    logits_np = torch.cat(all_logits).numpy()
    probs_np = torch.cat(all_probs).numpy()
    preds_np = torch.cat(all_preds).numpy()

    # ---- Build Results DataFrame ----
    results_df = df.copy()
    results_df['pred_label'] = preds_np
    results_df['binding_probability'] = probs_np
    results_df['logits_0'] = logits_np[:, 0]
    results_df['logits_1'] = logits_np[:, 1]

    os.makedirs(args.output_dir, exist_ok=True)
    output_csv = os.path.join(args.output_dir, args.output_name)
    results_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to: {output_csv}")

    # ---- Metrics (if labels available) ----
    if all_labels:
        labels_np = torch.cat(all_labels).numpy()
        metrics = compute_metrics(labels_np, probs_np, preds_np)

        print("\n" + "=" * 45)
        print(" EVALUATION SUMMARY")
        print("=" * 45)
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  AUROC:     {metrics['auroc']:.4f}")
        print(f"  AUPRC:     {metrics['auprc']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print("=" * 45)

        cm = confusion_matrix(labels_np, preds_np)
        print("\nConfusion Matrix:")
        print(f"  TN: {cm[0, 0]:4d}  FP: {cm[0, 1]:4d}")
        print(f"  FN: {cm[1, 0]:4d}  TP: {cm[1, 1]:4d}")

        # Save metrics to a text file
        metrics_path = os.path.join(args.output_dir, "metrics_summary.txt")
        with open(metrics_path, 'w') as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v:.6f}\n")
            f.write("\nConfusion Matrix:\n")
            f.write(f"TN: {cm[0, 0]}  FP: {cm[0, 1]}\n")
            f.write(f"FN: {cm[1, 0]}  TP: {cm[1, 1]}\n")
        print(f"\nMetrics saved to: {metrics_path}")
    else:
        print("\nNo labels found in data. Predictions saved without evaluation.")

    print("\nInference complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TCR-pHLA Binding Inference with Sliding Cross-Attention")

    # Data
    parser.add_argument("--data_path", type=str,
                        default="../../data/pMHC_TCR/test.csv",
                        help="Path to test CSV (columns: antigen.epitope, mhc.seq, cdr3.beta, label)")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--output_dir", type=str,
                        default="../../inference_result/sliding",
                        help="Directory to save predictions")
    parser.add_argument("--output_name", type=str, default="sliding_predictions.csv",
                        help="Output CSV filename")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples for inference")

    # Model architecture (must match the trained checkpoint)
    parser.add_argument("--phla_model_path", type=str, required=True,
                        help="Path to fine-tuned pHLA ESM weights")
    parser.add_argument("--phla_model_name", type=str,
                        default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--tcr_model_name", type=str,
                        default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--L_Q", type=int, default=23)
    parser.add_argument("--L_K", type=int, default=43)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--window_T", type=float, default=None)
    parser.add_argument("--n_iter", type=int, default=3)

    # Tokenization
    parser.add_argument("--phla_max_length", type=int, default=50)
    parser.add_argument("--tcr_max_length", type=int, default=35)

    # Resources
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
