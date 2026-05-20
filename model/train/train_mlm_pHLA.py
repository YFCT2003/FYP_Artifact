# -*- coding: UTF-8 -*-
"""
@Project：PLM-interact
@File：train_mlm_pHLA.py
@Date：2026/3/13 16:49
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import csv
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import EsmForMaskedLM, EsmTokenizer, get_cosine_schedule_with_warmup
from transformers import DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
from typing import List, Union, Any, Dict

# Import utility functions
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    load_sequences, EarlyStopping, get_device, logger,
    save_checkpoint, load_checkpoint, plot_training_curves, calculate_mlm_accuracy,
    ddp_setup, ddp_cleanup
)


# ==========================================
# 1. pHLA MLM Model Definition
# ==========================================
class PHLAForMLM(nn.Module):
    """
    pHLA Epitope MLM fine-tuning model.
    Built on ESM-2 with custom Dropout injection, only performing Masked Language Modeling on the Epitope region.
    After training, the encoder weights are loaded by PHLA_encoder in Sliding_attention.py for downstream binding prediction tasks.
    """

    def __init__(self, model_name, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1):
        super().__init__()
        self.esm_mlm = EsmForMaskedLM.from_pretrained(
            model_name,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob
        )

    def forward(self, input_ids, attention_mask, labels=None):
        return self.esm_mlm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


# ==========================================
# 2. Custom Epitope Masking DataCollator
# ==========================================
class EpitopeOnlyDataCollator(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Dynamic padding
        batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        token_labels = batch["input_ids"].clone()
        probability_matrix = torch.full(token_labels.shape, 0.0)
        eos_token_id = self.tokenizer.eos_token_id

        for i in range(token_labels.size(0)):
            eos_positions = (token_labels[i] == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) >= 2:
                # Locate Epitope position (content between the last two delimiters)
                start_idx = eos_positions[-2] + 1
                end_idx = eos_positions[-1]
                probability_matrix[i, start_idx:end_idx] = self.mlm_probability

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
            token_labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Ensure each sequence's Epitope region has at least one token masked
        for i in range(masked_indices.size(0)):
            if masked_indices[i].sum() == 0:
                valid_candidates = (probability_matrix[i] > 0).nonzero(as_tuple=True)[0]
                if len(valid_candidates) > 0:
                    random_idx = valid_candidates[torch.randint(0, len(valid_candidates), (1,)).item()]
                    masked_indices[i, random_idx] = True

        # Core: set label to -100 for non-masked tokens (HLA, Pad, and unmasked Epitope)
        token_labels[~masked_indices] = -100

        # 80-10-10 masking strategy
        indices_replaced = torch.bernoulli(torch.full(token_labels.shape, 0.8)).bool() & masked_indices
        batch["input_ids"][indices_replaced] = self.tokenizer.mask_token_id

        indices_random = torch.bernoulli(
            torch.full(token_labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), token_labels.shape, dtype=torch.long)
        batch["input_ids"][indices_random] = random_words[indices_random]

        batch["labels"] = token_labels
        return batch


# ==========================================
# 3. Dataset Definition
# ==========================================
class PHLADataset(Dataset):
    def __init__(self, epitope, hla, tokenizer, max_length=50):
        self.epitope = epitope.values if hasattr(epitope, 'values') else epitope
        self.hla = hla.values if hasattr(hla, 'values') else hla
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.epitope)

    def __getitem__(self, item):
        encoded = self.tokenizer(
            self.hla[item],
            self.epitope[item],
            truncation="longest_first",
            max_length=self.max_length
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }


# ==========================================
# 4. Main Training Process
# ==========================================
def main(args):
    ddp_setup()
    rank = int(os.environ["RANK"])
    device = int(os.environ["LOCAL_RANK"])

    if rank == 0:
        logger.info(f"Starting DDP training on {torch.cuda.device_count()} GPUs.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Folder Setup (only on main process)
    if rank == 0:
        model_short = re.search(r'esm2_t\d+_(\w+)_UR50D', args.model_name).group(1)
        folder_name = f"ESM2_{model_short}_Batch_{args.batch_size_train}_lr_{args.lr}_mlm_{args.mlm_probability}_grad_accum_{args.grad_accumulation_steps}_weight_decay_{args.weight_decay}_dropout_{args.hidden_dropout_prob}"
        experiment_dir = os.path.join(args.weights_root, folder_name)
        checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
        curves_dir = os.path.join(experiment_dir, "curves")
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(curves_dir, exist_ok=True)
        best_model_path = os.path.join(experiment_dir, "best_model.pt")
        history_csv_path = os.path.join(experiment_dir, "training_history.csv")
    else:
        best_model_path = ""  # Only needed for rank 0
        history_csv_path = ""

    # 2. Load Tokenizer and Model
    start_epoch = 0
    train_history = []
    val_history = []

    tokenizer = EsmTokenizer.from_pretrained(args.model_name)

    model = PHLAForMLM(
        args.model_name,
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model = DDP(model, device_ids=[device], find_unused_parameters=True)

    # 3. Data Loading
    df = load_sequences(args.data_path)
    train_df, val_df = train_test_split(df, test_size=args.val_split, random_state=args.seed)

    train_dataset = PHLADataset(train_df["Epitope"], train_df["HLA_sequence"], tokenizer, args.max_length)
    val_dataset = PHLADataset(val_df["Epitope"], val_df["HLA_sequence"], tokenizer, args.max_length)

    mlm_collator = EpitopeOnlyDataCollator(tokenizer=tokenizer, mlm_probability=args.mlm_probability)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size_train, collate_fn=mlm_collator,
                                  num_workers=args.num_workers, sampler=train_sampler, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size_train, collate_fn=mlm_collator,
                                num_workers=args.num_workers, sampler=val_sampler, pin_memory=True)

    # 4. Optimizer, Scheduler, Early Stopping, and AMP
    scaler = torch.amp.GradScaler('cuda')  # Updated to the latest API

    # Fix 2: Correctly calculate total steps considering gradient accumulation
    actual_update_steps_per_epoch = max(1, len(train_dataloader) // args.grad_accumulation_steps)
    total_training_steps = args.epochs * actual_update_steps_per_epoch

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=total_training_steps)

    # Fix 1: Consolidated load_checkpoint logic here so optimizer and scheduler states are correctly restored
    if args.resume_from and os.path.exists(args.resume_from):
        if rank == 0: logger.info(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint_data = load_checkpoint(args.resume_from, model, optimizer=optimizer, scheduler=scheduler,
                                          device=device)
        start_epoch = checkpoint_data['epoch']
        train_history = checkpoint_data.get('train_history', [])
        val_history = checkpoint_data.get('val_history', [])
        if rank == 0: logger.info(f"Resumed from Epoch {start_epoch}")

    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=best_model_path)

    # 5. Training Loop
    if rank == 0:
        logger.info("================ Starting MLM Training with DDP & AMP ================")

        # Fix 3: Use 'a' (append) mode if resuming to prevent overwriting previous CSV logs
        file_mode = 'a' if (args.resume_from and os.path.exists(history_csv_path)) else 'w'
        csv_file = open(history_csv_path, mode=file_mode, newline='')
        csv_writer = csv.writer(csv_file)
        if file_mode == 'w':
            csv_writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for step, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Fix 4: Force a gradient update on the very last step of the epoch to prevent leftover gradients
            is_last_step = (step + 1) == len(train_dataloader)
            is_accumulating = ((step + 1) % args.grad_accumulation_steps != 0) and not is_last_step

            if is_accumulating:
                with model.no_sync():
                    with torch.amp.autocast('cuda'):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss / args.grad_accumulation_steps
                scaler.scale(loss).backward()

            else:
                # Update phase: normal backward, triggers DDP synchronization
                with torch.amp.autocast('cuda'):  # Updated to the latest API
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / args.grad_accumulation_steps
                scaler.scale(loss).backward()

                # ================= MODIFIED =================
                # Gradient clipping and parameter update, and fix the warning about premature scheduler.step() call under AMP
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Record the current scale value
                scale_before = scaler.get_scale()

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Only execute scheduler.step() when the scale has not decreased (indicating the optimizer was not skipped)
                scale_after = scaler.get_scale()
                if scale_before <= scale_after:
                    scheduler.step()
                # ============================================

            with torch.no_grad():
                acc = calculate_mlm_accuracy(outputs.logits, labels)
            train_loss += loss.item() * args.grad_accumulation_steps
            train_acc += acc

            if rank == 0 and (step + 1) % args.log_interval == 0:
                logger.info(
                    f"Epoch [{epoch + 1}/{args.epochs}], Step [{step + 1}/{len(train_dataloader)}], Loss: {loss.item() * args.grad_accumulation_steps:.4f}, Acc: {acc:.4f}")

        avg_train_loss = train_loss / len(train_dataloader)
        avg_train_acc = train_acc / len(train_dataloader)

        # 6. Validation Cycle
        model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad(), torch.amp.autocast('cuda'):
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                val_acc += calculate_mlm_accuracy(outputs.logits, labels)

        # ======== Modified DDP metrics aggregation logic ========
        world_size = torch.distributed.get_world_size()

        # 1. First calculate the current GPU's local average Loss and Acc (divided by the local number of batches)
        local_avg_val_loss = val_loss / len(val_dataloader)
        local_avg_val_acc = val_acc / len(val_dataloader)

        # 2. Convert to Tensor for cross-GPU communication
        val_loss_tensor = torch.tensor(local_avg_val_loss).to(device)
        val_acc_tensor = torch.tensor(local_avg_val_acc).to(device)

        # 3. Sum the local averages from all GPUs
        torch.distributed.all_reduce(val_loss_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(val_acc_tensor, op=torch.distributed.ReduceOp.SUM)

        # 4. Divide by the number of GPUs (world_size) to obtain the true global average
        avg_val_loss = val_loss_tensor.item() / world_size
        avg_val_acc = val_acc_tensor.item() / world_size
        # ==========================================

        if rank == 0:
            train_history.append({'epoch': epoch + 1, 'loss': avg_train_loss, 'accuracy': avg_train_acc})
            val_history.append({'epoch': epoch + 1, 'loss': avg_val_loss, 'accuracy': avg_val_acc})
            csv_writer.writerow([epoch + 1, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc])
            csv_file.flush()

            logger.info(f"==== Epoch {epoch + 1} Summary ===="
                        f" Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")

            if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
                checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                save_checkpoint(epoch + 1, model, optimizer, scheduler, train_history, val_history, args,
                                checkpoint_path)

        # Keep as is, do not modify parameters
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            # logger.warning("Early stopping triggered!")
            break

    if rank == 0:
        csv_file.close()
        logger.info("================ Generating Plots ================")
        plot_training_curves(train_history, val_history, curves_dir, "mlm")
        logger.info(f"Training completed. Best model saved to: {best_model_path}")

    ddp_cleanup()


def add_args_func(parser):
    # Basic Params
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data_path", type=str, default=r"D:\Python\Python_project\FYP\data\HLA_ABC_pos_9_hla_seq.csv",
                        help="Input dataset path")
    parser.add_argument("--weights_root", type=str, default=r"D:\Python\Python_project\FYP\weights",
                        help="Root directory for saving weights")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs')
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument('--batch_size_train', type=int, default=64, help='Training batch size per GPU')
    parser.add_argument('--grad_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='Warmup steps')
    parser.add_argument('--mlm_probability', type=float, default=0.0001, help='MLM masking probability')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation data split ratio')

    # Dropout Parameters
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.15, help='Hidden layer dropout probability')
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.15,
                        help='Attention layer dropout probability')

    # Model & Resource Params
    parser.add_argument('--model_name', type=str, default="facebook/esm2_t6_8M_UR50D",
                        help='Pretrained ESM-2 model name')
    parser.add_argument('--max_length', type=int, default=50, help='Maximum sequence length')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers')

    # Logging & Saving
    parser.add_argument('--log_interval', type=int, default=100, help='Steps between logging')
    parser.add_argument('--save_interval', type=int, default=10, help='Epochs between saving checkpoints')

    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Epitope Masked Language Modeling Training with DDP and AMP")
    add_args_func(parser)
    args = parser.parse_args()
    main(args)