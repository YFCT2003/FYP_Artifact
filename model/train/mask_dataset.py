# -*- coding: UTF-8 -*-
"""
@Project：FYP
@File：mask_dataset.py
@Date：2026/3/19 20:22
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
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
# 1. Custom Epitope Masking DataCollator
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
# 2. Dataset Definition
# ==========================================
class PHLADataset(Dataset):
    def __init__(self, epitope, hla, tokenizer, max_length=1603):
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

model_name = "facebook/esm2_t6_8M_UR50D"
data_path = r"D:\Python\Python_project\FYP\data\HLA_ABC_pos_9_hla_seq.csv"

tokenizer = EsmTokenizer.from_pretrained(model_name)
model = EsmForMaskedLM.from_pretrained(model_name)

df = load_sequences(data_path)
train_df, val_df = train_test_split(df[:100], test_size=0.1, random_state=42)

train_dataset = PHLADataset(train_df["Epitope"], train_df["HLA_sequence"], tokenizer, 50)
val_dataset = PHLADataset(val_df["Epitope"], val_df["HLA_sequence"], tokenizer, 50)

train_dataset = PHLADataset(train_df["Epitope"], train_df["HLA_sequence"], tokenizer, 50)
val_dataset = PHLADataset(val_df["Epitope"], val_df["HLA_sequence"], tokenizer, 50)

mlm_collator = EpitopeOnlyDataCollator(tokenizer=tokenizer, mlm_probability=0.22)

train_dataloader = DataLoader(train_dataset, batch_size=16, collate_fn=mlm_collator)
val_dataloader = DataLoader(val_dataset, batch_size=16, collate_fn=mlm_collator)

for item in train_dataloader:
    print(item)

print()