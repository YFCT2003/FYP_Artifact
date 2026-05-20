# -*- coding: UTF-8 -*-
"""
@Project：FYP
@File：pHLA_attention_matrix.py
@Date：2026/4/15 15:08

Extract self-attention from the fine-tuned ESM-2 model, focusing on the HLA ↔ Epitope interaction regions,
average over multiple samples, then plot and save heatmaps.

Input layout after tokenizer encoding:
  [CLS] HLA_residues... [EOS] Epitope_residues... [EOS]

We extract the last layer attention, average over all heads,
and crop the HLA(34) × Epitope(9) sub-matrix to plot heatmaps.
"""

import os
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import EsmForMaskedLM, EsmTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'


# ==========================================
# 1. Dataset
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
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "epitope_seq": self.epitope[item],
            "hla_seq": self.hla[item]
        }


# ==========================================
# 2. Locate the indices of HLA / Epitope in the token sequence
# ==========================================
def locate_regions(input_ids, eos_token_id):
    """
    Returns the index range of HLA and Epitope residues in a single sample (excluding special tokens).
    Layout: [CLS] HLA... [EOS] Epitope... [EOS]
    Returns:
        hla_range:     (start, end)  left-inclusive, right-exclusive
        epitope_range: (start, end)  left-inclusive, right-exclusive
        Returns None if localization fails
    """
    eos_positions = (input_ids == eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) < 2:
        return None
    # HLA: from index 1 (skip CLS) to the first EOS
    hla_start = 1
    hla_end = eos_positions[-2].item()
    # Epitope: from first EOS+1 to the second EOS
    epi_start = hla_end + 1
    epi_end = eos_positions[-1].item()
    return (hla_start, hla_end), (epi_start, epi_end)


# ==========================================
# 3. Main
# ==========================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Tokenizer & Model
    tokenizer = EsmTokenizer.from_pretrained(args.model_name)

    print(f"Loading model from {args.model_path}...")
    model = EsmForMaskedLM.from_pretrained(args.model_name)

    checkpoint = torch.load(args.model_path, map_location="cpu")
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    state_dict = {k.replace('module.', '').replace('esm_mlm.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 2. Load Data
    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    df = df[df["antigen.epitope"].str.len() == 9].reset_index(drop=True)

    # Limit number of samples to control computational cost
    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)
    print(f"Using {len(df)} samples for attention analysis")

    dataset = PHLADataset(df["antigen.epitope"], df["mhc.seq"], tokenizer, args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    eos_token_id = tokenizer.eos_token_id

    # Read HLA pseudo-sequence length (all samples share the same HLA allele)
    hla_len = len(df["mhc.seq"].iloc[0])
    epi_len = 9

    # 3. Accumulate attention matrices
    # (a) HLA→Epitope: [hla_len, epi_len]
    attn_hla_to_epi = np.zeros((hla_len, epi_len), dtype=np.float64)
    # (b) Epitope→HLA: [epi_len, hla_len]
    attn_epi_to_hla = np.zeros((epi_len, hla_len), dtype=np.float64)
    # (c) Epitope→Epitope: [epi_len, epi_len]
    attn_epi_to_epi = np.zeros((epi_len, epi_len), dtype=np.float64)
    count = 0

    print("Extracting attention matrices...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Attention"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            output_attentions=True)
            # attentions: tuple of (B, n_heads, seq_len, seq_len) per layer
            # Take the last layer, average over all heads
            last_layer_attn = outputs.attentions[-1]  # [B, H, S, S]
            avg_attn = last_layer_attn.mean(dim=1)     # [B, S, S]

            for i in range(input_ids.size(0)):
                regions = locate_regions(input_ids[i], eos_token_id)
                if regions is None:
                    continue
                (hla_s, hla_e), (epi_s, epi_e) = regions
                cur_hla_len = hla_e - hla_s
                cur_epi_len = epi_e - epi_s

                if cur_hla_len != hla_len or cur_epi_len != epi_len:
                    continue

                attn_mat = avg_attn[i].cpu().numpy()  # [S, S]

                # HLA→Epitope (query=HLA, key=Epitope)
                attn_hla_to_epi += attn_mat[hla_s:hla_e, epi_s:epi_e]
                # Epitope→HLA
                attn_epi_to_hla += attn_mat[epi_s:epi_e, hla_s:hla_e]
                # Epitope→Epitope
                attn_epi_to_epi += attn_mat[epi_s:epi_e, epi_s:epi_e]
                count += 1

    if count == 0:
        print("No valid samples found!")
        return

    attn_hla_to_epi /= count
    attn_epi_to_hla /= count
    attn_epi_to_epi /= count
    print(f"Averaged over {count} samples")

    # 4. Plot heatmaps
    hla_seq = df["mhc.seq"].iloc[0]
    hla_labels = [f"{aa}{i+1}" for i, aa in enumerate(hla_seq)]
    epi_labels = [f"P{i+1}" for i in range(epi_len)]

    # (a) HLA → Epitope attention heatmap
    _plot_heatmap(
        attn_hla_to_epi,
        x_labels=epi_labels, y_labels=hla_labels,
        xlabel="Epitope Position", ylabel="HLA Pseudo-sequence Position",
        title="Average Attention: HLA → Epitope (Last Layer)",
        save_path=os.path.join(args.output_dir, "attention_hla_to_epitope.png")
    )

    # (b) Epitope → HLA attention heatmap
    _plot_heatmap(
        attn_epi_to_hla,
        x_labels=hla_labels, y_labels=epi_labels,
        xlabel="HLA Pseudo-sequence Position", ylabel="Epitope Position",
        title="Average Attention: Epitope → HLA (Last Layer)",
        save_path=os.path.join(args.output_dir, "attention_epitope_to_hla.png")
    )

    # (c) Epitope → Epitope attention heatmap
    _plot_heatmap(
        attn_epi_to_epi,
        x_labels=epi_labels, y_labels=epi_labels,
        xlabel="Epitope Position (Key)", ylabel="Epitope Position (Query)",
        title="Average Attention: Epitope ↔ Epitope (Last Layer)",
        save_path=os.path.join(args.output_dir, "attention_epitope_self.png")
    )


def _plot_heatmap(matrix, x_labels, y_labels, xlabel, ylabel, title, save_path):
    """Generic heatmap plotting function"""
    fig, ax = plt.subplots(figsize=(max(8, len(x_labels) * 0.4), max(6, len(y_labels) * 0.25)))
    im = ax.imshow(matrix, cmap='magma', aspect='auto', interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Attention Score', fontsize=11)

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="pHLA Attention Matrix Analysis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", type=str,
                        default=r"D:\Python\Python_project\FYP\data\train_val_test_for_pMHC\test_A0201_pMHC.csv")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to best_model.pt or checkpoint_epoch_X.pt")
    parser.add_argument('--model_name', type=str, default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_length', type=int, default=50)
    parser.add_argument('--max_samples', type=int, default=1000,
                        help="Max samples for attention analysis (None = use all)")
    parser.add_argument('--output_dir', type=str,
                        default=r"D:\Python\Python_project\FYP\inference_result")

    args = parser.parse_args()
    main(args)
