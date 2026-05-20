# -*- coding: UTF-8 -*-
"""
@Project：FYP
@File：inference_mlm_33layers.py

Leave-One-Out Inference + 33-Layer Deep Attention Analysis
Extract and save all 33 layers of Attention, explore the real Contact Heads!
"""

import os
import argparse
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import EsmForMaskedLM, EsmTokenizer
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 0. Global Plot Style
# ==========================================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.format": "png",  # Batch plotting in png is faster; important charts can be manually changed to svg
})
sns.set_theme(style="ticks")
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
})


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
            self.hla[item], self.epitope[item],
            truncation="longest_first", max_length=self.max_length
        )
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "epitope_seq": self.epitope[item],
            "hla_seq": self.hla[item]
        }


def locate_regions(input_ids, eos_token_id):
    eos_positions = (input_ids == eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) < 2: return None
    return (1, eos_positions[-2].item()), (eos_positions[-2].item() + 1, eos_positions[-1].item())


def plot_attention_heatmap(matrix, x_labels, y_labels, title, save_path):
    fig, ax = plt.subplots(figsize=(max(8, len(x_labels) * 0.4), max(6, len(y_labels) * 0.25)))
    sns.heatmap(matrix, xticklabels=x_labels, yticklabels=y_labels,
                cmap='Reds', square=False, cbar_kws={'label': 'Accumulative Attention'}, ax=ax)
    ax.set_title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_all_heads_grid(layer_matrix, x_labels, y_labels, layer_idx, save_path):
    """Plot 4x5 grid layout, showing 20 Heads of a given layer"""
    fig, axes = plt.subplots(4, 5, figsize=(25, 16))
    axes = axes.flatten()
    for head_idx in range(20):
        sns.heatmap(layer_matrix[:, head_idx, :], cmap='Reds', cbar=False, ax=axes[head_idx])
        axes[head_idx].set_title(f"Layer {layer_idx + 1} - Head {head_idx + 1}")
        axes[head_idx].set_xticks([])
        axes[head_idx].set_yticks([])
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


# ==========================================
# Main
# ==========================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create multi-level output directories
    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
    base_out_dir = os.path.join(args.output_dir, dataset_name)
    layer_heatmaps_dir = os.path.join(base_out_dir, "layer_heatmaps_1_to_33")
    os.makedirs(layer_heatmaps_dir, exist_ok=True)

    tokenizer = EsmTokenizer.from_pretrained(args.model_name)
    standard_aas = list("ACDEFGHIKLMNPQRSTVWY")
    aa_to_id = {aa: tokenizer.get_vocab()[aa] for aa in standard_aas}
    eos_token_id = tokenizer.eos_token_id

    print(f"Loading 33-Layer model from {args.model_path}...")
    model = EsmForMaskedLM.from_pretrained(args.model_name)
    checkpoint = torch.load(args.model_path, map_location="cpu")
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    state_dict = {k.replace('module.', '').replace('esm_mlm.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    df = pd.read_csv(args.data_path)
    df = df[df["Epitope"].str.len() == 9].reset_index(drop=True)
    test_dataloader = DataLoader(PHLADataset(df["Epitope"], df["HLA_sequence"], tokenizer, args.max_length),
                                 batch_size=args.batch_size, shuffle=False)

    # ==========================================
    # Ultimate data structure: [Epi_pos(9), Layer(33), Head(20), HLA_pos(34)]
    # ==========================================
    hla_len = len(df["HLA_sequence"].iloc[0])
    num_layers = model.config.num_hidden_layers  # ESM-2 650M has 33 layers
    num_heads = model.config.num_attention_heads  # 20 heads per layer

    # Core tensor
    ultimate_attn = np.zeros((9, num_layers, num_heads, hla_len), dtype=np.float64)
    attn_count = 0

    print("Starting Deep Inference & Full Attention Extraction...")
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            for pos in range(9):
                masked_input_ids = input_ids.clone()
                valid_mask, mask_indices = [], []

                for i in range(input_ids.size(0)):
                    eos_positions = (input_ids[i] == eos_token_id).nonzero(as_tuple=True)[0]
                    if len(eos_positions) >= 2 and (eos_positions[-1] - (eos_positions[-2] + 1)) == 9:
                        target_idx = eos_positions[-2] + 1 + pos
                        masked_input_ids[i, target_idx] = tokenizer.mask_token_id
                        valid_mask.append(True)
                        mask_indices.append(target_idx.item())
                    else:
                        valid_mask.append(False)
                        mask_indices.append(-1)

                if not any(valid_mask): continue

                outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask, output_attentions=True)

                # Iterate over all 33 layers
                for layer_idx in range(num_layers):
                    layer_attn = outputs.attentions[layer_idx]  # [Batch, 20, SeqLen, SeqLen]

                    for i, is_valid in enumerate(valid_mask):
                        if is_valid:
                            regions = locate_regions(input_ids[i], eos_token_id)
                            if regions:
                                (hla_s, hla_e), (epi_s, epi_e) = regions
                                if (hla_e - hla_s) == hla_len:
                                    # Extract [20_heads, HLA_len]
                                    attn_vector = layer_attn[i, :, mask_indices[i], hla_s:hla_e].cpu().numpy()
                                    ultimate_attn[pos, layer_idx, :, :] += attn_vector

                                    if pos == 0 and layer_idx == 0:
                                        attn_count += 1

    print(f"\nAccumulative attention extracted over {attn_count} valid samples.")

    # ==========================================
    # Ultimate data saving and visualization
    # ==========================================
    # 1. Save this all-encompassing [9, 33, 20, 34] numpy matrix
    np.save(os.path.join(base_out_dir, "ultimate_attention_all_layers_heads.npy"), ultimate_attn)
    print("Saved Ultimate Tensor (9x33x20x34) -> ultimate_attention_all_layers_heads.npy")

    hla_labels = [f"{aa}{i + 1}" for i, aa in enumerate(df["HLA_sequence"].iloc[0])]
    epi_labels = [f"P{i + 1}" for i in range(9)]

    # 2. Iterate over 33 layers: save per-layer mean matrix and plot 33 heatmaps
    print("Generating 33 layer heatmaps...")
    for layer_idx in tqdm(range(num_layers), desc="Plotting Layers"):
        # Take the mean across 20 heads for this layer: shape -> [9, 34]
        layer_mean_matrix = ultimate_attn[:, layer_idx, :, :].mean(axis=1)

        # Save the single-layer matrix
        np.save(os.path.join(layer_heatmaps_dir, f"layer_{layer_idx + 1}_mean_attn.npy"), layer_mean_matrix)

        # Plot
        plot_attention_heatmap(
            layer_mean_matrix, hla_labels, epi_labels,
            f"Layer {layer_idx + 1} Mean Attention: Epitope → HLA",
            os.path.join(layer_heatmaps_dir, f"layer_{layer_idx + 1}_heatmap.png")
        )

    # 3. Plot 20-head grid for the specified layer
    target_layer_idx = args.target_layer - 1  # User provides 1-indexed, convert to 0-indexed
    target_matrix = ultimate_attn[:, target_layer_idx, :, :]  # [9, 20, 34]

    plot_all_heads_grid(
        target_matrix, hla_labels, epi_labels, target_layer_idx,
        os.path.join(base_out_dir, f"layer_{args.target_layer}_all_heads_explorer.png")
    )
    print(f"Per-head grid saved for Layer {args.target_layer}.")

    print("\nAll visualizations and .npy files saved successfully!")
    print(f"Check directory: {base_out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default=r"D:\Python\Python_project\FYP\data\train_val_test_for_pMHC\test_A0201_pMHC.csv")
    parser.add_argument("--model_path", type=str, required=True)
    # Note: the default is changed to the 33-layer 650M model
    parser.add_argument('--model_name', type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument('--batch_size', type=int, default=32)  # Large model may need smaller batch_size to prevent OOM
    parser.add_argument('--max_length', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default=r"D:\Python\Python_project\FYP\inference_result")
    parser.add_argument('--target_layer', type=int, default=33,
                        help="Layer to visualize per-head grid (1-indexed, default: last layer)")

    args = parser.parse_args()
    main(args)
