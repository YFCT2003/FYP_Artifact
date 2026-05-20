# -*- coding: UTF-8 -*-
"""
@Project：FYP
@File：plot_from_csv.py
@Date：2026/4/18

Replot all figures directly from the existing inference_results.csv without needing to re-run inference.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 0. Global Plot Style
# ==========================================
# Step 1: Call Seaborn to set the base style theme
sns.set_theme(style="ticks")

# Step 2: Apply custom settings uniformly to override Seaborn defaults
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
STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")


# ==========================================
# 1. Plot: Accuracy & Loss
# ==========================================
def plot_position_metrics(df, save_path):
    positions = list(range(1, 10))
    accuracies, losses = [], []
    for pos in positions:
        sub = df[df["Masked_Position"] == pos]
        accuracies.append(sub["Is_Correct"].mean() if len(sub) > 0 else 0.0)
        losses.append(sub["Loss"].mean() if len(sub) > 0 else 0.0)

    spacing = 0.5
    x_pos = [spacing * i for i in range(1, 10)]
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 6))

    bars = ax1.bar(x_pos, accuracies, color='#C377E4', alpha=0.8, width=width, label='Accuracy')
    ax1.set_xlabel('Epitope Position')
    ax1.set_ylabel('Accuracy', color='#C377E4')
    ax1.tick_params(axis='y', labelcolor='#C377E4')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(positions)
    ax1.set_xlim(0.2, x_pos[-1] + 0.5)
    ax1.set_ylim(0, max(accuracies) * 1.25 if max(accuracies) > 0 else 1.0)

    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{acc:.3f}', ha='center', va='bottom', color='#C377E4')

    ax2 = ax1.twinx()
    ax2.plot(x_pos, losses, color='#D52A58', marker='o', linewidth=2, markersize=5, label='Loss')
    ax2.set_ylabel('Average Loss', color='#D52A58')
    ax2.tick_params(axis='y', labelcolor='#D52A58')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Epitope Recovery')
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


# ==========================================
# 2. Plot: AA Preference Heatmap
# ==========================================
def plot_aa_preference(df, save_path):
    matrix = np.zeros((len(STANDARD_AAS), 9))

    for pos_idx, pos in enumerate(range(1, 10)):
        sub = df[df["Masked_Position"] == pos]
        for row in sub["All_AA_Probs"]:
            aa_probs = json.loads(row)
            for aa_idx, aa in enumerate(STANDARD_AAS):
                matrix[aa_idx, pos_idx] += aa_probs.get(aa, 0.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, xticklabels=[str(i+1) for i in range(9)],
                yticklabels=STANDARD_AAS, cmap='Reds', square=True,
                cbar_kws={'label': 'Predicted Probability', 'pad': 0.02}, ax=ax)
    ax.set_xlabel("Epitope Position")
    ax.set_ylabel("Amino Acid")
    ax.set_title("HLA-A*02:01")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


# ==========================================
# 3. Plot: Attention Heatmap
# ==========================================
# Real PDB numbering for standard HLA-I 34-mer
HLA_REAL_POSITIONS = [
    7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80,
    81, 84, 95, 97, 99, 114, 116, 118, 143, 147, 150, 152, 156,
    158, 159, 163, 167, 171
]

# ==========================================
# 3. Plot: Attention Heatmap (update only the mapping part)
# ==========================================
def plot_attention_heatmap(attn_matrix, hla_seq, save_path):
    # Modify here: replace i+1 with HLA_REAL_POSITIONS[i]
    hla_labels = [f"{aa}{HLA_REAL_POSITIONS[i]}" for i, aa in enumerate(hla_seq)]
    epi_labels = [f"P{i+1}" for i in range(9)]

    # The following parameters fully preserve the original code
    fig, ax = plt.subplots(figsize=(max(8, len(hla_labels) * 0.4), 4))
    sns.heatmap(attn_matrix, xticklabels=hla_labels, yticklabels=epi_labels,
                cmap='Reds', square=False,
                cbar_kws={'label': 'Attention Score', 'pad': 0.01}, ax=ax)
    ax.set_xlabel("HLA Pseudo-sequence Position")
    ax.set_ylabel("Epitope Position")
    ax.set_title("Attention: Epitope → HLA")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


# ==========================================
# 4. Main
# ==========================================
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.csv_path}...")
    df = pd.read_csv(args.csv_path)

    plot_position_metrics(df, os.path.join(args.output_dir, "position_accuracy_loss.svg"))
    plot_aa_preference(df, os.path.join(args.output_dir, "aa_preference.svg"))

    # If attn_epi_to_hla.npy exists, directly load and replot
    npy_path = os.path.join(args.output_dir, "layer_30_mean_attn.npy")
    if os.path.exists(npy_path):
        attn_matrix = np.load(npy_path)
        hla_seq = df["HLA"].iloc[0]  # Use the HLA sequence recorded in the CSV
        plot_attention_heatmap(attn_matrix, hla_seq,
                               os.path.join(args.output_dir, "attention_epitope_to_hla.svg"))
    else:
        print(f"No attn_epi_to_hla.npy found in {args.output_dir}, skipping attention heatmap.")

    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Re-plot from inference_results.csv")
    parser.add_argument("--csv_path", type=str, default=r"D:\Python\Python_project\FYP\inference_result\train_A0201_pMHC\inference_results.csv",
                        help="Path to inference_results.csv")
    parser.add_argument("--output_dir", type=str, default=r"D:\Python\Python_project\FYP\inference_result\train_A0201_pMHC",
                        help="Directory to save plots")
    args = parser.parse_args()
    main(args)
