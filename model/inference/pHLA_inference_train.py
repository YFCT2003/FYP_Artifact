# -*- coding: UTF-8 -*-
"""
@Project：FYP
@File：inference_mlm.py
@Date：2026/3/20 15:01

Leave-One-Out Inference + Attention Analysis for Epitope-HLA MLM
1. Per-position mask inference, calculate recovery accuracy / loss, plot dual-Y-axis chart
2. Extract self-attention from unmasked input, plot HLA↔Epitope heatmap
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
    # --- Font ---
    # Nature Methods requires sans-serif font; Arial / Helvetica preferred
    "font.family":          "Arial",
    "font.size":            11,
    "axes.titlesize":       13,
    "axes.labelsize":       12,
    "xtick.labelsize":      11,
    "ytick.labelsize":      11,
    "legend.fontsize":      10,
    "legend.title_fontsize":11,

    # --- Line & Tick ---
    "axes.linewidth":       0.75,    # Axis line width
    "lines.linewidth":      1.0,
    "patch.linewidth":      0.5,
    "xtick.major.width":    0.75,
    "ytick.major.width":    0.75,
    "xtick.minor.width":    0.5,
    "ytick.minor.width":    0.5,
    "xtick.major.size":     3.0,
    "ytick.major.size":     3.0,
    "xtick.minor.size":     1.5,
    "ytick.minor.size":     1.5,
    "xtick.direction":      "out",
    "ytick.direction":      "out",

    # --- Spines ---
    "axes.spines.top":      False,
    "axes.spines.right":    False,

    # --- Legend ---
    "legend.frameon":       False,   # Remove legend border

    # --- Figure & Save ---
    # Single column 89mm ≈ 3.5in, double column 183mm ≈ 7.2in
    "figure.dpi":           150,     # Screen preview
    "savefig.dpi":          1200,    # Journal requires line art 600, halftone 300, combination 600
    "savefig.bbox":         "tight",
    "savefig.format":       "svg",   # Vector format, preferred for poster/PPT editing; specify extension at call site when PNG is needed
})
sns.set_theme(style="ticks")
plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ==========================================
# 1. Dataset Definition
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
# 2. Helper: Locate HLA / Epitope regions
# ==========================================
def locate_regions(input_ids, eos_token_id):
    """
    Returns the index range of HLA and Epitope in a single sample (left-inclusive, right-exclusive, excluding special tokens).
    Layout: [CLS] HLA... [EOS] Epitope... [EOS]
    """
    eos_positions = (input_ids == eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) < 2:
        return None
    hla_start = 1
    hla_end = eos_positions[-2].item()
    epi_start = hla_end + 1
    epi_end = eos_positions[-1].item()
    return (hla_start, hla_end), (epi_start, epi_end)


# ==========================================
# 3. Visualization: Accuracy & Loss
# ==========================================
def plot_position_metrics(pos_correct, pos_loss_sum, pos_count, save_path):
    """Plot dual-Y-axis chart: bar = Accuracy (left), line = Loss (right)"""
    positions = list(range(1, 10))
    accuracies = []
    losses = []
    for pos in range(9):
        if pos_count[pos] > 0:
            accuracies.append(pos_correct[pos] / pos_count[pos])
            losses.append(pos_loss_sum[pos] / pos_count[pos])
        else:
            accuracies.append(0.0)
            losses.append(0.0)

    spacing = 0.5
    x_pos = [spacing * i for i in range(1, 10)]
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 6))

    bars = ax1.bar(x_pos, accuracies, color='#4C72B0', alpha=0.8, width=width, label='Accuracy')
    ax1.set_xlabel('Epitope Position')
    ax1.set_ylabel('Accuracy', color='#4C72B0')
    ax1.tick_params(axis='y', labelcolor='#4C72B0')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(positions)
    ax1.set_xlim(0.2, x_pos[-1] + 0.5)
    ax1.set_ylim(0, max(accuracies) * 1.25 if max(accuracies) > 0 else 1.0)

    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{acc:.3f}', ha='center', va='bottom', color='#4C72B0')

    ax2 = ax1.twinx()
    ax2.plot(x_pos, losses, color='#C44E52', marker='o', linewidth=2, markersize=7, label='Loss')
    ax2.set_ylabel('Average Loss', color='#C44E52')
    ax2.tick_params(axis='y', labelcolor='#C44E52')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Per-Position Epitope Recovery: Accuracy & Loss')
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to: {save_path}")


# ==========================================
# 4. Visualization: Attention Heatmap
# ==========================================
def plot_aa_preference(pos_aa_prob_sum, standard_aas, save_path):
    """X-axis: Epitope P1-P9, Y-axis: 20 amino acids, value: cumulative predicted probability"""
    matrix = np.array([
        [pos_aa_prob_sum[pos][aa] for pos in range(9)]
        for aa in standard_aas
    ])  # [20, 9]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, xticklabels=[str(i+1) for i in range(9)],
                yticklabels=standard_aas, cmap='Reds', square=True,
                cbar_kws={'label': 'Accumulative Predicted Probability', 'pad': 0.02}, ax=ax)
    ax.set_xlabel("Epitope Position")
    ax.set_ylabel("Amino Acid")
    ax.set_title("HLA-A*02:01")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"AA preference heatmap saved to: {save_path}")


def plot_attention_heatmap(matrix, x_labels, y_labels, xlabel, ylabel, title, save_path):
    fig, ax = plt.subplots(figsize=(max(8, len(x_labels) * 0.4), max(6, len(y_labels) * 0.25)))
    sns.heatmap(matrix, xticklabels=x_labels, yticklabels=y_labels,
                cmap='Reds', square=False, cbar_kws={'label': 'Accumulative Attention Score', 'pad': 0.01}, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=90)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Heatmap saved to: {save_path}")


# ==========================================
# 5. Main
# ==========================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create subfolder using the dataset filename to avoid results from different datasets overwriting each other
    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
    output_dir = os.path.join(args.output_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Tokenizer & Standard AAs
    tokenizer = EsmTokenizer.from_pretrained(args.model_name)
    standard_aas = list("ACDEFGHIKLMNPQRSTVWY")
    aa_to_id = {aa: tokenizer.get_vocab()[aa] for aa in standard_aas}
    eos_token_id = tokenizer.eos_token_id

    # 2. Load Model
    print(f"Loading model from {args.model_path}...")
    model = EsmForMaskedLM.from_pretrained(args.model_name)

    checkpoint = torch.load(args.model_path, map_location="cpu")
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    state_dict = {k.replace('module.', '').replace('esm_mlm.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. Load Test Data
    print(f"Loading test data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    df = df[df["Epitope"].str.len() == 9].reset_index(drop=True)
    print(f"Total 9-mer samples: {len(df)}")

    test_dataset = PHLADataset(df["Epitope"], df["HLA_sequence"], tokenizer, args.max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 4. Tracking
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    pos_correct = {pos: 0 for pos in range(9)}
    pos_loss_sum = {pos: 0.0 for pos in range(9)}
    pos_count = {pos: 0 for pos in range(9)}
    # Amino acid preference accumulation: pos -> aa -> cumulative probability
    pos_aa_prob_sum = {pos: {aa: 0.0 for aa in standard_aas} for pos in range(9)}
    results_list = []

    # Attention accumulation matrix
    hla_len = len(df["HLA_sequence"].iloc[0])
    epi_len = 9
    attn_epi_to_hla = np.zeros((epi_len, hla_len), dtype=np.float64)
    attn_count = 0

    # 5. Inference Loop
    print("Starting Inference + Attention Extraction...")
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            epitopes = batch['epitope_seq']
            hlas = batch['hla_seq']

            # Per-position mask forward: inference evaluation + extract attention from [MASK] positions to HLA
            for pos in range(9):
                masked_input_ids = input_ids.clone()
                labels = torch.full_like(input_ids, -100)
                valid_mask = []
                mask_indices = []

                for i in range(input_ids.size(0)):
                    eos_positions = (input_ids[i] == eos_token_id).nonzero(as_tuple=True)[0]
                    if len(eos_positions) >= 2:
                        start_idx = eos_positions[-2] + 1
                        end_idx = eos_positions[-1]
                        if (end_idx - start_idx) == 9:
                            target_idx = start_idx + pos
                            labels[i, target_idx] = input_ids[i, target_idx]
                            masked_input_ids[i, target_idx] = tokenizer.mask_token_id
                            valid_mask.append(True)
                            mask_indices.append(target_idx.item())
                        else:
                            valid_mask.append(False)
                            mask_indices.append(-1)
                    else:
                        valid_mask.append(False)
                        mask_indices.append(-1)

                if not any(valid_mask):
                    continue

                outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask,
                                output_attentions=True)
                logits = outputs.logits
                last_layer_attn = outputs.attentions[-1].mean(dim=1)  # [B, S, S]

                active_loss = labels.view(-1) != -100
                active_logits = logits.view(-1, model.config.vocab_size)[active_loss]
                active_labels = labels.view(-1)[active_loss]

                losses = loss_fct(active_logits, active_labels)
                probs = F.softmax(active_logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                active_idx = 0
                for i, is_valid in enumerate(valid_mask):
                    if is_valid:
                        true_label = active_labels[active_idx].item()
                        pred_label = preds[active_idx].item()
                        is_correct = (true_label == pred_label)
                        loss_val = losses[active_idx].item()
                        aa_probs = {aa: float(probs[active_idx, aa_to_id[aa]].item()) for aa in standard_aas}

                        pos_correct[pos] += int(is_correct)
                        pos_loss_sum[pos] += loss_val
                        pos_count[pos] += 1
                        for aa in standard_aas:
                            pos_aa_prob_sum[pos][aa] += aa_probs[aa]

                        # Extract attention from [MASK] position to HLA region
                        regions = locate_regions(input_ids[i], eos_token_id)
                        if regions is not None:
                            (hla_s, hla_e), (epi_s, epi_e) = regions
                            if (hla_e - hla_s) == hla_len and (epi_e - epi_s) == epi_len:
                                attn_vector = last_layer_attn[i, mask_indices[i], hla_s:hla_e].cpu().numpy()
                                attn_epi_to_hla[pos, :] += attn_vector
                                if pos == 0:
                                    attn_count += 1

                        results_list.append({
                            "HLA": hlas[i],
                            "Epitope": epitopes[i],
                            "Masked_Position": pos + 1,
                            "True_AA": tokenizer.decode([true_label]),
                            "Pred_AA": tokenizer.decode([pred_label]),
                            "True_AA_Prob": aa_probs.get(tokenizer.decode([true_label]).strip(), 0.0),
                            "Pred_AA_Prob": float(probs[active_idx, pred_label].item()),
                            "Is_Correct": is_correct,
                            "Loss": loss_val,
                            "All_AA_Probs": json.dumps(aa_probs)
                        })
                        active_idx += 1

    # ==========================================
    # 6. Save & Print Inference Results
    # ==========================================
    output_csv = os.path.join(output_dir, "inference_results.csv")
    pd.DataFrame(results_list).to_csv(output_csv, index=False)

    print("\n" + "=" * 40)
    print(" INFERENCE SUMMARY")
    print("=" * 40)

    total_correct = sum(pos_correct.values())
    total_samples = sum(pos_count.values())
    overall_acc = total_correct / total_samples if total_samples > 0 else 0
    print(f"Overall Prediction Accuracy: {overall_acc:.4f} ({total_correct}/{total_samples})")
    print("-" * 40)

    for pos in range(9):
        if pos_count[pos] > 0:
            acc = pos_correct[pos] / pos_count[pos]
            avg_loss = pos_loss_sum[pos] / pos_count[pos]
            print(f"Position {pos + 1}: Accuracy = {acc:.4f}, Average Loss = {avg_loss:.4f}")
        else:
            print(f"Position {pos + 1}: No data.")
    print("=" * 40)

    # ==========================================
    # 7. Plot: Accuracy & Loss
    # ==========================================
    plot_position_metrics(pos_correct, pos_loss_sum, pos_count,
                          os.path.join(output_dir, "position_accuracy_loss.png"))

    # ==========================================
    # 8. Plot: AA Preference Heatmap
    # ==========================================
    plot_aa_preference(pos_aa_prob_sum, standard_aas,
                       os.path.join(output_dir, "aa_preference.png"))

    # ==========================================
    # 9. Plot: Attention Heatmaps
    # ==========================================
    if attn_count > 0:
        print(f"\nAccumulative attention over {attn_count} samples")

        # Save accumulation matrix; can be directly loaded for replotting next time without needing to re-run inference
        np.save(os.path.join(output_dir, "attn_epi_to_hla.npy"), attn_epi_to_hla)

        hla_seq = df["HLA_sequence"].iloc[0]
        hla_labels = [f"{aa}{i+1}" for i, aa in enumerate(hla_seq)]
        epi_labels = [f"P{i+1}" for i in range(epi_len)]

        plot_attention_heatmap(
            attn_epi_to_hla, hla_labels, epi_labels,
            "HLA Pseudo-sequence Position", "Epitope Position",
            "Accumulative Attention: Epitope → HLA (Last Layer)",
            os.path.join(output_dir, "attention_epitope_to_hla.png"))

    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Leave-One-Out Inference + Attention Analysis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", type=str,
                        default=r"D:\Python\Python_project\FYP\data\train_val_test_for_pMHC\test_A0201_pMHC.csv")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to best_model.pt or checkpoint_epoch_X.pt")
    parser.add_argument('--model_name', type=str, default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_length', type=int, default=50)
    parser.add_argument('--output_dir', type=str,
                        default=r"D:\Python\Python_project\FYP\inference_result")

    args = parser.parse_args()
    main(args)
