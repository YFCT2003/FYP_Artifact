# TCR-pHLA Binding Prediction using Protein Language Models

A two-stage deep learning framework for predicting TCR-pHLA binding affinity using ESM-2 protein language models and a sliding cross-attention mechanism.

## Project Structure

```
FYP/
├── data/                       # Training and testing datasets
│   ├── pMHC/                   # Epitope-HLA binding data (MLM task)
│   └── pMHC_TCR/               # TCR-pHLA binding data (binding prediction task)
├── weights/                    # Trained model weights (root level)
├── inference_result/           # Inference outputs and visualizations
├── model/                      # Core model code
│   ├── Architecture/           # Model architecture definitions
│   │   └── Sliding_attention.py         # Dual-stream sliding cross-attention model
│   ├── train/                  # Training scripts
│   │   ├── train_mlm_pHLA.py            # Stage 1: Epitope MLM fine-tuning
│   │   └── train_sliding_attention.py   # Stage 2: TCR-pHLA binding training
│   ├── inference/              # Inference and analysis scripts
│   │   ├── inference_mlm.py             # MLM leave-one-out inference (pMHC_TCR)
│   │   ├── pHLA_inference_train.py      # MLM inference for pMHC data
│   │   ├── inference_sliding.py         # Sliding attention binding inference
│   │   ├── Layer33_attention.py         # 33-layer deep attention analysis
│   │   ├── pHLA_attention_matrix.py     # Attention matrix extraction
│   │   └── plot_from_csv.py             # Re-plotting from saved results
│   ├── scripts/                # SLURM job submission scripts (.sh)
│   ├── utils/                  # Shared utility functions
│   │   └── utils.py
│   └── logs/                   # Training logs
└── README.md
```

## Overview

This project implements a two-stage approach for TCR-pHLA binding prediction:

### Stage 1: Epitope-HLA Masked Language Modeling (MLM)

Fine-tune ESM-2 on epitope-HLA paired sequences using masked language modeling. The model learns contextual representations of how epitopes bind to HLA molecules by predicting masked epitope residues conditioned on HLA context.

- **Script**: `model/train/train_mlm_pHLA.py`
- **Architecture**: ESM-2 (with Epitope-only masking strategy)
- **Output**: Fine-tuned pHLA encoder weights

### Stage 2: TCR-pHLA Binding Prediction

A dual-stream sliding cross-attention model that:
1. Encodes pHLA pairs using the fine-tuned ESM-2 (from Stage 1, frozen)
2. Encodes TCR CDR3-beta sequences using a separate pre-trained ESM-2 (frozen)
3. Aligns the two representations via a novel sliding cross-attention mechanism with spatial bias
4. Predicts binding probability through a classification head

- **Script**: `model/train/train_sliding_attention.py`
- **Architecture**: `model/Architecture/Sliding_attention.py`
- **Output**: Binding classifier

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (recommended for training, optional for inference)

### Setup

**Option 1: Conda (recommended)**

```bash
# Clone the repository
git clone 
cd FYP

# Create environment from exported yml
conda env create -f environment.yml
conda activate Biopython
```

**Option 2: pip**

```bash
# Clone the repository
git clone 
cd FYP

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install core dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.5.1 | Deep learning framework (CUDA 11.8) |
| `transformers` | 4.57.3 | HuggingFace ESM-2 models |
| `pandas` | 2.3.3 | Data processing |
| `numpy` | 2.3.5 | Numerical computing |
| `scikit-learn` | 1.7.2 | Evaluation metrics |
| `matplotlib` | 3.9.4 | Plotting |
| `seaborn` | 0.13.2 | Statistical visualization |
| `tqdm` | 4.67.1 | Progress bars |
| `safetensors` | 0.7.0 | Safe tensor serialization |

## Data Formats

The project uses CSV data in two formats:

### pMHC (Epitope-HLA MLM)

Used by Stage 1 MLM training and inference. Located in `data/pMHC/`.

| Column | Description |
|---|---|
| `Epitope` | 9-mer epitope peptide sequence |
| `HLA_sequence` | HLA pseudo-sequence (34 amino acids) |
| `Qualitative Measurement` | Binding status (optional) |

### pMHC_TCR (TCR-pHLA Binding)

Used by Stage 2 binding prediction. Located in `data/pMHC_TCR/`.

| Column | Description |
|---|---|
| `cdr3.beta` | TCR CDR3-beta sequence |
| `antigen.epitope` | 9-mer epitope peptide sequence |
| `mhc.seq` | HLA pseudo-sequence (34 amino acids) |
| `label` | Binary binding label (1 = binding, 0 = non-binding) |

## Usage

### Data Preparation

Place your data in the `data/` directory:

- **For MLM training**: CSV with columns `Epitope`, `HLA_sequence`
- **For binding training**: CSV with columns `antigen.epitope`, `mhc.seq`, `cdr3.beta`, `label`

### Stage 1: MLM Fine-tuning

```bash
cd model
python train/train_mlm_pHLA.py \
    --data_path ../data/pMHC/train.csv \
    --weights_root ../weights \
    --model_name facebook/esm2_t33_650M_UR50D \
    --epochs 100 \
    --batch_size_train 64 \
    --lr 5e-5
```

For multi-GPU training on a cluster:

```bash
cd model/scripts
sbatch train.sh      # GPU4090 queue
sbatch train_mlm_a800.sh  # A800 queue
```

### Stage 2: Binding Prediction Training

```bash
cd model
python train/train_sliding_attention.py \
    --data_path ../data/pMHC_TCR/train.csv \
    --phla_model_path ../weights/ESM2_*/best_model.pt \
    --phla_model_name facebook/esm2_t33_650M_UR50D \
    --tcr_model_name facebook/esm2_t33_650M_UR50D \
    --weights_root ../weights
```

For cluster submission:

```bash
cd model/scripts
sbatch train_sliding.sh
sbatch train_sliding_hparam.sh   # Hyperparameter grid search
```

### Inference

**TCR-pHLA binding prediction:**

```bash
cd model
python inference/inference_sliding.py \
    --data_path ../data/pMHC_TCR/test.csv \
    --checkpoint_path ../weights/SlidingAttn_*/best_model.pt \
    --phla_model_path ../weights/ESM2_*/best_model.pt
```

**MLM epitope recovery analysis (pMHC_TCR data):**

```bash
cd model
python inference/inference_mlm.py \
    --data_path ../data/pMHC_TCR/test.csv \
    --model_path ../weights/ESM2_*/best_model.pt
```

**MLM inference (pMHC data):**

```bash
cd model
python inference/pHLA_inference_train.py \
    --data_path ../data/pMHC/test.csv \
    --model_path ../weights/ESM2_*/best_model.pt \
    --model_name facebook/esm2_t33_650M_UR50D
```

**Attention analysis:**

```bash
cd model
python inference/pHLA_attention_matrix.py \
    --data_path ../data/pMHC_TCR/test.csv \
    --model_path ../weights/ESM2_*/best_model.pt
```

**33-layer deep attention analysis:**

```bash
cd model
python inference/Layer33_attention.py \
    --data_path ../data/pMHC/test.csv \
    --model_path ../weights/ESM2_*/best_model.pt \
    --model_name facebook/esm2_t33_650M_UR50D \
    --target_layer 33
```

**Re-plot from saved results:**

```bash
cd model
python inference/plot_from_csv.py \
    --csv_path ../inference_result/<dataset>/inference_results.csv \
    --output_dir ../inference_result/<dataset>
```

### Cluster Submission

All SLURM scripts are in `model/scripts/`. Submit with:

```bash
cd model/scripts
sbatch train.sh
sbatch train_sliding.sh
sbatch inference_layer33.sh
```

## Model Architecture

### Sliding Cross-Attention

The core architecture (`model/Architecture/Sliding_attention.py`) consists of:

1. **pHLA Encoder** — Fine-tuned ESM-2 (frozen) for HLA-epitope pair encoding
2. **TCR Encoder** — Pre-trained ESM-2 (frozen) for CDR3-beta encoding
3. **Sliding Cross-Attention** — Dual-stream mechanism:
   - Forward stream: TCR queries pHLA
   - Reverse stream: pHLA queries TCR
   - Spatial bias: Gaussian kernel with iterative position refinement
4. **Classification Head** — MLP on pooled representations

## Outputs

- **Training**: Model checkpoints and training curves saved to `weights/`
- **Inference**: Prediction CSVs and metrics saved to `inference_result/`
- **Attention analysis**: Heatmaps and `.npy` attention matrices saved to `inference_result/`

## License

Academic research use.
