#!/bin/bash
#SBATCH --job-name=sliding_attn
#SBATCH --partition=gpua800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --output=logs/sliding_attn_%j.out
#SBATCH --error=logs/sliding_attn_%j.err

set -e

ml anaconda3
source activate Biopython

# ---- Paths (modify as needed) ----
PROJECT_ROOT="/gpfs/work/bio/yuchongyan22/PythonProject/FYP"
DATA_PATH="${PROJECT_ROOT}/data/tc-hard/train_9_all.csv"
PHLA_MODEL_PATH="${PROJECT_ROOT}/model/weights/ESM2_650M_Batch_64_lr_5e-05_mlm_0.0001_grad_accum_1_weight_decay_0.05_dropout_0.25/best_model.pt"  # Stage 1 weights
WEIGHT_ROOT="${PROJECT_ROOT}/model/weights"

mkdir -p logs

# ---- Training ----
python ../train/train_sliding_attention.py \
    --data_path "$DATA_PATH" \
    --phla_model_path "$PHLA_MODEL_PATH" \
    --phla_model_name "facebook/esm2_t33_650M_UR50D" \
    --tcr_model_name "facebook/esm2_t33_650M_UR50D" \
    --weights_root "$WEIGHT_ROOT" \
    --epochs 50 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --batch_size_train 512 \
    --batch_size_val 512 \
    --grad_accumulation_steps 1 \
    --warmup_steps 1000 \
    --patience 5 \
    --val_split 0.1 \
    --hard_split_cols antigen.epitope \
    --d_model 512 \
    --n_heads 8 \
    --n_iter 3 \
    --sigma 1.0 \
    --classifier_dropout 0.4 \
    --focal_gamma 0 \
    --seed 42 \
    --num_workers 2 \
    --log_interval 100 \
    --save_interval 10
