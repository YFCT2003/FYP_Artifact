#!/bin/bash
#SBATCH --job-name=sliding_hparam
#SBATCH --partition=gpua800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --output=hparam_%j.out
#SBATCH --error=hparam_%j.err

set -e

ml anaconda3
source activate Biopython

# ---- Fixed paths ----
PROJECT_ROOT="/gpfs/work/bio/yuchongyan22/PythonProject/FYP"
DATA_PATH="${PROJECT_ROOT}/data/tc-hard/train_9_all.csv"
PHLA_MODEL_PATH="${PROJECT_ROOT}/model/weights/ESM2_650M_Batch_64_lr_5e-05_mlm_0.0001_grad_accum_1_weight_decay_0.05_dropout_0.25/best_model.pt"
WEIGHT_ROOT="${PROJECT_ROOT}/model/weights"

mkdir -p logs

# ---- Search space (modify as needed) ----
LR_LIST=(5e-4 5e-5)
D_MODEL_LIST=(512 1280)
SIGMA_LIST=(1.0 2.0)

TOTAL=$(( ${#LR_LIST[@]} * ${#D_MODEL_LIST[@]} * ${#SIGMA_LIST[@]} ))
CURRENT=0

echo "============================================"
echo " Hyperparameter Grid Search"
echo " lr:      ${LR_LIST[*]}"
echo " d_model: ${D_MODEL_LIST[*]}"
echo " sigma:   ${SIGMA_LIST[*]}"
echo " d_ff:    auto (4 × d_model)"
echo " Total combinations: ${TOTAL}"
echo "============================================"

for LR in "${LR_LIST[@]}"; do
  for D_MODEL in "${D_MODEL_LIST[@]}"; do
    for SIGMA in "${SIGMA_LIST[@]}"; do

    # d_model must be divisible by n_heads=8
    if (( D_MODEL % 8 != 0 )); then
      echo "[SKIP] d_model=${D_MODEL} not divisible by 8"
      continue
    fi

    CURRENT=$((CURRENT + 1))
    echo ""
    echo "===== [${CURRENT}/${TOTAL}] lr=${LR} d_model=${D_MODEL} sigma=${SIGMA} ====="

    python ../train/train_sliding_attention.py \
        --data_path "$DATA_PATH" \
        --phla_model_path "$PHLA_MODEL_PATH" \
        --phla_model_name "facebook/esm2_t33_650M_UR50D" \
        --tcr_model_name "facebook/esm2_t33_650M_UR50D" \
        --weights_root "$WEIGHT_ROOT" \
        --epochs 50 \
        --lr "$LR" \
        --weight_decay 0.01 \
        --batch_size_train 512 \
        --batch_size_val 512 \
        --grad_accumulation_steps 1 \
        --warmup_steps 1000 \
        --patience 5 \
        --val_split 0.1 \
        --hard_split_cols antigen.epitope \
        --d_model "$D_MODEL" \
        --n_heads 8 \
        --n_iter 3 \
        --sigma "$SIGMA" \
        --classifier_dropout 0.4 \
        --focal_gamma 0 \
        --seed 42 \
        --num_workers 2 \
        --log_interval 100 \
        --save_interval 10

    echo "===== [${CURRENT}/${TOTAL}] Done ====="
    done
  done
done

echo ""
echo "============================================"
echo " Grid search complete: ${CURRENT}/${TOTAL} runs"
echo " Results saved under ${WEIGHT_ROOT}/SlidingAttn_*"
echo "============================================"
