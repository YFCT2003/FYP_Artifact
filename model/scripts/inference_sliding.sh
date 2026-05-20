#!/bin/bash
#SBATCH --job-name=sliding_inference
#SBATCH --partition=gpu4090
#SBATCH --qos=4gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=sliding_inference.out
#SBATCH --error=sliding_inference.err

source activate Biopython

PROJECT_ROOT="/gpfs/work/bio/yuchongyan22/PythonProject/FYP"
DATA_PATH="${PROJECT_ROOT}/data/pMHC_TCR/test.csv"
MODEL_PATH="${PROJECT_ROOT}/model/weights/SlidingAttn_d512_h8_iter3_lr_0.0005_bs_512_drop_0.4_sigma_1.0_focal_gamma_0.0_wd_0.01/best_model.pt"
PHLA_MODEL_PATH="${PROJECT_ROOT}/model/weights/ESM2_650M_Batch_64_lr_5e-05_mlm_0.0001_grad_accum_1_weight_decay_0.05_dropout_0.25/best_model.pt"
OUTPUT_DIR="${PROJECT_ROOT}/inference_result/sliding"

mkdir -p "${OUTPUT_DIR}"

python ../inference/inference_sliding.py \
  --data_path "${DATA_PATH}" \
  --checkpoint_path "${MODEL_PATH}" \
  --phla_model_path "${PHLA_MODEL_PATH}" \
  --phla_model_name "facebook/esm2_t33_650M_UR50D" \
  --tcr_model_name "facebook/esm2_t33_650M_UR50D" \
  --d_model 512 \
  --n_heads 8 \
  --n_iter 3 \
  --sigma 1.0 \
  --batch_size 128 \
  --output_dir "${OUTPUT_DIR}"

echo "===== Sliding Inference Complete ====="
