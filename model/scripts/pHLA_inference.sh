#!/bin/bash

#SBATCH --job-name=pMHC_inference          # job name
#SBATCH --partition=gpu4090                # gpu4090 partition
#SBATCH --qos=4gpus                        # gpu4090 QOS
#SBATCH --nodes=1                          # number of nodes
#SBATCH --ntasks-per-node=1                # processes per node
#SBATCH --cpus-per-task=4                  # CPU cores per task
#SBATCH --gres=gpu:1                       # request 1 GPU
#SBATCH --output=inference.out             # standard output
#SBATCH --error=inference.err              # error output

source activate Biopython

# ========== Common Paths ==========
BASE="/gpfs/work/bio/yuchongyan22/PythonProject/FYP"
DATA_PATH="${BASE}/data/train_data_inference/train_A0201_pMHC.csv"
MODEL_NAME="facebook/esm2_t33_650M_UR50D"
MODEL_PATH="${BASE}/model/weights/ESM2_650M_Batch_64_lr_5e-05_mlm_0.0001_grad_accum_1_weight_decay_0.05_dropout_0.25/best_model.pt"
OUTPUT_DIR="${BASE}/inference_result"

mkdir -p "${OUTPUT_DIR}"

# ========== MLM Inference + Attention Analysis (merged in one script) ==========
python ../inference/pHLA_inference_train.py \
  --data_path "${DATA_PATH}" \
  --model_path "${MODEL_PATH}" \
  --model_name "${MODEL_NAME}" \
  --batch_size 64 \
  --max_length 50 \
  --output_dir "${OUTPUT_DIR}"

echo "===== All Done ====="
