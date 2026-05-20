#!/bin/bash

#SBATCH --job-name=pMHC_MLM             # job name
#SBATCH --partition=gpua8001t	        # gpua800 partition
#SBATCH --qos=4gpus                # gpua800 QOS
#SBATCH --nodes=1                  # number of nodes
#SBATCH --ntasks-per-node=1        # processes per node
#SBATCH --cpus-per-task=24         # Edit 1: 4GPU x 6 threads = 24 CPU cores
#SBATCH --gres=gpu:4               # Edit 2: request 4 GPUs (was incorrectly commented)
#SBATCH --output=train.out         # standard output
#SBATCH --error=train.err          # error output

ml anaconda3

source activate Biopython

# each GPU process uses 6 CPU threads
export OMP_NUM_THREADS=6            # Edit 3: explicitly set 6 threads per GPU instead of using SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6

NUM_GPUS=4

DATA_PATH="/gpfs/work/bio/yuchongyan22/PythonProject/FYP/data/HLA_ABC_pos_9_hla_seq.csv"
WEIGHT_ROOT="/gpfs/work/bio/yuchongyan22/PythonProject/FYP/model/weights/"

torchrun \
  --nproc-per-node=$NUM_GPUS \
  ../train/train_mlm_pHLA.py \
  --data_path "$DATA_PATH" \
  --weights_root "$WEIGHT_ROOT" \
  --model_name "facebook/esm2_t33_650M_UR50D" \
  --lr 3e-4 \
  --warmup_steps 1500 \
  --weight_decay 0.05 \
  --hidden_dropout_prob 0.1 \
  --attention_probs_dropout_prob 0.1 \
  --num_workers 0 \
  --batch_size_train 512 \
  --grad_accumulation_steps 1 \
  --mlm_probability 0.0001
