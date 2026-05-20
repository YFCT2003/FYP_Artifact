#!/bin/bash

#SBATCH --job-name=pMHC_MLM             # job name
#SBATCH --partition=gpu4090	        # gpu4090 partition
#SBATCH --qos=4gpus                # gpu4090 QOS
#SBATCH --nodes=1                  # number of nodes
#SBATCH --ntasks-per-node=1        # processes per node
#SBATCH --cpus-per-task=6          # CPU cores per task
#SBATCH --gres=gpu:4               # request 4 GPUs
#SBATCH --output=train.out         # standard output
#SBATCH --error=train.err          # error output

source ~/miniconda3/bin/activate Biopython

python ../train/train_mlm_pHLA.py \
  --data_path "/gpfs/work/bio/yuchongyan22/PythonProject/FYP/data/HLA_ABC_pos_9_hla_seq.csv" \
  --weights_root "/gpfs/work/bio/yuchongyan22/PythonProject/FYP/model/weights/" \
  --model_name "facebook/esm2_t33_650M_UR50D"