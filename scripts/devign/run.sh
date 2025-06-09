#!/bin/bash

#SBATCH --job-name=code_sae_devign
#SBATCH --mem=30G

#SBATCH --gres=shard:6
#SBATCH --time=300:00:00
#SBATCH --mincpus=1
#SBATCH --mail-type=all
#SBATCH --mail-user=rufimelo99@gmail.com
#SBATCH --output=/cfs/home/u021521/CodeSAE/logs/slurm-%x-%j.out
#SBATCH --error=/cfs/home/u021521/CodeSAE/logs/slurm-%x-%j.err

# Prepare Environment
source activate /cfs/home/u021521/anaconda3/envs/code_sae/
echo "Submitting job"

export WANDB_MODE=offline
BASE_DIR=/cfs/home/u021521/CodeSAE


python $BASE_DIR/code_sae/training.py \
  --config $BASE_DIR/scripts/devign/layer_0_residual_mid_gpt2_gated_fisher.json

python $BASE_DIR/code_sae/training.py \
  --config $BASE_DIR/scripts/devign/layer_5_residual_mid_gpt2_gated_fisher.json

python $BASE_DIR/code_sae/training.py \
  --config $BASE_DIR/scripts/devign/layer_10_residual_mid_gpt2_gated_fisher.json

wandb sync wandb/offline-run-*