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

# export WANDB_MODE=offline
BASE_DIR=/cfs/home/u021521/CodeSAE

BASE_CONFIG=(
  $BASE_DIR/scripts/base_residual_mid_gpt2.json
)

CONFIG_DIR=$BASE_DIR/scripts/generated_configs
mkdir -p $CONFIG_DIR

HOOK_POINTS=("hook_resid_mid" "hook_mlp_out" "hook_resid_pre")
USE_FISHER_OPTIONS=("true" "false")
HOOK_LAYERS=(0 5 10)
ARCHITECTURES=("gated" "jumprelu" "topk" "standard")

# Paired project names and dataset paths
WANDB_PROJECTS=("Devign" "TinyStories" "TheStackPython" "TheStackJava")
DATASET_PATHS=("TQRG/devign_gpt2_tokenized" "apollo-research/roneneldan-TinyStories-tokenizer-gpt2" "TQRG/bigcode_stack_dedup_python_gpt2_tokenized" "TQRG/reset23_stack_java_gpt2_tokenized")

counter=0

# Iterate over index to pair wandb_project and dataset_path
for i in "${!WANDB_PROJECTS[@]}"; do
  wandb_project="${WANDB_PROJECTS[$i]}"
  dataset_path="${DATASET_PATHS[$i]}"

  for use_fisher in "${USE_FISHER_OPTIONS[@]}"; do
    for hook_layer in "${HOOK_LAYERS[@]}"; do
      for hook_point in "${HOOK_POINTS[@]}"; do
        for arch in "${ARCHITECTURES[@]}"; do
          config_name="config_${counter}.json"
          config_path="$CONFIG_DIR/$config_name"

          jq \
            --argjson use_fisher $use_fisher \
            --argjson hook_layer $hook_layer \
            --arg hook_point "$hook_point" \
            --arg arch "$arch" \
            --arg wandb_project "$wandb_project" \
            --arg dataset_path "$dataset_path" \
            '
            .use_fisher = $use_fisher |
            .hook_layer = $hook_layer |
            .hook_name = ("blocks." + ($hook_layer|tostring) + "." + $hook_point) |
            .architecture = $arch |
            .wandb_project = $wandb_project |
            .dataset_path = $dataset_path |
            .run_name = ("layer_" + ($hook_layer|tostring) + "_" + $hook_point + "_gpt2_" + $arch + "_fisher_" + ($use_fisher|tostring) + "_project_" + $wandb_project)
            ' "${BASE_CONFIG[0]}" > "$config_path"

          echo "Running config $config_name"
          python $BASE_DIR/code_sae/training.py --config "$config_path"

          counter=$((counter + 1))
        done
      done
    done
  done
done
