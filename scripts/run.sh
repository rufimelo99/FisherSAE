#!/bin/bash
echo "Submitting job"

BASE_DIR=${PWD}

BASE_CONFIG=(
  $BASE_DIR/scripts/base_residual_mid_gpt2.json
)

CONFIG_DIR=$BASE_DIR/scripts/generated_configs
mkdir -p $CONFIG_DIR

USE_FISHER_OPTIONS=("true" "false")
HOOK_LAYERS=(0 5 10)
ARCHITECTURES=("gated" "jumprelu" "topk" "standard")

# Paired project names and dataset paths
WANDB_PROJECTS=("Devign" "TinyStories")
DATASET_PATHS=("TQRG/devign_gpt2_tokenized" "apollo-research/roneneldan-TinyStories-tokenizer-gpt2")

counter=0

# Iterate over index to pair wandb_project and dataset_path
for i in "${!WANDB_PROJECTS[@]}"; do
  wandb_project="${WANDB_PROJECTS[$i]}"
  dataset_path="${DATASET_PATHS[$i]}"

  for use_fisher in "${USE_FISHER_OPTIONS[@]}"; do
    for hook_layer in "${HOOK_LAYERS[@]}"; do
      for arch in "${ARCHITECTURES[@]}"; do
        config_name="config_${counter}.json"
        config_path="$CONFIG_DIR/$config_name"

        jq \
          --argjson use_fisher $use_fisher \
          --argjson hook_layer $hook_layer \
          --arg arch "$arch" \
          --arg wandb_project "$wandb_project" \
          --arg dataset_path "$dataset_path" \
          '
          .use_fisher = $use_fisher |
          .hook_layer = $hook_layer |
          .hook_name = ("blocks." + ($hook_layer|tostring) + ".hook_resid_mid") |
          .architecture = $arch |
          .wandb_project = $wandb_project |
          .dataset_path = $dataset_path |
          .run_name = ("layer_" + ($hook_layer|tostring) + "_residual_mid_gpt2_" + $arch + "_fisher_" + ($use_fisher|tostring) + "_project_" + $wandb_project)
          ' "${BASE_CONFIG[0]}" > "$config_path"

        echo "Running config $config_name"
        # python $BASE_DIR/code_sae/training.py --config "$config_path"

        counter=$((counter + 1))
      done
    done
  done
done
