#!/bin/bash

# SAE releases to evaluate (different training token amounts)
SAE_RELEASES=(
    "rufimelo/vulnerable_code_qwen_coder_standard_16384_1M"
    # "rufimelo/vulnerable_code_qwen_coder_standard_16384_5M"
    # "rufimelo/vulnerable_code_qwen_coder_standard_16384_10M"
    # "rufimelo/vulnerable_code_qwen_coder_standard_16384_25M"
    # "rufimelo/vulnerable_code_qwen_coder_standard_16384_50M"
    # "rufimelo/vulnerable_code_qwen_coder_standard_16384_100M"
    # "rufimelo/vulnerable_code_qwen_coder_standard_16384_200M"
)

SAE_ID="blocks.11.hook_resid_post"
CONFIG_FILE="artifacts/DeltaSeccommits_QwenCoder_vulnerable/inference_config.json"

for sae_release in "${SAE_RELEASES[@]}"; do
    echo "=============================================="
    echo "Running inference for: $sae_release"
    echo "=============================================="

    python -m code_sae.inference \
        --config "$CONFIG_FILE" \
        --kwargs "{\"sae_release\": \"$sae_release\", \"sae_id\": \"$SAE_ID\", \"wandb_run_name\": \"$sae_release\"}"

    echo "Completed: $sae_release"
    echo ""
done

echo "All experiments completed!"