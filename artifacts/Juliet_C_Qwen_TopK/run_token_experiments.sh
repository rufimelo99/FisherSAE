#!/bin/bash
# Token scaling study for TopK SAE on Juliet C (vulnerable) across layers 0, 14, 27.
# Sweeps token budgets to find saturation point before committing to a full run.

set -euo pipefail

token_amounts="100000000"
LAYERS="11"
BASE_CONFIG="artifacts/Juliet_C_Qwen_TopK/_training_config_layer0_topk_16384_lr_2e-4.json"

for layer in $LAYERS; do
    for tokens in $token_amounts; do
        echo "===== Juliet-C TopK — layer ${layer} — ${tokens} tokens ====="
        tmp_cfg="/tmp/juliet_c_topk_layer${layer}_tokens${tokens}.json"
        sed -e "s/blocks\.[0-9]*\.hook_resid_post/blocks.${layer}.hook_resid_post/" \
            -e "s/\"training_tokens\": [0-9]*/\"training_tokens\": ${tokens}/" \
            "$BASE_CONFIG" \
            > "$tmp_cfg"
        python -m code_sae.training --config "$tmp_cfg"
        echo "===== Done — layer ${layer} — ${tokens} tokens ====="
    done
done
