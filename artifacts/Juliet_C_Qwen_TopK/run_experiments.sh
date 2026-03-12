#!/bin/bash
# Train TopK SAEs (k=64, 4.57x expansion, layer_norm) on all 8 sampled layers.
# Dataset: Juliet C/C++ 1.3 — vulnerable C functions (NIST SARD)
# Architecture mirrors DeltaSeccommits TopK run:
#   type:                   topk
#   d_sae:                  16384       (4.57x expansion over d_in=3584)
#   k:                      64          (guaranteed sparsity)
#   normalize_activations:  layer_norm
#   lr:                     2e-4
#   lr_warm_up_steps:       500
#   training_tokens:        50M

set -euo pipefail

layers="0 3 7 11 15 19 23 27"
CONFIG_FILE="artifacts/Juliet_C_Qwen_TopK/_training_config_layer0_topk_16384_lr_2e-4.json"

for layer in $layers; do
    echo "===== Training TopK SAE — Juliet C — layer ${layer} ====="
    sed "s/blocks\.[0-9]*\.hook_resid_post/blocks.${layer}.hook_resid_post/" \
        "$CONFIG_FILE" \
        > /tmp/juliet_c_topk_layer${layer}.json
    python -m code_sae.training --config /tmp/juliet_c_topk_layer${layer}.json
    echo "===== Done — layer ${layer} ====="
done
