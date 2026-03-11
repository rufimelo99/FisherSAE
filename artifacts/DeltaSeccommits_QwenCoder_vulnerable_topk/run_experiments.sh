#!/bin/bash
# Train TopK SAEs (k=64, 8x expansion, layer_norm) on all 8 sampled layers.
# Architecture: TopK with aux loss — replaces ReLU+L1 from the standard run.
# Key differences vs. standard config:
#   type:                   topk        (was: standard)
#   d_sae:                  28672       (was: 16384  — 8x vs 4.57x expansion)
#   k:                      64          (guaranteed sparsity; replaces λ=1.0 L1)
#   aux_loss_coefficient:   0.03125     (dead-feature prevention; replaces resampling only)
#   normalize_activations:  layer_norm  (was: none)
#   lr:                     2e-4        (was: 3e-5)
#   lr_warm_up_steps:       500         (was: 0)
#   training_tokens:        100M        (was: 50M)

set -euo pipefail

layers="0 3 7 11 15 19 23 27"
CONFIG_FILE="artifacts/DeltaSeccommits_QwenCoder_vulnerable_topk/_training_config_layer0_topk_28672.json"

for layer in $layers; do
    echo "===== Training TopK SAE — layer ${layer} ====="
    sed "s/blocks\.[0-9]*\.hook_resid_post/blocks.${layer}.hook_resid_post/" \
        "$CONFIG_FILE" \
        > /tmp/topk_config_layer${layer}.json
    python code_sae/training.py --config /tmp/topk_config_layer${layer}.json
    echo "===== Done — layer ${layer} ====="
done
