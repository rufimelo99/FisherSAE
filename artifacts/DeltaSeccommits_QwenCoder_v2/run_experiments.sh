#!/bin/bash
layers="0 3 7 11 15 19 23 27"
CONFIG_FILE="artifacts/DeltaSeccommits_QwenCoder_v2/_training_config_layer0_standard_16384_lr_1e-4.json"

for layer in $layers; do
    echo "Training layer $layer..."
    sed "s/blocks\.[0-9]*\.hook_resid_post/blocks.${layer}.hook_resid_post/" \
        artifacts/DeltaSeccommits_QwenCoder_v2/_training_config_layer0_standard_16384_lr_1e-4.json \
        > /tmp/config_layer${layer}.json
    python code_sae/training.py --config /tmp/config_layer${layer}.json
done
