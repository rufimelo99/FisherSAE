#!/bin/bash
# Train SAEs on Qwen/Qwen2.5-Coder-14B-Instruct across all 8 target layers.
# Qwen2.5-Coder-14B has 48 layers total; layers [0,3,7,11,15,19,23,27] use the
# same absolute positions as the primary Qwen2.5-7B experiments for direct comparison.
# Note: Qwen2.5-Coder-14B uses the same tokenizer as Qwen2.5-7B-Instruct —
# check whether the existing TQRG tokenized dataset is reusable before re-tokenizing.
#
# GPU: ~28GB bfloat16 (autocast_lm=true) — fits on 1× A100 40GB

layers="0 3 7 11 15 19 23 27"
layers="15"
CONFIG_FILE="artifacts/DeltaSecommits_Qwen2.5-14B_vulnerable/_training_config_layer0_standard_16384_lr_1e-4.json"

for layer in $layers; do
    echo "Training layer $layer..."
    sed "s/blocks\.[0-9]*\.hook_resid_post/blocks.${layer}.hook_resid_post/" \
        "$CONFIG_FILE" \
        > /tmp/config_qwencoder14b_layer${layer}.json
    python code_sae/training.py --config /tmp/config_qwencoder14b_layer${layer}.json
done
