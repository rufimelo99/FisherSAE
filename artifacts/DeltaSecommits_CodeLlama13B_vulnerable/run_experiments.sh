#!/bin/bash
# Train SAEs on codellama/CodeLlama-13b-Instruct-hf across all 8 target layers.
# CodeLlama-13b has 40 layers total; layers [0,3,7,11,15,19,23,27] cover the
# same absolute positions used for Qwen2.5-7B, enabling direct cross-model comparison.
#
# Before running: tokenize DeltaSecommits with the CodeLlama tokenizer using
#   _tokenization_config_DeltaSecommits_CodeLlama13B.json
# and push to TQRG/DeltaSecommits_codellama-13b-instruct_tokenized_v2_vulnerable
#
# GPU: ~26GB bfloat16 (autocast_lm=true) — fits on 1× A100 40GB

layers="0 3 7 11 15 19 23 27"
layers="15"
CONFIG_FILE="artifacts/DeltaSecommits_CodeLlama13B_vulnerable/_training_config_layer0_standard_16384_lr_1e-4.json"

for layer in $layers; do
    echo "Training layer $layer..."
    sed "s/blocks\.[0-9]*\.hook_resid_post/blocks.${layer}.hook_resid_post/" \
        "$CONFIG_FILE" \
        > /tmp/config_codellama13b_layer${layer}.json
    python code_sae/training.py --config /tmp/config_codellama13b_layer${layer}.json
done
