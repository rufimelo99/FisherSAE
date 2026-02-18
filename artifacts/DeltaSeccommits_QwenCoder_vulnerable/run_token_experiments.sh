#!/bin/bash
token_amounts="1000000 5000000 10000000 25000000 50000000 100000000 200000000"
CONFIG_FILE="artifacts/DeltaSeccommits_QwenCoder_vulnerable/_training_config_layer0_standard_16384_lr_1e-4.json"

for tokens in $token_amounts; do
    echo "Training with $tokens tokens on layer 11..."
    sed -e "s/blocks\.[0-9]*\.hook_resid_post/blocks.11.hook_resid_post/" \
        -e "s/\"training_tokens\": [0-9]*/\"training_tokens\": ${tokens}/" \
        $CONFIG_FILE \
        > /tmp/config_layer11_tokens${tokens}.json
    python code_sae/training.py --config /tmp/config_layer11_tokens${tokens}.json
done
