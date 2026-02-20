#!/bin/bash
token_amounts="50000000"
CONFIG_FILE="artifacts/DeltaSeccommits_QwenCoder_vulnerable/_training_config_layer0_standard_16384_lr_1e-4.json"
seeds="0 1 2 3 4"

for seed in $seeds; do
    for tokens in $token_amounts; do
        echo "Training with $tokens tokens on layer 11... (seed $seed)"
        sed -e "s/blocks\.[0-9]*\.hook_resid_post/blocks.11.hook_resid_post/" \
            -e "s/\"training_tokens\": [0-9]*/\"training_tokens\": ${tokens}/" \
            $CONFIG_FILE \
            --seed $seed \
            > /tmp/config_layer11_tokens${tokens}_seed${seed}.json
        python code_sae/training.py --config /tmp/config_layer11_tokens${tokens}_seed${seed}.json
    done
done