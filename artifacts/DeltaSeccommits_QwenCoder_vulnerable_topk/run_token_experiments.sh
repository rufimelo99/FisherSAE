#!/bin/bash
# Token scaling study for the TopK SAE at layer 11.
# Mirrors the standard-config scaling study but with the new architecture.
# Run this before committing to 100M tokens — check if saturation occurs earlier.

set -euo pipefail

token_amounts="1000000 5000000 10000000 25000000 50000000 100000000"
CONFIG_FILE="artifacts/DeltaSeccommits_QwenCoder_vulnerable_topk/_training_config_layer0_topk_28672.json"

for tokens in $token_amounts; do
    echo "===== TopK SAE — layer 11 — ${tokens} tokens ====="
    sed -e "s/blocks\.[0-9]*\.hook_resid_post/blocks.11.hook_resid_post/" \
        -e "s/\"training_tokens\": [0-9]*/\"training_tokens\": ${tokens}/" \
        "$CONFIG_FILE" \
        > /tmp/topk_config_layer11_tokens${tokens}.json
    python code_sae/training.py --config /tmp/topk_config_layer11_tokens${tokens}.json
    echo "===== Done — ${tokens} tokens ====="
done
