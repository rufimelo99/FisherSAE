#!/bin/bash
# Tokenize DeltaSecommits with Llama-2-13b-chat tokenizer for SAE training
# This uses SAELens pretokenize_runner which properly chunks sequences to context_size=100

set -e

CONFIG_FILE="artifacts/DeltaSecommits_Llama2-13B_vulnerable/_pretokenize_config_DeltaSecommits_Llama2_13b.json"

echo "Pretokenizing DeltaSecommits dataset with Llama-2-13b-chat tokenizer..."
echo "Config: $CONFIG_FILE"
echo ""

python -m sae_lens.pretokenize_runner --config "$CONFIG_FILE"

echo ""
echo "✓ Pretokenization complete!"
echo "Dataset saved to HuggingFace Hub: TQRG/DeltaSecommits_llama-2-13b-chat_tokenized_v2_vulnerable"
