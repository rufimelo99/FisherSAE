#!/bin/bash
# Train SAEs on Llama-2-13B across multiple layers
# Layers: 0, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39

set -e

LAYERS="0 3 7 11 15 19 23 27 31 35 39"
CONFIG_DIR="artifacts/DeltaSecommits_Llama2-13B_vulnerable"

echo "========================================"
echo "Training Llama-2-13B SAEs"
echo "Layers: $LAYERS"
echo "========================================"
echo ""

for layer in $LAYERS; do
    CONFIG_FILE="$CONFIG_DIR/_training_config_layer${layer}_standard_16384_lr_1e-4.json"

    if [ ! -f "$CONFIG_FILE" ]; then
        echo "⚠ Config not found: $CONFIG_FILE"
        continue
    fi

    echo "========================================"
    echo "Training layer $layer..."
    echo "========================================"
    python code_sae/training.py --config "$CONFIG_FILE"

    echo "✓ Layer $layer training complete"
    echo ""
done

echo "========================================"
echo "✓ All layer training complete!"
echo "========================================"
