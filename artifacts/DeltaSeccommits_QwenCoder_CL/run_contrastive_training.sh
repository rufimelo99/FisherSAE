#!/bin/bash
#SBATCH --job-name=contrastive-sae-deltasec
#SBATCH --output=logs/contrastive_sae_deltasec_%j.log
#SBATCH --error=logs/contrastive_sae_deltasec_%j.err
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAYERS=(0 14 27)

echo "=============================================="
echo "Contrastive SAE Training Pipeline"
echo "Dataset: DeltaSeccommits (QwenCoder)"
echo "Layers: ${LAYERS[*]}"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# Create directories
mkdir -p data/activations
mkdir -p logs

for LAYER in "${LAYERS[@]}"; do
    echo ""
    echo "############################################"
    echo "# Processing Layer ${LAYER}"
    echo "############################################"
    echo ""

    # Config files in this folder
    PRECOMPUTE_CONFIG="${SCRIPT_DIR}/_precompute_layer${LAYER}.json"
    TRAINING_CONFIG="${SCRIPT_DIR}/_training_config_layer${LAYER}_topk_cl_16384_lr_1e-4_fast.json"

    echo "Precompute config: ${PRECOMPUTE_CONFIG}"
    echo "Training config: ${TRAINING_CONFIG}"

    # Validate config files exist
    if [[ ! -f "$PRECOMPUTE_CONFIG" ]]; then
        echo "ERROR: Precompute config not found: $PRECOMPUTE_CONFIG"
        exit 1
    fi

    if [[ ! -f "$TRAINING_CONFIG" ]]; then
        echo "ERROR: Training config not found: $TRAINING_CONFIG"
        exit 1
    fi

    mkdir -p checkpoints/contrastive_sae_deltasec_layer${LAYER}

    # Step 1: Pre-compute activations (skip if already done)
    POSITIVE_ACTS="data/activations/positive_acts_layer${LAYER}.pt"
    NEGATIVE_ACTS="data/activations/negative_acts_layer${LAYER}.pt"

    if [[ -f "$POSITIVE_ACTS" && -f "$NEGATIVE_ACTS" ]]; then
        echo "[Layer ${LAYER} - Step 1] SKIPPED - Activations already exist"
        echo "  - $POSITIVE_ACTS"
        echo "  - $NEGATIVE_ACTS"
    else
        echo "[Layer ${LAYER} - Step 1] Pre-computing activations..."
        echo "  This may take 2-4 hours for a 7B model."

        python code_sae/precompute_activations.py \
            --config "$PRECOMPUTE_CONFIG"

        echo "[Layer ${LAYER} - Step 1] DONE - Activations saved"
    fi

    # Step 2: Train SAE with pre-computed activations
    echo "[Layer ${LAYER} - Step 2] Training contrastive SAE..."

    python code_sae/contrastive_training_fast.py \
        --config "$TRAINING_CONFIG"

    echo "[Layer ${LAYER}] Training complete!"
done

echo ""
echo "=============================================="
echo "All layers complete!"
echo "Dataset: DeltaSeccommits | Layers: ${LAYERS[*]}"
echo "End time: $(date)"
echo "=============================================="
