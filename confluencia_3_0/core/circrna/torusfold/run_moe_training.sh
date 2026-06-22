#!/bin/bash
# run_moe_training.sh — Train TorusFold MOE model
#
# Prerequisites:
#   1. Pretrained expert checkpoints in models/torusfold/
#   2. Pseudo-labels in data/ directory
#
# Usage:
#   bash run_moe_training.sh
#   bash run_moe_training.sh --labels data/circrna_3d_merged
#   bash run_moe_training.sh --top-k 3 --fusion-mode stacked_refine

set -e

# Defaults
LABELS_DIR="${LABELS_DIR:-data/circrna_3d_merged}"
PRETRAINED_DIR="${PRETRAINED_DIR:-models/torusfold}"
OUTPUT_DIR="${OUTPUT_DIR:-models/torusfold_moe}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TOP_K="${TOP_K:-2}"
FUSION_MODE="${FUSION_MODE:-confidence}"
GATE_LR="${GATE_LR:-5e-4}"
D_HIDDEN="${D_HIDDEN:-128}"

# Parse args
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --labels)     LABELS_DIR="$2"; shift 2 ;;
        --pretrained) PRETRAINED_DIR="$2"; shift 2 ;;
        --output)     OUTPUT_DIR="$2"; shift 2 ;;
        --top-k)      TOP_K="$2"; shift 2 ;;
        --fusion)     FUSION_MODE="$2"; shift 2 ;;
        --device)     DEVICE="$2"; shift 2 ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

SCRIPT="python -m confluencia_3_0.core.circrna.torusfold.train_moe"

echo "============================================================"
echo "  TorusFold MOE Training Pipeline"
echo "============================================================"
echo "  Labels:     $LABELS_DIR"
echo "  Pretrained: $PRETRAINED_DIR"
echo "  Output:     $OUTPUT_DIR"
echo "  Device:     $DEVICE"
echo "  Top-K:      $TOP_K"
echo "  Fusion:     $FUSION_MODE"
echo "============================================================"

# Check pretrained experts exist
if [ ! -d "$PRETRAINED_DIR" ]; then
    echo "ERROR: Pretrained directory not found: $PRETRAINED_DIR"
    echo "Run train_all_schemes.py first to generate expert checkpoints."
    exit 1
fi

n_checkpoints=$(ls "$PRETRAINED_DIR"/scheme*_best.pt "$PRETRAINED_DIR"/scheme*.pt 2>/dev/null | wc -l)
echo "  Found $n_checkpoints expert checkpoints"

if [ "$n_checkpoints" -eq 0 ]; then
    echo "WARNING: No expert checkpoints found!"
    echo "  Train experts first:"
    echo "  python train_all_schemes.py --schemes 1 3 5 7 --labels $LABELS_DIR"
fi

# Phase 2: Train gating + fusion (experts frozen)
echo ""
echo "============================================================"
echo "  Phase 2: Training Gating + Fusion (experts frozen)"
echo "============================================================"

$SCRIPT \
    --labels "$LABELS_DIR" \
    --pretrained-dir "$PRETRAINED_DIR" \
    --output "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --top-k "$TOP_K" \
    --fusion-mode "$FUSION_MODE" \
    --d-hidden "$D_HIDDEN" \
    --phase 2 \
    --phase2-epochs 30 \
    --gate-lr "$GATE_LR" \
    $EXTRA_ARGS

# Phase 3: End-to-end fine-tuning (optional, uncomment to enable)
# echo ""
# echo "============================================================"
# echo "  Phase 3: End-to-end Fine-tuning"
# echo "============================================================"
#
# $SCRIPT \
#     --labels "$LABELS_DIR" \
#     --pretrained-dir "$PRETRAINED_DIR" \
#     --output "$OUTPUT_DIR" \
#     --device "$DEVICE" \
#     --batch-size "$BATCH_SIZE" \
#     --top-k "$TOP_K" \
#     --fusion-mode "$FUSION_MODE" \
#     --d-hidden "$D_HIDDEN" \
#     --phase 3 \
#     --phase3-epochs 10 \
#     --finetune-lr 1e-5 \
#     $EXTRA_ARGS

echo ""
echo "============================================================"
echo "  MOE Training Complete"
echo "============================================================"
echo "  Model: $OUTPUT_DIR/torusfold_moe_best.pt"
echo "  Config: $OUTPUT_DIR/moe_config.json"
echo ""
echo "  To evaluate:"
echo "  python evaluate_casp.py --model moe --checkpoint $OUTPUT_DIR/torusfold_moe_best.pt"
echo "============================================================"
