#!/bin/bash
# run_training_sequential.sh — 串联训练所有 TorusFold schemes
# 按显存占用从少到多排序，顺序执行，避免 OOM
#
# 用法:
#   bash run_training_sequential.sh
#   bash run_training_sequential.sh --labels /path/to/pseudo_labels
#   bash run_training_sequential.sh --schemes 1 3 5

set -e

# ── 默认参数 ──
LABELS_DIR="${LABELS_DIR:-data/circbase_real_3d}"
OUTPUT_DIR="${OUTPUT_DIR:-models/torusfold_real}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"
D_HIDDEN="${D_HIDDEN:-128}"
N_LAYERS="${N_LAYERS:-4}"
LR="${LR:-1e-3}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-100}"
SEED="${SEED:-42}"

# 额外参数透传
EXTRA_ARGS=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --labels)   LABELS_DIR="$2"; shift 2 ;;
        --output)   OUTPUT_DIR="$2"; shift 2 ;;
        --device)   DEVICE="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --d-hidden) D_HIDDEN="$2"; shift 2 ;;
        --schemes)
            # 用户指定 schemes，覆盖默认顺序
            SCHEMES_OVERRIDE="$2"
            shift 2
            ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# ── 训练顺序（显存占用从少到多）──
# Scheme 2: 无训练（纯物理求解器）
# Scheme 3: Transformer + PaxNet (~2GB)
# Scheme 1: EGNN (~3GB)
# Scheme 5: CircPairformer (~4GB)
# Scheme 4: DDPM+EGNN (~6GB)
# Scheme 6: GNN Latent Diffusion (~8GB)

if [ -n "$SCHEMES_OVERRIDE" ]; then
    SCHEMES=($SCHEMES_OVERRIDE)
else
    SCHEMES=(2 3 1 5 4 6)
fi

SCRIPT="python -m confluencia_3_0.core.circrna.torusfold.train_all_schemes"

echo "============================================================"
echo "  TorusFold Sequential Training"
echo "============================================================"
echo "  Labels:  $LABELS_DIR"
echo "  Output:  $OUTPUT_DIR"
echo "  Device:  $DEVICE"
echo "  Batch:   $BATCH_SIZE"
echo "  Schemes: ${SCHEMES[*]}"
echo "============================================================"

# 检查伪标签是否存在
if [ ! -d "$LABELS_DIR" ]; then
    echo "ERROR: Labels directory not found: $LABELS_DIR"
    echo "Run generate_pseudo_labels.py first:"
    echo "  python -m confluencia_3_0.core.circrna.torusfold.generate_pseudo_labels --n 1000 --output $LABELS_DIR"
    exit 1
fi

if [ ! -f "$LABELS_DIR/sequences.json" ]; then
    echo "ERROR: sequences.json not found in $LABELS_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 记录开始时间
START_TIME=$(date +%s)

# ── 逐个训练 ──
FAILED=()
SUCCEEDED=()

for SCHEME in "${SCHEMES[@]}"; do
    echo ""
    echo "============================================================"
    echo "  [$(date '+%H:%M:%S')] Training Scheme $SCHEME"
    echo "============================================================"

    SCHEME_START=$(date +%s)

    # Scheme 2 不需要训练
    if [ "$SCHEME" -eq 2 ]; then
        echo "  Scheme 2: Batch+Physics Filter (no training needed)"
        echo "  Skipping..."
        SUCCEEDED+=($SCHEME)
        continue
    fi

    # Scheme-specific 参数调整
    case $SCHEME in
        1)
            # EGNN: k-NN sparse edges, manageable memory
            SCHEME_EPOCHS=50
            SCHEME_BATCH=$BATCH_SIZE
            ;;
        3)
            # Dual-Engine: 中等，50 epochs
            SCHEME_EPOCHS=50
            SCHEME_BATCH=$BATCH_SIZE
            ;;
        4)
            # DDPM: 重，100 epochs，小 batch
            SCHEME_EPOCHS=100
            SCHEME_BATCH=2
            ;;
        5)
            # CircPairformer: 中等，50 epochs
            SCHEME_EPOCHS=50
            SCHEME_BATCH=$BATCH_SIZE
            ;;
        6)
            # GNN Latent Diffusion: 最重，100 epochs，小 batch
            SCHEME_EPOCHS=100
            SCHEME_BATCH=2
            ;;
        *)
            SCHEME_EPOCHS=50
            SCHEME_BATCH=$BATCH_SIZE
            ;;
    esac

    # 运行训练
    CMD="$SCRIPT --schemes $SCHEME --labels $LABELS_DIR --device $DEVICE --batch-size $SCHEME_BATCH --epochs $SCHEME_EPOCHS --d-hidden $D_HIDDEN --n-layers $N_LAYERS --lr $LR --diffusion-steps $DIFFUSION_STEPS --seed $SEED --output $OUTPUT_DIR $EXTRA_ARGS"

    echo "  Command: $CMD"
    echo ""

    if $CMD; then
        SCHEME_END=$(date +%s)
        SCHEME_ELAPSED=$((SCHEME_END - SCHEME_START))
        echo "  Scheme $SCHEME completed in ${SCHEME_ELAPSED}s"
        SUCCEEDED+=($SCHEME)
    else
        SCHEME_END=$(date +%s)
        SCHEME_ELAPSED=$((SCHEME_END - SCHEME_START))
        echo "  Scheme $SCHEME FAILED after ${SCHEME_ELAPSED}s"
        FAILED+=($SCHEME)
    fi

    # 清理 GPU 缓存
    if [ "$DEVICE" = "cuda" ]; then
        python -c "import torch; torch.cuda.empty_cache(); print('  GPU cache cleared')" 2>/dev/null || true
    fi
done

# ── 汇总 ──
END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "  Training Complete"
echo "============================================================"
echo "  Total time: ${TOTAL_ELAPSED}s ($((TOTAL_ELAPSED / 60))min)"
echo ""
echo "  Succeeded: ${SUCCEEDED[*]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  FAILED:    ${FAILED[*]}"
    echo ""
    echo "  To retry failed schemes:"
    for f in "${FAILED[@]}"; do
        echo "    $SCRIPT --schemes $f --labels $LABELS_DIR --device $DEVICE --output $OUTPUT_DIR"
    done
fi
echo ""
echo "  Model files:"
ls -lh "$OUTPUT_DIR"/*.pt 2>/dev/null || echo "    (none found)"
echo ""
echo "  Results: $OUTPUT_DIR/training_results.json"
echo "============================================================"
