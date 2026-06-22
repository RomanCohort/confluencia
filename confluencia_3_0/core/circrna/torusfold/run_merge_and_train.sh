#!/bin/bash
# run_merge_and_train.sh — Merge data + train all schemes on GPU
#
# Usage:
#   bash run_merge_and_train.sh

set -e

REPO_DIR="/root/autodl-tmp/confluencia"
cd "$REPO_DIR"

echo "============================================================"
echo "  Step 1: Check available data sources"
echo "============================================================"

# Count samples in each source
for d in data/circbase_real_3d confluencia_3_0/data/pdb_3d data/shape_3d data/shape_constrained data/shape_expanded data/medium_length_3d data/rfam_3d data/pseudo_labels; do
  if [ -f "$d/sequences.json" ]; then
    n=$(python3 -c "import json; print(len(json.load(open('$d/sequences.json'))))")
    echo "  $d: $n samples"
  else
    echo "  $d: NOT FOUND"
  fi
done

# Count coords
if [ -d "data/circbase_real_3d/coords" ]; then
  nc=$(ls data/circbase_real_3d/coords/*.npy 2>/dev/null | wc -l)
  echo "  circbase_real_3d coords: $nc .npy files"
fi

echo ""
echo "============================================================"
echo "  Step 2: Merge datasets"
echo "============================================================"

MERGED_DIR="data/circrna_3d_merged"

# Always remove old merged data to ensure fresh merge with all sources
if [ -d "$MERGED_DIR" ]; then
  echo "  Removing old merged dataset..."
  rm -rf "$MERGED_DIR"
fi

echo "  Merging all available sources..."

ARGS="--output $MERGED_DIR --skip-validation"

if [ -f "data/circbase_real_3d/sequences.json" ]; then
  ARGS="$ARGS --isrnacirc-dir data/circbase_real_3d"
fi
if [ -f "data/shape_3d/sequences.json" ]; then
  ARGS="$ARGS --shape-dir data/shape_3d"
elif [ -f "data/shape_constrained/sequences.json" ]; then
  ARGS="$ARGS --shape-dir data/shape_constrained"
fi
if [ -f "confluencia_3_0/data/pdb_3d/sequences.json" ]; then
  ARGS="$ARGS --pdb-dir confluencia_3_0/data/pdb_3d"
fi
if [ -f "data/medium_length_3d/sequences.json" ]; then
  ARGS="$ARGS --medium-dir data/medium_length_3d"
fi
if [ -f "data/rfam_3d/sequences.json" ]; then
  ARGS="$ARGS --rfam-dir data/rfam_3d"
fi
if [ -f "data/shape_expanded/sequences.json" ]; then
  ARGS="$ARGS --shape-exp-dir data/shape_expanded"
fi

echo "  Merge command args: $ARGS"
python3 confluencia_3_0/core/circrna/torusfold/merge_expanded_dataset.py $ARGS

echo ""
echo "============================================================"
echo "  Step 3: Train all schemes"
echo "============================================================"

# Use merged data if available, otherwise fall back to circbase_real_3d
if [ -f "$MERGED_DIR/sequences.json" ]; then
  LABELS_DIR="$MERGED_DIR"
else
  LABELS_DIR="data/circbase_real_3d"
fi

echo "  Training with: $LABELS_DIR"
echo ""

python3 confluencia_3_0/core/circrna/torusfold/train_all_schemes.py \
  --schemes 1 2 3 4 5 6 7 \
  --labels "$LABELS_DIR" \
  --device cuda \
  --epochs 50 \
  --batch-size 4

echo ""
echo "============================================================"
echo "  Done!"
echo "============================================================"
