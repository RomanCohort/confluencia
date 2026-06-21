#!/bin/bash
# push_data_to_github.sh — Push training data to GitHub via Git LFS.
#
# On the current GPU machine:
#   bash push_data_to_github.sh
#
# On the new machine:
#   git lfs install && git pull
#   # Data will be available automatically
#
# If total data > GitHub LFS free quota (1GB), only merged dataset is pushed.

set -e

echo "============================================================"
echo "  Push Training Data to GitHub"
echo "============================================================"

# ── Check prerequisites ──────────────────────────────────
if ! command -v git-lfs &> /dev/null; then
    echo "  Installing Git LFS..."
    apt-get install -y git-lfs 2>/dev/null || conda install -c conda-forge git-lfs -y 2>/dev/null
fi
git lfs install

# ── Check data sizes ─────────────────────────────────────
echo ""
echo "  Checking data sizes..."

CONFLUENCIA_DIR="/root/autodl-tmp/confluencia"
DATA_DIR="/root/autodl-tmp/data"

# Collect available datasets
DATASETS=()

if [ -d "$DATA_DIR/circrna_3d_merged_v2" ]; then
    SIZE=$(du -sm "$DATA_DIR/circrna_3d_merged_v2" | cut -f1)
    echo "  circrna_3d_merged_v2: ${SIZE} MB"
    DATASETS+=("$DATA_DIR/circrna_3d_merged_v2")
fi

if [ -d "$CONFLUENCIA_DIR/data/circbase_real_3d" ]; then
    SIZE=$(du -sm "$CONFLUENCIA_DIR/data/circbase_real_3d" | cut -f1)
    echo "  circbase_real_3d: ${SIZE} MB"
    DATASETS+=("$CONFLUENCIA_DIR/data/circbase_real_3d")
fi

if [ -d "$CONFLUENCIA_DIR/data/medium_length_3d" ]; then
    SIZE=$(du -sm "$CONFLUENCIA_DIR/data/medium_length_3d" | cut -f1)
    echo "  medium_length_3d: ${SIZE} MB"
    DATASETS+=("$CONFLUENCIA_DIR/data/medium_length_3d")
fi

if [ -d "$CONFLUENCIA_DIR/data/shape_3d" ]; then
    SIZE=$(du -sm "$CONFLUENCIA_DIR/data/shape_3d" | cut -f1)
    echo "  shape_3d: ${SIZE} MB"
    DATASETS+=("$CONFLUENCIA_DIR/data/shape_3d")
fi

if [ -d "$CONFLUENCIA_DIR/confluencia_3_0/data/pdb_3d" ]; then
    SIZE=$(du -sm "$CONFLUENCIA_DIR/confluencia_3_0/data/pdb_3d" | cut -f1)
    echo "  pdb_3d: ${SIZE} MB"
    DATASETS+=("$CONFLUENCIA_DIR/confluencia_3_0/data/pdb_3d")
fi

if [ -d "$DATA_DIR/shape_3d" ]; then
    SIZE=$(du -sm "$DATA_DIR/shape_3d" | cut -f1)
    echo "  shape_3d (data dir): ${SIZE} MB"
    DATASETS+=("$DATA_DIR/shape_3d")
fi

# ── Calculate total ──────────────────────────────────────
TOTAL=0
for ds in "${DATASETS[@]}"; do
    SIZE=$(du -sm "$ds" | cut -f1)
    TOTAL=$((TOTAL + SIZE))
done
echo ""
echo "  Total data: ${TOTAL} MB"

# ── Decide push strategy ─────────────────────────────────
LFS_QUOTA_MB=950  # GitHub free LFS quota ~1GB, leave some margin

cd "$CONFLUENCIA_DIR"

# Track .npy files with LFS
echo ""
echo "  Setting up Git LFS tracking..."
git lfs track "*.npy"
git lfs track "*.npz"
git add .gitattributes

if [ $TOTAL -le $LFS_QUOTA_MB ]; then
    echo "  Total under LFS quota. Pushing all datasets."
    for ds in "${DATASETS[@]}"; do
        # Copy into repo if not already there
        REL_PATH=$(realpath --relative-to="$CONFLUENCIA_DIR" "$ds")
        if [ ! -d "$CONFLUENCIA_DIR/$REL_PATH" ]; then
            echo "  Copying $ds -> $CONFLUENCIA_DIR/$REL_PATH"
            mkdir -p "$CONFLUENCIA_DIR/$REL_PATH"
            cp -r "$ds"/* "$CONFLUENCIA_DIR/$REL_PATH/"
        fi
        git add "$REL_PATH"
    done
else
    echo "  Total exceeds LFS quota. Pushing only merged dataset."
    # Copy merged dataset into repo data dir
    MERGED_SRC="$DATA_DIR/circrna_3d_merged_v2"
    MERGED_DST="$CONFLUENCIA_DIR/data/circrna_3d_merged"
    if [ ! -d "$MERGED_DST" ]; then
        echo "  Copying merged dataset -> $MERGED_DST"
        mkdir -p "$MERGED_DST"
        cp -r "$MERGED_SRC"/* "$MERGED_DST/"
    fi
    git add data/circrna_3d_merged/

    # Also push source datasets that are in repo already
    for ds in circbase_real_3d medium_length_3d; do
        if [ -d "$CONFLUENCIA_DIR/data/$ds" ]; then
            git add "data/$ds"
        fi
    done
fi

# ── Commit and push ──────────────────────────────────────
echo ""
echo "  Committing..."
git commit -m "Add circRNA 3D training data for TorusFold multi-scheme training"

echo "  Pushing to GitHub..."
git push origin main

echo ""
echo "============================================================"
echo "  Done! Data pushed to GitHub."
echo "============================================================"
echo ""
echo "  On new machine, run:"
echo "    git lfs install"
echo "    git pull"
echo "    # Data available in data/ directory"
echo ""
