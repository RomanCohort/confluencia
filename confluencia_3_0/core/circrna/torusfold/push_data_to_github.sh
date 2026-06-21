#!/bin/bash
# push_data_to_github.sh — Push training data to GitHub via Git LFS.
#
# On the current GPU machine:
#   bash push_data_to_github.sh
#
# On the new machine:
#   git lfs install && git pull
#   # Data will be available automatically

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

# ── Paths ────────────────────────────────────────────────
CONFLUENCIA_DIR="/root/autodl-tmp/confluencia"
DATA_DIR="/root/autodl-tmp/data"
REPO_DATA_DIR="$CONFLUENCIA_DIR/data"

cd "$CONFLUENCIA_DIR"

# ── Track large files with LFS ───────────────────────────
echo ""
echo "  Setting up Git LFS tracking..."
git lfs track "*.npy"
git lfs track "*.npz"
git add .gitattributes

# ── Copy external datasets into repo ─────────────────────
echo ""
echo "  Copying datasets into repo..."

# Map: source_dir -> repo_subdir_name
declare -A DATASETS

# External data (outside repo)
if [ -d "$DATA_DIR/circrna_3d_merged_v2" ]; then
    DATASETS["$DATA_DIR/circrna_3d_merged_v2"]="circrna_3d_merged"
fi
if [ -d "$DATA_DIR/shape_3d" ]; then
    DATASETS["$DATA_DIR/shape_3d"]="shape_3d"
fi

# Internal data (already in repo)
if [ -d "$CONFLUENCIA_DIR/data/circbase_real_3d" ]; then
    DATASETS["$CONFLUENCIA_DIR/data/circbase_real_3d"]=""
fi
if [ -d "$CONFLUENCIA_DIR/data/medium_length_3d" ]; then
    DATASETS["$CONFLUENCIA_DIR/data/medium_length_3d"]=""
fi
if [ -d "$CONFLUENCIA_DIR/confluencia_3_0/data/pdb_3d" ]; then
    DATASETS["$CONFLUENCIA_DIR/confluencia_3_0/data/pdb_3d"]=""
fi

# ── Copy and add ─────────────────────────────────────────
TOTAL=0
for src_dir in "${!DATASETS[@]}"; do
    repo_subdir="${DATASETS[$src_dir]}"

    # Determine destination in repo
    if [ -n "$repo_subdir" ]; then
        # External data: copy into repo/data/<subdir>
        dst_dir="$REPO_DATA_DIR/$repo_subdir"
        SIZE=$(du -sm "$src_dir" | cut -f1)
        TOTAL=$((TOTAL + SIZE))
        echo "  Copying $src_dir -> $dst_dir (${SIZE} MB)"
        mkdir -p "$dst_dir"
        cp -r "$src_dir"/* "$dst_dir/"
        git add "data/$repo_subdir/"
    else
        # Already in repo, just add
        # Find relative path from repo root
        rel_path=$(realpath --relative-to="$CONFLUENCIA_DIR" "$src_dir")
        SIZE=$(du -sm "$src_dir" | cut -f1)
        TOTAL=$((TOTAL + SIZE))
        echo "  Adding $rel_path (${SIZE} MB)"
        git add "$rel_path"
    fi
done

echo ""
echo "  Total data: ${TOTAL} MB"

# ── Commit and push ──────────────────────────────────────
echo ""
echo "  Committing..."
git commit -m "Add circRNA 3D training data for TorusFold multi-scheme training

Sources: IsRNAcirc (5663) + SHAPE experimental (1118) + medium-length (2000) + PDB (184)
Merged: 8139 unique samples, 43-1000 nt, mean 476 nt"

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
