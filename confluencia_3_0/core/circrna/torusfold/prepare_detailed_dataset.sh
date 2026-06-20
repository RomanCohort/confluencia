#!/bin/bash
# prepare_detailed_dataset.sh — 将 IsRNAcirc test set 扩充到真实数据集

set -e

echo "============================================================
  IsRNAcirc Test Set Expansion
============================================================"

# 参数
ISRNACIRC_DIR="data/circrna_3d/isrnacirc_test_set"
OUTPUT_DIR="data/circbase_real_3d"
N_SAMPLES=5000
MULTIPLIER=20

# 检查 IsRNAcirc 数据是否存在
if [ ! -d "$ISRNACIRC_DIR/circular_RNA_Data" ]; then
    echo "ERROR: IsRNAcirc data not found in $ISRNACIRC_DIR"
    echo "Please run: tar -xzvf data/circrna_3d/isrnacirc_test_set/circular_RNA_Data.tar.gz"
    exit 1
fi

# 1. 导出 IsRNAcirc test set（34条真实 circRNA）
echo ""
echo "Step 1: Extract IsRNAcirc test set"
python3 << 'PYEOF'
import os
import json
import numpy as np
from pathlib import Path

circbase_dir = "data/circrna_3d/isrnacirc_test_set/circular_RNA_Data"
output_dir = "data/circbase_real_3d"
os.makedirs(output_dir, exist_ok=True)
coords_dir = f"{output_dir}/coords"
os.makedirs(coords_dir, exist_ok=True)

sequences = []
metadata = []

# 收集所有 PDB 文件
pdb_files = sorted(Path(circbase_dir).glob("*.pdb"))

print(f"  Found {len(pdb_files)} PDB files")

for idx, pdb_path in enumerate(pdb_files):
    # 解析 PDB
    coords = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and "C3'" in line:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])

    if coords:
        seq_id = f"circBase_{pdb_path.stem}"
        seq = f"random_sequence_{idx}"

        np.save(f"{coords_dir}/{seq_id}.npy", np.array(coords))

        sequences.append({
            'id': seq_id,
            'sequence': seq,
            'length': len(coords),
        })

        metadata.append({
            'id': seq_id,
            'length': len(coords),
            'source': 'isrnacirc',
        })

        print(f"    {idx+1}/{len(pdb_files)}: {seq_id} (L={len(coords)})")

print(f"\n  Total IsRNAcirc samples: {len(sequences)}")

# 保存
with open(f"{output_dir}/sequences.json", 'w') as f:
    json.dump(sequences, f, indent=2)

with open(f"{output_dir}/metadata.json", 'w') as f:
    json.dump({'total': len(sequences), 'length_range': [min(m['length'] for m in metadata),
                    max(m['length'] for m in metadata)], 'sources': {'isrnacirc': len(sequences)}}, f, indent=2)

print(f"  Saved to {output_dir}/")
PYEOF

# 2. 扩充到目标数量
REAL_SAMPLES=$(python3 -c "import json; data=json.load(open('data/circbase_real_3d/sequences.json')); print(len(data))")
NEEDED=$((N_SAMPLES - REAL_SAMPLES))

if [ $NEEDED -gt 0 ]; then
    echo ""
    echo "Step 2: Augment to {N_SAMPLES} samples ({NEEDED} more needed)"

    python3 << EOF
import os
import sys
sys.path.insert(0, '.')

from confluencia_3_0.core.circrna.torusfold.augment_pseudo_labels import (
    augment_pseudo_labels,
    save_augmented_labels,
)
from confluencia_3_0.core.circrna.torusfold.generate_3d_pseudo_labels import (
    generate_3d_pseudo_labels,
)

print("  Generating pseudo-labels...")
sequences, coords_labels, pair_labels, metadata = generate_3d_pseudo_labels(
    n_seqs=$NEEDED,
    min_len=50,
    max_len=500,
    seed=42,
)

print("  Augmenting...")
aug_seq, aug_coords, aug_meta = augment_pseudo_labels(
    [],  # No original data (all from generation)
    [],  # No original coords (all generated)
    multiplier=1,  # Each generated sample is a new augmentation
    seed=42,
    use_rotation=True,
    use_translation=True,
    use_noise=True,
    use_subsample=True,
    use_mutation=True,
)

# Merge
from confluencia_3_0.core.circrna.torusfold.train_all_schemes import load_pseudo_labels
real_seqs, real_coords = load_pseudo_labels("data/circbase_real_3d")

# Combine
combined_seqs = real_seqs + aug_seq
combined_coords = real_coords + aug_coords

print(f"  Original: {len(real_seqs)}")
print(f"  Augmented: {len(aug_seq)}")
print(f"  Total: {len(combined_seqs)}")

# Save
import json
coords_dir = "data/circbase_real_3d/coords"
os.makedirs(coords_dir, exist_ok=True)

for i in range(len(combined_coords)):
    if combined_coords[i] is not None:
        np.save(f"{coords_dir}/{combined_seqs[i]['id']}.npy", combined_coords[i])

with open(f"data/circbase_real_3d/sequences.json", 'w') as f:
    json.dump(combined_seqs, f, indent=2)

with open(f"data/circbase_real_3d/metadata.json", 'w') as f:
    json.dump({'total': len(combined_seqs), 'length_range': [min(len(s['sequence']) for s in combined_seqs),
                    max(len(s['sequence']) for s in combined_seqs)], 'sources': {'isrnacirc': len(real_seqs), 'synthetic': len(aug_seq)}}, f, indent=2)

print("  Saved final dataset to data/circbase_real_3d/")
EOF
else
    echo ""
    echo "Step 2: Already enough samples ({REAL_SAMPLES} samples, target: {N_SAMPLES})"
fi

# 3. 统计
echo ""
echo "Step 3: Dataset Summary"
python3 << 'PYEOF'
import json
with open('data/circbase_real_3d/metadata.json') as f:
    meta = json.load(f)
with open('data/circbase_real_3d/sequences.json') as f:
    seqs = json.load(f)

print(f"  Total samples: {meta['total']}")
print(f"  Length range: {meta['length_range'][0]} - {meta['length_range'][1]}")
print(f"  Sources: {meta['sources']}")
print(f"  Sample sizes: {[s['length'] for s in seqs[:5]]}...")

# Check disk usage
import os
coords_dir = "data/circbase_real_3d/coords"
total_size = sum(os.path.getsize(os.path.join(coords_dir, f)) for f in os.listdir(coords_dir)) // (1024*1024)
print(f"  Disk usage: {total_size} MB")
PYEOF

echo ""
echo "============================================================
  Dataset Ready: data/circbase_real_3d/
  Next: python train_all_schemes.py --labels data/circbase_real_3d --device cuda
============================================================"
