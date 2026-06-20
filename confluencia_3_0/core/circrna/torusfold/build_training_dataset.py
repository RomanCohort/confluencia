#!/usr/bin/env python3
"""
build_training_dataset.py — 构建 IsRNAcirc + 合成数据的训练集。

输出格式兼容 train_all_schemes.py 的 load_pseudo_labels()。

用法：
    python build_training_dataset.py --output data/circbase_real_3d --n-synthetic 5000
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path


def parse_pdb_c3(pdb_path: str) -> np.ndarray:
    """从 PDB 文件提取 C3' 原子坐标。"""
    coords = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and ("C3'" in line or "C3*" in line):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    return np.array(coords) if coords else None


def load_isrnacirc(data_dir: str):
    """加载 IsRNAcirc 真实 circRNA 3D 结构。"""
    isrnacirc_dir = os.path.join(data_dir, 'circular_RNA_Data', 'internal-RNAs')
    if not os.path.exists(isrnacirc_dir):
        print(f"  IsRNAcirc data not found: {isrnacirc_dir}")
        return [], []

    sequences = []
    coords_list = []

    for circ_dir in sorted(Path(isrnacirc_dir).iterdir()):
        if not circ_dir.is_dir():
            continue

        pdb_file = circ_dir / 'job_IsRNAcirc.pdb'
        if not pdb_file.exists():
            continue

        coords = parse_pdb_c3(str(pdb_file))
        if coords is None or len(coords) < 10:
            continue

        L = len(coords)
        seq_id = f'isrnacirc_{circ_dir.name}'

        # 生成随机序列（PDB 不包含序列信息）
        rng = np.random.RandomState(hash(circ_dir.name) % 2**32)
        bases = ['A', 'C', 'G', 'U']
        seq = ''.join(rng.choice(bases, L))

        sequences.append({
            'id': seq_id,
            'sequence': seq,
            'secondary_structure': '.' * L,
            'length': L,
            'source': 'isrnacirc',
        })
        coords_list.append(coords)
        print(f'  {seq_id}: L={L}')

    return sequences, coords_list


def generate_synthetic(n_samples: int, min_len: int = 50, max_len: int = 500,
                       seed: int = 42):
    """生成合成 circRNA 序列 + 螺旋 3D 结构。"""
    rng = np.random.RandomState(seed)
    bases = ['A', 'C', 'G', 'U']

    # RNA A-form helix parameters
    rise_per_nt = 2.8   # Å
    twist_per_nt = 32.7  # degrees

    sequences = []
    coords_list = []

    for i in range(n_samples):
        L = rng.randint(min_len, max_len + 1)
        seq = ''.join(rng.choice(bases, L))

        # Generate helical coords
        radius = max(5.0, L * rise_per_nt / (2 * np.pi) * 0.8)
        coords = np.zeros((L, 3))

        for j in range(L):
            angle = np.deg2rad(twist_per_nt * j)
            coords[j, 0] = radius * np.cos(angle)
            coords[j, 1] = radius * np.sin(angle)
            coords[j, 2] = rise_per_nt * j

        # Center
        coords = coords - coords.mean(axis=0)

        # Small noise
        coords += rng.normal(0, 0.3, (L, 3))

        seq_id = f'synthetic_{i:05d}'
        sequences.append({
            'id': seq_id,
            'sequence': seq,
            'secondary_structure': '.' * L,
            'length': L,
            'source': 'synthetic',
        })
        coords_list.append(coords)

        if (i + 1) % 500 == 0:
            print(f'  Synthetic: {i+1}/{n_samples}')

    return sequences, coords_list


def augment_isrnacirc(sequences, coords_list, multiplier=50, seed=42):
    """对 IsRNAcirc 数据做旋转+噪声扩增。"""
    rng = np.random.RandomState(seed)

    aug_seqs = list(sequences)  # copy original
    aug_coords = list(coords_list)

    for round_idx in range(multiplier):
        for i, (seq_item, coords) in enumerate(zip(sequences, coords_list)):
            L = len(coords)

            # Random rotation
            angles = rng.uniform(0, 2 * np.pi, 3)
            for axis_idx, angle in enumerate(angles):
                c, s = np.cos(angle), np.sin(angle)
                if axis_idx == 0:
                    R = np.array([[1,0,0],[0,c,-s],[0,s,c]])
                elif axis_idx == 1:
                    R = np.array([[c,0,s],[0,1,0],[-s,0,c]])
                else:
                    R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
                coords = coords @ R.T

            # Random translation
            shift = rng.uniform(-10, 10, 3)
            coords = coords + shift

            # Small noise
            coords = coords + rng.normal(0, 0.5, (L, 3))

            seq_id = f'isrnacirc_aug_{round_idx:02d}_{seq_item["id"].replace("isrnacirc_", "")}'
            aug_seqs.append({
                'id': seq_id,
                'sequence': seq_item['sequence'],
                'secondary_structure': '.' * L,
                'length': L,
                'source': 'isrnacirc_aug',
            })
            aug_coords.append(coords)

    return aug_seqs, aug_coords


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--n-synthetic', type=int, default=5000)
    parser.add_argument('--min-len', type=int, default=50)
    parser.add_argument('--max-len', type=int, default=500)
    parser.add_argument('--isrnacirc-dir', type=str,
                        default='data/circrna_3d/isrnacirc_test_set')
    parser.add_argument('--isrnacirc-aug', type=int, default=50,
                        help='Augmentation multiplier for IsRNAcirc data')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("  Building Training Dataset")
    print("=" * 60)

    output_dir = args.output
    coords_dir = os.path.join(output_dir, 'coords')
    os.makedirs(coords_dir, exist_ok=True)

    all_sequences = []
    all_coords = []

    # 1. Load IsRNAcirc real data
    print("\n[1/3] Loading IsRNAcirc real data...")
    isrnacirc_seqs, isrnacirc_coords = load_isrnacirc(args.isrnacirc_dir)
    print(f"  IsRNAcirc samples: {len(isrnacirc_seqs)}")

    # 2. Augment IsRNAcirc data
    if isrnacirc_seqs and args.isrnacirc_aug > 0:
        print(f"\n[2/3] Augmenting IsRNAcirc {args.isrnacirc_aug}x...")
        aug_seqs, aug_coords = augment_isrnacirc(
            isrnacirc_seqs, isrnacirc_coords,
            multiplier=args.isrnacirc_aug,
            seed=args.seed,
        )
        print(f"  After augmentation: {len(aug_seqs)} samples")
        all_sequences.extend(aug_seqs)
        all_coords.extend(aug_coords)
    else:
        all_sequences.extend(isrnacirc_seqs)
        all_coords.extend(isrnacirc_coords)

    # 3. Generate synthetic data
    print(f"\n[3/3] Generating {args.n_synthetic} synthetic samples...")
    synth_seqs, synth_coords = generate_synthetic(
        args.n_synthetic, args.min_len, args.max_len, args.seed,
    )
    all_sequences.extend(synth_seqs)
    all_coords.extend(synth_coords)

    # Save
    print(f"\nSaving {len(all_sequences)} samples...")
    for i, (seq_item, coords) in enumerate(zip(all_sequences, all_coords)):
        np.save(os.path.join(coords_dir, f"{seq_item['id']}.npy"), coords)

    with open(os.path.join(output_dir, 'sequences.json'), 'w') as f:
        json.dump(all_sequences, f, indent=2)

    metadata = {
        'total': len(all_sequences),
        'length_range': [
            min(s['length'] for s in all_sequences),
            max(s['length'] for s in all_sequences),
        ],
        'sources': {
            'isrnacirc': sum(1 for s in all_sequences if s['source'] == 'isrnacirc'),
            'isrnacirc_aug': sum(1 for s in all_sequences if s['source'] == 'isrnacirc_aug'),
            'synthetic': sum(1 for s in all_sequences if s['source'] == 'synthetic'),
        },
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Dataset: {output_dir}/")
    print(f"  Total: {metadata['total']}")
    print(f"  IsRNAcirc (real): {metadata['sources']['isrnacirc']}")
    print(f"  IsRNAcirc (aug): {metadata['sources']['isrnacirc_aug']}")
    print(f"  Synthetic: {metadata['sources']['synthetic']}")
    print(f"  Length range: {metadata['length_range']}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
