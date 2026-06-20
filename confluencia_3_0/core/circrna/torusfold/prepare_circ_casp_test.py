#!/usr/bin/env python3
"""
prepare_circ_casp_test.py — 准备 Circ-CASP 测试集。

从 IsRNAcirc 数据中选择 30 个结构作为测试集。
测试集序列公开，真实结构保密。

用法：
    python prepare_circ_casp_test.py --output data/circ_casp_test --n-test 30
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
import hashlib


def parse_pdb_c3(pdb_path: str) -> np.ndarray:
    """从 PDB 提取 C3' 坐标。"""
    coords = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and ("C3'" in line or "C3*" in line):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    return np.array(coords) if coords else None


def generate_random_sequence(length: int, seed: int) -> str:
    """生成随机 RNA 序列。"""
    rng = np.random.RandomState(seed)
    bases = ['A', 'C', 'G', 'U']
    return ''.join(rng.choice(bases, length))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--isrnacirc-dir', type=str,
                        default='data/circrna_3d/isrnacirc_test_set/circular_RNA_Data/internal-RNAs')
    parser.add_argument('--output', type=str, default='data/circ_casp_test')
    parser.add_argument('--n-test', type=int, default=30)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()

    print("=" * 60)
    print("  Circ-CASP Test Set Preparation")
    print("=" * 60)

    os.makedirs(args.output, exist_ok=True)
    coords_dir = os.path.join(args.output, 'coords')
    pairs_dir = os.path.join(args.output, 'pairs')
    public_dir = os.path.join(args.output, 'public')
    os.makedirs(coords_dir, exist_ok=True)
    os.makedirs(pairs_dir, exist_ok=True)
    os.makedirs(public_dir, exist_ok=True)

    # 收集所有 IsRNAcirc PDB
    isrnacirc_dir = Path(args.isrnacirc_dir)
    all_pdb_files = list(isrnacirc_dir.glob("*/job_IsRNAcirc.pdb"))
    print(f"  Found {len(all_pdb_files)} IsRNAcirc structures")

    # 随机选择 N 个作为测试集
    rng = np.random.RandomState(args.seed)
    selected_indices = rng.choice(len(all_pdb_files), min(args.n_test, len(all_pdb_files)), replace=False)
    selected_pdbs = [all_pdb_files[i] for i in selected_indices]

    print(f"  Selected {len(selected_pdbs)} for test set")

    # 生成测试集 ID 映射（保密）
    id_mapping = {}
    test_sequences = []

    for idx, pdb_path in enumerate(selected_pdbs):
        coords = parse_pdb_c3(str(pdb_path))
        if coords is None or len(coords) < 50:
            print(f"  Skipping {pdb_path.parent.name}: invalid coords")
            continue

        L = len(coords)

        # 生成测试 ID（不暴露原始名称）
        circ_id = f"circ_{idx+1:03d}"

        # 使用哈希作为秘密映射
        secret_hash = hashlib.md5(f"{pdb_path.parent.name}_{args.seed}".encode()).hexdigest()[:8]
        id_mapping[circ_id] = {
            'original_name': pdb_path.parent.name,
            'secret_hash': secret_hash,
            'length': L,
        }

        # 生成随机序列
        seed = hash(circ_id) % 2**32
        sequence = generate_random_sequence(L, seed)

        # 保存完整结构（保密）
        np.save(os.path.join(coords_dir, f"{circ_id}.npy"), coords)

        # 保存配对信息（如果有）
        # TODO: 从二级结构文件提取配对信息
        with open(os.path.join(pairs_dir, f"{circ_id}.json"), 'w') as f:
            json.dump([], f)  # 暂时空

        # 公开信息：仅序列
        test_sequences.append({
            'id': circ_id,
            'sequence': sequence,
            'length': L,
        })

        print(f"  {circ_id}: L={L}, hash={secret_hash}")

    # 保存公开序列
    with open(os.path.join(public_dir, 'sequences.json'), 'w') as f:
        json.dump(test_sequences, f, indent=2)

    # 保存 ID 映射（保密）
    with open(os.path.join(args.output, 'id_mapping_secret.json'), 'w') as f:
        json.dump(id_mapping, f, indent=2)

    # 保存元数据
    metadata = {
        'n_targets': len(test_sequences),
        'length_range': [min(s['length'] for s in test_sequences),
                        max(s['length'] for s in test_sequences)],
        'seed': args.seed,
        'source': 'IsRNAcirc',
    }
    with open(os.path.join(args.output, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Test set prepared: {args.output}/")
    print(f"  Targets: {len(test_sequences)}")
    print(f"  Length range: {metadata['length_range']}")
    print(f"  Public data: {public_dir}/sequences.json")
    print(f"  Ground truth: {coords_dir}/ (SECRET)")
    print(f"{'='*60}")
    print(f"\n  ⚠️  WARNING: Do not share coords/ and id_mapping_secret.json!")
    print(f"  Only share public/ directory with participants.")


if __name__ == '__main__':
    main()
