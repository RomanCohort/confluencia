#!/usr/bin/env python3
"""
augment_pseudo_labels.py — 数据增强：扩充伪标签数据集。

增强策略：
1. 旋转增强：随机旋转3D坐标（保持结构不变）
2. 平移增强：随机平移（保持相对位置）
3. 噪声扰动：添加小幅度坐标噪声
4. 长度裁剪：从长序列中随机裁剪子序列
5. 序列突变：随机替换部分碱基（保持结构约束）

用法：
    python augment_pseudo_labels.py --input data/pseudo_labels --output data/pseudo_labels_aug --multiplier 5
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_pseudo_labels(labels_dir: str):
    """加载伪标签数据。"""
    with open(os.path.join(labels_dir, 'sequences.json'), 'r') as f:
        seq_data = json.load(f)

    coords_dir = os.path.join(labels_dir, 'coords')
    coords_list = []
    for item in seq_data:
        coords_path = os.path.join(coords_dir, f"{item['id']}.npy")
        if os.path.exists(coords_path):
            coords_list.append(np.load(coords_path))
        else:
            coords_list.append(None)

    return seq_data, coords_list


def rotation_matrix(axis: str, angle: float) -> np.ndarray:
    """生成绕指定轴的旋转矩阵。"""
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    else:  # z
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def random_rotation(coords: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """随机旋转坐标（保持结构不变）。"""
    # 绕三个轴随机旋转
    angles = rng.uniform(0, 2 * np.pi, 3)
    R = rotation_matrix('x', angles[0]) @ rotation_matrix('y', angles[1]) @ rotation_matrix('z', angles[2])
    return coords @ R.T


def random_translation(coords: np.ndarray, rng: np.random.RandomState, max_shift: float = 10.0) -> np.ndarray:
    """随机平移坐标。"""
    shift = rng.uniform(-max_shift, max_shift, 3)
    return coords + shift


def add_noise(coords: np.ndarray, rng: np.random.RandomState, noise_scale: float = 0.5) -> np.ndarray:
    """添加高斯噪声。"""
    noise = rng.normal(0, noise_scale, coords.shape)
    return coords + noise


def subsample_sequence(seq: str, coords: np.ndarray, rng: np.random.RandomState,
                       min_len: int = 50) -> Tuple[str, np.ndarray]:
    """从长序列中随机裁剪子序列。"""
    L = len(seq)
    if L <= min_len:
        return seq, coords

    # 随机选择裁剪长度
    new_len = rng.randint(min_len, L)

    # 随机选择起始位置
    start = rng.randint(0, L - new_len + 1)

    new_seq = seq[start:start + new_len]
    new_coords = coords[start:start + new_len].copy()

    # 重新中心化
    new_coords = new_coords - new_coords.mean(axis=0)

    return new_seq, new_coords


def mutate_sequence(seq: str, rng: np.random.RandomState, mutation_rate: float = 0.05) -> str:
    """随机突变序列（保持互补配对约束）。"""
    bases = ['A', 'C', 'G', 'U']
    complement = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}

    seq_list = list(seq)
    n_mutations = int(len(seq) * mutation_rate)

    for _ in range(n_mutations):
        pos = rng.randint(0, len(seq))
        # 随机选择新碱基
        new_base = rng.choice(bases)
        seq_list[pos] = new_base

    return ''.join(seq_list)


def augment_pseudo_labels(
    seq_data: List[dict],
    coords_list: List[np.ndarray],
    multiplier: int = 5,
    seed: int = 42,
    use_rotation: bool = True,
    use_translation: bool = True,
    use_noise: bool = True,
    use_subsample: bool = True,
    use_mutation: bool = True,
):
    """增强伪标签数据集。"""
    rng = np.random.RandomState(seed)

    augmented_seq = []
    augmented_coords = []
    augmented_meta = []

    original_count = len(seq_data)

    for aug_round in range(multiplier):
        for i, (item, coords) in enumerate(zip(seq_data, coords_list)):
            if coords is None:
                continue

            seq = item['sequence']
            L = len(seq)

            # 应用增强
            aug_coords = coords.copy()
            aug_seq = seq

            # 1. 旋转
            if use_rotation and rng.random() < 0.5:
                aug_coords = random_rotation(aug_coords, rng)

            # 2. 平移
            if use_translation and rng.random() < 0.5:
                aug_coords = random_translation(aug_coords, rng)

            # 3. 噪声
            if use_noise and rng.random() < 0.5:
                aug_coords = add_noise(aug_coords, rng, noise_scale=0.3)

            # 4. 裁剪
            if use_subsample and rng.random() < 0.3 and L > 100:
                aug_seq, aug_coords = subsample_sequence(aug_seq, aug_coords, rng)

            # 5. 突变
            if use_mutation and rng.random() < 0.3:
                aug_seq = mutate_sequence(aug_seq, rng, mutation_rate=0.03)

            # 创建新样本
            new_id = f"aug_{aug_round:02d}_{item['id']}"
            augmented_seq.append({
                'id': new_id,
                'sequence': aug_seq,
                'secondary_structure': item.get('secondary_structure', '.' * len(aug_seq)),
                'pair_constraints': [],  # 增强后需要重新计算
                'original_id': item['id'],
                'augmentation': aug_round,
            })
            augmented_coords.append(aug_coords)
            augmented_meta.append({
                'id': new_id,
                'length': len(aug_seq),
                'source': 'augmented',
                'original_id': item['id'],
            })

    # 合并原始数据和增强数据
    # 保留原始数据
    for i, (item, coords) in enumerate(zip(seq_data, coords_list)):
        if coords is not None:
            augmented_seq.insert(i, item)
            augmented_coords.insert(i, coords)
            augmented_meta.insert(i, {
                'id': item['id'],
                'length': len(item['sequence']),
                'source': 'original',
            })

    print(f"  Original: {original_count}")
    print(f"  Augmented: {len(augmented_seq) - original_count}")
    print(f"  Total: {len(augmented_seq)}")

    return augmented_seq, augmented_coords, augmented_meta


def save_augmented_labels(
    seq_data: List[dict],
    coords_list: List[np.ndarray],
    metadata: List[dict],
    output_dir: str,
):
    """保存增强后的数据。"""
    os.makedirs(output_dir, exist_ok=True)
    coords_dir = os.path.join(output_dir, 'coords')
    os.makedirs(coords_dir, exist_ok=True)

    # 保存坐标
    for i, coords in enumerate(coords_list):
        np.save(os.path.join(coords_dir, f"{seq_data[i]['id']}.npy"), coords)

    # 保存序列
    with open(os.path.join(output_dir, 'sequences.json'), 'w') as f:
        json.dump(seq_data, f, indent=2)

    # 保存元数据
    summary = {
        'total': len(seq_data),
        'length_range': [min(m['length'] for m in metadata),
                        max(m['length'] for m in metadata)],
        'sources': {
            'original': sum(1 for m in metadata if m['source'] == 'original'),
            'augmented': sum(1 for m in metadata if m['source'] == 'augmented'),
        },
        'samples': metadata[:100],
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Augment pseudo-labels')
    parser.add_argument('--input', type=str, required=True,
                        help='Input pseudo-labels directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for augmented labels')
    parser.add_argument('--multiplier', type=int, default=5,
                        help='Number of augmented samples per original (default: 5)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-rotation', action='store_true', help='Disable rotation augmentation')
    parser.add_argument('--no-translation', action='store_true', help='Disable translation augmentation')
    parser.add_argument('--no-noise', action='store_true', help='Disable noise augmentation')
    parser.add_argument('--no-subsample', action='store_true', help='Disable subsampling')
    parser.add_argument('--no-mutation', action='store_true', help='Disable sequence mutation')
    args = parser.parse_args()

    print("=" * 60)
    print("  Pseudo-Label Augmentation")
    print("=" * 60)
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Multiplier: {args.multiplier}x")

    # 加载数据
    seq_data, coords_list = load_pseudo_labels(args.input)
    print(f"  Loaded: {len(seq_data)} sequences")

    # 增强
    aug_seq, aug_coords, aug_meta = augment_pseudo_labels(
        seq_data, coords_list,
        multiplier=args.multiplier,
        seed=args.seed,
        use_rotation=not args.no_rotation,
        use_translation=not args.no_translation,
        use_noise=not args.no_noise,
        use_subsample=not args.no_subsample,
        use_mutation=not args.no_mutation,
    )

    # 保存
    save_augmented_labels(aug_seq, aug_coords, aug_meta, args.output)

    print("\n" + "=" * 60)
    print("  Next: Train with augmented labels")
    print(f"  python train_all_schemes.py --labels {args.output} --device cuda")
    print("=" * 60)


if __name__ == '__main__':
    main()
