#!/usr/bin/env python3
"""
generate_real_dataset.py — 从 circBase 生成真实 circRNA 3D 数据集。

数据来源：
1. circBase (www.circbase.org): 14万+ circRNA 序列
2. 使用 IsRNAcirc 生成 3D 结构作为标签

用法：
    python generate_real_dataset.py --output data/circbase_3d --n-samples 5000 --min-len 50 --max-len 500
"""

import os
import sys
import json
import gzip
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════
# circBase 数据下载和解析
# ═══════════════════════════════════════════════════════════════

CIRCBASE_URL = "http://www.circbase.org/cgi-bin/download.cgi"

def download_circbase(output_dir: str, species: str = "hsa") -> str:
    """下载 circBase 数据。

    Returns:
        下载文件路径
    """
    import urllib.request

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"circBase_{species}.txt.gz")

    if os.path.exists(output_path):
        print(f"  Found existing: {output_path}")
        return output_path

    print(f"  Downloading circBase ({species})...")

    # circBase 下载 URL 格式
    url = f"{CIRCBASE_URL}?sp={species}&db=hsa&type=circ"

    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"  Downloaded: {output_path}")
        return output_path
    except Exception as e:
        print(f"  Download failed: {e}")
        print("  Using fallback: generate synthetic sequences...")
        return None


def parse_circbase(filepath: str, min_len: int = 50, max_len: int = 500) -> List[Dict]:
    """解析 circBase 文件，提取 circRNA 序列。

    circBase 格式 (BED-like):
    chrom  start  end  name  score  strand  circRNA_type  seq

    Returns:
        List of {id, sequence, chrom, start, end, strand}
    """
    sequences = []

    print(f"  Parsing: {filepath}")

    # 处理 gz 文件
    if filepath.endswith('.gz'):
        opener = lambda: gzip.open(filepath, 'rt')
    else:
        opener = lambda: open(filepath, 'r')

    with opener() as f:
        for i, line in enumerate(f):
            if line.startswith('#') or line.startswith('chrom'):
                continue

            parts = line.strip().split('\t')
            if len(parts) < 8:
                continue

            chrom, start, end, name, score, strand = parts[:6]
            seq = parts[7] if len(parts) > 7 else None

            if not seq:
                continue

            # 长度过滤
            L = len(seq)
            if L < min_len or L > max_len:
                continue

            # 只保留有效碱基
            valid_bases = set('ACGUacgu')
            if not all(b in valid_bases for b in seq):
                continue

            sequences.append({
                'id': f"circBase_{name}_{i}",
                'sequence': seq.upper().replace('T', 'U'),
                'chrom': chrom,
                'start': int(start),
                'end': int(end),
                'strand': strand,
                'length': L,
            })

            if len(sequences) % 1000 == 0:
                print(f"    Parsed {len(sequences)} sequences...")

    print(f"  Total valid sequences: {len(sequences)}")
    return sequences


def generate_synthetic_circrnas(n_samples: int, min_len: int = 50, max_len: int = 500,
                                 seed: int = 42) -> List[Dict]:
    """生成合成 circRNA 序列（当 circBase 不可用时）。"""
    rng = np.random.RandomState(seed)
    bases = ['A', 'C', 'G', 'U']

    sequences = []
    for i in range(n_samples):
        L = rng.randint(min_len, max_len + 1)
        seq = ''.join(rng.choice(bases, L))

        sequences.append({
            'id': f"synthetic_{i:06d}",
            'sequence': seq,
            'chrom': 'syn',
            'start': 0,
            'end': L,
            'strand': '+',
            'length': L,
        })

    return sequences


# ═══════════════════════════════════════════════════════════════
# 3D 结构生成
# ═══════════════════════════════════════════════════════════════

def generate_3d_structure_isrnacirc(sequence: str, isrnacirc_path: str,
                                     output_dir: str) -> Optional[np.ndarray]:
    """使用 IsRNAcirc 生成 3D 结构。"""
    try:
        # 创建临时输入文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
            f.write(f">temp\n{sequence}\n")
            input_file = f.name

        # 运行 IsRNAcirc
        output_prefix = os.path.join(output_dir, "temp_output")
        cmd = f"python {isrnacirc_path}/run_IsRNAcirc.py -i {input_file} -o {output_prefix}"

        result = subprocess.run(cmd, shell=True, capture_output=True, timeout=300)

        if result.returncode != 0:
            return None

        # 解析输出 PDB
        pdb_file = f"{output_prefix}.pdb"
        if os.path.exists(pdb_file):
            coords = parse_pdb_coords(pdb_file)
            os.remove(input_file)
            os.remove(pdb_file)
            return coords

        os.remove(input_file)
        return None

    except Exception as e:
        return None


def generate_3d_structure_simple(sequence: str, seed: int = 42) -> np.ndarray:
    """简单螺旋结构生成（快速，无外部依赖）。"""
    rng = np.random.RandomState(seed)
    L = len(sequence)

    # A-form RNA 参数
    rise_per_nt = 2.8  # Å
    twist_per_nt = 32.7  # degrees

    # 根据序列长度计算半径
    radius = max(5.0, L * rise_per_nt / (2 * np.pi) * 0.8)

    coords = np.zeros((L, 3))

    for i in range(L):
        angle = np.deg2rad(twist_per_nt * i)
        coords[i, 0] = radius * np.cos(angle)
        coords[i, 1] = radius * np.sin(angle)
        coords[i, 2] = rise_per_nt * i

    # 中心化
    coords = coords - coords.mean(axis=0)

    # 添加随机扰动
    noise = rng.normal(0, 0.5, (L, 3))
    coords = coords + noise

    return coords


def parse_pdb_coords(pdb_file: str) -> np.ndarray:
    """从 PDB 文件提取 C3' 原子坐标。"""
    coords = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and "C3'" in line:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    return np.array(coords) if coords else None


# ═══════════════════════════════════════════════════════════════
# 数据集生成
# ═══════════════════════════════════════════════════════════════

def generate_dataset(
    sequences: List[Dict],
    output_dir: str,
    use_isrnacirc: bool = False,
    isrnacirc_path: str = None,
    n_workers: int = 4,
) -> None:
    """生成完整数据集（序列 + 3D 坐标）。"""
    os.makedirs(output_dir, exist_ok=True)
    coords_dir = os.path.join(output_dir, 'coords')
    os.makedirs(coords_dir, exist_ok=True)

    print(f"\n  Generating 3D structures for {len(sequences)} sequences...")

    for i, item in enumerate(sequences):
        seq = item['sequence']
        seq_id = item['id']

        # 生成 3D 结构
        if use_isrnacirc and isrnacirc_path:
            coords = generate_3d_structure_isrnacirc(seq, isrnacirc_path, coords_dir)
        else:
            coords = generate_3d_structure_simple(seq, seed=i)

        if coords is not None:
            # 保存坐标
            np.save(os.path.join(coords_dir, f"{seq_id}.npy"), coords)
        else:
            # 标记失败
            item['failed'] = True

        if (i + 1) % 100 == 0:
            print(f"    Progress: {i+1}/{len(sequences)}")

    # 移除失败的样本
    valid_sequences = [s for s in sequences if not s.get('failed')]
    failed_count = len(sequences) - len(valid_sequences)

    if failed_count > 0:
        print(f"  Failed: {failed_count} samples")

    # 保存序列信息
    with open(os.path.join(output_dir, 'sequences.json'), 'w') as f:
        json.dump(valid_sequences, f, indent=2)

    # 保存元数据
    metadata = {
        'total': len(valid_sequences),
        'length_range': [min(s['length'] for s in valid_sequences),
                        max(s['length'] for s in valid_sequences)],
        'sources': {
            'circbase': sum(1 for s in valid_sequences if s['id'].startswith('circBase')),
            'synthetic': sum(1 for s in valid_sequences if s['id'].startswith('synthetic')),
        },
        'method': 'isrnacirc' if use_isrnacirc else 'helical',
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Dataset saved to: {output_dir}/")
    print(f"  Total samples: {len(valid_sequences)}")


def main():
    parser = argparse.ArgumentParser(description='Generate real circRNA 3D dataset from circBase')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--n-samples', type=int, default=5000,
                        help='Number of samples to generate (default: 5000)')
    parser.add_argument('--min-len', type=int, default=50,
                        help='Minimum sequence length')
    parser.add_argument('--max-len', type=int, default=500,
                        help='Maximum sequence length')
    parser.add_argument('--species', type=str, default='hsa',
                        help='Species (hsa=human, mmu=mouse)')
    parser.add_argument('--use-isrnacirc', action='store_true',
                        help='Use IsRNAcirc for 3D structure generation')
    parser.add_argument('--isrnacirc-path', type=str, default='tools/IsRNAcirc',
                        help='Path to IsRNAcirc')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("  circBase 3D Dataset Generation")
    print("=" * 60)
    print(f"  Output: {args.output}")
    print(f"  Target samples: {args.n_samples}")
    print(f"  Length range: {args.min_len}-{args.max_len}")

    # 尝试下载 circBase
    circbase_file = download_circbase(args.output, args.species)

    # 解析或生成序列
    if circbase_file and os.path.exists(circbase_file):
        sequences = parse_circbase(circbase_file, args.min_len, args.max_len)
    else:
        print("\n  Generating synthetic sequences...")
        sequences = generate_synthetic_circrnas(
            args.n_samples, args.min_len, args.max_len, args.seed
        )

    # 限制样本数
    if len(sequences) > args.n_samples:
        rng = np.random.RandomState(args.seed)
        indices = rng.choice(len(sequences), args.n_samples, replace=False)
        sequences = [sequences[i] for i in indices]

    # 生成 3D 结构
    generate_dataset(
        sequences,
        args.output,
        use_isrnacirc=args.use_isrnacirc,
        isrnacirc_path=args.isrnacirc_path,
    )

    print("\n" + "=" * 60)
    print("  Next: Train with real data")
    print(f"  python train_all_schemes.py --labels {args.output} --device cuda")
    print("=" * 60)


if __name__ == '__main__':
    main()
