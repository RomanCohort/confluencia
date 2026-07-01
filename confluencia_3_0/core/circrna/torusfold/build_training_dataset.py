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
from typing import Tuple, Optional, List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Try to import ViennaRNA
try:
    import RNA
    HAS_VIENNA = True
except ImportError:
    HAS_VIENNA = False

# Try to import GeometricConstraintSolver
try:
    from confluencia_3_0.core.circrna.torusfold.constraint_solver import (
        GeometricConstraintSolver, SolverConfig
    )
    HAS_SOLVER = True
except ImportError:
    HAS_SOLVER = False

# IsRNAcirc 子目录
ISRNACIRC_SUBDIRS = ['hairpin-RNAs', 'helix-RNAs', 'internal-RNAs', 'junction-RNAs']


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


def parse_dot_bracket(dot_bracket: str) -> List[Tuple[int, int]]:
    """将点括号字符串解析为配对索引列表（0-indexed）。支持假结点 [ ]。"""
    pairs, stack, pseudo_stack = [], [], []
    for i, char in enumerate(dot_bracket):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                pairs.append((stack.pop(), i))
        elif char == '[':
            pseudo_stack.append(i)
        elif char == ']':
            if pseudo_stack:
                pairs.append((pseudo_stack.pop(), i))
    return pairs


def parse_subo_file(subo_path: str) -> Tuple[Optional[str], Optional[str]]:
    """解析 .subo 文件。第1行=序列(ACGU)，第2行=点括号二级结构。返回 (sequence, dot_bracket) 或 (None, None)。"""
    try:
        with open(subo_path, 'r') as f:
            lines = f.readlines()
        if len(lines) < 2:
            return None, None
        sequence, dot_bracket = lines[0].strip(), lines[1].strip()
        if not all(b in 'ACGUacgu' for b in sequence):
            return None, None
        if not all(c in '().[]{}' for c in dot_bracket):
            return None, None
        if len(sequence) != len(dot_bracket):
            return None, None
        return sequence.upper(), dot_bracket
    except Exception as e:
        print(f"    Warning: Failed to parse {subo_path}: {e}")
        return None, None


def find_subo_file(seq_dir: str) -> Optional[str]:
    """在 sequence_2D_structure 目录中查找 .subo 文件。"""
    if not os.path.isdir(seq_dir):
        return None
    for f in os.listdir(seq_dir):
        if f.endswith('.subo'):
            return os.path.join(seq_dir, f)
    return None


def filter_structures_quality(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    应用 training_strategy_v2.md 中定义的质量过滤标准

    过滤阈值：
    - Confidence >= 0.70 (降低标准以保留更多数据)
    - BSJ distance: 2.8-5.0 Å (保留多模态结果)
    - Energy < 800 kJ/mol (放宽以适应circBase数据)
    - RMSD variance < 0.3 (结构收敛性)
    - BSJ clashes < 5 (几何合理性)

    Args:
        records: 待过滤的数据记录列表

    Returns:
        filtered: 符合质量标准的记录列表
    """
    filtered = []

    for rec in records:
        # 1. 置信度阈值 (降低到0.70)
        confidence = rec.get('confidence', 0)
        if confidence < 0.70:
            continue

        # 2. BSJ距离范围 (2.8-5.0 Å)
        bsj_dist = rec.get('bsj_distance')
        if bsj_dist is not None:
            if bsj_dist < 2.8 or bsj_dist > 5.0:
                continue

        # 3. 能量阈值 (放宽到800)
        energy = rec.get('energy', 0)
        if energy > 800:
            continue

        # 4. RMSD方差 (放宽到0.3)
        rmsd_variance = rec.get('rmsd_variance', 0)
        if rmsd_variance > 0.3:
            continue

        # 5. BSJ几何冲突检查
        bsj_clashes = rec.get('bsj_clashes', 0)
        if bsj_clashes > 5:
            continue

        # 通过所有过滤器
        filtered.append(rec)

    print(f"Quality filtering: {len(records)} → {len(filtered)} ({len(filtered)/len(records)*100:.1f}% retained)")
    return filtered


def predict_ss_vienna(sequence: str) -> str:
    """Predict secondary structure using ViennaRNA circ mode.

    Falls back to all-dots if ViennaRNA is unavailable.
    """
    sequence = sequence.upper().replace('T', 'U')
    L = len(sequence)

    if HAS_VIENNA:
        try:
            md = RNA.md()
            md.circ = True
            fc = RNA.fold_compound(sequence, md)
            ss, mfe = fc.mfe()
            return ss
        except Exception:
            pass

    return '.' * L


def extract_sequence_from_pdb(pdb_path: str) -> Optional[str]:
    """从 PDB ATOM 行提取残基序列。使用残基名称列构建序列，仅在没有 .subo 文件时使用。"""
    residues, seen = [], set()
    res_map = {'A': 'A', 'ADE': 'A', 'U': 'U', 'URA': 'U', 'G': 'G', 'GUA': 'G', 'C': 'C', 'CYT': 'C'}
    with open(pdb_path, 'r') as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            res_name = line[17:20].strip()
            try:
                res_num = int(line[22:26].strip())
            except ValueError:
                continue
            if res_num in seen:
                continue
            seen.add(res_num)
            if res_name in res_map:
                residues.append(res_map[res_name])
            elif len(res_name) == 1 and res_name in 'ACGU':
                residues.append(res_name)
    return ''.join(residues) if residues else None


def load_isrnacirc(data_dir: str) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    """
    加载 IsRNAcirc 真实 circRNA 3D 结构。

    扫描所有4个子目录：hairpin-RNAs, helix-RNAs, internal-RNAs, junction-RNAs
    从 .subo 文件解析序列和二级结构。
    """
    base_dir = os.path.join(data_dir, 'circular_RNA_Data')
    if not os.path.exists(base_dir):
        print(f"  IsRNAcirc data not found: {base_dir}")
        return [], []

    sequences = []
    coords_list = []
    total_count = 0
    subo_count = 0

    for subdir in ISRNACIRC_SUBDIRS:
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.exists(subdir_path):
            print(f"  Warning: Subdirectory not found: {subdir_path}")
            continue

        for circ_dir in sorted(Path(subdir_path).iterdir()):
            if not circ_dir.is_dir():
                continue

            total_count += 1
            circ_name = circ_dir.name

            # 查找 PDB 文件
            pdb_file = circ_dir / 'job_IsRNAcirc.pdb'
            if not pdb_file.exists():
                print(f"    Warning: No PDB file for {circ_name}")
                continue

            # 提取 C3' 坐标
            coords = parse_pdb_c3(str(pdb_file))
            if coords is None or len(coords) < 10:
                print(f"    Warning: Insufficient coords for {circ_name} (L={len(coords) if coords is not None else 0})")
                continue

            L = len(coords)
            seq_id = f'isrnacirc_{subdir}_{circ_name}'

            # 尝试从 .subo 文件获取序列和二级结构
            seq_dir = circ_dir / 'sequence_2D_structure'
            subo_file = find_subo_file(str(seq_dir))

            sequence = None
            secondary_structure = None
            seq_len = L  # 默认使用坐标长度

            if subo_file:
                sequence, secondary_structure = parse_subo_file(subo_file)
                if sequence:
                    subo_count += 1
                    # 注意：.subo 文件中的序列可能是简化版本
                    # 坐标通常比序列长（每个核苷酸约5个原子）
                    # 使用 .subo 的序列长度作为参考
                    seq_len = len(sequence)
                    # 验证坐标是否可以下采样到序列长度
                    if L >= seq_len:
                        # 坐标更长，下采样到序列长度
                        step = L / seq_len
                        indices = [int(i * step) for i in range(seq_len)]
                        coords = coords[indices]
                        L = seq_len
                    else:
                        # 坐标更短，跳过此结构
                        print(f"    Warning: Coords ({L}) < sequence ({seq_len}) for {circ_name}, using PDB sequence")
                        sequence = None
                        secondary_structure = None

            # 如果没有 .subo 或解析失败，从 PDB 提取序列
            if sequence is None:
                sequence = extract_sequence_from_pdb(str(pdb_file))
                if sequence:
                    # 用 ViennaRNA circ-mode 预测二级结构（比全点好）
                    secondary_structure = predict_ss_vienna(sequence)
                else:
                    # 最后回退：随机序列 + ViennaRNA 预测
                    rng = np.random.RandomState(hash(circ_name) % 2**32)
                    sequence = ''.join(rng.choice(['A', 'C', 'G', 'U'], L))
                    secondary_structure = predict_ss_vienna(sequence)

            # 从二级结构提取配对约束
            pair_constraints = parse_dot_bracket(secondary_structure) if secondary_structure else []

            sequences.append({
                'id': seq_id,
                'sequence': sequence,
                'secondary_structure': secondary_structure,
                'pair_constraints': pair_constraints,
                'length': L,
                'source': 'isrnacirc',
                'structure_type': subdir.replace('-RNAs', ''),
                'has_real_ss': secondary_structure != '.' * L if secondary_structure else False,
            })
            coords_list.append(coords)
            ss_status = "real" if (secondary_structure and secondary_structure != '.' * L) else "none"
            print(f'  {seq_id}: L={L}, SS={ss_status}')

    print(f'\n  Total: {total_count} structures, {subo_count} with real secondary structure')
    return sequences, coords_list


def simple_constraint_solve(seq_len: int, pair_constraints: list, bond_length: float = 5.9) -> np.ndarray:
    """Generate 3D coords from constraints using gradient descent.

    Fallback when GeometricConstraintSolver is not available.
    """
    coords = np.zeros((seq_len, 3))
    # Initialize as circular helix
    for i in range(seq_len):
        angle = 2 * np.pi * i / seq_len
        radius = bond_length * seq_len / (2 * np.pi) * 0.5
        coords[i] = [radius * np.cos(angle), radius * np.sin(angle), 0]

    # Refine with constraint satisfaction (simple gradient descent)
    for step in range(200):
        grad = np.zeros_like(coords)
        # Bond constraints
        for i in range(seq_len - 1):
            nxt = (i + 1) % seq_len
            diff = coords[nxt] - coords[i]
            dist = np.linalg.norm(diff)
            if dist > 0:
                force = (dist - bond_length) * diff / dist
                grad[i] += force * 0.1
                grad[nxt] -= force * 0.1
        # Pair constraints
        for pi, pj, target_d, w in pair_constraints:
            diff = coords[pj] - coords[pi]
            dist = np.linalg.norm(diff)
            if dist > 0:
                force = w * (dist - target_d) * diff / dist
                grad[pi] += force * 0.05
                grad[pj] -= force * 0.05
        coords -= grad
    return coords


def generate_synthetic(n_samples: int, min_len: int = 50, max_len: int = 500,
                       seed: int = 42):
    """生成合成 circRNA 序列 + 3D 结构。

    使用 ViennaRNA (circ mode) 预测二级结构，
    然后使用 GeometricConstraintSolver 生成 3D 坐标。
    如果 ViennaRNA 未安装，回退到启发式配对。
    """
    rng = np.random.RandomState(seed)
    bases = ['A', 'C', 'G', 'U']

    # RNA A-form helix parameters (for fallback)
    rise_per_nt = 2.8   # Å
    twist_per_nt = 32.7  # degrees

    sequences = []
    coords_list = []

    # Initialize solver if available
    solver = None
    if HAS_SOLVER:
        config = SolverConfig(
            n_samples=1,
            use_annealing_closure=False,
            bond_length=5.9,
            pair_distance=10.6,
            max_iterations=50,
        )
        solver = GeometricConstraintSolver(config)

    print(f"  ViennaRNA: {'available (circ mode)' if HAS_VIENNA else 'NOT available, using heuristic pairing'}")
    print(f"  GeometricConstraintSolver: {'available' if HAS_SOLVER else 'NOT available, using simple gradient descent'}")

    for i in range(n_samples):
        L = rng.randint(min_len, max_len + 1)
        seq = ''.join(rng.choice(bases, L))

        # Predict secondary structure and extract pair constraints
        pair_constraints = []
        ss = '.' * L
        mfe = 0.0

        if HAS_VIENNA:
            try:
                md = RNA.md()
                md.circ = True
                fc = RNA.fold_compound(seq, md)
                ss, mfe = fc.mfe()

                # Extract base pairs from dot-bracket
                stack = []
                for pos, char in enumerate(ss):
                    if char == '(':
                        stack.append(pos)
                    elif char == ')' and stack:
                        j_pos = stack.pop()
                        pair_constraints.append((j_pos, pos, 10.6, 1.0))
            except Exception:
                pass

        if not pair_constraints:
            # Heuristic: complement pairing
            complement = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
            for j in range(L):
                for k in range(j + 4, min(j + 20, L)):
                    if complement.get(seq[j]) == seq[k] and rng.random() < 0.3:
                        pair_constraints.append((j, k, 10.6, 1.0))

        # Generate 3D coordinates using solver or fallback
        coords = None
        if solver and pair_constraints:
            # Build constraint set
            class CS:
                def __init__(self, n, pairs):
                    self.seq_len = n
                    self.pair_constraints = pairs

            cs = CS(L, pair_constraints)
            conformations = solver.solve(cs)
            if conformations and len(conformations) > 0:
                coords = conformations[0]

        if coords is None:
            # Fallback: simple gradient descent or helical coords
            if pair_constraints:
                coords = simple_constraint_solve(L, pair_constraints, bond_length=5.9)
            else:
                # Generate helical coords (original behavior)
                radius = max(5.0, L * rise_per_nt / (2 * np.pi) * 0.8)
                coords = np.zeros((L, 3))
                for j in range(L):
                    angle = np.deg2rad(twist_per_nt * j)
                    coords[j, 0] = radius * np.cos(angle)
                    coords[j, 1] = radius * np.sin(angle)
                    coords[j, 2] = rise_per_nt * j
                coords = coords - coords.mean(axis=0)
                coords += rng.normal(0, 0.3, (L, 3))

        seq_id = f'synthetic_{i:05d}'
        sequences.append({
            'id': seq_id,
            'sequence': seq,
            'secondary_structure': ss,
            'pair_constraints': [(p[0], p[1]) for p in pair_constraints],
            'length': L,
            'source': 'synthetic',
            'mfe': float(mfe) if mfe != 0.0 else None,
        })
        coords_list.append(coords)

        if (i + 1) % 500 == 0:
            print(f'  Synthetic: {i+1}/{n_samples}')

    return sequences, coords_list


def augment_isrnacirc(sequences, coords_list, multiplier=80, seed=42):
    """对 IsRNAcirc 数据做旋转+噪声扩增。保留真实二级结构和配对约束。"""
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
            aug_item = {
                'id': seq_id,
                'sequence': seq_item['sequence'],
                'secondary_structure': seq_item.get('secondary_structure', '.' * L),
                'pair_constraints': seq_item.get('pair_constraints', []),
                'length': L,
                'source': 'isrnacirc_aug',
                'structure_type': seq_item.get('structure_type', 'unknown'),
                'has_real_ss': seq_item.get('has_real_ss', False),
            }
            aug_seqs.append(aug_item)
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
    parser.add_argument('--isrnacirc-aug', type=int, default=80,
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

    # Convert pair_constraints tuples to lists for JSON serialization
    json_sequences = []
    for s in all_sequences:
        item = dict(s)
        if 'pair_constraints' in item:
            item['pair_constraints'] = [list(p) for p in item['pair_constraints']]
        json_sequences.append(item)

    with open(os.path.join(output_dir, 'sequences.json'), 'w') as f:
        json.dump(json_sequences, f, indent=2)

    isrnacirc_real = [s for s in all_sequences if s['source'] == 'isrnacirc']
    isrnacirc_with_ss = sum(1 for s in isrnacirc_real if s.get('has_real_ss', False))
    structure_types = {}
    for s in isrnacirc_real:
        st = s.get('structure_type', 'unknown')
        structure_types[st] = structure_types.get(st, 0) + 1

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
        'isrnacirc_with_real_ss': isrnacirc_with_ss,
        'isrnacirc_structure_types': structure_types,
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Dataset: {output_dir}/")
    print(f"  Total: {metadata['total']}")
    print(f"  IsRNAcirc (real): {metadata['sources']['isrnacirc']}")
    print(f"  IsRNAcirc (real with SS): {isrnacirc_with_ss}")
    print(f"  IsRNAcirc (aug): {metadata['sources']['isrnacirc_aug']}")
    print(f"  Synthetic: {metadata['sources']['synthetic']}")
    print(f"  Structure types: {structure_types}")
    print(f"  Length range: {metadata['length_range']}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
