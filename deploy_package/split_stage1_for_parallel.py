#!/usr/bin/env python3
"""把 stage1_merged 切成 N 份，用软链接避免占空间，用于 Stage 2 并行处理"""

import os
import sys
import math

def split_stage1(src_dir='stage1_merged', n_chunks=30, prefix='stage1_chunk'):
    """把 src_dir 里的 circ_* 目录切成 n_chunks 份（软链接）"""
    if not os.path.isdir(src_dir):
        print(f"错误: {src_dir} 不存在")
        sys.exit(1)

    all_seqs = sorted([d for d in os.listdir(src_dir) if d.startswith('circ_')])
    n = len(all_seqs)
    chunk_size = math.ceil(n / n_chunks)

    print(f"总序列数: {n}")
    print(f"切分成 {n_chunks} 份，每份约 {chunk_size} 条\n")

    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n)
        chunk_seqs = all_seqs[start:end]

        chunk_dir = f'{prefix}_{i:02d}'
        os.makedirs(chunk_dir, exist_ok=True)

        # 清空旧链接
        for old in os.listdir(chunk_dir):
            old_path = os.path.join(chunk_dir, old)
            if os.path.islink(old_path):
                os.unlink(old_path)

        for seq_id in chunk_seqs:
            src = os.path.abspath(os.path.join(src_dir, seq_id))
            dst = os.path.join(chunk_dir, seq_id)
            if not os.path.exists(dst):
                os.symlink(src, dst)

        print(f"{chunk_dir}: {len(chunk_seqs)} sequences")

    print(f"\n完成！共 {n_chunks} 份")
    print(f"\n下一步并行运行 Stage 2:")
    print(f"  for i in $(seq -w 0 {n_chunks-1}); do")
    print(f"    CUDA_VISIBLE_DEVICES=$((i % 4)) python run_stage2_only.py \\")
    print(f"      --input {prefix}_$i --output stage2_output_$i &")
    print(f"  done")

if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else 'stage1_merged'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    split_stage1(src, n)
