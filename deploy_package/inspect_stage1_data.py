#!/usr/bin/env python3
"""检查 Stage 1 输出数据结构 + 统计序列长度分布"""

import json
import os
from collections import Counter

def check_structure(stage1_dir='stage1_merged', sample_size=1000):
    """检查 Stage 1 输出结构"""
    if not os.path.isdir(stage1_dir):
        print(f"错误: {stage1_dir} 不存在")
        return

    # 1. 查看第一条数据的完整结构
    seq_ids = sorted([d for d in os.listdir(stage1_dir) if d.startswith('circ_')])
    print(f"总序列数: {len(seq_ids)}")
    print(f"\n=== 第一条数据结构 (circ_000000) ===")
    path = os.path.join(stage1_dir, seq_ids[0], 'stage1_result.json')
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        print(json.dumps(data, indent=2)[:500])
        print(f"\n字段: {list(data.keys())}")

    # 2. 统计长度分布（抽样）
    print(f"\n=== 长度分布 (抽样 {sample_size} 条) ===")
    lengths = []
    sampled = 0
    for seq_id in seq_ids[:sample_size]:
        path = os.path.join(stage1_dir, seq_id, 'stage1_result.json')
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            lengths.append(len(data['sequence']))
            sampled += 1

    if not lengths:
        print("无数据")
        return

    print(f"抽样: {sampled} 条")
    print(f"长度范围: {min(lengths)} - {max(lengths)} nt")
    print(f"平均长度: {sum(lengths)/len(lengths):.0f} nt")
    print(f"中位数: {sorted(lengths)[len(lengths)//2]} nt")

    print(f"\n长度分布:")
    buckets = [(0,500),(500,1000),(1000,1500),(1500,2000),(2000,3000),(3000,5000),(5000,99999)]
    for lo, hi in buckets:
        cnt = sum(1 for l in lengths if lo <= l < hi)
        pct = cnt/len(lengths)*100
        bar = '█' * int(pct/2)
        print(f"  {lo:5d}-{hi:5d}: {cnt:4d} ({pct:5.1f}%) {bar}")

    # 3. 长序列占比（>2000nt 会被 Stage 2 跳过）
    long_count = sum(1 for l in lengths if l > 2000)
    print(f"\n长序列 (>2000nt): {long_count}/{sampled} ({long_count/sampled*100:.1f}%)")
    print(f"短序列 (≤2000nt): {sampled-long_count}/{sampled} ({(sampled-long_count)/sampled*100:.1f}%)")

if __name__ == '__main__':
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else 'stage1_merged'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    check_structure(src, n)
