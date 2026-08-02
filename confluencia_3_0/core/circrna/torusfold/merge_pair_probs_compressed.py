"""merge_pair_probs_compressed.py — 流式合并 42 个 precompute 分片成压缩 npz。

背景: precompute_pair_probs.py 的合并步骤把 42 分片的 bp_probs 全部 extend 进
内存再 np.savez → 165GB 峰值 OOM 崩溃（本地）。分片 raw 共 165GB 也太大,
无法直接上传 A800。

本脚本: 逐分片逐样本 gzip.compress → 存进 object 数组。压缩率约 5%
(165GB → ~8GB), 峰值内存只占一个分片 (~7GB), 流式友好。

产物: circrna_3d_all_pair_probs_compressed.npz
  - ids:       object array, 82106 个 str (与 CG npz 对齐)
  - lengths:   int32 array, 82106
  - bp_probs:  object array, 82106 个 gzip-compressed bytes
    (解压后是 [L,L] float32, 即 ViennaRNA bpp)

训练脚本读取处 (train_s10_curriculum.py ~line 274) 需配合解压:
    pp_data = np.load(path, allow_pickle=True)
    pp_ids = {str(x): i for i, x in enumerate(pp_data['ids'])}
    pp_arr = pp_data['bp_probs']
    pair_probs = [np.frombuffer(gzip.decompress(pp_arr[idx]), dtype=np.float32).reshape(L, L)
                  if cid in pp_ids else None for cid, L in ...]

用法:
    python merge_pair_probs_compressed.py \\
        --shard-dir ../../../../data/.precompute_tmp \\
        --output ../../../../data/circrna_3d_all_pair_probs_compressed.npz \\
        --n-shards 42
"""
from __future__ import annotations

import argparse
import gzip
import os
import sys
import time
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True, help=".precompute_tmp 目录")
    ap.add_argument("--output", required=True, help="压缩后输出 npz 路径")
    ap.add_argument("--n-shards", type=int, default=42)
    ap.add_argument("--n-threads", type=int, default=8)
    args = ap.parse_args()

    shard_dir = Path(args.shard_dir)
    t0 = time.time()

    all_ids: list[str] = []
    all_lens: list[int] = []
    all_bp: list[bytes] = []

    for ci in range(args.n_shards):
        shard = shard_dir / f"bp_{ci}.npz"
        if not shard.exists():
            print(f"  [skip] {shard.name} 不存在 (已算分片不足 {args.n_shards})")
            continue
        d = np.load(shard, allow_pickle=True)
        ids = list(d["ids"])
        lens = list(d["lengths"])
        bps = d["bp_probs"]
        n = len(ids)
        for j in range(n):
            all_ids.append(str(ids[j]))
            all_lens.append(int(lens[j]))
            all_bp.append(gzip.compress(bps[j].tobytes(), compresslevel=3))
        print(f"  [{ci+1}/{args.n_shards}] {shard.name}: {n} 样本压缩 "
              f"(elapsed {time.time()-t0:.0f}s, 累计 {len(all_bp)})")

    print(f"\n总计 {len(all_bp)} 样本, 压缩后 {sum(len(b) for b in all_bp)/1e9:.2f}GB "
          f"(raw {sum(np.prod(np.fromstring(b, dtype=np.uint8).shape, dtype=int) for b in []):.0f})")
    print(f"写入 {args.output} ...")
    np.savez_compressed(
        args.output,
        ids=np.array(all_ids, dtype=object),
        lengths=np.array(all_lens, dtype=np.int32),
        bp_probs=np.array(all_bp, dtype=object),
    )
    sz = os.path.getsize(args.output) / 1e9
    print(f"完成: {args.output} = {sz:.2f}GB, 耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
