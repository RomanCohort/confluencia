"""regenerate_cg_coords.py — 用 Phase 0 模型 + physics_refine 重生成 CG 坐标.

背景: circrna_3d_all_consolidated.npz 的 coords 98.5% 是随机噪声 (generate_32_
workers 无预训练模型时 fallback rng.randn). Phase 1-4 在噪声上训练无意义.

本脚本: 读 CG npz 的 ids/lengths + circbase FASTA 序列, 用 Phase 0 checkpoint
推理坐标 + physics_refine (20 步), 生成与 npz 对齐的新 coords 数组, 写回新 npz.

多进程: 每 worker 一个 GPU 推理循环 (CPU-bound collate 用多进程并行).
用法 (A800, torusfold 目录):
    python regenerate_cg_coords.py \
        --ckpt models/s10_curriculum/phase0_end_full.pt \
        --n-workers 16 --max-len 2500 \
        --out ../../../../data/circrna_3d_all_consolidated_v2.npz

速度参考 (本地 ROCm): ~3.4s/条 (含 refine), 82106 条 16 worker ≈ 4.8h.
A800 (更强 GPU) 预计 ~1-2h.
"""
from __future__ import annotations

import argparse
import gzip
import os
import sys
import time
from multiprocessing import Process, Queue, cpu_count

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_fasta_ids(path: str) -> dict:
    seq_map = {}
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as f:
        name = None; buf = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if name: seq_map[name] = ''.join(buf)
                name = line[1:].split('|')[0].split()[0]
                buf = []
            else:
                buf.append(line)
        if name: seq_map[name] = ''.join(buf)
    return seq_map


def worker_fn(wid: int, in_q: Queue, out_q: Queue, ckpt: str, device_id: int, refine_steps: int):
    """One worker: pull (idx, seq, L) tasks, run model+refine, push (idx, coords)."""
    import torch
    from scheme10_equivariant import EquivariantS10Config, StrictlyEquivariantS10
    from physics_refine import refine_coords

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = EquivariantS10Config(
        d_model=256, d_inv=64, d_eq=32, n_layers=4,
        k_theta=4, k_phi=2, use_coord_diffusion=True, n_diffusion_steps=20,
        d_coord_hidden=128, cfg_dropout_prob=0.1,
        use_s8_refine=True, use_adaptive_k=True,
        d_model_inv=64, d_model_eq=64, dropout=0.1,
        n_tokens=5, bond_length=5.9, r_scale=300.0,
    )
    model = StrictlyEquivariantS10(cfg).to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=True)
    sd = ck.get('model_state_dict', ck)
    model.load_state_dict(sd)
    model.eval()
    tok = {'A': 0, 'U': 1, 'G': 2, 'C': 3}

    print(f'  worker {wid} ready on {device}', flush=True)
    while True:
        task = in_q.get()
        if task is None:
            break
        idx, seq, L = task
        try:
            s_ids = torch.tensor([tok.get(b, 4) for b in seq], dtype=torch.long).unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    pred = model(s_ids, return_loss=False)
                if isinstance(pred, tuple):
                    pred = pred[0]
                coords = pred[0, :L].cpu().float().unsqueeze(0)
                if refine_steps > 0:
                    l_t = torch.tensor([L], dtype=torch.long).to(device)
                    coords = refine_coords(coords.to(device), l_t,
                                           n_steps=refine_steps, project_bonds=True)
            c = coords[0].cpu().numpy()
            if np.isnan(c).any() or np.isinf(c).any():
                raise RuntimeError('NaN in output')
            out_q.put((idx, c))
        except Exception as e:
            out_q.put((idx, None, str(e)[:80]))
            print(f'  worker {wid} err idx={idx} L={L}: {e}', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='models/s10_curriculum/phase0_end_full.pt')
    ap.add_argument('--data', default=None, help='CG npz (默认 data/circrna_3d_all_consolidated.npz)')
    ap.add_argument('--fasta', default=None, help='circbase FASTA')
    ap.add_argument('--n-workers', type=int, default=16)
    ap.add_argument('--max-len', type=int, default=2500, help='跳过超长序列 (保留原坐标)')
    ap.add_argument('--refine-steps', type=int, default=20)
    ap.add_argument('--out', default=None, help='输出 npz (默认 *_regenerated.npz)')
    ap.add_argument('--resume', default=None, help='已有部分结果 npy 目录 (续跑)')
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 条 (测试用)')
    args = ap.parse_args()

    # torusfold → circrna → core → confluencia_3_0 → 项目根 (parents[4])
    base = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[os.pardir]*4))
    data_path = args.data or os.path.join(base, 'data', 'circrna_3d_all_consolidated.npz')
    fasta_path = args.fasta or os.path.join(base, 'data', 'circrna', 'circbase_seqs.fa.gz')
    out_path = args.out or data_path.replace('.npz', '_regenerated.npz')
    print(f'Data:  {data_path}')
    print(f'FASTA: {fasta_path}')
    print(f'Out:   {out_path}')

    npz = np.load(data_path, allow_pickle=True)
    ids = [str(x) for x in npz['ids']]
    lens = [int(x) for x in npz['lengths']]
    n = len(ids)
    seq_map = load_fasta_ids(fasta_path)
    print(f'  {n} samples, {len(seq_map)} FASTA seqs, '
          f'{sum(1 for x in ids if x in seq_map)} matched')

    # ── 组装任务 ──────────────────────────────────────────────────────
    tasks = []
    for idx, (cid, L) in enumerate(zip(ids, lens)):
        if L > args.max_len:
            continue  # 超长跳过 (保留原坐标)
        seq = seq_map.get(cid)
        if not seq:
            continue
        seq = seq[:L].upper().replace('T', 'U')
        tasks.append((idx, seq, L))
    if args.limit > 0:
        tasks = tasks[:args.limit]
    print(f'  待重生成: {len(tasks)} 条 (跳过超长/无序列 {n - len(tasks)})')

    # ── 多进程分发 ─────────────────────────────────────────────────────
    in_q, out_q = Queue(), Queue()
    procs = []
    for w in range(min(args.n_workers, cpu_count())):
        p = Process(target=worker_fn, args=(w, in_q, out_q, args.ckpt, 0, args.refine_steps),
                    daemon=True)
        p.start()
        procs.append(p)
    for t in tasks:
        in_q.put(t)
    for _ in procs:
        in_q.put(None)  # sentinel

    # ── 收集结果 (内存中保留新 coords, 因为要写回 npz) ──────────────────
    new_coords = [None] * n
    done = 0
    t0 = time.time()
    while done < len(tasks):
        res = out_q.get()
        if len(res) == 3:
            idx, _, err = res
            print(f'  [{done}/{len(tasks)}] err {idx}: {err}', flush=True)
        else:
            idx, c = res
            new_coords[idx] = c
        done += 1
        if done % 500 == 0:
            el = time.time() - t0
            print(f'  {done}/{len(tasks)} done, {el:.0f}s, '
                  f'ETA {(el/done)*(len(tasks)-done)/60:.0f}min', flush=True)

    for p in procs:
        p.join()

    # ── 写回 ──────────────────────────────────────────────────────────
    old_coords = npz['coords']
    final = []
    n_new = 0
    for idx in range(n):
        c = new_coords[idx]
        if c is not None:
            final.append(c)
            n_new += 1
        else:
            final.append(old_coords[idx])
    print(f'  新坐标: {n_new}/{n}, 保留原: {n - n_new}')
    np.savez(out_path,
             ids=np.array(ids, dtype=object),
             lengths=np.array(lens, dtype=np.int32),
             coords=np.array(final, dtype=object))
    print(f'  写入 {out_path} ({os.path.getsize(out_path)/1e9:.2f}GB)')


if __name__ == '__main__':
    main()
