"""validate_cg_pseudolabel.py — 验证 Phase 0 模型在 CG 序列上的泛化质量.

背景: CG 训练数据 (circrna_3d_all_consolidated.npz) 98.5% 是随机噪声坐标
(generate_32_workers.py 无预训练模型时 rng.randn(L,3)*10 fallback). 训练
Phase 1-4 前必须先确认 Phase 0 模型能否对 CG 序列生成合理结构.

本脚本: 加载 phase0 checkpoint, 对随机抽的 N 条 CG 序列做推理, 检查:
  1. 键长分布 (P-P 理想 5.9Å, 合理范围 4.5-7.5)
  2. 坐标尺度 / 无 NaN
  3. 与现有噪声数据的对比

用法 (在 A800, torusfold 目录):
    python validate_cg_pseudolabel.py [--ckpt models/s10_curriculum/phase0_end_full.pt] \
        [--n 30] [--max-len 2000]
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import torch

# 确保能 import torusfold 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scheme10_equivariant import EquivariantS10Config, StrictlyEquivariantS10  # noqa: E402
from physics_refine import refine_coords  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='models/s10_curriculum/phase0_end_full.pt')
    ap.add_argument('--n', type=int, default=30, help='采样序列数')
    ap.add_argument('--max-len', type=int, default=2000, help='过滤超长序列 (推理慢)')
    ap.add_argument('--data', default=None, help='CG npz 路径 (默认 data/circrna_3d_all_consolidated.npz)')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')

    # ── 加载 Phase 0 模型 ──────────────────────────────────────────────
    cfg = EquivariantS10Config(
        d_model=256, d_inv=64, d_eq=32, n_layers=4,
        k_theta=4, k_phi=2, use_coord_diffusion=True, n_diffusion_steps=20,
        d_coord_hidden=128, cfg_dropout_prob=0.1,
        use_s8_refine=True, use_adaptive_k=True,
        d_model_inv=64, d_model_eq=64, dropout=0.1,
        n_tokens=5, bond_length=5.9, r_scale=300.0,
    )
    model = StrictlyEquivariantS10(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    sd = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(sd)
    model.eval()
    print(f'  Model loaded: {sum(p.numel() for p in model.parameters()):,} params')

    # ── 加载 CG 数据 (只取 ids/seqs, 不碰 coords) ───────────────────────
    data_path = args.data or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..',
        'data', 'circrna_3d_all_consolidated.npz')
    data_path = os.path.abspath(data_path)
    print(f'  Data: {data_path}')
    npz = np.load(data_path, allow_pickle=True)
    ids, lens = npz['ids'], npz['lengths']

    # 需要序列 — CG npz 没有 seq, 用 FASTA
    fasta_path = os.path.join(os.path.dirname(data_path), 'circrna', 'circbase_seqs.fa.gz')
    seq_map = {}
    if os.path.isfile(fasta_path):
        import gzip
        with gzip.open(fasta_path, 'rt') as f:
            name = None; buf = []
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if name: seq_map[name] = ''.join(buf)
                    name = line[1:].split('|')[0].split()[0]  # 取第一个 | 前
                    buf = []
                else:
                    buf.append(line)
            if name: seq_map[name] = ''.join(buf)
        print(f'  FASTA: {len(seq_map)} seqs')

    tok = {'A': 0, 'U': 1, 'G': 2, 'C': 3}

    # ── 采样并推理 ─────────────────────────────────────────────────────
    pool = [i for i in range(len(ids)) if int(lens[i]) <= args.max_len]
    rng = random.Random(42)
    sample_idx = rng.sample(pool, min(args.n, len(pool)))
    print(f'\n采样 {len(sample_idx)} 条 (L <= {args.max_len}):\n')
    print(f'{"idx":>6} {"len":>5} {"bond_mean":>9} {"bond_std":>9} {"clash%":>7} {"status":>10}')
    print('-' * 55)

    n_ok, n_nan, n_clash = 0, 0, 0
    total_start = time.time()
    for i in sample_idx:
        cid = str(ids[i]); L = int(lens[i])
        seq_str = seq_map.get(cid)
        if not seq_str:
            print(f'{i:>6} {L:>5}  {"no_seq":>20}')
            continue
        seq_str = seq_str[:L].upper().replace('T', 'U')
        s_ids = torch.tensor([tok.get(b, 4) for b in seq_str], dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                pred = model(s_ids, return_loss=False)
            if isinstance(pred, tuple):
                pred = pred[0]
            coords = pred.cpu().numpy()[0, :L]  # (L, 3)

        if np.isnan(coords).any() or np.isinf(coords).any():
            n_nan += 1
            print(f'{i:>6} {L:>5}  {"NaN":>20}')
            continue

        # [refine] physics refine: 20 步 stereo energy 精修 (project_bonds=True 保键长)
        c_t = torch.tensor(coords, dtype=torch.float32).unsqueeze(0).to(device)
        l_t = torch.tensor([L], dtype=torch.long).to(device)
        ref = refine_coords(c_t, l_t, n_steps=20, project_bonds=True)
        coords = ref[0].cpu().numpy()[:L]

        # 键长统计
        d = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
        bm, bs = d.mean(), d.std()
        # clash: 非相邻距离 < 2.5 (原子重叠)
        if L > 50:
            sub = coords[::4]  # 采样避免 O(L²)
            dist = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
            mask = np.triu(np.ones_like(dist, dtype=bool), k=2)
            clash = float((dist[mask] < 2.5).mean() * 100)
        else:
            clash = float('nan')

        is_real = 4.5 < bm < 7.5 and bs < 3.0
        status = 'REAL-LIKE' if is_real else 'noise-like'
        if is_real: n_ok += 1
        if clash > 5: n_clash += 1
        print(f'{i:>6} {L:>5}  {bm:>9.2f} {bs:>9.2f} {clash:>6.1f}% {status:>10}')

    print('-' * 55)
    dt = time.time() - total_start
    print(f'\n结果: {n_ok}/{len(sample_idx)} 键长合理, {n_nan} NaN, {n_clash} 高clash')
    print(f'耗时 {dt:.0f}s ({dt/max(len(sample_idx),1):.1f}s/条)')
    if n_ok / max(len(sample_idx), 1) > 0.5:
        print('\n=> 泛化良好: 可重新生成 CG pseudo-label (走 generate_32_workers.py)')
    else:
        print('\n=> 泛化差: Phase 0 模型对 CG 长序列不适用, 需考虑只用 PDB 数据或改进模型')


if __name__ == '__main__':
    main()
