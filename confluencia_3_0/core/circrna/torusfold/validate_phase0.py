"""validate_phase0.py — Phase 0 小验证（方案B：独立脚本，不动训练主循环）

目标：PDB 环化数据能否让 Encoder latent 编码真实 3D 结构。
三个验证点：
  1. 可学习性 — diffusion loss 在真实 PDB 3D coords 上几十步内下降
  2. latent 编码 3D — 线性探针用 latent_inv mean-pool 区分高/低 z_span，
     训练前应 ~50%（随机），训练后显著 >50%
  3. anchor_aux（含 CRBPSA 氢键加权）— 数值正常、梯度流向 scorer

用法：
    python validate_phase0.py [--steps 80] [--batch 8] [--lr 5e-4]
    # Reads:  ../data/pdb_cyclized/consolidated.npz
"""
import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

BASE = os.path.abspath('.')
DEPLOY_ROOT = os.path.normpath(os.path.join(BASE, '..', '..', '..', '..'))
NPZ_PATH = os.path.join(DEPLOY_ROOT, 'data', 'pdb_cyclized', 'consolidated.npz')

from scheme10_equivariant import EquivariantS10Config, StrictlyEquivariantS10

SEQ_MAP = {'A': 0, 'U': 1, 'G': 2, 'C': 3}   # must match train_s10_curriculum collate


def make_config():
    return EquivariantS10Config(
        d_model=256, d_inv=64, d_eq=32, n_layers=4,
        k_theta=4, k_phi=2, use_coord_diffusion=True, n_diffusion_steps=20,
        d_coord_hidden=128, cfg_dropout_prob=0.1,
        use_s8_refine=True, use_adaptive_k=True,
        d_model_inv=64, d_model_eq=64, dropout=0.1,
        n_tokens=5, bond_length=5.9, r_scale=300.0,
        use_dynamic_anchors=True,   # LAMA + CRBPSA hbond path
    )


# ═══════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════
def load_data():
    data = np.load(NPZ_PATH, allow_pickle=True)
    ids, lengths, coords, seqs = data['ids'], data['lengths'], data['coords'], data['seqs']
    n = len(lengths)
    # z_span per sample (3D-ness proxy) — valid region only
    zsp = np.array([float(np.ptp(coords[i, :int(lengths[i]), 2])) for i in range(n)])
    return ids, lengths, coords, seqs, zsp


def build_pair_probs(coords, lengths, decay=4.0):
    """Spatial-neighborhood pair signal: exp(-d/decay), diag=0.
    PDB has no ViennaRNA bpp, so the nearest-neighbor topology from real
    coords stands in — enough for topk neighbor selection + anchor hotspot."""
    B, L, _ = coords.shape
    device = coords.device
    pp = torch.zeros(B, L, L, device=device)
    for b in range(B):
        Lb = int(lengths[b])
        if Lb < 2:
            continue
        d = torch.cdist(coords[b, :Lb], coords[b, :Lb])
        w = torch.exp(-d / decay)
        w.fill_diagonal_(0)
        pp[b, :Lb, :Lb] = w
    return pp


def collate(indices, coords, lengths, seqs, device='cpu'):
    """Pad to max_L in batch; return (seq_ids, target, lengths, pair_probs)."""
    max_L = max(int(lengths[i]) for i in indices)
    B = len(indices)
    seq_ids = torch.zeros(B, max_L, dtype=torch.long)
    target = torch.zeros(B, max_L, 3)
    lens = torch.zeros(B, dtype=torch.long)
    for k, i in enumerate(indices):
        L = int(lengths[i])
        s = [SEQ_MAP.get(b, 4) for b in seqs[i][:L]]
        seq_ids[k, :L] = torch.tensor(s, dtype=torch.long)
        target[k, :L] = torch.tensor(np.asarray(coords[i, :L], dtype=np.float32))
        lens[k] = L
    pp = build_pair_probs(target, lens)
    return seq_ids.to(device), target.to(device), lens.to(device), pp.to(device)


# ═══════════════════════════════════════════════════════════════
# Latent probe (z_span separability)
# ═══════════════════════════════════════════════════════════════
class LatentProbe(nn.Module):
    def __init__(self, d_in, d_hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def extract_latent_pooled(model, probe_idx, coords, lengths, seqs, device, bs=8):
    """mean-pool latent_inv over valid region → [N, d_inv]. Batched (fast)."""
    model.eval()
    capture = {}

    def hook(mod, inp, out):
        capture['inv'] = out[0].detach()

    handle = model.latent.register_forward_hook(hook)
    pooled, zspans = [], []
    with torch.no_grad():
        for start in range(0, len(probe_idx), bs):
            batch = probe_idx[start:start + bs]
            seq_ids, tgt, lens, pp = collate(batch, coords, lengths, seqs, device)
            model(seq_ids, target_coords=tgt, pair_probs=pp, return_loss=True)
            lat = capture['inv']                       # [B, L, d_inv]
            B, L, D = lat.shape
            valid = torch.arange(L, device=lat.device).unsqueeze(0) < lens.unsqueeze(-1)
            pooled_b = (lat * valid.unsqueeze(-1)).sum(1) / \
                       lens.float().clamp(min=1).unsqueeze(-1)   # [B, D]
            for k, i in enumerate(batch):
                pooled.append(pooled_b[k])             # [D] per sample
                Lb = int(lengths[i])
                zspans.append(float(np.ptp(np.asarray(coords[i, :Lb, 2]))))
    handle.remove()
    return torch.stack(pooled), torch.tensor(zspans, dtype=torch.float32)


def probe_accuracy(z_encoded, labels, n_fold=5, lr=1e-3, iters=200, seed=0):
    """5-fold linear-probe classification accuracy of high vs low z_span."""
    rng = np.random.RandomState(seed)
    N = len(labels)
    accs = []
    idx = np.arange(N)
    rng.shuffle(idx)
    fold = np.array_split(idx, n_fold)
    for f in range(n_fold):
        test_i, train_i = fold[f], np.concatenate([x for j, x in enumerate(fold) if j != f])
        probe = LatentProbe(z_encoded.shape[1]).to(z_encoded.device)
        opt = torch.optim.Adam(probe.parameters(), lr=lr)
        lossf = nn.BCEWithLogitsLoss()
        xt, yt = z_encoded[train_i], labels[train_i].float()
        xe, ye = z_encoded[test_i], labels[test_i].float()
        probe.train()
        for _ in range(iters):
            opt.zero_grad()
            loss = lossf(probe(xt), yt)
            loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            pred = (torch.sigmoid(probe(xe)) > 0.5).float()
            accs.append((pred == ye).float().mean().item())
    return float(np.mean(accs))


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=80)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print('=' * 64)
    print(f'  Phase 0 validation — PDB 3D pretrain (方案B)')
    print(f'  Device: {device}  steps={args.steps} batch={args.batch} lr={args.lr}')
    print('=' * 64)

    # Data
    ids, lengths, coords, seqs, zsp = load_data()
    print(f'  Loaded {len(lengths)} PDB samples; z_span mean={zsp.mean():.1f}')

    # Model
    cfg = make_config()
    model = StrictlyEquivariantS10(cfg).to(device)
    print(f'  Model: {sum(p.numel() for p in model.parameters()):,} params')

    # Probe pool: balanced high/low z_span, fixed indices (before/after contrast)
    med = np.median(zsp)
    hi = np.where(zsp > med)[0]; lo = np.where(zsp <= med)[0]
    np.random.shuffle(hi); np.random.shuffle(lo)
    N_probe = min(150, len(hi), len(lo))
    probe_idx = list(hi[:N_probe]) + list(lo[:N_probe])
    probe_labels = torch.tensor([1.0] * N_probe + [0.0] * N_probe)
    print(f'  Probe pool: {len(probe_idx)} samples ({N_probe} high + {N_probe} low z_span), median={med:.1f}')

    # ── Before training ──
    print('\n[latent] extracting BEFORE training...')
    z_before, _ = extract_latent_pooled(model, probe_idx, coords, lengths, seqs, device)
    acc_before = probe_accuracy(z_before.to(device), probe_labels.to(device))
    print(f'  probe acc (random-init latent): {acc_before*100:.1f}%   (chance ~50%)')

    # Warmup forward (compile/init adaptive_k etc.)
    warm_idx = list(range(min(args.batch, len(lengths))))
    w_seq, w_tgt, w_len, w_pp = collate(warm_idx, coords, lengths, seqs, device)
    model.train()
    with torch.no_grad():
        model(w_seq, target_coords=w_tgt, pair_probs=w_pp, return_loss=True)

    # ── Train a few steps ──
    print(f'\n[train] {args.steps} steps...')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_pool = np.arange(len(lengths))
    t0 = time.time()
    for step in range(args.steps):
        idx = np.random.choice(train_pool, args.batch, replace=False)
        seq_ids, tgt, lens, pp = collate(idx, coords, lengths, seqs, device)
        model.train()
        diff_loss, x0_pred, contact_pred, anchor_loss = model(
            seq_ids, target_coords=tgt, pair_probs=pp, return_loss=True)
        if torch.isnan(diff_loss):
            print(f'  [step {step}] NaN diff_loss — abort'); return
        loss = diff_loss + 0.5 * (anchor_loss if anchor_loss is not None else torch.zeros_like(diff_loss))
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 20 == 0:
            au = anchor_loss.item() if anchor_loss is not None else float('nan')
            print(f'  step {step+1}/{args.steps}  diff_loss={diff_loss.item():.4f}  anchor_aux={au:.4f}  '
                  f'({time.time()-t0:.1f}s)')

    # ── After training ──
    print('\n[latent] extracting AFTER training...')
    z_after, _ = extract_latent_pooled(model, probe_idx, coords, lengths, seqs, device)
    acc_after = probe_accuracy(z_after.to(device), probe_labels.to(device))
    print(f'  probe acc (trained latent):   {acc_after*100:.1f}%')

    print('\n' + '=' * 64)
    print('  VERDICT')
    print('=' * 64)
    print(f'  probe acc: {acc_before*100:.1f}% → {acc_after*100:.1f}%')
    if acc_after > acc_before + 0.10:
        print('  ✓ Encoder latent learned 3D structure (z_span separable)')
    elif acc_after > 0.6:
        print('  ✓ latent encodes some 3D (modest separation)')
    else:
        print('  ✗ latent NOT encoding 3D — check data/signal')
    print(f'  Done in {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
