"""train_s10_curriculum.py - S10 curriculum training with stratified length mixing.

Framework: train_stratified_curriculum.py (4 phases, each contains all length scales,
  ratios drift from short-heavy to long-heavy over phase).

Verified ABC+D geometry priors ported from train_s10_82k.py (smoke-tested, all pass).

Phases:
  Phase 1: short=90% medium=8% long=2% (conf>=0.8, high-quality)
  Phase 2: short=70% medium=20% long=10% (conf>=0.5)
  Phase 3: short=40% medium=40% long=20% (conf>=0.5)
  Phase 4: short=20% medium=30% long=50% (conf>=0.3, BSJ stress)

No phase isolation - all length scales present every epoch -> no representation collapse.
"""
# ── MUST come before any torch import ──
import os
# A800-80G NVLink: expandable_segments reduces fragmentation on large contiguous allocs
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys, time, json, gc, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import logging

logging.basicConfig(
    level=logging.WARNING, format='%(levelname)s %(filename)s:%(lineno)d %(message)s'
)

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, '.')
sys.path.insert(0, os.path.join('.', 'circrna_3d_pipeline'))
sys.path.insert(0, os.path.join('.', 'rl'))

from scheme10_equivariant import EquivariantS10Config, StrictlyEquivariantS10

# A+B+C geometry prior modules (verified in train_s10_82k.py)
from stereochemistry_losses import get_stereo_loss_breakdown
from physics_decoupled_loss import PhysicsDecoupledLoss
from physics_loss import PhysicsLoss
from contact_map_aux_head import ContactMapAuxHead, generate_contact_map
from chirality_embedding import ChiralityAwareEmbedding
from cartesian_to_torus import cartesian_to_torus, major_ring_radius
from contrastive_circrna import GeometryContrastiveLoss
from physics_bridge import ConstraintExtractor, WC_PAIRS
from physics_distillation import ContactMapDistillationLoss
from bias_annealing import apply_bias_annealing
from train_curriculum import BSJGeometryLoss
from train_all_schemes import kabsch_rmsd

# Phase definitions (4-bucket, aligned with circrna_3d_all full distribution)
# short<=200, medium 201-500, long 501-1000, xlong >1000
LENGTH_BUCKET = {
    "short": (151, 200),
    "medium": (201, 500),
    "long": (501, 1000),
    "xlong": (1001, 5000),
}
PHASES = {
    1: {"conf_min": 0.8,
        "ratios": {"short": 0.60, "medium": 0.30, "long": 0.08, "xlong": 0.02},
        "detach_frac": 0.25,  # [v4] freeze Encoder via stop-grad for first 25% of phase
        "desc": "Short-dominant core geometry (high quality)"},
    2: {"conf_min": 0.5,
        "ratios": {"short": 0.40, "medium": 0.40, "long": 0.15, "xlong": 0.05},
        "detach_frac": 0.0,
        "desc": "Shift to medium, introduce long"},
    3: {"conf_min": 0.5,
        "ratios": {"short": 0.20, "medium": 0.35, "long": 0.35, "xlong": 0.10},
        "detach_frac": 0.0,
        "desc": "Long-dominant, all medium+ quality"},
    4: {"conf_min": 0.3,
        "ratios": {"short": 0.10, "medium": 0.25, "long": 0.40, "xlong": 0.25},
        "detach_frac": 0.0,
        "desc": "Long+xlong heavy with BSJ stress"},
}
DEFAULT_PHASE_EPOCHS = {1: 10, 2: 10, 3: 10, 4: 10}  # 10 per phase

LOSS_WEIGHTS = {
    'coord': 10.0, 'closure': 5.0, 'bond': 2.0, 'diffusion': 1.0,
    'stereo': 1.0, 'physics_decoupled': 1.0, 'physics_pairing': 1.0,
    'contact_aux': 1.0, 'torus': 0.5, 'chirality': 0.5,
    'contrastive': 0.1, 'physics_bridge': 0.1, 'distillation': 0.1,
    'anchor_aux': 0.5,  # dynamic anchor scorer supervision
}

# ═══════════════════════════════════════════════════════════════
# MC-Dropout Uncertainty Weighting (from train_uncertainty_weighted.py)
# ═══════════════════════════════════════════════════════════════
#
# Each epoch: K forward passes per sample (dropout active) -> per-bucket variance
# -> bucket weight = 1 + temperature * (bucket_var / max_var)
#
# Mechanically independent of phase ratios: phases control SAMPLING,
# MC-Dropout controls GRADIENT scaling. No empirical thresholds.
#
MC_N_SAMPLES = 5           # K MC-Dropout samples per sample
MC_TEMPERATURE = 2.0       # weight scaling temperature
MC_MAX_SAMPLES = 8          # reduced: 200 caused CPU-bound collate to block GPU for minutes per epoch
MC_START_EPOCH = 0         # start MC-Dropout from epoch 0
BUCKET_NAMES = ["short", "medium", "long", "xlong"]


def length_bucket(L):
    """Four-bucket classification aligned with circrna_3d_all distribution.

    Same logic as length_bucket_full / used in PHASES ratios.
    L<150 samples are classified as short (open-end, not excluded).
    """
    if L <= 200:
        return "short"
    elif L <= 500:
        return "medium"
    elif L <= 1000:
        return "long"
    else:
        return "xlong"


# length_bucket_full kept as alias for backward compat in estimate_bucket_uncertainty
length_bucket_full = length_bucket


def estimate_bucket_uncertainty(bucket_groups, model, collate, device,
                                 mc_n, mc_temp, mc_max):
    """
    MC-Dropout uncertainty estimation per bucket.

    For each bucket, draw max_samples indices, do K MC-forward passes,
    compute per-sample coordinate variance, aggregate to bucket variance.

    Returns: bucket_weights dict {bname: weight}
    """
    # Keep dropout active for MC sampling
    model.train()

    bucket_var = {}
    bucket_count = {}

    for bname in BUCKET_NAMES:
        pool = bucket_groups.get(bname, [])
        if len(pool) == 0:
            bucket_var[bname] = 0.0
            bucket_count[bname] = 0
            continue

        # xlong now runs normal MC-Dropout (import bug fixed in multiscale_equivariant)
        n_draw = min(mc_max, len(pool))
        draw_idx = np.random.choice(len(pool), size=n_draw, replace=False)
        bucket_vars = []

        # Batched MC estimation: process batch_size samples at once
        for start in range(0, len(draw_idx), batch_size):
            batch_d_idx = draw_idx[start:start + batch_size]
            batch_indices = [pool[i] for i in batch_d_idx]
            seq_ids, target_s, lengths, _ = collate(batch_indices)
            B_eff = len(batch_indices)

            # K MC-Dropout forward passes (entire batch)
            preds_list = []
            with torch.no_grad():
                for _ in range(mc_n):
                    pred = model(seq_ids, return_loss=False)
                    preds_list.append(pred)
            preds = torch.stack(preds_list, dim=0)  # [K, B_eff, L, 3]

            # Per-sample variance [B_eff, L]
            pred_var = preds.var(dim=0).sum(dim=-1)
            for b in range(B_eff):
                L = int(lengths[b].item())
                sample_var = pred_var[b, :L].mean().item()
                bucket_vars.append(sample_var)

        bucket_var[bname] = float(np.mean(bucket_vars))
        bucket_count[bname] = n_draw

    # Normalize variance -> weights
    global_max = max(bucket_var.values()) + 1e-8
    bucket_weights = {}
    for bname in BUCKET_NAMES:
        norm_var = min(bucket_var[bname] / global_max, 1.0)
        bucket_weights[bname] = 1.0 + mc_temp * norm_var

    return bucket_weights, bucket_var, bucket_count

BASE = os.path.abspath('.')
DEPLOY_ROOT = os.path.normpath(os.path.join(BASE, '..', '..', '..', '..'))
device = 'cuda'
output_dir = os.path.join(BASE, 'models', 's10_curriculum')
os.makedirs(output_dir, exist_ok=True)

batch_size = 16      # A800 80GB: ample memory, large batch
grad_accum_steps = 1  # Effective batch = 16 (no accumulation needed on A800)

print('=' * 60)
print('  S10 Curriculum Training (stratified length mixing)')
print('=' * 60)
print(f'Device: {device}, GPU: {torch.cuda.get_device_name(0)}')
print(f'Output: {output_dir}, Batch size: {batch_size}')

# Load data from consolidated npz (82k samples, ~0.55 GB)
print(f'Loading data from consolidated npz ...')
t0 = time.time()
npz_path = os.path.join(DEPLOY_ROOT, 'data', 'circrna_3d_all_consolidated.npz')
if os.path.isfile(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    ids_arr = data['ids']
    lengths_arr = data['lengths']
    coords_arr = data['coords']
    n = len(lengths_arr)
    # P1 fix: load real sequences from circBase FASTA (id -> sequence).
    # FASTA is the upstream source of circrna_3d_all/.npy; IDs match 1:1.
    # Fallback to synthetic ACGU repeat only if FASTA missing or id not found.
    fasta_path = os.path.join(DEPLOY_ROOT, 'data', 'circrna', 'circbase_seqs.fa.gz')
    seq_map = {}
    if os.path.isfile(fasta_path):
        import gzip
        cur_id, cur_seq = None, ''
        with gzip.open(fasta_path, 'rt') as f:
            for line in f:
                if line.startswith('>'):
                    if cur_id is not None:
                        seq_map[cur_id] = cur_seq
                    cur_id = line.strip()[1:].split('|')[0]
                    cur_seq = ''
                else:
                    cur_seq += line.strip().upper().replace('T', 'U')
        if cur_id is not None:
            seq_map[cur_id] = cur_seq
        print(f'  FASTA loaded: {len(seq_map)} sequences')
    else:
        logging.warning(f"FASTA not found: {fasta_path}; falling back to synthetic ACGU repeats")
    seqs = []
    n_fallback = 0
    for cid, L in zip(ids_arr, lengths_arr):
        s = seq_map.get(str(cid))
        if s is None:
            n_fallback += 1
            s = ('ACGU' * (int(L) // 4 + 1))
        seqs.append(s[:int(L)])
    if n_fallback > 0:
        logging.warning(f"{n_fallback}/{n} samples missing in FASTA; used synthetic ACGU fallback")
    coords = [np.asarray(c[:int(L)], dtype=np.float32) for c, L in zip(coords_arr, lengths_arr)]
    meta = [{'id': str(cid), 'length': int(L)} for cid, L in zip(ids_arr, lengths_arr)]
    print(f'  Loaded {n} samples in {time.time()-t0:.2f}s')

    # Pair probabilities: precomputed ViennaRNA bpp or geometric fallback.
    # Generated by precompute_pair_probs.py on the A800 node.
    # If missing, each sample gets torch.zeros(L,L) (graceful degradation).
    pair_probs_path = os.path.join(DEPLOY_ROOT, 'data', 'circrna_3d_all_pair_probs.npz')
    pair_probs = None
    if os.path.isfile(pair_probs_path):
        t_pp = time.time()
        pp_data = np.load(pair_probs_path, allow_pickle=True)
        pp_ids = {str(x): i for i, x in enumerate(pp_data['ids'])}
        pp_arr = pp_data['bp_probs']
        pair_probs = [pp_arr[pp_ids.get(str(cid))] if cid in pp_ids else None for cid in ids_arr]
        n_pp = sum(1 for x in pair_probs if x is not None)
        print(f'  pair_probs loaded: {n_pp}/{n} with data, {time.time()-t_pp:.2f}s')
    else:
        pair_probs = [None] * n
        logging.warning(f"pair_probs not found: {pair_probs_path}; "
                        f"run precompute_pair_probs.py or use geometric fallback")
else:
    raise FileNotFoundError(f"Consolidated npz not found: {npz_path}. Run consolidate_npy_to_npz.py first.")

n = len(seqs)
val_idx = list(range(max(0, n - 50), n))
train_idx = [i for i in range(n) if i not in val_idx]
t_seq = [seqs[i] for i in train_idx]
t_coords = [coords[i] for i in train_idx]
t_meta = [meta[i] for i in train_idx]
t_pair_probs = [pair_probs[i] for i in train_idx]

# Length bucket assignment (uses length_bucket / length_bucket_full defined above)
bucket_groups = defaultdict(list)
for i in range(len(t_seq)):
    bucket_groups[length_bucket(t_meta[i]['length'])].append(i)
print(f'  Buckets: {dict(sorted((k, len(v)) for k, v in bucket_groups.items()))}')

# Model
cfg = EquivariantS10Config(
    d_model=256, d_inv=64, d_eq=32, n_layers=4,
    k_theta=4, k_phi=2, use_coord_diffusion=True, n_diffusion_steps=20,  # 100 -> 20: 5x faster forward
    d_coord_hidden=128, cfg_dropout_prob=0.1,
    use_s8_refine=True, use_adaptive_k=True,
    d_model_inv=64, d_model_eq=64, dropout=0.1,
    n_tokens=5, bond_length=5.9, r_scale=300.0
)
model = StrictlyEquivariantS10(cfg).to(device)
print(f'  Model: {sum(p.numel() for p in model.parameters()):,} params')

# Prior modules (A+B+C, all verified)
pd_loss_fn = PhysicsDecoupledLoss(w_geo=1.0, w_phys=0.1, use_rg_loss=True,
                                   use_clash_loss=True, use_angle_loss=True).to(device)
physics_loss_fn = PhysicsLoss(n_tokens=5, device=device).to(device)
contact_head = ContactMapAuxHead(d_inv=64, d_hidden=32).to(device)
contact_proj = nn.Linear(3, 64, bias=False).to(device)
chirality_emb = ChiralityAwareEmbedding(n_tokens=5, d_model=64).to(device)
chirality_proj = nn.Linear(64, 64, bias=False).to(device)
contrastive_loss_fn = GeometryContrastiveLoss(temperature=0.1).to(device)
constraint_extractor = ConstraintExtractor(c_z=64, n_rbf=16, bond_length=5.9,
                                            pair_distance=8.0, pair_threshold=0.5,
                                            bsj_weight_boost=2.0).to(device)
distillation_loss_fn = None
try:
    distillation_loss_fn = ContactMapDistillationLoss(config=None).to(device)
except Exception:
    pass
# BSJGeometryLoss is single-sample only (L,3) from train_curriculum.py
# We wrap it for batched input (B,L,3)
bsj_geom_loss_single = BSJGeometryLoss(
    target_angle=108.0, target_dihedral=180.0, target_distance=3.5,
    angle_weight=2.0, dihedral_weight=1.0, distance_weight=5.0,
)

def bsj_geom_loss_batched(p_denorm, lengths):
    """Compute BSJ geometry loss for a batch of samples."""
    B = p_denorm.shape[0]
    total = torch.tensor(0.0, device=device)
    valid = 0
    for b in range(B):
        L = int(lengths[b].item())
        if L < 4: continue
        coords = p_denorm[b, :L]  # (L, 3)
        bsj_indices = torch.tensor([0, L-1], device=device)
        val = bsj_geom_loss_single(coords, bsj_indices)
        if torch.is_tensor(val) and not torch.isnan(val):
            total += val
            valid += 1
    return total / max(valid, 1)

prior_params = (list(contact_head.parameters()) + list(contact_proj.parameters()) +
                list(chirality_emb.parameters()) + list(chirality_proj.parameters()) +
                list(contrastive_loss_fn.parameters()) +
                list(constraint_extractor.parameters()) +
                list(pd_loss_fn.parameters()) +
                list(physics_loss_fn.parameters()))
if distillation_loss_fn is not None:
    prior_params += list(distillation_loss_fn.parameters())
all_params = list(model.parameters()) + prior_params
# AdamW with higher lr for faster convergence (loss was stuck at ~555k with SGD lr=1e-4)
optimizer = torch.optim.AdamW(all_params, lr=1e-3, weight_decay=1e-3)
scaler = torch.amp.GradScaler('cuda', enabled=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, patience=8
)
total_params = sum(p.numel() for p in all_params)
print(f'  + geometry prior params: {total_params - sum(p.numel() for p in model.parameters()):,}')
print(f'  Total trainable: {total_params:,} params')

# Collate function (returns: seq_ids, target, lengths, pair_probs)
def collate(indices):
    max_L = max(t_meta[i]['length'] for i in indices)
    bs, bc, lengths, pp_batch = [], [], [], []
    for i in indices:
        L = t_meta[i]['length']
        seq_ids = torch.tensor(
            [{'A': 0, 'U': 1, 'G': 2, 'C': 3}.get(b, 4) for b in t_seq[i]],
            dtype=torch.long)
        seq_pad = torch.zeros(max_L, dtype=torch.long)
        seq_pad[:L] = seq_ids
        c = torch.zeros(max_L, 3)
        c[:L] = torch.tensor(t_coords[i], dtype=torch.float32)
        bs.append(seq_pad); bc.append(c); lengths.append(L)

        # Pair probabilities matrix (padded to max_L)
        pp = t_pair_probs[i]
        if pp is not None:
            pp_mat = torch.zeros(max_L, max_L)
            pp_mat[:L, :L] = torch.tensor(pp[:L, :L], dtype=torch.float32)
        else:
            pp_mat = torch.zeros(max_L, max_L)
        pp_batch.append(pp_mat)
    return (torch.stack(bs).to(device), torch.stack(bc).to(device),
            torch.tensor(lengths, dtype=torch.long).to(device),
            torch.stack(pp_batch).to(device))

# Warmup
print(f'  Warmup forward (compile kernels)...')
warmup_seq, warmup_tgt, warmup_len, warmup_pp = collate(bucket_groups["short"][:2])
model.train()
t0 = time.time()
with torch.no_grad():
    model(warmup_seq, target_coords=warmup_tgt, return_loss=True)
torch.cuda.synchronize()
print(f'  Warmup done in {time.time()-t0:.1f}s, GPU: {torch.cuda.max_memory_allocated()/1e9:.1f}GB')
with torch.no_grad():
    model(warmup_seq, target_coords=warmup_tgt, return_loss=True)
torch.cuda.synchronize()
print(f'  Second warmup done, GPU: {torch.cuda.max_memory_allocated()/1e9:.1f}GB')
print()

# ═══════════════════════════════════════════════════════════════
# Loss computation (all ABC+D, verified in train_s10_82k.py)
# ═══════════════════════════════════════════════════════════════

def compute_all_losses(p_denorm, seq_ids, target, lengths, pair_probs, model=None):
    """Compute all ABC+D loss terms. Returns (total_loss, loss_dict).

    Args:
        pair_probs: (B, Lc, Lc) ViennaRNA base-pair probabilities (padded with 0).
        model: if provided, uses Kendall uncertainty weighting for the 6 core
               geometry+physics terms (coord/bond/stereo/physics_pairing/
               contact_aux/physics_bridge) via model.uncertainty_log_vars.
               Other terms keep fixed LOSS_WEIGHTS.
    """
    B, Lc, _ = p_denorm.shape  # Bc == B
    loss = torch.tensor(0.0, device=device)
    loss_dict = {}

    # [v4] Kendall uncertainty weighting helper.
    # weighted_L = 0.5 * exp(-log_var) * L + 0.5 * log_var
    # log_var = log(σ²); uniform init (0 → σ²=1, weight=0.5); light L2 reg prevents
    # a term from being turned off entirely (σ² → ∞).
    UW_TERMS = {'coord', 'bond', 'stereo', 'physics_pairing', 'contact_aux', 'physics_bridge'}
    UW_REG = 0.01  # light regularizer on log_var drift
    log_vars = getattr(model, 'uncertainty_log_vars', None) if model is not None else None

    def uw_add(term, value):
        """Apply Kendall UW to a core term, else fall back to fixed weight."""
        if log_vars is not None and term in UW_TERMS and term in log_vars:
            lv = log_vars[term]
            precision = torch.exp(-lv)            # 1/σ²
            reg = UW_REG * (lv ** 2)
            weighted = 0.5 * precision * value + 0.5 * lv + reg
            return weighted, precision.item(), lv.item()
        return value * LOSS_WEIGHTS.get(term, 1.0), None, None

    # Pre-compute normalized coords (mask padding per sample)
    valid_mask = torch.arange(Lc, device=device).unsqueeze(0) < lengths.unsqueeze(-1)  # [B, Lc]
    t_sum = (target * valid_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / valid_mask.sum(dim=-1, keepdim=True).clamp(min=1)
    p_sum = (p_denorm * valid_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / valid_mask.sum(dim=-1, keepdim=True).clamp(min=1)
    t_c = target - t_sum
    t_scale = torch.norm(t_c * valid_mask.unsqueeze(-1), dim=(1, 2), keepdim=True).clamp(min=1.0)
    t_norm = t_c / t_scale
    p_c = p_denorm - p_sum
    p_norm = p_c / t_scale

    # 1. Coordinate loss
    coord_loss = torch.tensor(0.0, device=device)
    for b in range(B):
        vL = int(lengths[b].item())
        coord_loss += torch.mean((p_norm[b, :vL] - t_norm[b, :vL]) ** 2)
    coord_loss /= B
    w, p, lv = uw_add('coord', coord_loss)
    loss = loss + w
    loss_dict['coord'] = coord_loss.item()
    if p is not None:
        loss_dict['coord_sigma2'] = float(np.exp(lv)); loss_dict['coord_prec'] = p

    # 2. Closure loss (P9 fix: use true last residue per sample, not padded tail)
    last_idx = (lengths - 1).clamp(min=0).long()  # [B]
    last_coords = p_denorm[torch.arange(B, device=device), last_idx]  # [B, 3]
    closure_dists = torch.norm(p_denorm[:, 0] - last_coords, dim=-1)
    cm = torch.where(lengths >= 2, 1.0, 0.0).to(device)
    closure_loss = (cm * (closure_dists - 5.9) ** 2).sum() / cm.sum().clamp(min=1.0)
    loss = loss + closure_loss * LOSS_WEIGHTS['closure']
    loss_dict['closure'] = closure_loss.item()

    # 3. Bond loss
    bond_loss = torch.tensor(0.0, device=device); nb = 0
    for b in range(B):
        vL = int(lengths[b].item())
        if vL < 4: continue
        bonds = torch.norm(p_denorm[b, 1:vL] - p_denorm[b, :vL - 1], dim=-1)
        bsj = torch.norm(p_denorm[b, 0] - p_denorm[b, vL - 1])
        all_b = torch.cat([bonds, bsj.unsqueeze(0)])
        # Manual MSE
        bond_loss += ((all_b - 5.9) ** 2).mean()
        nb += 1
    bond_loss /= max(nb, 1)
    w, p, lv = uw_add('bond', bond_loss)
    loss = loss + w
    loss_dict['bond'] = bond_loss.item()
    if p is not None:
        loss_dict['bond_sigma2'] = float(np.exp(lv))

    # 4. Stereochemistry loss
    try:
        stereo_result = get_stereo_loss_breakdown(p_denorm, lengths)
        stereo_total = stereo_result['total']
        w, p, lv = uw_add('stereo', stereo_total)
        loss = loss + w
        loss_dict['stereo'] = stereo_total.item()
        if p is not None:
            loss_dict['stereo_sigma2'] = float(np.exp(lv))
    except Exception as e:
        logging.warning(f"stereochemistry loss failed (sampled): {e}")
        loss_dict['stereo'] = 0.0

    # 5. Physics decoupled loss
    try:
        pd_result = pd_loss_fn(p_denorm, target, lengths, None)
        pd_loss = pd_result['loss'] if isinstance(pd_result, dict) else pd_result
        loss = loss + pd_loss * LOSS_WEIGHTS['physics_decoupled']
        loss_dict['physics_decoupled'] = pd_loss.item()
    except Exception as e:
        logging.warning(f"physics_decoupled loss failed: {e}")
        loss_dict['physics_decoupled'] = 0.0

    # 6. Physics pairing loss (P2 fix: pass real pair_probs for pairing_consistency)
    try:
        phys_loss = physics_loss_fn(p_denorm, seq_ids, pair_probs, lengths=lengths)
        w, p, lv = uw_add('physics_pairing', phys_loss)
        loss = loss + w
        loss_dict['physics_pairing'] = phys_loss.item()
        if p is not None:
            loss_dict['physics_pairing_sigma2'] = float(np.exp(lv))
    except Exception as e:
        logging.warning(f"physics_pairing loss failed: {e}")
        loss_dict['physics_pairing'] = 0.0

    # 7. Contact map auxiliary loss
    try:
        latent_contact = contact_proj(p_norm)
        contact_pred = contact_head(latent_contact)
        with torch.no_grad():
            contact_target = generate_contact_map(target, threshold=8.0)
        # binary_cross_entropy = -[y*log(x) + (1-y)*log(1-x)]
        eps = 1e-7
        contact_loss = -(contact_target * (contact_pred + eps).log() +
                         (1 - contact_target) * (1 - contact_pred + eps).log()).mean()
        w, p, lv = uw_add('contact_aux', contact_loss)
        loss = loss + w
        loss_dict['contact_aux'] = contact_loss.item()
        if p is not None:
            loss_dict['contact_aux_sigma2'] = float(np.exp(lv))
    except Exception as e:
        logging.warning(f"contact_aux loss failed: {e}")
        loss_dict['contact_aux'] = 0.0

    # 8. Torus coordinate loss
    try:
        R_target = major_ring_radius(lengths.float(), bond_length=5.9)
        theta_pred, phi_pred, r_pred = cartesian_to_torus(p_denorm, R_target)
        theta_tgt, phi_tgt, r_tgt = cartesian_to_torus(target, R_target)
        dtheta = torch.remainder(theta_pred - theta_tgt + math.pi, 2 * math.pi) - math.pi
        dphi = torch.remainder(phi_pred - phi_tgt + math.pi, 2 * math.pi) - math.pi
        dr = r_pred - r_tgt
        torus_loss = (dtheta ** 2).mean() + (dphi ** 2).mean() + (dr ** 2).mean()
        loss = loss + torus_loss * LOSS_WEIGHTS['torus']
        loss_dict['torus'] = torus_loss.item()
    except Exception as e:
        logging.warning(f"torus loss failed: {e}")
        loss_dict['torus'] = 0.0

    # 9. Chirality loss
    try:
        seq_float = seq_ids.float()
        chirality_out = chirality_proj(chirality_emb(seq_float))
        chirality_loss = (chirality_out - contact_proj(p_norm)).pow(2).mean()
        loss = loss + chirality_loss * LOSS_WEIGHTS['chirality']
        loss_dict['chirality'] = chirality_loss.item()
    except Exception as e:
        logging.warning(f"chirality loss failed: {e}")
        loss_dict['chirality'] = 0.0

    # 10. Contrastive loss (B >= 4)
    if B >= 4:
        try:
            mid = B // 2
            contrastive_loss = contrastive_loss_fn(p_denorm[:mid],
                                                    p_denorm[mid:2*mid] if 2*mid <= B else p_denorm[:mid],
                                                    lengths[:mid])
            loss = loss + contrastive_loss * LOSS_WEIGHTS['contrastive']
            loss_dict['contrastive'] = contrastive_loss.item()
        except Exception:
            loss_dict['contrastive'] = 0.0

    # 11. Physics bridge constraint violation loss (B >= 2)
    # FIX: constraint_extractor returns ConstraintSet (dataclass), not tensor.
    # Old code `constraints.pow(2).mean()` always crashed silently.
    # Correct approach: extract constraint lists, compute distance violations
    # against the predicted structure, per sample.
    if B >= 2:
        try:
            Bc, Lc, _ = p_denorm.shape
            # [v4 fix] Removed dead O(L²) allocation: pair_repr = torch.zeros(B, Lc, Lc, 64)
            # was 16 GB at L=2000/B=16, but the `constraints` object it produced was
            # never read — violations below recompute distances from cmat + constants
            # (constraint_extractor.bond_length / pair_distance). Deleting it changes
            # no behavior, just frees the O(L²·c_z) peak that OOM'd long sequences.
            tok2base = {0: 'A', 1: 'U', 2: 'G', 3: 'C', 4: 'N'}
            seq0 = seq_ids[0]
            L0 = int(lengths[0].item())
            seq_str = ''.join(tok2base.get(int(t), 'N') for t in seq0[:L0])
        except Exception as e:
            logging.warning(f"physics_bridge: setup failed: {e}")
            loss_dict['physics_bridge'] = 0.0
            constraint_loss = torch.tensor(0.0, device=device)
        else:
            # Compute constraint violation per sample (vectorized).
            try:
                violations = []
                for bi in range(B):
                    vL = int(lengths[bi].item())
                    cmat = torch.cdist(p_denorm[bi, :vL], p_denorm[bi, :vL])  # (vL, vL)
                    pp_sample = pair_probs[bi, :vL, :vL]

                    # Bond violations: backbone bonds (i, i+1) target 5.9 Å (circular)
                    bond_tgt = constraint_extractor.bond_length
                    next_idx = torch.roll(torch.arange(vL), -1)
                    bond_d = cmat[torch.arange(vL), next_idx]
                    bond_v = ((bond_d - bond_tgt) ** 2).mean()

                    # Pair violations (vectorized): target = WC 10.6, others fallback 10.6
                    # Build mask: prob >= 0.1 AND j >= i+4 (upper triangle offset 4)
                    pair_tgt = constraint_extractor.pair_distance
                    # Upper triangle with offset 4
                    triu_mask = torch.triu(torch.ones(vL, vL, dtype=torch.bool, device=device),
                                           diagonal=4)
                    prob_mask = (pp_sample >= 0.1) & triu_mask
                    if prob_mask.any():
                        probs = pp_sample[prob_mask]
                        dists = cmat[prob_mask]
                        # BSJ-crossing boost: circ_dist >= L/2
                        ii, jj = torch.where(prob_mask)
                        circ_dist = torch.minimum((ii - jj).abs(), vL - (ii - jj).abs())
                        bsj_boost = torch.where(circ_dist >= vL // 2,
                                                constraint_extractor.bsj_weight_boost,
                                                1.0)
                        w = probs * bsj_boost
                        pair_v = (w * (dists - pair_tgt) ** 2).sum() / w.sum().clamp(min=1e-8)
                    else:
                        pair_v = torch.tensor(0.0, device=device)

                    violations.append(float(bond_v + pair_v))

                constraint_loss = torch.tensor(float(np.mean(violations)), device=device)
                loss_dict['physics_bridge'] = float(constraint_loss.item())
            except Exception as e:
                logging.warning(f"physics_bridge: violation computation failed: {e}")
                constraint_loss = torch.tensor(0.0, device=device)
                loss_dict['physics_bridge'] = 0.0

        if constraint_loss > 0:
            w, p, lv = uw_add('physics_bridge', constraint_loss)
            loss = loss + w
            if p is not None:
                loss_dict['physics_bridge_sigma2'] = float(np.exp(lv))

    # 12. Contact distillation loss
    if distillation_loss_fn is not None:
        try:
            teacher_contact = generate_contact_map(target, threshold=8.0)
            # P6 fix: use real pair_probs as confidence instead of all-ones
            confidence = pair_probs.clamp(min=0.0, max=1.0)
            distill_loss = distillation_loss_fn(contact_pred, teacher_contact, confidence)
            loss = loss + distill_loss * LOSS_WEIGHTS['distillation']
            loss_dict['distillation'] = distill_loss.item()
        except Exception:
            loss_dict['distillation'] = 0.0

    return loss, loss_dict

# ═══════════════════════════════════════════════════════════════
# Curriculum training loop
# ═══════════════════════════════════════════════════════════════

def build_epoch_batches(phase, epoch, n_phase_epochs):
    """Build batches for one epoch using phase's length-mixing ratios."""
    ratios = PHASES[phase]["ratios"]
    t = epoch / max(n_phase_epochs - 1, 1)
    short_final = max(0.05, ratios["short"] * (1.0 - t * 0.1))
    long_final = ratios["long"] * (1.0 + t * 0.05)
    xlong_final = max(0.0, ratios["xlong"] * (1.0 + t * 0.05))
    medium_final = max(0.05, 1.0 - short_final - long_final - xlong_final)
    norm = short_final + medium_final + long_final + xlong_final
    ratios = {
        "short": short_final/norm,
        "medium": medium_final/norm,
        "long": long_final/norm,
        "xlong": xlong_final/norm,
    }

    # Build per-bucket pools (skip empty buckets)
    bucket_indices = {}
    for bname in ["short", "medium", "long", "xlong"]:
        pool = bucket_groups[bname].copy()
        if len(pool) == 0:
            continue  # skip if no samples in this bucket
        np.random.shuffle(pool)
        bucket_indices[bname] = pool

    if not bucket_indices:
        return []

    N_TARGET = sum(len(v) for v in bucket_indices.values()) // batch_size
    epoch_batches = []
    for bname, ratio in ratios.items():
        pool = bucket_indices.get(bname)
        if pool is None or len(pool) == 0:
            continue  # skip empty bucket
        n_from = int(round(N_TARGET * ratio))
        if n_from <= 1:
            continue
        # Shuffle pool once, then slice into clean batches — no boundary loss,
        # no replace=True duplication.  n_from is # of samples, rounded up to
        # fill an integer number of full-size batches.
        np.random.shuffle(pool)
        n_batches = (n_from + batch_size - 1) // batch_size
        for i in range(n_batches):
            batch = pool[i * batch_size:(i + 1) * batch_size]
            if len(batch) >= 2:
                epoch_batches.append(batch)
    np.random.shuffle(epoch_batches)
    return epoch_batches

def train_one_phase(phase, n_phase_epochs):
    """Train one phase of the curriculum."""
    ratios = PHASES[phase]["ratios"]
    print(f'\n  === Phase {phase}: {n_phase_epochs} epochs ===')
    print(f'  {PHASES[phase]["desc"]}')
    print(f'  Length mixing: short={ratios["short"]:.0%} medium={ratios["medium"]:.0%} long={ratios["long"]:.0%} xlong={ratios["xlong"]:.0%}')

    best_val = float('inf')
    phase_history = []
    patience = 0

    for epoch in range(n_phase_epochs):
        model.train()

        # [v4] Stop-Gradient schedule: detach latent→diffusion edge for the first
        # detach_frac of Phase 1, so the Encoder learns structure from self-supervision
        # (anchor_aux / contact_aux) instead of being pulled by coordinate losses.
        detach_frac = PHASES[phase].get("detach_frac", 0.0)
        frac_done = epoch / max(n_phase_epochs - 1, 1)
        model.detach_latent = (detach_frac > 0.0 and frac_done < detach_frac)
        if model.detach_latent and (epoch == 0 or frac_done < detach_frac <= frac_done + 1.0/n_phase_epochs):
            print(f'    [stop-grad] latent→diffusion detached (epoch {epoch+1}, releases at {int(detach_frac*100)}%)')

        # MC-Dropout uncertainty weighting per bucket
        bucket_weights, bucket_var, bucket_count = estimate_bucket_uncertainty(
            bucket_groups, model, collate, device,
            mc_n=MC_N_SAMPLES, mc_temp=MC_TEMPERATURE, mc_max=MC_MAX_SAMPLES
        )
        weight_str = ' '.join(f'{b}={bucket_weights[b]:.2f}' for b in BUCKET_NAMES)

        epoch_batches = build_epoch_batches(phase, epoch, n_phase_epochs)
        if not epoch_batches: continue
        print(f'  [P{phase}] Epoch {epoch+1}/{n_phase_epochs} ({len(epoch_batches)} batches) UQ({weight_str})')

        epoch_frac = epoch / max(n_phase_epochs - 1, 1)
        bsj_weight = max(0.0, 3.0 * (1.0 - epoch_frac))
        anneal_sigma = 2.0 * (1.0 - epoch_frac)

        train_loss = 0.0; n_batches = 0; nan_batches = 0
        loss_acc = defaultdict(float)  # accumulate weighted loss terms for breakdown
        epoch_t0 = time.time()

        for step, batch_indices in enumerate(epoch_batches):
            seq_ids, target, lengths, batch_pp = collate(batch_indices)
            B = target.shape[0]
            if torch.isnan(target).any():
                nan_batches += 1; continue

            # Bias annealing on target (valid region only, respect lengths)
            if anneal_sigma > 0.01:
                Lc = target.shape[1]
                valid_mask = torch.arange(Lc, device=device).unsqueeze(0) < lengths.unsqueeze(-1)
                target = target + torch.randn_like(target) * anneal_sigma * valid_mask.float().unsqueeze(-1)

            # autocast ON: fp16 activations cut memory, scaler handles grad scaling
            # v4: coord diffusion — diff_loss is MSE on (B,L,3); x0_pred is the
            # single-step denoised coords (differentiable) for geometric losses.
            # v4.1: 4-tuple return with anchor_aux_loss for dynamic anchor supervision.
            with torch.amp.autocast('cuda'):
                diff_loss, x0_pred, contact_pred_latent, anchor_aux_loss = model(
                    seq_ids, target_coords=target, pair_probs=batch_pp,
                    return_loss=True,
                )
                if torch.isnan(diff_loss):
                    nan_batches += 1; continue

                # x0_pred: (B, Lc, 3) — single-step denoised coordinate prediction
                pred = x0_pred
                if pred is None or torch.isnan(pred).any():
                    nan_batches += 1; continue

                # Normalize (valid_mask: P10 padding fix)
                Lc = target.shape[1]
                valid_mask = torch.arange(Lc, device=device).unsqueeze(0) < lengths.unsqueeze(-1)
                t_sum = (target * valid_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / valid_mask.sum(dim=-1, keepdim=True).clamp(min=1)
                p_sum = (pred * valid_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / valid_mask.sum(dim=-1, keepdim=True).clamp(min=1)
                t_c = target - t_sum
                t_scale = torch.norm(t_c * valid_mask.unsqueeze(-1), dim=(1, 2), keepdim=True).clamp(min=1.0)
                p_c = pred - p_sum
                p_scale = torch.norm(p_c * valid_mask.unsqueeze(-1), dim=(1, 2), keepdim=True).clamp(min=1e-6)
                p_denorm = p_c / p_scale * t_scale + t_sum

                # All ABC+D losses (geometric regularizers on x0_pred)
                loss, loss_dict = compute_all_losses(p_denorm, seq_ids, target, lengths, batch_pp, model=model)

                # Diffusion loss (primary — direct 3D coordinate MSE)
                if diff_loss is not None:
                    loss = loss + diff_loss * LOSS_WEIGHTS['diffusion']
                    loss_dict['diffusion'] = diff_loss.item()

                # Dynamic anchor auxiliary loss (supervises scorer via pair_probs hotspot)
                if anchor_aux_loss is not None and not torch.isnan(anchor_aux_loss):
                    loss = loss + anchor_aux_loss * LOSS_WEIGHTS['anchor_aux']
                    loss_dict['anchor_aux'] = anchor_aux_loss.item()

                # [v4] Latent-direct contact loss — only in detach phase.
                # All p_denorm-based supervision is cut by stop-grad, so this head
                # (latent_inv → contact map, NOT through diffusion) is the Encoder's
                # structural signal. Uses real coords as target (no_grad).
                if model.detach_latent and contact_pred_latent is not None:
                    try:
                        contact_target = generate_contact_map(target, threshold=8.0)
                        eps = 1e-7
                        latent_contact_loss = -(
                            contact_target * (contact_pred_latent + eps).log() +
                            (1 - contact_target) * (1 - contact_pred_latent + eps).log()
                        ).mean()
                        if not torch.isnan(latent_contact_loss):
                            loss = loss + latent_contact_loss * LOSS_WEIGHTS['contact_aux']
                            loss_dict['latent_contact'] = latent_contact_loss.item()
                    except Exception as e:
                        logging.warning(f'latent-direct contact loss failed: {e}')

                # BSJ geometry loss (dynamic weight, decays over phase)
                if bsj_weight > 0:
                    try:
                        bsj_loss_val = bsj_geom_loss_batched(p_denorm, lengths)
                        if not torch.isnan(bsj_loss_val):
                            loss = loss + bsj_loss_val * bsj_weight
                            loss_dict['bsj_geometry'] = bsj_loss_val.item()
                    except Exception:
                        pass

                # MC-Dropout uncertainty weight
                bw = 0.0
                for b in range(B):
                    bname = length_bucket_full(int(lengths[b].item()))
                    bw += bucket_weights[bname]
                uq_weight = bw / max(B, 1)
                loss = loss * uq_weight
                loss_dict['uq_weight'] = uq_weight

            # Backward + optimizer step with GradScaler
            # Order: scale → backward → unscale → clip → step → update
            loss = loss / grad_accum_steps
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                scaler.update()  # abnormal branch also updates, don't break the chain
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0 or step == len(epoch_batches) - 1:
                has_nan = any(p.grad is not None and torch.isnan(p.grad).any() for p in all_params)
                if has_nan:
                    nan_batches += 1
                    scaler.unscale_(optimizer)
                    optimizer.zero_grad()
                else:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            n_batches += 1
            train_loss += loss.item() * grad_accum_steps
            # Accumulate weighted loss terms for breakdown
            for k, v in loss_dict.items():
                weight = LOSS_WEIGHTS.get(k, 1.0)
                loss_acc[k] += v * weight

            # Periodic cache clear to keep VRAM footprint stable
            if step > 0 and step % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            if step > 0 and step % 500 == 0:
                avg = train_loss / n_batches
                lr = optimizer.param_groups[0]['lr']
                cur_mem = torch.cuda.memory_allocated() / 1e9
                max_mem = torch.cuda.max_memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                # Weighted loss breakdown (divided by n_batches for avg per-step)
                n_valid = max(n_batches, 1)
                parts = [f'{k}={loss_acc[k]/n_valid:.1f}' for k in ['coord','closure','bond','diffusion','stereo','physics_decoupled','physics_pairing','contact_aux','torus','chirality','contrastive','physics_bridge','distillation'] if k in loss_acc]
                print(f'    step {step}/{len(epoch_batches)} loss={avg:.2f} nan={nan_batches} '
                      f'lr={lr:.1e} GPU: cur={cur_mem:.2f}GB peak={max_mem:.2f}GB reserved={reserved:.2f}GB')
                print(f'      BREAKDOWN: ' + ' | '.join(parts))
                # [v4] Kendall UW learned σ² per core term (tracks how the model balances geometry vs physics)
                uw_parts = [f"{k}:σ²={float(np.exp(model.uncertainty_log_vars[k].item())):.2f}"
                            for k in ['coord','bond','stereo','physics_pairing','contact_aux','physics_bridge']]
                print(f'      UW-σ²: ' + ' | '.join(uw_parts))
                loss_acc.clear()
                torch.cuda.reset_peak_memory_stats(device)
                gc.collect()
                torch.cuda.empty_cache()

        # Validation
        avg_train = train_loss / max(n_batches, 1)
        model.eval()
        val_rmsd = 0.0; n_val = 0
        with torch.no_grad():
            for vi in range(min(20, 50)):
                i = val_idx[vi]
                L = meta[i]['length']
                if L < 4: continue
                s_ids = torch.tensor(
                    [{'A': 0, 'U': 1, 'G': 2, 'C': 3}.get(b, 4) for b in seqs[i]],
                    dtype=torch.long).unsqueeze(0).to(device)
                t_val = torch.tensor(coords[i], dtype=torch.float32).unsqueeze(0).to(device)
                p_val = model(s_ids, return_loss=False)
                t_c = t_val - t_val.mean(dim=1, keepdim=True)
                t_scale = torch.norm(t_c, dim=(1, 2), keepdim=True).clamp(min=1.0)
                p_c = p_val - p_val.mean(dim=1, keepdim=True)
                p_scale = torch.norm(p_c, dim=(1, 2), keepdim=True).clamp(min=1e-6)
                p_denorm = p_c / p_scale * t_scale + t_val.mean(dim=1, keepdim=True)
                p_s = p_denorm[0, :L]  # (L, 3)
                t_s = t_val[0, :L]     # (L, 3)
                if p_s.abs().sum() > 1e-6 and t_s.abs().sum() > 1e-6:
                    rmsd = kabsch_rmsd(p_s, t_s)
                    if not (np.isnan(rmsd) or np.isinf(rmsd)):
                        val_rmsd += rmsd; n_val += 1
        avg_val = val_rmsd / max(n_val, 1)

        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val; patience = 0
            torch.save({'model_state_dict': model.state_dict(), 'phase': phase,
                         'epoch': epoch+1, 'val_rmsd': avg_val},
                        os.path.join(output_dir, f'phase{phase}_best.pt'))
            print(f'    -> Best model saved (val_rmsd={avg_val:.1f}A)')
        else:
            patience += 1

        # Save full checkpoint every 5 epochs (resume-ready)
        if (epoch + 1) % 5 == 0:
            ckpt = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'phase': phase, 'epoch': epoch + 1,
                'best_val': best_val, 'patience': patience,
                'val_rmsd': avg_val, 'avg_train': avg_train,
                'bucket_weights': bucket_weights,
                'history': phase_history,
            }
            ckpt_path = os.path.join(output_dir, f'phase{phase}_epoch{epoch+1:03d}_full.pt')
            torch.save(ckpt, ckpt_path)
            print(f'    -> Full checkpoint saved: {os.path.basename(ckpt_path)}')

        lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_t0
        print(f'  [P{phase}] Epoch {epoch+1}/{n_phase_epochs} '
              f'train_loss={avg_train:.2f} val_rmsd={avg_val:.1f}A lr={lr:.1e} '
              f'time={epoch_time/60:.1f}m pat={patience}')
        phase_history.append({'phase': phase, 'epoch': epoch+1, 'train_loss': avg_train,
                               'val_rmsd': avg_val, 'lr': lr, 'loss_breakdown': loss_dict})

        if patience >= 10:
            print(f'  [P{phase}] Early stopping at epoch {epoch+1}')
            break

        gc.collect(); torch.cuda.empty_cache()

    # Phase-end full checkpoint (resume-ready)
    phase_end_ckpt = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'phase': phase, 'epoch': len(phase_history),
        'best_val': best_val, 'patience': patience,
        'val_rmsd': avg_val, 'avg_train': avg_train,
        'bucket_weights': bucket_weights,
        'history': phase_history,
    }
    ckpt_path = os.path.join(output_dir, f'phase{phase}_end_full.pt')
    torch.save(phase_end_ckpt, ckpt_path)
    print(f'  Phase-end checkpoint saved: {os.path.basename(ckpt_path)}')

    print(f'  Phase {phase} done: best val_rmsd={best_val:.1f}A')
    return best_val, phase_history


# ═══════════════════════════════════════════════════════════════
# Run all phases
# ═══════════════════════════════════════════════════════════════

all_history = []
for phase in range(1, 5):
    n_ep = DEFAULT_PHASE_EPOCHS[phase]
    best_val, history = train_one_phase(phase, n_ep)
    all_history.append({'phase': phase, 'best_val': best_val, 'history': history})
    with open(os.path.join(output_dir, f'phase{phase}_history.json'), 'w') as f:
        json.dump(history, f, indent=2, default=str)

with open(os.path.join(output_dir, 'all_history.json'), 'w') as f:
    json.dump(all_history, f, indent=2, default=str)

print()
print('=' * 60)
print('  S10 Curriculum Training Complete')
print('=' * 60)
for h in all_history:
    print(f'  Phase {h["phase"]}: best val_rmsd={h["best_val"]:.1f}A')
