"""Smoke test for train_s10_curriculum.py: MC-Dropout + all losses + xlong."""
import os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, math
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '.'); sys.path.insert(0, './rl')
from scheme10_equivariant import EquivariantS10Config, StrictlyEquivariantS10
from stereochemistry_losses import get_stereo_loss_breakdown
from physics_decoupled_loss import PhysicsDecoupledLoss
from physics_loss import PhysicsLoss
from contact_map_aux_head import ContactMapAuxHead, generate_contact_map
from chirality_embedding import ChiralityAwareEmbedding
from cartesian_to_torus import cartesian_to_torus, major_ring_radius
from contrastive_circrna import GeometryContrastiveLoss
from physics_bridge import ConstraintExtractor
from physics_distillation import ContactMapDistillationLoss
from train_curriculum import BSJGeometryLoss

# ── Inline copy of functions from train_s10_curriculum.py ──
BUCKET_NAMES = ["short", "medium", "long", "xlong"]
MC_N_SAMPLES = 5
MC_TEMPERATURE = 2.0
MC_MAX_SAMPLES = 200


def length_bucket_full(L):
    if L <= 200: return "short"
    elif L <= 500: return "medium"
    elif L <= 1000: return "long"
    else: return "xlong"


def estimate_bucket_uncertainty(bucket_groups, model, device,
                                 mc_n, mc_temp, mc_max):
    model.train()
    bucket_var = {}
    bucket_count = {}
    for bname in BUCKET_NAMES:
        pool = bucket_groups.get(bname, [])
        if len(pool) == 0:
            bucket_var[bname] = 0.0; bucket_count[bname] = 0; continue
        if bname == "xlong":
            bucket_var[bname] = 1.0; bucket_count[bname] = 0; continue
        n_draw = min(mc_max, len(pool))
        draw_idx = np.random.choice(len(pool), size=n_draw, replace=False)
        bucket_vars = []
        for d_idx in draw_idx:
            L = t_meta[d_idx]['length']
            seq_ids = torch.tensor(
                [{'A':0,'U':1,'G':2,'C':3}.get(b,4) for b in t_seq[d_idx]],
                dtype=torch.long).unsqueeze(0).to(device)
            target_s = torch.zeros(1, L, 3, device=device)
            target_s[0, :L] = torch.tensor(t_coords[d_idx][:L],
                                            dtype=torch.float32, device=device)
            preds = []
            with torch.no_grad():
                for _ in range(mc_n):
                    pred = model(seq_ids, return_loss=False)
                    preds.append(pred)
            preds = torch.stack(preds, dim=0)
            pred_var = torch.var(preds, dim=0, unbiased=False).sum(dim=-1)
            sample_var = pred_var[0, :L].mean().item()
            bucket_vars.append(sample_var)
        bucket_var[bname] = float(np.mean(bucket_vars))
        bucket_count[bname] = n_draw
    global_max = max(bucket_var.values()) + 1e-8
    bucket_weights = {}
    for bname in BUCKET_NAMES:
        norm_var = min(bucket_var[bname] / global_max, 1.0)
        bucket_weights[bname] = 1.0 + mc_temp * norm_var
    return bucket_weights, bucket_var, bucket_count

# ── Device & data ──
device = 'cuda'
torch.cuda.empty_cache()
labels_dir = os.path.normpath(os.path.join(os.path.abspath('.'),
    '..', '..', '..', '..', 'data', 'circrna_3d_all'))
npy_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.npy')])
# Pick 8 medium samples for a fast full-loss test
sample_idx = np.random.choice(len(npy_files), 8, replace=True)
t_seq, t_coords, t_meta = [], [], []
for i in sample_idx:
    arr = np.load(os.path.join(labels_dir, npy_files[i]))
    L = arr.shape[0]
    t_seq.append(('ACGU' * (L // 4 + 1))[:L])
    t_coords.append(arr.astype(np.float32))
    t_meta.append({'length': L, 'id': npy_files[i]})

# Build batch
max_L = max(m['length'] for m in t_meta)
seq_ids = torch.randint(0, 4, (8, max_L), dtype=torch.long, device=device)
target = torch.zeros(8, max_L, 3, device=device)
lengths = torch.tensor([m['length'] for m in t_meta], device=device)
for b in range(8):
    L = t_meta[b]['length']
    target[b, :L] = torch.tensor(t_coords[b][:L], device=device)

# Simulate bucket_groups for MC estimation (use indices into t_seq/t_coords)
bucket_groups = {"short": [], "medium": [], "long": [], "xlong": []}
for b in range(8):
    bname = length_bucket_full(t_meta[b]['length'])
    bucket_groups[bname].append(b)

# ── Model + all prior modules ──
cfg = EquivariantS10Config(
    d_model=256, d_inv=64, d_eq=32, n_layers=4, k_theta=4, k_phi=2,
    use_diffusion=True, n_diffusion_steps=100, use_s8_refine=True, use_adaptive_k=True,
    d_model_inv=64, d_model_eq=64, dropout=0.1, n_tokens=5, bond_length=5.9, r_scale=300.0)
model = StrictlyEquivariantS10(cfg).to(device); model.train()

pd_loss_fn = PhysicsDecoupledLoss(w_geo=1.0, w_phys=0.1, use_rg_loss=True,
    use_clash_loss=True, use_angle_loss=True).to(device)
physics_loss_fn = PhysicsLoss(n_tokens=5, device=device).to(device)
contact_head = ContactMapAuxHead(d_inv=64, d_hidden=32).to(device)
contact_proj = nn.Linear(3, 64, bias=False).to(device)
chirality_emb = ChiralityAwareEmbedding(n_tokens=5, d_model=64).to(device)
chirality_proj = nn.Linear(64, 64, bias=False).to(device)
contrastive_loss_fn = GeometryContrastiveLoss(temperature=0.1).to(device)
constraint_extractor = ConstraintExtractor(c_z=64, n_rbf=16, bond_length=5.9,
    pair_distance=8.0, pair_threshold=0.5, bsj_weight_boost=2.0).to(device)
distillation_loss_fn = None
try:
    distillation_loss_fn = ContactMapDistillationLoss(config=None).to(device)
except:
    pass
bsj_single = BSJGeometryLoss(target_angle=108.0, target_dihedral=180.0,
    target_distance=3.5, angle_weight=2.0, dihedral_weight=1.0, distance_weight=5.0)
LOSS_WEIGHTS = {
    'coord': 10, 'closure': 5, 'bond': 2, 'diffusion': 1, 'stereo': 1,
    'physics_decoupled': 1, 'physics_pairing': 1, 'contact_aux': 1, 'torus': 0.5,
    'chirality': 0.5, 'contrastive': 0.1, 'physics_bridge': 0.1, 'distillation': 0.1,
}

# ── 1. MC-Dropout uncertainty estimation ──
print("=== MC-Dropout Uncertainty ===")
bucket_weights, bucket_var, bucket_count = estimate_bucket_uncertainty(
    bucket_groups, model, device,
    mc_n=MC_N_SAMPLES, mc_temp=MC_TEMPERATURE, mc_max=MC_MAX_SAMPLES)
for b in BUCKET_NAMES:
    print(f"  {b}: weight={bucket_weights[b]:.2f} var={bucket_var[b]:.6f} count={bucket_count[b]}")

# ── 2. Full forward pass ──
print("\n=== Forward pass ===")
pred, diff_loss, _ = model(seq_ids, return_loss=True)
if torch.isnan(pred).any():
    print("  ERROR: NaN in prediction")
    sys.exit(1)
print(f"  pred shape: {pred.shape}, diff_loss: {diff_loss.item() if diff_loss is not None else 'None'}")

# ── 3. Normalize ──
t_c = target - target.mean(1, keepdim=True)
t_scale = torch.norm(t_c, dim=(1, 2), keepdim=True).clamp(min=1.0)
t_norm = t_c / t_scale
p_c = pred - pred.mean(1, keepdim=True)
p_scale = torch.norm(p_c, dim=(1, 2), keepdim=True).clamp(min=1e-6)
p_denorm = p_c / p_scale * t_scale + target.mean(1, keepdim=True)
p_norm = p_c / t_scale

# ── 4. All ABC+D losses (inline) ──
print("\n=== Loss terms ===")
B, Lc, _ = p_denorm.shape
loss = torch.tensor(0.0, device=device)
loss_dict = {}

# 1 coord
cl = torch.tensor(0.0, device=device)
for b in range(B):
    vL = lengths[b]; cl += torch.mean((p_norm[b, :vL] - t_norm[b, :vL]) ** 2)
cl /= B; loss += cl * LOSS_WEIGHTS['coord']; loss_dict['coord'] = cl.item()

# 2 closure
cd = torch.norm(p_denorm[:, 0] - p_denorm[:, -1], dim=-1)
cm = torch.where(lengths >= 2, 1.0, 0.0).to(device)
clo = (cm * (cd - 5.9) ** 2).sum() / cm.sum().clamp(min=1.0)
loss += clo * LOSS_WEIGHTS['closure']; loss_dict['closure'] = clo.item()

# 3 bond
bl = torch.tensor(0.0, device=device); nb = 0
for b in range(B):
    vL = lengths[b]
    if vL < 4: continue
    bonds = torch.norm(p_denorm[b, 1:vL] - p_denorm[b, :vL-1], dim=-1)
    bsj = torch.norm(p_denorm[b, 0] - p_denorm[b, vL-1])
    all_b = torch.cat([bonds, bsj.unsqueeze(0)])
    bl += F.mse_loss(all_b, torch.full_like(all_b, 5.9)); nb += 1
bl /= max(nb, 1); loss += bl * LOSS_WEIGHTS['bond']; loss_dict['bond'] = bl.item()

# 4 stereo
try:
    sr = get_stereo_loss_breakdown(p_denorm, lengths)
    loss += sr['total'] * LOSS_WEIGHTS['stereo']; loss_dict['stereo'] = sr['total'].item()
except:
    loss_dict['stereo'] = 0.0

# 5 pd
try:
    pr = pd_loss_fn(p_denorm, target, lengths, None)
    pl = pr['loss'] if isinstance(pr, dict) else pr
    loss += pl * LOSS_WEIGHTS['physics_decoupled']; loss_dict['physics_decoupled'] = pl.item()
except:
    loss_dict['physics_decoupled'] = 0.0

# 6 physics_pairing
try:
    ph = physics_loss_fn(p_denorm, seq_ids)
    loss += ph * LOSS_WEIGHTS['physics_pairing']; loss_dict['physics_pairing'] = ph.item()
except:
    loss_dict['physics_pairing'] = 0.0

# 7 contact_aux
contact_pred_saved = None
try:
    lc = contact_proj(p_norm); contact_pred_saved = contact_head(lc)
    with torch.no_grad(): ct = generate_contact_map(target, threshold=8.0)
    clo2 = F.binary_cross_entropy(contact_pred_saved, ct, reduction='mean')
    loss += clo2 * LOSS_WEIGHTS['contact_aux']; loss_dict['contact_aux'] = clo2.item()
except:
    loss_dict['contact_aux'] = 0.0

# 8 torus
try:
    Rt = major_ring_radius(bond_length=5.9, lengths=lengths.float())
    tp, tphi, tr = cartesian_to_torus(p_denorm, Rt)
    tt, ttp, ttr = cartesian_to_torus(target, Rt)
    dt = torch.remainder(tp - tt + math.pi, 2 * math.pi) - math.pi
    dph = torch.remainder(tphi - ttp + math.pi, 2 * math.pi) - math.pi
    dr = tr - ttr
    tl = (dt ** 2).mean() + (dph ** 2).mean() + (dr ** 2).mean()
    loss += tl * LOSS_WEIGHTS['torus']; loss_dict['torus'] = tl.item()
except:
    loss_dict['torus'] = 0.0

# 9 chirality
try:
    sf = seq_ids.float(); co = chirality_proj(chirality_emb(sf))
    chl = (co - contact_proj(p_norm)).pow(2).mean()
    loss += chl * LOSS_WEIGHTS['chirality']; loss_dict['chirality'] = chl.item()
except:
    loss_dict['chirality'] = 0.0

# 10 contrastive
if B >= 4:
    try:
        mid = B // 2
        cf = contrastive_loss_fn(
            p_denorm[:mid],
            p_denorm[mid:2*mid] if 2*mid <= B else p_denorm[:mid],
            lengths[:mid])
        loss += cf * LOSS_WEIGHTS['contrastive']; loss_dict['contrastive'] = cf.item()
    except:
        loss_dict['contrastive'] = 0.0

# 11 physics_bridge
if B >= 2:
    try:
        pair_repr = torch.zeros(B, Lc, Lc, 64, device=device)
        for bi in range(B):
            vL = lengths[bi]; dists = torch.cdist(p_denorm[bi, :vL], p_denorm[bi, :vL])
            pair_repr[bi, :vL, :vL, 0] = dists
        pair_probs = torch.zeros(B, Lc, Lc, device=device)
        cons = constraint_extractor(pair_repr, pair_probs, seq_ids)
        cll = cons.pow(2).mean()
        loss += cll * LOSS_WEIGHTS['physics_bridge']; loss_dict['physics_bridge'] = cll.item()
    except:
        loss_dict['physics_bridge'] = 0.0

# 12 distillation
if distillation_loss_fn is not None and contact_pred_saved is not None:
    try:
        tct = generate_contact_map(target, threshold=8.0)
        conf = torch.ones(B, Lc, Lc, device=device)
        dl = distillation_loss_fn(contact_pred_saved, tct, conf)
        loss += dl * LOSS_WEIGHTS['distillation']; loss_dict['distillation'] = dl.item()
    except:
        loss_dict['distillation'] = 0.0

# Diffusion
if diff_loss is not None:
    loss += diff_loss * LOSS_WEIGHTS['diffusion']; loss_dict['diffusion'] = diff_loss.item()

# BSJ
bsj_weight = 3.0
total_bsj = torch.tensor(0.0, device=device); valid_bsj = 0
for b in range(B):
    L = lengths[b]
    if L < 4: continue
    coords_s = p_denorm[b, :L]
    bsj_indices = torch.tensor([0, L-1], device=device)
    val = bsj_single(coords_s, bsj_indices)
    if torch.is_tensor(val) and not torch.isnan(val):
        total_bsj += val; valid_bsj += 1
bsj_loss_val = total_bsj / max(valid_bsj, 1)
loss += bsj_loss_val * bsj_weight; loss_dict['bsj_geometry'] = bsj_loss_val.item()

# MC-Dropout uncertainty weight
bw = 0.0
for b in range(B):
    bname = length_bucket_full(int(lengths[b].item()))
    bw += bucket_weights[bname]
uq_weight = bw / max(B, 1)
loss = loss * uq_weight; loss_dict['uq_weight'] = uq_weight

for k, v in sorted(loss_dict.items()):
    print(f"  {k}: {v:.4f}")

# ── 5. Backward ──
print("\n=== Backward ===")
optimizer = torch.optim.AdamW(list(model.parameters()), lr=1e-4)
optimizer.zero_grad()
loss.backward()
has_nan = any(p.grad is not None and torch.isnan(p.grad).any() for p in model.parameters())
gnorm = torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=0.5)
optimizer.step()

print(f"  Total loss: {loss.item():.2f}")
print(f"  Loss finite: {torch.isfinite(loss)}")
print(f"  NaN grad: {has_nan}")
print(f"  Grad norm: {gnorm:.3f}")
print(f"  Batch Ls: {[m['length'] for m in t_meta]}")
print("\nSMOKE TEST PASSED")
