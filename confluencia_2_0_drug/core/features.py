from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

# Import canonical AA constants from shared
from confluencia_shared.features import AA_ORDER, AA_TO_IDX

# ---------------------------------------------------------------------------
# circRNA constants
# ---------------------------------------------------------------------------
NT_ORDER: Tuple[str, ...] = ("A", "U", "G", "C")
NT_TO_IDX: Dict[str, int] = {nt: i for i, nt in enumerate(NT_ORDER)}

# Literature-derived thermodynamic nearest-neighbor parameters (kcal/mol, 37°C)
# Based on Turner 1999 / Xia et al. 1998 for RNA.
_NN_PARAMS: Dict[Tuple[str, str], float] = {
    ("A", "U"): -0.93, ("U", "A"): -0.93,
    ("G", "C"): -2.24, ("C", "G"): -2.24,
    ("G", "U"): -1.44, ("U", "G"): -1.44,
    ("A", "A"): -0.55, ("A", "C"): -1.10,
    ("C", "A"): -1.10, ("C", "C"): -2.10,
    ("G", "A"): -1.33, ("A", "G"): -1.33,
    ("C", "U"): -1.44, ("U", "C"): -1.44,
    ("G", "G"): -2.08, ("U", "U"): -1.09,
}
_INIT_PARAMS: Dict[str, float] = {"G": -0.40, "C": -0.40, "A": 0.20, "U": 0.20}

# Known IRES consensus motifs and their relative strengths (literature-derived)
_IRES_MOTIFS: List[Tuple[str, float]] = [
    ("GCGCC", 0.85), ("CCUG", 0.72), ("GGGG", 0.68),  # strong
    ("UUGU", 0.60), ("AUGG", 0.55), ("GCCC", 0.50),  # moderate
    ("CUGA", 0.45), ("GCAA", 0.42),                    # weak
]

# Kozak consensus: (gcc)gccRccAUGG (uppercase = highly conserved)
_KOZAK_CORE = "AUG"
_KOZAK_OPTIMAL_3PRIME = "G"  # +4 position

# Modification half-life extension factors vs unmodified (literature priors)
_MOD_STABILITY_FACTOR: Dict[str, float] = {
    "none": 1.0,
    "m6A": 1.8,   # N6-methyladenosine: ~1.5-2x stability
    "Ψ": 2.5,     # Pseudouridine: ~2-3x stability
    "5mC": 2.0,   # 5-methylcytidine: ~2x stability
    "ms2m6A": 3.0, # N6,2'-O-dimethyladenosine (cap analog): ~3x
}

# Modification innate-immune evasion factors (0 = no evasion, 1 = full evasion)
_MOD_IMMUNE_EVASION: Dict[str, float] = {
    "none": 0.0,
    "m6A": 0.25,
    "Ψ": 0.55,
    "5mC": 0.35,
    "ms2m6A": 0.50,
}

# Delivery vector baseline parameters
_DELIVERY_PARAMS: Dict[str, Dict[str, float]] = {
    "LNP_standard": {"bioavailability": 0.65, "liver_target": 0.80, "spleen_target": 0.10,
                     "muscle_target": 0.03, "endosomal_escape": 0.02, "half_life_base_h": 6.0},
    "LNP_liver":    {"bioavailability": 0.75, "liver_target": 0.90, "spleen_target": 0.05,
                     "muscle_target": 0.01, "endosomal_escape": 0.03, "half_life_base_h": 8.0},
    "LNP_spleen":   {"bioavailability": 0.60, "liver_target": 0.35, "spleen_target": 0.50,
                     "muscle_target": 0.02, "endosomal_escape": 0.02, "half_life_base_h": 5.0},
    "AAV":          {"bioavailability": 0.90, "liver_target": 0.60, "spleen_target": 0.15,
                     "muscle_target": 0.10, "endosomal_escape": 0.90, "half_life_base_h": 168.0},
    "naked":        {"bioavailability": 0.05, "liver_target": 0.20, "spleen_target": 0.10,
                     "muscle_target": 0.05, "endosomal_escape": 0.005, "half_life_base_h": 0.5},
}


@dataclass(frozen=True)
class MixedFeatureSpec:
    smiles_hash_dim: int = 128
    smiles_rdkit_bits: int = 2048
    smiles_rdkit_version: int = 2
    prefer_rdkit: bool = True
    # ── Enhanced feature flags ──────────────────────────────
    use_gnn: bool = False
    use_chemberta: bool = False
    use_esm2: bool = False
    use_pk_prior: bool = False
    use_dose_response: bool = False
    use_feature_selection: bool = False
    use_cross_features: bool = False
    use_auxiliary_labels: bool = False
    target_transform: str = "none"  # "none" or "logit"
    online_mode: bool = False  # attempt online download for pretrained encoders
    feature_selection_top_k: int = 512
    feature_selection_correl_thresh: float = 0.95
    gnn_hidden_dim: int = 128
    chemberta_model: str = "seyonec/ChemBERTa-zinc-base-v1"
    esm2_model_size: str = "650M"
    cache_dir: str = "./.cache"


def build_feature_names(spec: MixedFeatureSpec | None = None, env_cols: List[str] | None = None) -> List[str]:
    spec = spec or MixedFeatureSpec()
    env_cols = list(env_cols or [])

    if bool(spec.prefer_rdkit):
        smiles_names = [f"smiles_morgan_{i:04d}" for i in range(int(spec.smiles_rdkit_bits))]
        if int(spec.smiles_rdkit_version) >= 2:
            smiles_names.extend(
                [
                    "smiles_desc_mol_wt",
                    "smiles_desc_logp",
                    "smiles_desc_tpsa",
                    "smiles_desc_hbd",
                    "smiles_desc_hba",
                    "smiles_desc_rot_bonds",
                    "smiles_desc_rings",
                    "smiles_desc_frac_csp3",
                ]
            )
    else:
        smiles_names = [f"smiles_hash_{i:04d}" for i in range(int(spec.smiles_hash_dim))]

    # ── Epitope block ─────────────────────────────────────
    if bool(spec.use_esm2):
        epi_dim = 1280 if str(spec.esm2_model_size) == "650M" else 320
        epi_names = [f"esm2_epi_{i:04d}" for i in range(epi_dim)]
    else:
        epi_names = [f"epitope_frac_{aa}" for aa in AA_ORDER]
        epi_names.extend(["epitope_hydrophobic_frac", "epitope_polar_frac",
                          "epitope_acidic_frac", "epitope_basic_frac"])

    # ── GNN block ─────────────────────────────────────────
    gnn_names = []
    if bool(spec.use_gnn):
        gnn_names = [f"gnn_emb_{i:04d}" for i in range(int(spec.gnn_hidden_dim))]

    # ── ChemBERTa block ──────────────────────────────────
    chemberta_names = []
    if bool(spec.use_chemberta):
        chemberta_names = [f"chemberta_{i:04d}" for i in range(768)]

    # ── Dose-response block ──────────────────────────────
    dr_names = []
    if bool(spec.use_dose_response):
        dr_names = [
            "dr_cumul_dose", "dr_dose_intensity", "dr_time_above_thresh",
            "dr_emax", "dr_ec50", "dr_hill", "dr_auc_dose",
            "dr_cmax_dose", "dr_onset_time", "dr_duration",
            "dr_mic_thresh", "dr_rate_accum",
        ]

    # ── PK prior block ────────────────────────────────────
    pk_names = []
    if bool(spec.use_pk_prior):
        pk_names = [
            "pk_lipinski_violations", "pk_rot_ratio", "pk_sa_score",
            "pk_esol_logs", "pk_half_life_cat", "pk_bioavail_cat",
            "pk_ppb_estimate", "pk_hba_norm", "pk_hbd_norm",
        ]

    # ── Cross-features block ──────────────────────────────
    cross_names = []
    if bool(spec.use_cross_features):
        cross_names = [
            "x_dose_x_binding", "x_dose_x_immune", "x_dose_per_freq",
            "x_freq_x_time", "x_binding_x_immune", "x_dose_sq",
            "x_log_dose", "x_dose_x_time", "x_cumul_x_binding",
        ]

    # ── Auxiliary labels block ────────────────────────────
    aux_names = []
    if bool(spec.use_auxiliary_labels):
        aux_names = ["aux_target_binding", "aux_immune_activation"]

    env_names = [f"env_{str(c)}" for c in env_cols]
    return [*smiles_names, *gnn_names, *chemberta_names, *epi_names,
            *dr_names, *pk_names, *cross_names, *aux_names, *env_names]


def _rdkit_total_dim(n_bits: int, version: int) -> int:
    return int(n_bits) + (8 if int(version) >= 2 else 0)


def _esm2_embedding_dim(model_size: str) -> int:
    return 1280 if str(model_size) == "650M" else 320


def _feature_block_dims(spec: MixedFeatureSpec) -> Dict[str, int]:
    """Return dimension of each feature block for a spec."""
    return {
        "smiles": _rdkit_total_dim(int(spec.smiles_rdkit_bits), int(spec.smiles_rdkit_version)),
        "gnn": int(spec.gnn_hidden_dim) if bool(spec.use_gnn) else 0,
        "chemberta": 768 if bool(spec.use_chemberta) else 0,
        "epitope": _esm2_embedding_dim(str(spec.esm2_model_size)) if bool(spec.use_esm2) else (len(AA_ORDER) + 4),
        "dose_response": 12 if bool(spec.use_dose_response) else 0,
        "pk_prior": 9 if bool(spec.use_pk_prior) else 0,
        "cross_features": 9 if bool(spec.use_cross_features) else 0,
        "auxiliary_labels": 2 if bool(spec.use_auxiliary_labels) else 0,
    }


# ===================================================================
# Dose-response curve features
# ===================================================================

def logit_transform(y: np.ndarray) -> np.ndarray:
    """Apply logit transform to bounded [0,1] targets.

    logit(e) = log(e / (1 - e))
    Stabilizes variance for bounded efficacy targets by mapping [0,1] → (-∞, +∞).
    Clips values away from 0 and 1 to avoid infinite logit values.
    """
    eps = 1e-4
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    y_clipped = np.clip(y, eps, 1.0 - eps)
    return np.log(y_clipped / (1.0 - y_clipped)).astype(np.float32)


def inverse_logit(z: np.ndarray) -> np.ndarray:
    """Inverse logit: sigmoid mapping (-∞, +∞) → [0, 1].

    sigmoid(z) = 1 / (1 + exp(-z))
    """
    z = np.asarray(z, dtype=np.float32).reshape(-1)
    return (1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))).astype(np.float32)


def compute_dose_response_features(
    dose: np.ndarray,
    freq: np.ndarray,
    treatment_time: np.ndarray,
    mic_threshold: float = 1.0,
    hill: float = 1.0,
) -> np.ndarray:
    """Derive dose-response curve features from dosing parameters.

    Features (12-dim):
      cumul_dose = dose * freq * treatment_time
      dose_intensity = dose / treatment_time
      time_above_thresh = treatment_time * sigmoid((dose - mic) / (dose + eps))
      emax_proxy = dose^hill / (ec50^hill + dose^hill)  [normalized Emax]
      ec50_proxy = treatment_time * 0.5  [simplified EC50]
      hill (raw)
      auc_dose_proxy = cumul_dose * 0.8
      cmax_dose_proxy = dose * freq
      onset_time = log1p(treatment_time) * 0.3
      duration = log1p(treatment_time)
      mic_thresh (raw)
      rate_accum = cumul_dose / (treatment_time + eps)
    """
    eps = 1e-9
    dose = np.asarray(dose, dtype=np.float32).reshape(-1)
    freq = np.asarray(freq, dtype=np.float32).reshape(-1)
    tt = np.asarray(treatment_time, dtype=np.float32).reshape(-1)
    n = len(dose)

    hill_arr = np.full(n, float(hill), dtype=np.float32)
    mic_arr = np.full(n, float(mic_threshold), dtype=np.float32)

    cumul_dose = dose * freq * tt
    dose_intensity = dose / (tt + eps)
    dose_exp = np.power(dose + eps, hill_arr + eps)
    ec50_exp = np.power(mic_arr + eps, hill_arr + eps)
    emax_vals = dose_exp / (ec50_exp + dose_exp + eps)
    sigmoid_term = 1.0 / (1.0 + np.exp(-(dose - mic_arr) / (dose + eps)))
    time_above = tt * sigmoid_term
    auc_proxy = cumul_dose * 0.8
    cmax_proxy = dose * freq
    onset = np.log1p(tt) * 0.3
    duration = np.log1p(tt)
    rate_accum = cumul_dose / (tt + eps)

    return np.column_stack([
        cumul_dose, dose_intensity, time_above,
        emax_vals, ec50_exp, hill_arr,
        auc_proxy, cmax_proxy, onset, duration,
        mic_arr, rate_accum,
    ]).astype(np.float32)


# ===================================================================
# Cross-features (dose/freq/binding interactions)
# ===================================================================

def compute_cross_features(
    dose: np.ndarray,
    freq: np.ndarray,
    treatment_time: np.ndarray,
    target_binding: np.ndarray | None = None,
    immune_activation: np.ndarray | None = None,
) -> np.ndarray:
    """Compute interaction features between dose, freq, binding, and immune.

    Features (9-dim):
      0. dose × target_binding (synergistic efficacy potential)
      1. dose × immune_activation (dose-dependent immune response)
      2. dose / freq (dose per administration)
      3. freq × treatment_time (exposure frequency)
      4. target_binding × immune_activation (binding-immune synergy)
      5. dose² (dose nonlinearity)
      6. log(dose + 1) (log-scale dose)
      7. dose × treatment_time (cumulative exposure proxy)
      8. (dose × freq × time) × target_binding (context-aware binding)
    """
    eps = 1e-9
    n = len(dose)
    dose = np.asarray(dose, dtype=np.float32).reshape(-1)
    freq = np.asarray(freq, dtype=np.float32).reshape(-1)
    tt = np.asarray(treatment_time, dtype=np.float32).reshape(-1)

    # Default missing binding/immune to 0.5 (neutral prior)
    if target_binding is None:
        target_binding = np.full(n, 0.5, dtype=np.float32)
    else:
        target_binding = np.asarray(target_binding, dtype=np.float32).reshape(-1)

    if immune_activation is None:
        immune_activation = np.full(n, 0.5, dtype=np.float32)
    else:
        immune_activation = np.asarray(immune_activation, dtype=np.float32).reshape(-1)

    out = np.zeros((n, 9), dtype=np.float32)
    out[:, 0] = dose * target_binding  # dose × binding
    out[:, 1] = dose * immune_activation  # dose × immune
    out[:, 2] = dose / (freq + eps)  # dose per administration
    out[:, 3] = freq * tt  # exposure frequency
    out[:, 4] = target_binding * immune_activation  # binding × immune
    out[:, 5] = dose * dose  # dose²
    out[:, 6] = np.log1p(dose)  # log(dose + 1)
    out[:, 7] = dose * tt  # cumulative exposure proxy
    out[:, 8] = dose * freq * tt * target_binding  # context-aware binding

    return out


# ===================================================================
# PK prior (ADMET-lite) features
# ===================================================================

def compute_pk_prior_features(mol_list: List[Any]) -> np.ndarray:
    """Compute ADMET-lite features from RDKit mol objects.

    Features (9-dim):
      0. Lipinski rule-of-5 violations count (MW>500, LogP>5, HBD>5, HBA>10, rotB>5)
      1. Rotatable bonds / heavy atoms ratio
      2. Synthetic accessibility proxy (based on rings + heteroatoms)
      3. ESOL LogS: logS = 0.16 - 0.63*LogP - 0.0062*MW + 0.066*RB - 0.74*AP
      4. Estimated half-life category (0-3: short/medium/long/extended)
      5. Estimated bioavailability category (0-3: poor/fair/good/excellent)
      6. Plasma protein binding estimate (0-1)
      7. HBA / (HBA + HBD) normalized
      8. HBD / (HBA + HBD) normalized
    """
    n = len(mol_list)
    out = np.zeros((n, 9), dtype=np.float32)
    eps = 1e-9

    try:
        Descriptors = importlib.import_module("rdkit.Chem.Descriptors")
        rdMolDescriptors = importlib.import_module("rdkit.Chem.rdMolDescriptors")
    except Exception:
        return out

    mol_wt_f = getattr(Descriptors, "MolWt", None)
    mol_logp_f = getattr(Descriptors, "MolLogP", None)
    n_rot_f = getattr(Descriptors, "NumRotatableBonds", None)
    calc_hbd = getattr(rdMolDescriptors, "CalcNumHBD", None)
    calc_hba = getattr(rdMolDescriptors, "CalcNumHBA", None)
    calc_tpsa = getattr(rdMolDescriptors, "CalcTPSA", None)
    calc_rings = getattr(rdMolDescriptors, "CalcNumRings", None)

    for i, mol in enumerate(mol_list):
        if mol is None or mol_wt_f is None:
            continue
        try:
            mw = float(mol_wt_f(mol))
            logp = float(mol_logp_f(mol))
            hbd = float(calc_hbd(mol)) if calc_hbd else 0.0
            hba = float(calc_hba(mol)) if calc_hba else 0.0
            rot_b = float(n_rot_f(mol)) if n_rot_f else 0.0
            heavy = float(Descriptors.HeavyAtomCount(mol))
            tpsa = float(calc_tpsa(mol)) if calc_tpsa else 0.0
            rings = float(calc_rings(mol)) if calc_rings else 0.0
        except Exception:
            continue

        # 0. Lipinski violations
        viol = 0
        if mw > 500: viol += 1
        if logp > 5: viol += 1
        if hbd > 5: viol += 1
        if hba > 10: viol += 1
        if rot_b > 5: viol += 1
        out[i, 0] = float(viol)

        # 1. Rotatable ratio
        out[i, 1] = rot_b / max(heavy, 1.0)

        # 2. SA proxy (rings + heteroatom fraction)
        ap = tpsa / 100.0
        sa = min(10.0, rings * 0.5 + (heavy - rings) * 0.01)
        out[i, 2] = sa

        # 3. ESOL LogS
        logs = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rot_b - 0.74 * ap
        out[i, 3] = logs

        # 4. Half-life category (rough proxy: MW + LogP + HBD)
        hl_score = (mw / 500.0) + (logp / 5.0) + (hbd / 5.0)
        out[i, 4] = min(3.0, hl_score / 3.0) * 3.0

        # 5. Bioavailability category
        score = 5 - viol
        out[i, 5] = max(0.0, min(3.0, score))

        # 6. PPB estimate: LogP > 3 → ~95%, LogP < 1 → ~10%
        ppb = np.clip(0.5 * logp + 0.5, 0.0, 1.0)
        out[i, 6] = float(ppb)

        # 7-8. HBA/HBD normalized
        total = hba + hbd + eps
        out[i, 7] = hba / total
        out[i, 8] = hbd / total

    return out


# ===================================================================
# Main feature matrix builder
# ===================================================================

def _resolve_label(df: pd.DataFrame, col: str) -> np.ndarray | None:
    """Try to resolve a regression label column from the DataFrame."""
    if col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() > 0:
            return vals.fillna(0.0).to_numpy(dtype=np.float32)
    return None


def _safe_str(v: object) -> str:
    if v is None:
        return ""
    return str(v).strip()


def encode_epitope(seq: str) -> np.ndarray:
    s = _safe_str(seq).upper().replace(" ", "")
    arr = np.zeros(len(AA_ORDER) + 4, dtype=np.float32)
    if not s:
        return arr

    counts = np.zeros(len(AA_ORDER), dtype=np.float32)
    hydrophobic = set("AVILMFWY")
    polar = set("STNQCY")
    acidic = set("DE")
    basic = set("KRH")

    hydro_c = 0.0
    polar_c = 0.0
    acidic_c = 0.0
    basic_c = 0.0

    for ch in s:
        idx = AA_TO_IDX.get(ch)
        if idx is not None:
            counts[idx] += 1.0
        if ch in hydrophobic:
            hydro_c += 1.0
        if ch in polar:
            polar_c += 1.0
        if ch in acidic:
            acidic_c += 1.0
        if ch in basic:
            basic_c += 1.0

    length = float(len(s))
    arr[: len(AA_ORDER)] = counts / max(length, 1.0)
    arr[len(AA_ORDER) + 0] = hydro_c / max(length, 1.0)
    arr[len(AA_ORDER) + 1] = polar_c / max(length, 1.0)
    arr[len(AA_ORDER) + 2] = acidic_c / max(length, 1.0)
    arr[len(AA_ORDER) + 3] = basic_c / max(length, 1.0)
    return arr


def encode_smiles_hash(smiles: str, dim: int = 128) -> np.ndarray:
    s = _safe_str(smiles)
    vec = np.zeros(dim, dtype=np.float32)
    if not s:
        return vec
    # Lightweight hash-based tokenization to avoid hard dependency on RDKit in the prototype.
    for i, ch in enumerate(s):
        slot = (hash((ch, i % 8)) % dim + dim) % dim
        vec[slot] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32)


def _try_encode_smiles_rdkit(smiles: str, n_bits: int = 2048, version: int = 2) -> Tuple[np.ndarray, bool]:
    """1.0-style featurization: Morgan bits + selected RDKit descriptors.

    Returns (vector, ok). If RDKit is unavailable or parsing fails, returns zeros and False.
    """
    s = _safe_str(smiles)
    if not s:
        dim = _rdkit_total_dim(n_bits=n_bits, version=version)
        return np.zeros((dim,), dtype=np.float32), False

    try:
        Chem: Any = importlib.import_module("rdkit.Chem")
        DataStructs: Any = importlib.import_module("rdkit.DataStructs")
        AllChem: Any = importlib.import_module("rdkit.Chem.AllChem")
        Descriptors: Any = importlib.import_module("rdkit.Chem.Descriptors")
        rdMolDescriptors: Any = importlib.import_module("rdkit.Chem.rdMolDescriptors")
        rdFingerprintGenerator: Any = importlib.import_module("rdkit.Chem.rdFingerprintGenerator")
    except Exception:
        dim = _rdkit_total_dim(n_bits=n_bits, version=version)
        return np.zeros((dim,), dtype=np.float32), False

    mol_from_smiles = getattr(Chem, "MolFromSmiles", None)
    if mol_from_smiles is None:
        dim = _rdkit_total_dim(n_bits=n_bits, version=version)
        return np.zeros((dim,), dtype=np.float32), False

    mol = mol_from_smiles(s)
    if mol is None:
        dim = _rdkit_total_dim(n_bits=n_bits, version=version)
        return np.zeros((dim,), dtype=np.float32), False

    convert_np = getattr(DataStructs, "ConvertToNumpyArray", None)
    if convert_np is None:
        dim = _rdkit_total_dim(n_bits=n_bits, version=version)
        return np.zeros((dim,), dtype=np.float32), False

    # Prefer MorganGenerator to avoid RDKit deprecation warnings; fallback keeps compatibility.
    fp = None
    get_morgan_generator = getattr(rdFingerprintGenerator, "GetMorganGenerator", None)
    if get_morgan_generator is not None:
        generator = get_morgan_generator(radius=2, fpSize=int(n_bits))
        get_fp_new = getattr(generator, "GetFingerprint", None)
        if get_fp_new is not None:
            fp = get_fp_new(mol)
    if fp is None:
        get_fp_old = getattr(AllChem, "GetMorganFingerprintAsBitVect", None)
        if get_fp_old is None:
            dim = _rdkit_total_dim(n_bits=n_bits, version=version)
            return np.zeros((dim,), dtype=np.float32), False
        fp = get_fp_old(mol, 2, nBits=int(n_bits))
    arr = np.zeros((int(n_bits),), dtype=np.int8)
    convert_np(fp, arr)
    out = arr.astype(np.float32)

    if int(version) >= 2:
        mol_wt = getattr(Descriptors, "MolWt", None)
        mol_logp = getattr(Descriptors, "MolLogP", None)
        n_rot = getattr(Descriptors, "NumRotatableBonds", None)
        calc_tpsa = getattr(rdMolDescriptors, "CalcTPSA", None)
        calc_hbd = getattr(rdMolDescriptors, "CalcNumHBD", None)
        calc_hba = getattr(rdMolDescriptors, "CalcNumHBA", None)
        calc_rings = getattr(rdMolDescriptors, "CalcNumRings", None)
        calc_csp3 = getattr(rdMolDescriptors, "CalcFractionCSP3", None)
        if None in (mol_wt, mol_logp, n_rot, calc_tpsa, calc_hbd, calc_hba, calc_rings, calc_csp3):
            # Keep dimension stable even if some descriptor callables are unavailable.
            desc_zeros = np.zeros((8,), dtype=np.float32)
            return np.concatenate([out, desc_zeros], axis=0).astype(np.float32), True

        mol_wt_f = cast(Any, mol_wt)
        mol_logp_f = cast(Any, mol_logp)
        n_rot_f = cast(Any, n_rot)
        calc_tpsa_f = cast(Any, calc_tpsa)
        calc_hbd_f = cast(Any, calc_hbd)
        calc_hba_f = cast(Any, calc_hba)
        calc_rings_f = cast(Any, calc_rings)
        calc_csp3_f = cast(Any, calc_csp3)

        desc = np.array(
            [
                float(mol_wt_f(mol)),
                float(mol_logp_f(mol)),
                float(calc_tpsa_f(mol)),
                float(calc_hbd_f(mol)),
                float(calc_hba_f(mol)),
                float(n_rot_f(mol)),
                float(calc_rings_f(mol)),
                float(calc_csp3_f(mol)),
            ],
            dtype=np.float32,
        )
        out = np.concatenate([out, desc], axis=0).astype(np.float32)

    return out, True


def build_feature_matrix(df: pd.DataFrame, spec: MixedFeatureSpec | None = None) -> Tuple[np.ndarray, List[str], str]:
    """Build complete feature matrix from DataFrame.

    Blocks concatenated in order:
      [smiles_rdkit/hash, gnn?, chemberta?, epitope/esm2?,
       dose_response?, pk_prior?, env_cols]

    Optionally applies feature selection (use_feature_selection=True).
    """
    spec = spec or MixedFeatureSpec()
    smiles_col = "smiles"
    epi_col = "epitope_seq"

    env_cols = [c for c in ["dose", "freq", "treatment_time"] if c in df.columns]
    rdkit_dim = _rdkit_total_dim(n_bits=int(spec.smiles_rdkit_bits), version=int(spec.smiles_rdkit_version))

    n = len(df)
    smiles_list = df[smiles_col].astype(str).tolist() if smiles_col in df.columns else [""] * n
    epi_list = df[epi_col].astype(str).tolist() if epi_col in df.columns else [""] * n

    blocks: List[np.ndarray] = []
    used_rdkit = 0

    # ── Block 1: SMILES RDKit / hash ────────────────────────
    if bool(spec.prefer_rdkit):
        sm_rows: List[np.ndarray] = []
        for s in smiles_list:
            e, ok = _try_encode_smiles_rdkit(s, n_bits=int(spec.smiles_rdkit_bits), version=int(spec.smiles_rdkit_version))
            sm_rows.append(e)
            if ok:
                used_rdkit += 1
        blocks.append(np.stack(sm_rows, axis=0))
    else:
        h_rows = [encode_smiles_hash(s, dim=int(spec.smiles_hash_dim)) for s in smiles_list]
        blocks.append(np.stack(h_rows, axis=0))

    backend = "rdkit" if used_rdkit > 0 else "hash"

    # ── Block 2: GNN embeddings ────────────────────────────
    if bool(spec.use_gnn):
        from .gnn_featurizer import GNNFeaturizer
        cache_dir = str(spec.cache_dir)
        gnn = GNNFeaturizer(hidden_dim=int(spec.gnn_hidden_dim), cache_dir=cache_dir, online=bool(spec.online_mode))
        blocks.append(gnn.transform(smiles_list))

    # ── Block 3: ChemBERTa embeddings ───────────────────────
    if bool(spec.use_chemberta):
        from .chemberta_encoder import ChemBERTaEncoder
        cache_dir = str(spec.cache_dir)
        ce = ChemBERTaEncoder(model_name=str(spec.chemberta_model), cache_dir=cache_dir, online=bool(spec.online_mode))
        blocks.append(ce.encode(smiles_list))

    # ── Block 4: Epitope (ESM-2 or AA composition) ──────────
    if bool(spec.use_esm2):
        from .esm2_mamba_fusion import ESM2Encoder
        enc = ESM2Encoder(model_size=str(spec.esm2_model_size))
        enc.load()  # ESM2Encoder has its own lazy loading + online mode
        blocks.append(enc.encode(epi_list).astype(np.float32))
    else:
        epi_rows = [encode_epitope(e) for e in epi_list]
        blocks.append(np.stack(epi_rows, axis=0))

    # ── Block 5: Dose-response features ─────────────────────
    if bool(spec.use_dose_response) and all(c in df.columns for c in ["dose", "freq", "treatment_time"]):
        dose_arr = pd.to_numeric(df["dose"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        freq_arr = pd.to_numeric(df["freq"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
        tt_arr = pd.to_numeric(df["treatment_time"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        blocks.append(compute_dose_response_features(dose_arr, freq_arr, tt_arr))

    # ── Block 6: PK prior features ──────────────────────────
    if bool(spec.use_pk_prior):
        try:
            Chem = importlib.import_module("rdkit.Chem")
            mol_from_smiles = getattr(Chem, "MolFromSmiles", None)
            mols = [mol_from_smiles(s) if mol_from_smiles else None for s in smiles_list]
        except Exception:
            mols = [None] * n
        blocks.append(compute_pk_prior_features(mols))

    # ── Block 7: Environment columns ─────────────────────────
    if env_cols:
        blocks.append(df[env_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32))
    else:
        blocks.append(np.zeros((n, 0), dtype=np.float32))

    # ── Block 8: Cross-features ──────────────────────────────
    if bool(spec.use_cross_features) and "dose" in df.columns:
        dose_arr = pd.to_numeric(df["dose"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        freq_arr = pd.to_numeric(df["freq"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
        tt_arr = pd.to_numeric(df["treatment_time"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        bind_arr = pd.to_numeric(df["target_binding"], errors="coerce").fillna(0.5).to_numpy(dtype=np.float32) if "target_binding" in df.columns else None
        imm_arr = pd.to_numeric(df["immune_activation"], errors="coerce").fillna(0.5).to_numpy(dtype=np.float32) if "immune_activation" in df.columns else None
        blocks.append(compute_cross_features(dose_arr, freq_arr, tt_arr, bind_arr, imm_arr))

    # ── Block 9: Auxiliary labels (target_binding, immune_activation as features) ──
    if bool(spec.use_auxiliary_labels):
        aux_arr = np.zeros((n, 2), dtype=np.float32)
        if "target_binding" in df.columns:
            aux_arr[:, 0] = pd.to_numeric(df["target_binding"], errors="coerce").fillna(0.5)
        if "immune_activation" in df.columns:
            aux_arr[:, 1] = pd.to_numeric(df["immune_activation"], errors="coerce").fillna(0.5)
        blocks.append(aux_arr)

    # ── Concatenate all blocks ───────────────────────────────
    X = np.concatenate(blocks, axis=1).astype(np.float32)
    return X, env_cols, backend


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "smiles" not in out.columns:
        out["smiles"] = ""
    if "epitope_seq" not in out.columns:
        out["epitope_seq"] = ""
    for c in ["dose", "freq", "treatment_time"]:
        if c not in out.columns:
            out[c] = 0.0
    if "group_id" not in out.columns:
        out["group_id"] = "G0"
    return out


# ===================================================================
# circRNA feature engineering
# ===================================================================

@dataclass(frozen=True)
class CircRNAFeatureSpec:
    """Configuration for circRNA feature extraction."""
    include_sequence: bool = True
    include_structure: bool = True
    include_functional: bool = True
    include_modification: bool = True
    include_delivery: bool = True


def _circrna_feature_names(spec: CircRNAFeatureSpec | None = None) -> List[str]:
    """Return ordered list of circRNA feature column names."""
    spec = spec or CircRNAFeatureSpec()
    names: List[str] = []
    if spec.include_sequence:
        names.extend([
            "crna_seq_len", "crna_gc_content", "crna_frac_A", "crna_frac_U",
            "crna_frac_G", "crna_frac_C", "crna_palindrome_ratio",
            "crna_bsj_flank_gc",
        ])
    if spec.include_structure:
        names.extend([
            "crna_mfe_per_nt", "crna_stem_loop_score", "crna_struct_stability",
        ])
    if spec.include_functional:
        names.extend([
            "crna_ires_score", "crna_orf_length", "crna_kozak_score",
            "crna_utr_conservation",
        ])
    if spec.include_modification:
        names.extend([
            "crna_mod_stability_factor", "crna_mod_immune_evasion",
        ])
    if spec.include_delivery:
        names.extend([
            "crna_bioavailability", "crna_liver_target", "crna_spleen_target",
            "crna_muscle_target", "crna_endosomal_escape", "crna_half_life_base_h",
        ])
    return names


def _is_valid_rna(seq: str) -> bool:
    """Check if sequence contains only valid RNA nucleotides."""
    return bool(seq) and all(ch in "AUGCaugc" for ch in seq)


def encode_cirrna_sequence(seq: str) -> np.ndarray:
    """Extract nucleotide-level sequence features from a circRNA sequence.

    Features (8-dim):
      - seq_len, gc_content, frac_A/U/G/C, palindrome_ratio, bsj_flank_gc
    """
    s = _safe_str(seq).upper().replace(" ", "")
    arr = np.zeros(8, dtype=np.float32)
    if not s:
        return arr

    n = float(len(s))
    arr[0] = n
    gc = sum(1 for ch in s if ch in "GC") / max(n, 1.0)
    arr[1] = gc
    arr[2] = s.count("A") / max(n, 1.0)
    arr[3] = s.count("U") / max(n, 1.0)
    arr[4] = s.count("G") / max(n, 1.0)
    arr[5] = s.count("C") / max(n, 1.0)

    # Palindrome ratio: fraction of positions that participate in a palindrome of length >= 4
    palindromes = 0
    checked: set[int] = set()
    for i in range(len(s) - 3):
        window = s[i : i + 4]
        if window == window[::-1] and i not in checked:
            palindromes += 1
            checked.update(range(i, i + 4))
    arr[6] = palindromes / max(n, 1.0)

    # BSJ flanking GC: GC content of the first and last 20 nt (backsplice junction region)
    flank_len = min(20, len(s) // 2)
    flank = s[:flank_len] + s[-flank_len:] if flank_len > 0 else s
    arr[7] = sum(1 for ch in flank if ch in "GC") / max(len(flank), 1.0)

    return arr


def _estimate_mfe(seq: str) -> float:
    """Estimate minimum free energy using simplified nearest-neighbor model.

    Uses Turner 1999 RNA NN parameters for a rough approximation.
    This avoids the external RNAfold dependency while giving reasonable estimates.
    Returns MFE in kcal/mol (more negative = more stable).
    """
    s = _safe_str(seq).upper().replace(" ", "")
    if len(s) < 4 or not _is_valid_rna(s):
        return 0.0

    # Simplified: assume all nucleotides can pair with a complementary partner.
    # We scan for maximal complementary pairing to estimate structure stability.
    n = len(s)
    comp = {"A": "U", "U": "A", "G": "C", "C": "G"}

    # Greedy pairing: scan for complementary base pairs
    total_energy = 0.0
    paired = [False] * n
    pairs_found = 0

    # Simple Nussinov-like greedy approach for pairing estimation
    for i in range(n - 4):  # min loop size of 4
        if paired[i]:
            continue
        # Look for best complement within reasonable distance
        best_j = -1
        best_dist = n
        for j in range(i + 4, n):
            if paired[j]:
                continue
            if s[j] == comp.get(s[i], ""):
                if j - i < best_dist:
                    best_j = j
                    best_dist = j - i
        if best_j >= 0:
            paired[i] = True
            paired[best_j] = True
            pairs_found += 1
            # Nearest-neighbor energy for the pair
            nn_key = (s[i], s[best_j])
            # Use average of initiation and nearest-neighbor
            init_e = _INIT_PARAMS.get(s[i], 0.0)
            nn_e = _NN_PARAMS.get(nn_key, -1.0)
            total_energy += (init_e + nn_e) * 0.5
            # Loop penalty: ~1.75 kcal/mol per unpaired nt in the loop
            loop_size = best_j - i - 1
            total_energy += 1.75 * math.log(max(loop_size, 1) + 1) * 0.3

    # Normalize by sequence length to get per-nucleotide MFE
    mfe_per_nt = total_energy / max(n, 1.0)
    return mfe_per_nt


def encode_cirrna_structure(seq: str) -> np.ndarray:
    """Estimate RNA secondary structure features without external tools.

    Features (3-dim):
      - mfe_per_nt: minimum free energy per nucleotide (kcal/mol, more negative = more stable)
      - stem_loop_score: estimated fraction of nucleotides in stem-loop structures
      - struct_stability: composite stability score (0-1, higher = more stable)
    """
    s = _safe_str(seq).upper().replace(" ", "")
    arr = np.zeros(3, dtype=np.float32)
    if not s or len(s) < 10:
        return arr

    mfe_per_nt = _estimate_mfe(s)
    arr[0] = mfe_per_nt

    # Stem-loop score: estimate fraction of bases that are paired
    comp = {"A": "U", "U": "A", "G": "C", "C": "G"}
    n = len(s)
    paired_count = 0
    paired = [False] * n

    for i in range(n - 4):
        if paired[i]:
            continue
        for j in range(i + 4, n):
            if paired[j]:
                continue
            if s[j] == comp.get(s[i], ""):
                paired[i] = True
                paired[j] = True
                paired_count += 1
                break

    stem_loop_ratio = (2.0 * paired_count) / max(n, 1.0)
    arr[1] = stem_loop_ratio

    # Composite stability: combine MFE, GC content, and stem-loop ratio
    gc = sum(1 for ch in s if ch in "GC") / max(n, 1.0)
    # More negative MFE = more stable; map to 0-1
    mfe_score = np.clip(1.0 - abs(mfe_per_nt) / 5.0, 0.0, 1.0)
    arr[2] = np.clip(0.4 * mfe_score + 0.3 * stem_loop_ratio + 0.3 * gc, 0.0, 1.0)

    return arr


def encode_cirrna_functional(seq: str, ires_type: str = "") -> np.ndarray:
    """Extract functional element features from circRNA sequence.

    Features (4-dim):
      - ires_score: IRES strength score (0-1)
      - orf_length: length of longest ORF (normalized by seq length)
      - kozak_score: Kozak consensus match score (0-1)
      - utr_conservation: estimated UTR conservation score (0-1)
    """
    s = _safe_str(seq).upper().replace(" ", "")
    arr = np.zeros(4, dtype=np.float32)
    if not s or len(s) < 20:
        return arr

    # --- IRES score ---
    # Combine sequence motif matching with known IRES type
    motif_score = 0.0
    motif_count = 0
    for motif, strength in _IRES_MOTIFS:
        count = s.count(motif)
        if count > 0:
            motif_score += strength * min(count, 3)  # cap contribution per motif
            motif_count += 1
    seq_ires = motif_score / max(len(_IRES_MOTIFS), 1.0)

    # Known IRES type bonus
    ires_type_map: Dict[str, float] = {
        "EMCV": 0.80, "HCV": 0.75, "CVB3": 0.70,
        "c-MYC": 0.65, "VEGF": 0.60, "BiP": 0.55,
        "custom": seq_ires, "": seq_ires,
    }
    type_bonus = ires_type_map.get(ires_type.strip().upper(), seq_ires)
    arr[0] = np.clip(0.4 * seq_ires + 0.6 * type_bonus, 0.0, 1.0)

    # --- ORF length ---
    # Find longest ORF (between AUG start and stop codons)
    stop_codons = {"UAA", "UAG", "UGA"}
    max_orf = 0
    for i in range(len(s) - 2):
        if s[i : i + 3] == "AUG":
            for j in range(i + 3, len(s) - 2, 3):
                codon = s[j : j + 3]
                if codon in stop_codons:
                    orf_len = j + 3 - i
                    if orf_len > max_orf:
                        max_orf = orf_len
                    break
            else:
                orf_len = len(s) - i
                if orf_len > max_orf:
                    max_orf = orf_len
    arr[1] = max_orf / max(len(s), 1.0)

    # --- Kozak score ---
    # Check for Kozak consensus around each AUG
    best_kozak = 0.0
    for i in range(len(s) - 6):
        if s[i + 3 : i + 6] == _KOZAK_CORE:
            # Check -3 and +4 positions for optimal Kozak
            if i >= 3:
                pos_minus3 = s[i]  # -3 position
                pos_plus4 = s[i + 6] if i + 6 < len(s) else "N"
                score = 0.0
                # -3 position: G or A is good
                if pos_minus3 in "GA":
                    score += 0.35
                # +4 position: G is optimal
                if pos_plus4 == _KOZAK_OPTIMAL_3PRIME:
                    score += 0.35
                # -6 to -4: purine-rich (G/C preferred)
                if i >= 3:
                    upstream = s[max(0, i - 3) : i]
                    purine_frac = sum(1 for ch in upstream if ch in "GA") / max(len(upstream), 1.0)
                    score += 0.3 * purine_frac
                best_kozak = max(best_kozak, score)
    arr[2] = best_kozak

    # --- UTR conservation ---
    # Estimate based on sequence regularity and known UTR motifs
    utr_motifs = ["AUUUA", "UUUUA", "AATAAA", "UUAUUUAUU"]  # ARE / polyA signal
    utr_hits = sum(1 for m in utr_motifs if m in s)
    arr[3] = np.clip(utr_hits / max(len(utr_motifs), 1.0), 0.0, 1.0)

    return arr


def encode_cirrna_modification(
    modification: str = "none",
    predicted_mod_density: float = 0.0,
) -> np.ndarray:
    """Encode modification-related features.

    Features (2-dim):
      - mod_stability_factor: how much the modification extends RNA half-life
      - mod_immune_evasion: how well the modification evades innate immune sensing
    """
    mod = _safe_str(modification).lower().strip()
    arr = np.zeros(2, dtype=np.float32)

    base_stability = _MOD_STABILITY_FACTOR.get(mod, _MOD_STABILITY_FACTOR["none"])
    base_evasion = _MOD_IMMUNE_EVASION.get(mod, _MOD_IMMUNE_EVASION["none"])

    # Scale by predicted modification density (fraction of nt modified, 0-1)
    density = float(np.clip(predicted_mod_density, 0.0, 1.0))
    if density <= 0.0:
        density = 0.3  # default assumed density for the modification type

    # Stability factor scales with density: factor = 1 + density * (base_factor - 1)
    arr[0] = 1.0 + density * (base_stability - 1.0)
    # Evasion scales linearly with density
    arr[1] = base_evasion * density

    return arr


def encode_cirrna_delivery(delivery_vector: str = "LNP_standard", route: str = "IV") -> np.ndarray:
    """Encode delivery system and route of administration features.

    Features (6-dim):
      - bioavailability, liver_target, spleen_target, muscle_target,
        endosomal_escape, half_life_base_h
    """
    vec = _safe_str(delivery_vector).strip()
    arr = np.zeros(6, dtype=np.float32)

    params = _DELIVERY_PARAMS.get(vec)
    if params is not None:
        arr[0] = params["bioavailability"]
        arr[1] = params["liver_target"]
        arr[2] = params["spleen_target"]
        arr[3] = params["muscle_target"]
        arr[4] = params["endosomal_escape"]
        arr[5] = params["half_life_base_h"]

    # Route adjustment factors
    route = _safe_str(route).upper().strip()
    route_factor: Dict[str, float] = {
        "IV": 1.0,    # intravenous: baseline
        "SC": 0.6,    # subcutaneous: lower bioavailability
        "IM": 0.7,    # intramuscular: moderate bioavailability
        "ID": 0.3,    # intradermal: low bioavailability but high local immune activation
    }
    rf = route_factor.get(route, 1.0)
    arr[0] *= rf  # bioavailability scales with route

    return arr


def build_cirrna_feature_vector(
    seq: str = "",
    modification: str = "none",
    delivery_vector: str = "LNP_standard",
    route: str = "IV",
    ires_type: str = "",
    spec: CircRNAFeatureSpec | None = None,
) -> np.ndarray:
    """Build complete circRNA feature vector.

    Returns a fixed-dimension vector regardless of which sub-features are available.
    """
    spec = spec or CircRNAFeatureSpec()
    parts: List[np.ndarray] = []

    if spec.include_sequence:
        parts.append(encode_cirrna_sequence(seq))
    if spec.include_structure:
        parts.append(encode_cirrna_structure(seq))
    if spec.include_functional:
        parts.append(encode_cirrna_functional(seq, ires_type=ires_type))
    if spec.include_modification:
        parts.append(encode_cirrna_modification(modification))
    if spec.include_delivery:
        parts.append(encode_cirrna_delivery(delivery_vector, route))

    if not parts:
        return np.array([], dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32)


def build_cirrna_feature_matrix(
    df: pd.DataFrame, spec: CircRNAFeatureSpec | None = None
) -> Tuple[np.ndarray, List[str]]:
    """Build circRNA feature matrix from a DataFrame.

    Reads columns: circrna_seq, modification, delivery_vector, route, ires_type.
    Returns (feature_matrix, feature_names).
    """
    spec = spec or CircRNAFeatureSpec()
    names = _circrna_feature_names(spec)

    xs: List[np.ndarray] = []
    for _, row in df.iterrows():
        x = build_cirrna_feature_vector(
            seq=str(row.get("circrna_seq", "")),
            modification=str(row.get("modification", "none")),
            delivery_vector=str(row.get("delivery_vector", "LNP_standard")),
            route=str(row.get("route", "IV")),
            ires_type=str(row.get("ires_type", "")),
            spec=spec,
        )
        xs.append(x)

    if not xs:
        return np.zeros((0, len(names)), dtype=np.float32), names
    return np.stack(xs, axis=0), names


def ensure_cirrna_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all circRNA-related columns exist in the DataFrame."""
    out = df.copy()
    if "circrna_seq" not in out.columns:
        out["circrna_seq"] = ""
    if "modification" not in out.columns:
        out["modification"] = "none"
    if "delivery_vector" not in out.columns:
        out["delivery_vector"] = "LNP_standard"
    if "route" not in out.columns:
        out["route"] = "IV"
    if "ires_type" not in out.columns:
        out["ires_type"] = ""
    return out


def build_combined_feature_matrix(
    df: pd.DataFrame,
    sm_spec: MixedFeatureSpec | None = None,
    cr_spec: CircRNAFeatureSpec | None = None,
) -> Tuple[np.ndarray, List[str], str]:
    """Build combined small-molecule + circRNA feature matrix.

    If circRNA columns are present and non-empty, appends circRNA features.
    Returns (feature_matrix, all_feature_names, smiles_backend).
    """
    sm_spec = sm_spec or MixedFeatureSpec()
    cr_spec = cr_spec or CircRNAFeatureSpec()

    sm_matrix, env_cols, backend = build_feature_matrix(df, spec=sm_spec)
    sm_names = build_feature_names(spec=sm_spec, env_cols=env_cols)

    # Check if circRNA features should be appended
    df_enhanced = ensure_cirrna_columns(df)
    has_cirrna = bool(
        (df_enhanced["circrna_seq"].notna() & (df_enhanced["circrna_seq"].str.len() > 0)).any()
    )

    if has_cirrna:
        cr_matrix, cr_names = build_cirrna_feature_matrix(df_enhanced, spec=cr_spec)
        combined = np.concatenate([sm_matrix, cr_matrix], axis=1)
        return combined, [*sm_names, *cr_names], backend

    return sm_matrix, sm_names, backend
