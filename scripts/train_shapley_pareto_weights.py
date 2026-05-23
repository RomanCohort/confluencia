"""
train_shapley_pareto_weights.py — Calibrate scoring weights via Shapley value
allocation + multi-objective Pareto optimization (NSGA-II).

Three-phase pipeline:
  Phase 1  Data preparation — simulate 5-modality inputs from survival + drug data
  Phase 2  Shapley value analysis — permutation Shapley for fusion modality weights
  Phase 3  NSGA-II Pareto optimization — calibrate ~40 sub-score weights
  Phase 4  Knee-point selection from Pareto front
  Phase 5  Visualization

Output: output/calibrated_weights.json (compatible with scoring_weights.json)

Usage (AutoDL):
    pip install pymoo lifelines scikit-learn matplotlib
    python scripts/train_shapley_pareto_weights.py \
        --survival-data data/gene_signature/cache/combined_raw_with_survival.csv \
        --drug-data confluencia-2.0-drug/data/breast_cancer_drug_dataset.csv \
        --output-dir output \
        --pop-size 200 \
        --n-gen 500 \
        --cv-folds 5 \
        --seed 42
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODALITIES = ["clinical", "binding", "kinetics", "gene_signature", "circ_rna"]

GENE_COLS_5 = ["TROP2", "NECTIN4", "LIV-1", "B7-H4", "TMEM65"]
CLINICAL_COLS = ["grade", "ER_positive", "HER2_positive", "PR_positive", "tumor_stage"]

# Default weight structure — must match scoring_weights.json keys
DEFAULT_WEIGHTS: Dict[str, Any] = {
    "version": 2,
    "source": "Default hardcoded weights",
    "fusion": {
        "clinical": 0.30, "binding": 0.20, "kinetics": 0.15,
        "gene_signature": 0.15, "circ_rna": 0.20,
    },
    "clinical_sub": {
        "efficacy": 0.40, "binding": 0.35, "immune": 0.25,
    },
    "clinical_safety": {
        "toxicity": 0.6, "inflammation": 0.4,
    },
    "binding_uncertainty_penalty": 0.3,
    "kinetics_sub": {
        "half_life": 0.25, "auc": 0.30, "therapeutic_index": 0.30, "cmax": 0.15,
    },
    "gene_signature_sub": {
        "efficacy": 0.30, "immune": 0.15, "proliferation": 0.15,
        "mito": 0.15, "risk_inverse": 0.15, "tide_inverse": 0.10,
    },
    "circ_rna_sub": {
        "immunotherapy": 0.20, "tumor_killing_index": 0.15,
        "immunogenicity": 0.15, "immune_cycle": 0.10,
        "tme": 0.10, "therapeutic_window": 0.10,
        "tide_inverse": 0.10, "ips_fraction": 0.10,
    },
    "tide_ips": {
        "tide_risk": 0.3, "tide_tmem65": 0.4, "tide_immune": -0.3,
        "ips_risk_inverse": 0.5, "ips_immune": 0.3,
    },
    "risk_adjustment": {
        "TROP2_high": 0.30, "TROP2_low": 0.15,
        "NECTIN4_high": 0.20, "NECTIN4_low": 0.10,
        "LIV-1_high": 0.15, "LIV-1_low": 0.08,
        "B7-H4_high": 0.10, "B7-H4_low": 0.05,
        "TMEM65_high": 0.25, "TMEM65_low": 0.13,
        "ddr_base": 0.2, "ddr_risk_weight": 0.4,
    },
    "clinical_uncertainty": {"inflammation": 0.2, "toxicity": 0.2},
    "binding_uncertainty": {"default": 0.3},
    "kinetics_uncertainty": {
        "hl_low": 0.5, "hl_high": 72.0, "implausible_penalty": 0.3,
        "cmax_high": 1000.0, "extreme_cmax_penalty": 0.2,
    },
    "gene_signature_uncertainty": {
        "extreme_high": 0.8, "extreme_low": 0.2, "extreme_penalty": 0.15,
    },
    "circ_rna_uncertainty": {
        "conflict_high": 0.6, "conflict_penalty": 0.2,
    },
    "go_threshold": 0.65,
    "conditional_threshold": 0.40,
    "safety_floor": 0.30,
}


# ---------------------------------------------------------------------------
# Phase 1 — Data Preparation
# ---------------------------------------------------------------------------

def load_survival_data(path: str) -> pd.DataFrame:
    """Load combined_raw_with_survival.csv."""
    p = Path(path)
    if not p.exists():
        print(f"ERROR: {path} not found"); sys.exit(1)
    df = pd.read_csv(p)
    if df["OS_status"].dtype == object:
        smap = {"Deceased": 1, "Living": 0, "DECEASED": 1, "LIVING": 0}
        df["OS_status"] = df["OS_status"].map(smap).fillna(df["OS_status"]).astype(int)
    print(f"[Phase1] Loaded survival data: {len(df)} samples, {len(df.columns)} cols")
    return df


def load_drug_data(path: str) -> pd.DataFrame:
    """Load breast_cancer_drug_dataset.csv."""
    p = Path(path)
    if not p.exists():
        print(f"WARNING: {path} not found, drug correlations will use defaults")
        return pd.DataFrame()
    df = pd.read_csv(p)
    print(f"[Phase1] Loaded drug data: {len(df)} rows")
    return df


def normalize_gene_expr(surv_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize log2 gene expression to 0-1 range (per gene quantile)."""
    df = surv_df.copy()
    for g in GENE_COLS_5:
        if g in df.columns:
            lo, hi = df[g].quantile(0.01), df[g].quantile(0.99)
            if hi > lo:
                df[f"{g}_norm"] = ((df[g] - lo) / (hi - lo)).clip(0, 1)
            else:
                df[f"{g}_norm"] = 0.5
        else:
            df[f"{g}_norm"] = 0.5
            print(f"  WARNING: gene {g} not found in data, using 0.5")
    return df


def estimate_drug_correlations(drug_df: pd.DataFrame) -> Dict[str, float]:
    """Estimate correlations between gene expression and drug metrics."""
    corrs = {
        "trop2_efficacy": 0.55,
        "trop2_binding": 0.50,
        "nectin4_efficacy": 0.40,
        "b7h4_immune": 0.45,
        "tmem65_efficacy": 0.35,
    }
    if drug_df.empty:
        return corrs
    try:
        if "efficacy" in drug_df.columns and "target_binding" in drug_df.columns:
            eb_corr = drug_df["efficacy"].corr(drug_df["target_binding"])
            if not np.isnan(eb_corr):
                corrs["trop2_binding"] = max(0.2, min(0.9, abs(eb_corr)))
    except Exception:
        pass
    return corrs


def simulate_modality_inputs(
    surv_df: pd.DataFrame,
    drug_corrs: Dict[str, float],
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """
    Simulate 5-modality scoring inputs for each sample.

    Strategy: use biological relationships, not random noise.
    - gene_signature: compute directly from gene expression via Cox-trained model
    - clinical: derive from grade/ER/HER2 + gene-efficacy correlation
    - binding: derive from target expression + MHC affinity correlation
    - kinetics: derive from efficacy -> PK parameter mapping
    - circ_rna: derive from immune scores + immunogenicity correlation
    """
    from confluencia_shared.gene_signature_enhanced import (
        compute_five_gene_signature_scores,
        predict_immunotherapy_response,
    )

    results = []
    for idx, (_, row) in enumerate(surv_df.iterrows()):
        t = row.get("TROP2_norm", 0.5)
        n = row.get("NECTIN4_norm", 0.5)
        l = row.get("LIV-1_norm", 0.5)
        b = row.get("B7-H4_norm", 0.5)
        m = row.get("TMEM65_norm", 0.5)

        # --- Gene signature scores (data-driven, Cox-trained) ---
        gs_scores = compute_five_gene_signature_scores(t, n, l, b, m, mode="yang2025")

        # Immunotherapy response -> TIDE/IPS
        imm = predict_immunotherapy_response(
            gs_scores["risk_score"], gs_scores["immune_score"], m,
        )

        gs_input = {
            "trop2": t, "nectin4": n, "liv1": l, "b7h4": b, "tmem65": m,
            "risk_score": gs_scores["risk_score"],
            "efficacy_score": gs_scores["efficacy_score"],
            "proliferation_score": gs_scores["proliferation_score"],
            "immune_score": gs_scores["immune_score"],
            "mito_score": gs_scores["mito_score"],
            "tide_score": imm["tide_score"],
            "ips_estimate": imm["ips_estimate"],
            "predicted_response": imm["predicted_response"],
            "dhe_recommended": m > 0.5 and t > 0.4,
        }

        # --- Clinical: efficacy from gene signature + clinical features ---
        grade = row.get("grade", 2.0)
        er = row.get("ER_positive", 0.5)
        her2 = row.get("HER2_positive", 0.5)
        grade_factor = 1.0 - (grade - 1) / 3.0 * 0.3
        er_factor = 0.7 + 0.3 * er
        her2_factor = 0.7 + 0.3 * her2
        base_efficacy = gs_scores["efficacy_score"]
        drug_efficacy = np.clip(
            base_efficacy * grade_factor * er_factor * her2_factor
            + rng.normal(0, 0.05), 0, 1,
        )

        drug_input = {
            "efficacy_pred": drug_efficacy,
            "target_binding_pred": np.clip(
                t * drug_corrs["trop2_binding"]
                + n * drug_corrs.get("nectin4_efficacy", 0.4) * 0.5
                + rng.normal(0, 0.05), 0, 1),
            "immune_activation_pred": np.clip(
                b * drug_corrs["b7h4_immune"]
                + (1 - gs_scores["risk_score"]) * 0.3
                + rng.normal(0, 0.05), 0, 1),
            "inflammation_risk_pred": np.clip(
                (1 - b) * 0.3 + gs_scores["risk_score"] * 0.3
                + rng.normal(0, 0.03), 0, 1),
            "genotoxicity_risk_pred": np.clip(
                gs_scores["risk_score"] * 0.2
                + rng.normal(0, 0.03), 0, 1),
        }

        # --- Binding: MHC affinity correlated with target expression ---
        epi_input = {
            "efficacy_pred": np.clip(
                t * 0.6 + m * 0.3 + rng.normal(0, 0.08), 0, 1),
            "pred_uncertainty": np.clip(
                0.1 + 0.3 * (1 - abs(t - 0.5))
                + rng.normal(0, 0.02), 0, 1),
        }

        # --- Kinetics: PK parameters derived from efficacy + biology ---
        pk_input = {
            "pkpd_cmax_mg_per_l": np.clip(
                2.0 + drug_efficacy * 6.0 + rng.normal(0, 1.0), 0.1, 50),
            "pkpd_tmax_h": np.clip(
                4.0 + rng.normal(0, 1.0), 0.5, 24),
            "pkpd_half_life_h": np.clip(
                8.0 + drug_efficacy * 10.0 + rng.normal(0, 3.0), 0.5, 72),
            "pkpd_auc_conc": np.clip(
                50 + drug_efficacy * 100 + rng.normal(0, 15), 10, 500),
            "pkpd_auc_effect": np.clip(
                30 + drug_efficacy * 80 + rng.normal(0, 10), 5, 200),
        }

        # --- circRNA: derive from immune/gene scores ---
        cr_input = {
            "immunotherapy_score": np.clip(
                imm["ips_estimate"] * 0.6 + (1 - imm["tide_score"]) * 0.4
                + rng.normal(0, 0.05), 0, 1),
            "therapeutic_window": np.clip(
                0.3 + drug_efficacy * 0.4 + rng.normal(0, 0.05), 0, 1),
            "tumor_killing_index": np.clip(
                gs_scores["proliferation_score"] * 0.5 + drug_efficacy * 0.3
                + rng.normal(0, 0.05), 0, 1),
            "overall_immunogenicity": np.clip(
                b * 0.3 + (1 - gs_scores["risk_score"]) * 0.4
                + rng.normal(0, 0.05), 0, 1),
            "rig_i_score": np.clip(
                0.3 + m * 0.3 + rng.normal(0, 0.05), 0, 1),
            "tlr_score": np.clip(
                0.2 + m * 0.4 + rng.normal(0, 0.05), 0, 1),
            "pkr_score": np.clip(
                0.2 + m * 0.3 + rng.normal(0, 0.05), 0, 1),
            "tide_score": imm["tide_score"],
            "ips": imm["ips_estimate"] * 10.0,
            "predicted_response": (
                "likely_responder" if imm["predicted_response"] == "CR/PR"
                else "likely_non_responder" if imm["predicted_response"] == "PD"
                else "intermediate"
            ),
            "immune_cycle_score": np.clip(
                0.3 + gs_scores["immune_score"] * 0.5
                + rng.normal(0, 0.05), 0, 1),
            "tme_score": np.clip(
                0.3 + (1 - gs_scores["risk_score"]) * 0.4
                + rng.normal(0, 0.05), 0, 1),
            "trained_model_risk": gs_scores["risk_score"],
        }

        results.append({
            "drug_input": drug_input,
            "epi_input": epi_input,
            "pk_input": pk_input,
            "gs_input": gs_input,
            "cr_input": cr_input,
            "OS_months": float(row.get("OS_months", np.nan)),
            "OS_status": int(row.get("OS_status", 0)),
        })

    return results


# ---------------------------------------------------------------------------
# Phase 2 — Shapley Value Analysis
# ---------------------------------------------------------------------------

def _build_default_inputs() -> Dict[str, Dict[str, float]]:
    """Build default modality inputs (population mean fill for missing modalities)."""
    return {
        "drug": {
            "efficacy_pred": 0.50, "target_binding_pred": 0.50,
            "immune_activation_pred": 0.50,
            "inflammation_risk_pred": 0.15, "genotoxicity_risk_pred": 0.10,
        },
        "epi": {"efficacy_pred": 0.50, "pred_uncertainty": 0.30},
        "pk": {
            "pkpd_cmax_mg_per_l": 5.0, "pkpd_tmax_h": 4.0,
            "pkpd_half_life_h": 12.0,
            "pkpd_auc_conc": 100.0, "pkpd_auc_effect": 60.0,
        },
    }


def _make_default_gs() -> Dict[str, Any]:
    """Default gene signature input."""
    return {
        "trop2": 0.5, "nectin4": 0.5, "liv1": 0.5, "b7h4": 0.5, "tmem65": 0.5,
        "risk_score": 0.5, "efficacy_score": 0.5,
        "proliferation_score": 0.5, "immune_score": 0.5, "mito_score": 0.5,
        "tide_score": 0.5, "ips_estimate": 0.5,
        "predicted_response": "SD", "dhe_recommended": False,
    }


def _make_default_cr() -> Dict[str, Any]:
    """Default circRNA input."""
    return {
        "immunotherapy_score": 0.5, "therapeutic_window": 0.5,
        "tumor_killing_index": 0.5, "overall_immunogenicity": 0.5,
        "rig_i_score": 0.3, "tlr_score": 0.3, "pkr_score": 0.3,
        "tide_score": 0.5, "ips": 5.0,
        "predicted_response": "intermediate",
        "immune_cycle_score": 0.5, "tme_score": 0.5,
        "trained_model_risk": 0.5,
    }


def _compute_composites(
    sample_inputs: List[Dict[str, Any]],
    active_modalities: List[str],
    engine: Any = None,
) -> np.ndarray:
    """Compute composite scores using only active modalities (fill others with defaults)."""
    from confluencia_joint.scoring import JointScoringEngine

    defaults = _build_default_inputs()
    default_gs = _make_default_gs()
    default_cr = _make_default_cr()
    if engine is None:
        engine = JointScoringEngine()

    scores = []
    for inp in sample_inputs:
        drug = inp["drug_input"] if "clinical" in active_modalities else defaults["drug"]
        epi = inp["epi_input"] if "binding" in active_modalities else defaults["epi"]
        pk = inp["pk_input"] if "kinetics" in active_modalities else defaults["pk"]
        gs = inp["gs_input"] if "gene_signature" in active_modalities else default_gs
        cr = inp["cr_input"] if "circ_rna" in active_modalities else default_cr

        result = engine.score(drug, epi, pk, gs, cr)
        scores.append(result.composite)

    return np.array(scores)


def permutation_shapley(
    sample_inputs: List[Dict[str, Any]],
    durations: np.ndarray,
    events: np.ndarray,
    rng: np.random.Generator,
    n_perms: int = 100,
) -> Dict[str, float]:
    """
    Compute permutation Shapley values for 5 fusion modalities.

    For each permutation of modalities, compute the marginal C-index gain
    when adding a modality to the coalition before it.

    Returns: {modality: Shapley_value} normalized to sum=1.
    """
    from lifelines.utils import concordance_index
    from confluencia_joint.scoring import JointScoringEngine

    n_mod = len(MODALITIES)
    shapley = {m: 0.0 for m in MODALITIES}

    # Create one engine and reuse it across all permutations
    engine = JointScoringEngine()

    print(f"[Phase2] Computing permutation Shapley ({n_perms} permutations)...")
    for perm_i in range(n_perms):
        if (perm_i + 1) % 20 == 0:
            print(f"  Permutation {perm_i + 1}/{n_perms}")

        perm = list(rng.permutation(MODALITIES))
        coalition: List[str] = []
        prev_c = 0.5  # baseline

        for mod in perm:
            coalition.append(mod)
            composites = _compute_composites(sample_inputs, coalition, engine)
            cur_c = concordance_index(durations, -composites, events)

            marginal = cur_c - prev_c
            shapley[mod] += marginal
            prev_c = cur_c

    # Average over permutations
    shapley = {m: v / n_perms for m, v in shapley.items()}

    # Shift to non-negative
    min_val = min(shapley.values())
    if min_val < 0:
        shapley = {m: v - min_val for m, v in shapley.items()}

    # Normalize to sum=1
    total = sum(shapley.values())
    if total > 0:
        shapley = {m: v / total for m, v in shapley.items()}
    else:
        shapley = {m: 1.0 / n_mod for m in MODALITIES}

    return shapley


def _fast_composite_vectorized(
    sample_inputs: List[Dict[str, Any]],
    w: Dict[str, Any],
) -> np.ndarray:
    """
    Fast vectorized composite score computation for NSGA-II.

    Instead of calling JointScoringEngine.score() per sample (which creates
    dataclasses, generates interpretation strings, etc.), we compute the
    composite directly with numpy array operations.

    This is ~50-100x faster than calling score() in a loop.
    """
    n = len(sample_inputs)

    # Extract arrays
    eff = np.array([s["drug_input"]["efficacy_pred"] for s in sample_inputs])
    tb  = np.array([s["drug_input"]["target_binding_pred"] for s in sample_inputs])
    imm = np.array([s["drug_input"]["immune_activation_pred"] for s in sample_inputs])
    infl = np.array([s["drug_input"].get("inflammation_risk_pred", 0) for s in sample_inputs])
    tox  = np.array([s["drug_input"].get("genotoxicity_risk_pred", 0) for s in sample_inputs])

    epi_eff = np.array([s["epi_input"]["efficacy_pred"] for s in sample_inputs])
    epi_unc = np.array([s["epi_input"]["pred_uncertainty"] for s in sample_inputs])

    hl = np.array([s["pk_input"]["pkpd_half_life_h"] for s in sample_inputs])
    auc_c = np.array([s["pk_input"]["pkpd_auc_conc"] for s in sample_inputs])
    auc_e = np.array([s["pk_input"]["pkpd_auc_effect"] for s in sample_inputs])
    cmax = np.array([s["pk_input"]["pkpd_cmax_mg_per_l"] for s in sample_inputs])

    gs_risk = np.array([s["gs_input"]["risk_score"] for s in sample_inputs])
    gs_eff  = np.array([s["gs_input"]["efficacy_score"] for s in sample_inputs])
    gs_prol = np.array([s["gs_input"]["proliferation_score"] for s in sample_inputs])
    gs_imm  = np.array([s["gs_input"]["immune_score"] for s in sample_inputs])
    gs_mito = np.array([s["gs_input"]["mito_score"] for s in sample_inputs])
    gs_tide = np.array([s["gs_input"]["tide_score"] for s in sample_inputs])
    gs_ips  = np.array([s["gs_input"]["ips_estimate"] for s in sample_inputs])

    cr_imm_score = np.array([s["cr_input"]["immunotherapy_score"] for s in sample_inputs])
    cr_tki = np.array([s["cr_input"]["tumor_killing_index"] for s in sample_inputs])
    cr_immu = np.array([s["cr_input"]["overall_immunogenicity"] for s in sample_inputs])
    cr_cycle = np.array([s["cr_input"]["immune_cycle_score"] for s in sample_inputs])
    cr_tme = np.array([s["cr_input"]["tme_score"] for s in sample_inputs])
    cr_tw = np.array([s["cr_input"]["therapeutic_window"] for s in sample_inputs])
    cr_tide = np.array([s["cr_input"]["tide_score"] for s in sample_inputs])
    cr_ips = np.array([s["cr_input"]["ips"] for s in sample_inputs]) / 10.0

    # --- Clinical score ---
    csw = w["clinical_sub"]
    clinical = (csw["efficacy"] * eff + csw["binding"] * tb + csw["immune"] * imm)
    clinical = np.clip(clinical, 0, 1)

    # Safety penalty
    cssw = w["clinical_safety"]
    safety_pen = np.clip(cssw["toxicity"] * tox + cssw["inflammation"] * infl, 0, 1)

    # --- Binding score ---
    bup = w["binding_uncertainty_penalty"]
    binding = np.clip(epi_eff * (1 - bup * epi_unc), 0, 1)

    # --- Kinetics score ---
    def _score_hl(h):
        s = np.where(np.isnan(h) | (h <= 0), 0.0,
              np.where((h >= 4) & (h <= 24), 1.0,
              np.where(h < 4, h / 4.0,
              np.exp(-0.1 * (h - 24)))))
        return s

    def _score_auc(a):
        s = np.where(np.isnan(a) | (a <= 0), 0.0,
              np.where((a >= 10) & (a <= 200), 1.0,
              np.where(a < 10, a / 10.0,
              np.exp(-0.005 * (a - 200)))))
        return s

    def _score_ti(auc_c_arr, auc_e_arr):
        ti = np.where(auc_c_arr > 0, auc_e_arr / auc_c_arr, 0.0)
        return np.clip(ti / (1.0 + ti), 0, 1)

    def _score_cmax(c):
        s = np.where(np.isnan(c) | (c <= 0), 0.0,
              np.where((c >= 0.5) & (c <= 10), 1.0,
              np.where(c < 0.5, c / 0.5,
              np.exp(-0.2 * (c - 10)))))
        return s

    ksw = w["kinetics_sub"]
    kinetics = (ksw["half_life"] * _score_hl(hl) +
                ksw["auc"] * _score_auc(auc_c) +
                ksw["therapeutic_index"] * _score_ti(auc_c, auc_e) +
                ksw["cmax"] * _score_cmax(cmax))
    kinetics = np.clip(kinetics, 0, 1)

    # --- Gene signature score ---
    gsw = w["gene_signature_sub"]
    gene_sig = (gsw["efficacy"] * gs_eff +
                gsw["immune"] * gs_imm +
                gsw["proliferation"] * gs_prol +
                gsw["mito"] * gs_mito +
                gsw["risk_inverse"] * (1.0 - gs_risk) +
                gsw["tide_inverse"] * (1.0 - gs_tide))
    gene_sig = np.clip(gene_sig, 0, 1)

    # --- circRNA score ---
    crsw = w["circ_rna_sub"]
    circ_rna = (crsw["immunotherapy"] * cr_imm_score +
                crsw["tumor_killing_index"] * cr_tki +
                crsw["immunogenicity"] * cr_immu +
                crsw["immune_cycle"] * cr_cycle +
                crsw["tme"] * cr_tme +
                crsw["therapeutic_window"] * cr_tw +
                crsw["tide_inverse"] * (1.0 - cr_tide) +
                crsw["ips_fraction"] * cr_ips)
    circ_rna = np.clip(circ_rna, 0, 1)

    # --- Uncertainty-adaptive dynamic weights ---
    cu = w["clinical_uncertainty"]
    pk_unc = w["kinetics_uncertainty"]
    gs_unc = w["gene_signature_uncertainty"]
    cr_unc = w["circ_rna_uncertainty"]
    bup_val = w["binding_uncertainty_penalty"]

    unc_clinical = np.clip(1.0 - (3 - 3 * np.ones(n)) / 3 + cu["inflammation"] * infl + cu["toxicity"] * tox, 0, 1)
    unc_binding = np.full(n, bup_val) * epi_unc  # simplified
    unc_kinetics = np.clip(
        1.0 - 3 / 3 +  # all PK params available by construction
        np.where((hl < pk_unc["hl_low"]) | (hl > pk_unc["hl_high"]), pk_unc["implausible_penalty"], 0) +
        np.where((cmax > pk_unc["cmax_high"]) | (cmax <= 0), pk_unc["extreme_cmax_penalty"], 0),
        0, 1)
    unc_gene_sig = np.clip(gs_unc["extreme_penalty"] * (
        (gs_risk > gs_unc["extreme_high"]) | (gs_risk < gs_unc["extreme_low"]) |
        (gs_tide > gs_unc["extreme_high"]) | (gs_tide < gs_unc["extreme_low"])).astype(float), 0, 1)
    unc_circ = np.clip(cr_unc["conflict_penalty"] * (
        (cr_tide > cr_unc["conflict_high"]) & (cr_ips > cr_unc["conflict_high"])).astype(float), 0, 1)

    ps = w.get("penalty_slope", 2.0)
    base = {
        "clinical": w["fusion"].get("clinical", 0.30),
        "binding": w["fusion"].get("binding", 0.20),
        "kinetics": w["fusion"].get("kinetics", 0.15),
        "gene_signature": w["fusion"].get("gene_signature", 0.15),
        "circ_rna": w["fusion"].get("circ_rna", 0.20),
    }

    # Vectorized adaptive weight computation
    cred = {
        "clinical": 1.0 - unc_clinical,
        "binding": 1.0 - unc_binding,
        "kinetics": 1.0 - unc_kinetics,
        "gene_signature": 1.0 - unc_gene_sig,
        "circ_rna": 1.0 - unc_circ,
    }
    adj = {}
    for dim in base:
        adj[dim] = base[dim] * (cred[dim] ** ps)
    adj_total = sum(adj[dim] for dim in adj)
    if isinstance(adj_total, np.ndarray):
        adj_total_safe = np.where(adj_total <= 0, 1.0, adj_total)
    elif adj_total <= 0:
        adj_total_safe = 1.0
    else:
        adj_total_safe = adj_total
    eff_w = {dim: adj[dim] / adj_total_safe for dim in adj}

    # Weighted composite
    composite = (eff_w["clinical"] * clinical +
                 eff_w["binding"] * binding +
                 eff_w["kinetics"] * kinetics +
                 eff_w["gene_signature"] * gene_sig +
                 eff_w["circ_rna"] * circ_rna)

    # Safety override: force low composite for unsafe drugs
    sf = w.get("safety_floor", 0.30)
    unsafe_mask = safety_pen > sf
    composite = np.where(unsafe_mask, np.minimum(composite, 0.3), composite)

    return np.clip(composite, 0, 1)


# ---------------------------------------------------------------------------
# Phase 3 — NSGA-II Multi-objective Pareto Optimization
# ---------------------------------------------------------------------------

# Variable vector layout (50 dimensions total):
#
#  [0:3]   clinical_sub          3 values -> softmax -> sum=1
#  [3:5]   clinical_safety       2 values -> softmax -> sum=1
#  [5:9]   kinetics_sub          4 values -> softmax -> sum=1
#  [9:15]  gene_signature_sub    6 values -> softmax -> sum=1
#  [15:23] circ_rna_sub          8 values -> softmax -> sum=1
#  [23:28] tide_ips              5 values, unconstrained (can be negative)
#  [28:40] risk_adjustment       12 values, non-negative (abs)
#  [40:49] uncertainty params    9 values, non-negative (abs)
#  [49]    penalty_slope         1 value, positive (max(0.1, x))
N_VAR = 50


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - x.max())
    return e / e.sum()


def _decode_variable_vector(x: np.ndarray) -> Dict[str, Any]:
    """Decode flat variable vector (50 dims) into weight groups."""

    w: Dict[str, Any] = {}

    # clinical_sub (3 values -> softmax)
    cs = _softmax(x[0:3])
    w["clinical_sub"] = {
        "efficacy": float(cs[0]),
        "binding": float(cs[1]),
        "immune": float(cs[2]),
    }

    # clinical_safety (2 values -> softmax)
    css = _softmax(x[3:5])
    w["clinical_safety"] = {
        "toxicity": float(css[0]),
        "inflammation": float(css[1]),
    }

    # kinetics_sub (4 values -> softmax)
    ks = _softmax(x[5:9])
    w["kinetics_sub"] = {
        "half_life": float(ks[0]),
        "auc": float(ks[1]),
        "therapeutic_index": float(ks[2]),
        "cmax": float(ks[3]),
    }

    # gene_signature_sub (6 values -> softmax)
    gs = _softmax(x[9:15])
    w["gene_signature_sub"] = {
        "efficacy": float(gs[0]),
        "immune": float(gs[1]),
        "proliferation": float(gs[2]),
        "mito": float(gs[3]),
        "risk_inverse": float(gs[4]),
        "tide_inverse": float(gs[5]),
    }

    # circ_rna_sub (8 values -> softmax)
    cr = _softmax(x[15:23])
    w["circ_rna_sub"] = {
        "immunotherapy": float(cr[0]),
        "tumor_killing_index": float(cr[1]),
        "immunogenicity": float(cr[2]),
        "immune_cycle": float(cr[3]),
        "tme": float(cr[4]),
        "therapeutic_window": float(cr[5]),
        "tide_inverse": float(cr[6]),
        "ips_fraction": float(cr[7]),
    }

    # tide_ips (5 values, unconstrained)
    w["tide_ips"] = {
        "tide_risk": float(x[23]),
        "tide_tmem65": float(x[24]),
        "tide_immune": float(x[25]),
        "ips_risk_inverse": float(x[26]),
        "ips_immune": float(x[27]),
    }

    # risk_adjustment (12 values, non-negative)
    ra_keys = [
        "TROP2_high", "TROP2_low", "NECTIN4_high", "NECTIN4_low",
        "LIV-1_high", "LIV-1_low", "B7-H4_high", "B7-H4_low",
        "TMEM65_high", "TMEM65_low", "ddr_base", "ddr_risk_weight",
    ]
    w["risk_adjustment"] = {k: float(abs(x[28 + i])) for i, k in enumerate(ra_keys)}

    # uncertainty params (9 values, non-negative)
    unc_keys = [
        "clinical_inf", "clinical_tox",
        "binding_default",
        "kin_impl", "kin_cmax",
        "gs_extreme", "gs_penalty",
        "cr_conflict", "cr_penalty",
    ]
    w["uncertainty"] = {k: float(abs(x[40 + i])) for i, k in enumerate(unc_keys)}

    # penalty_slope
    w["penalty_slope"] = max(0.1, float(x[49]))

    return w


def _build_fake_weights(
    decoded: Dict[str, Any],
    fusion_weights: Dict[str, float],
) -> Dict[str, Any]:
    """Build a complete weights dict that matches scoring_weights.json schema."""
    w = copy.deepcopy(DEFAULT_WEIGHTS)

    # Fusion
    w["fusion"] = fusion_weights

    # Sub-weights
    w["clinical_sub"] = decoded["clinical_sub"]
    w["clinical_safety"] = decoded["clinical_safety"]
    w["kinetics_sub"] = decoded["kinetics_sub"]
    w["gene_signature_sub"] = decoded["gene_signature_sub"]
    w["circ_rna_sub"] = decoded["circ_rna_sub"]
    w["tide_ips"] = decoded["tide_ips"]
    w["risk_adjustment"] = decoded["risk_adjustment"]

    # Uncertainty
    cu = decoded["uncertainty"]
    w["clinical_uncertainty"] = {
        "inflammation": cu["clinical_inf"],
        "toxicity": cu["clinical_tox"],
    }
    w["binding_uncertainty"] = {"default": cu["binding_default"]}
    w["binding_uncertainty_penalty"] = cu["binding_default"]
    w["kinetics_uncertainty"] = {
        "hl_low": 0.5, "hl_high": 72.0,
        "implausible_penalty": cu["kin_impl"],
        "cmax_high": 1000.0, "extreme_cmax_penalty": cu["kin_cmax"],
    }
    w["gene_signature_uncertainty"] = {
        "extreme_high": 0.8, "extreme_low": 0.2,
        "extreme_penalty": cu["gs_penalty"],
    }
    w["circ_rna_uncertainty"] = {
        "conflict_high": 0.6, "conflict_penalty": cu["cr_penalty"],
    }

    return w


def _score_samples_with_fake_weights(
    sample_inputs: List[Dict[str, Any]],
    fake_weights: Dict[str, Any],
    penalty_slope: float = 2.0,
) -> np.ndarray:
    """
    Score all samples using fake weights by monkey-patching weight_loader cache.

    This is the critical integration point: we override weight_loader._cached_weights
    so that all get_sub_weights()/get_weight() calls inside JointScoringEngine.score()
    use our candidate weights instead of the JSON file.
    """
    import confluencia_shared.weight_loader as wl
    from confluencia_joint.scoring import JointScoringEngine

    # Save original state
    orig_cached = wl._cached_weights

    # Inject fake weights
    wl._cached_weights = fake_weights

    try:
        engine = JointScoringEngine()
        # Override fusion weights directly on engine
        fw = fake_weights["fusion"]
        engine.clinical_weight = fw.get("clinical", 0.30)
        engine.binding_weight = fw.get("binding", 0.20)
        engine.kinetics_weight = fw.get("kinetics", 0.15)
        engine.gene_signature_weight = fw.get("gene_signature", 0.15)
        engine.circrna_weight = fw.get("circ_rna", 0.20)

        # Override penalty_slope via monkey-patching _adaptive_weights
        # Default is 2.0; if different, we patch the method
        _orig_adaptive = JointScoringEngine._adaptive_weights
        if penalty_slope != 2.0:
            @staticmethod
            def _patched_adaptive(base_w, unc_w, _ps=penalty_slope):
                adjusted = {}
                for dim in ["clinical", "binding", "kinetics", "gene_signature", "circrna"]:
                    cred = 1.0 - unc_w[dim]
                    adjusted[dim] = base_w[dim] * (cred ** _ps)
                total = sum(adjusted.values())
                if total <= 0:
                    n = len(base_w)
                    return {k: 1.0 / n for k in base_w}
                return {dim: adjusted[dim] / total for dim in adjusted}
            engine._adaptive_weights = _patched_adaptive.__func__

        composites = []
        for inp in sample_inputs:
            result = engine.score(
                inp["drug_input"], inp["epi_input"], inp["pk_input"],
                inp["gs_input"], inp["cr_input"],
            )
            composites.append(result.composite)
        return np.array(composites)

    finally:
        # Restore original cache
        wl._cached_weights = orig_cached


def _objective_function(
    x: np.ndarray,
    sample_inputs: List[Dict[str, Any]],
    durations: np.ndarray,
    events: np.ndarray,
    fusion_weights: Dict[str, float],
    cv_folds: int,
    rng_seed: int,
) -> Tuple[float, float, float, float]:
    """
    Evaluate 4 objectives for NSGA-II.

    Returns: (1-C, instability, 1-safety_AUC, 1/dispersion)
    All to be minimized.

    Uses _fast_composite_vectorized for ~50-100x speedup over per-sample score().
    """
    from lifelines.utils import concordance_index
    from sklearn.model_selection import StratifiedKFold

    decoded = _decode_variable_vector(x)
    fake_w = _build_fake_weights(decoded, fusion_weights)

    n = len(sample_inputs)
    indices = np.arange(n)

    # Compute all composites at once using fast vectorized path
    all_composites = _fast_composite_vectorized(sample_inputs, fake_w)

    # Replace any NaN with 0 (safety)
    all_composites = np.nan_to_num(all_composites, nan=0.0)

    # Stratified K-fold CV for C-index stability
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=rng_seed)
    c_indices = []
    for train_idx, val_idx in skf.split(indices, events):
        val_comp = all_composites[val_idx]
        val_dur = durations[val_idx]
        val_evt = events[val_idx]
        # Skip fold if NaN in durations
        valid_fold = ~np.isnan(val_dur) & (val_dur > 0)
        if valid_fold.sum() < 5:
            c_indices.append(0.5)
            continue
        c = concordance_index(val_dur[valid_fold], -val_comp[valid_fold], val_evt[valid_fold])
        c_indices.append(c)

    # Obj 1: 1 - mean C-index
    mean_c = np.mean(c_indices)
    obj1 = 1.0 - mean_c

    # Obj 2: Instability = Var / Mean
    comp_var = np.var(all_composites)
    comp_mean = max(np.mean(all_composites), 1e-6)
    obj2 = comp_var / comp_mean

    # Obj 3: Safety sensitivity — AUC for flagging high-risk samples
    cs_w = decoded["clinical_safety"]
    infl_arr = np.array([np.clip(s["drug_input"].get("inflammation_risk_pred", 0), 0, 1) for s in sample_inputs])
    tox_arr = np.array([np.clip(s["drug_input"].get("genotoxicity_risk_pred", 0), 0, 1) for s in sample_inputs])
    sp_arr = cs_w["toxicity"] * tox_arr + cs_w["inflammation"] * infl_arr
    safety_labels = (sp_arr > 0.25).astype(float)

    n_pos = safety_labels.sum()
    if n_pos > 0 and n_pos < n:
        from sklearn.metrics import roc_auc_score
        try:
            safety_auc = roc_auc_score(safety_labels, all_composites)
        except ValueError:
            safety_auc = 0.5
    else:
        safety_auc = 0.5

    obj3 = 1.0 - safety_auc

    # Obj 4: Score dispersion — maximize IQR
    q75, q25 = np.percentile(all_composites, [75, 25])
    iqr = q75 - q25
    obj4 = 1.0 / max(iqr, 1e-6)

    return obj1, obj2, obj3, obj4


def run_nsga2(
    sample_inputs: List[Dict[str, Any]],
    durations: np.ndarray,
    events: np.ndarray,
    fusion_weights: Dict[str, float],
    pop_size: int = 200,
    n_gen: int = 500,
    cv_folds: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run NSGA-II to optimize sub-score weights.

    Returns: (Pareto_X, Pareto_F) where
      Pareto_X: (n_pareto, N_VAR) variable matrix
      Pareto_F: (n_pareto, 4) objective matrix
    """
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rlk import RandomLinKnotSampling
    from pymoo.optimize import minimize as pymoo_minimize
    from pymoo.termination import get_termination

    class WeightOptProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(
                n_var=N_VAR, n_obj=4, n_constr=0,
                xl=np.full(N_VAR, -3.0),
                xu=np.full(N_VAR, 3.0),
            )

        def _evaluate(self, x, out, *args, **kwargs):
            objs = _objective_function(
                x, sample_inputs, durations, events,
                fusion_weights, cv_folds, seed,
            )
            out["F"] = np.array(objs)

    problem = WeightOptProblem()

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=RandomLinKnotSampling(),
        crossover=SBX(eta=20, prob=0.9),
        mutation=PM(eta=15, prob=1.0 / N_VAR),
    )

    termination = get_termination("n_gen", n_gen)

    print(f"[Phase3] Running NSGA-II: pop={pop_size}, gen={n_gen}, cv={cv_folds}")
    res = pymoo_minimize(problem, algorithm, termination, seed=seed, verbose=True)

    return res.X, res.F


# ---------------------------------------------------------------------------
# Phase 4 — Knee Point Selection
# ---------------------------------------------------------------------------

def select_knee_point(
    pareto_F: np.ndarray,
    strategy: str = "knee",
) -> int:
    """
    Select a solution from the Pareto front.

    Strategies:
      - "knee": maximum distance from the line connecting objective extremes
      - "survival": best C-index (minimize obj1)
      - "safety": best safety sensitivity (minimize obj3)
    """
    if strategy == "survival":
        return int(np.argmin(pareto_F[:, 0]))
    elif strategy == "safety":
        return int(np.argmin(pareto_F[:, 2]))
    else:
        # Knee point: maximum perpendicular distance from the diagonal line
        # Normalize each objective to [0, 1]
        F_norm = pareto_F.copy()
        for j in range(pareto_F.shape[1]):
            lo, hi = F_norm[:, j].min(), F_norm[:, j].max()
            if hi > lo:
                F_norm[:, j] = (F_norm[:, j] - lo) / (hi - lo)
            else:
                F_norm[:, j] = 0.0

        # Line from min-point to max-point in normalized space
        p_min = F_norm.min(axis=0)
        p_max = F_norm.max(axis=0)
        line_vec = p_max - p_min
        line_len = np.linalg.norm(line_vec)

        if line_len < 1e-10:
            return 0

        line_unit = line_vec / line_len
        distances = []
        for i in range(len(F_norm)):
            v = F_norm[i] - p_min
            proj_len = np.dot(v, line_unit)
            perp = v - proj_len * line_unit
            distances.append(np.linalg.norm(perp))

        return int(np.argmax(distances))


# ---------------------------------------------------------------------------
# Phase 5 — Visualization
# ---------------------------------------------------------------------------

def plot_shapley_values(shapley: Dict[str, float], output_dir: Path) -> None:
    """Bar chart of Shapley values."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    mods = list(shapley.keys())
    vals = [shapley[m] for m in mods]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    bars = ax.bar(mods, vals, color=colors[: len(mods)], edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Shapley Value (normalized)", fontsize=12)
    ax.set_title("Phase 2: Shapley Values — Modality Contribution to C-index", fontsize=13)
    ax.set_ylim(0, max(vals) * 1.25 if vals else 1.0)
    plt.tight_layout()
    fig.savefig(output_dir / "figures" / "shapley_values.png", dpi=150)
    plt.close(fig)
    print("[Phase5] Saved shapley_values.png")


def plot_pareto_front(pareto_F: np.ndarray, knee_idx: int, output_dir: Path) -> None:
    """3D scatter of Pareto front (first 3 objectives)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    x, y, z = pareto_F[:, 0], pareto_F[:, 1], pareto_F[:, 2]
    ax.scatter(x, y, z, c="steelblue", alpha=0.5, s=20, label="Pareto solutions")
    ax.scatter([x[knee_idx]], [y[knee_idx]], [z[knee_idx]],
              c="red", s=120, marker="*", label=f"Knee point (idx={knee_idx})")
    ax.set_xlabel("1 - C-index", fontsize=10)
    ax.set_ylabel("Instability", fontsize=10)
    ax.set_zlabel("1 - Safety AUC", fontsize=10)
    ax.set_title("Phase 3: Pareto Front (3D projection)", fontsize=13)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "figures" / "pareto_front.png", dpi=150)
    plt.close(fig)
    print("[Phase5] Saved pareto_front.png")


def plot_weight_comparison(
    default_w: Dict[str, Any],
    calibrated_w: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Side-by-side bar chart of default vs calibrated sub-weights."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = ["clinical_sub", "clinical_safety", "kinetics_sub",
              "gene_signature_sub", "circ_rna_sub"]
    labels_short = ["Clinical", "Safety", "Kinetics", "GeneSig", "circRNA"]

    all_keys, default_vals, cal_vals = [], [], []
    for grp, short in zip(groups, labels_short):
        d = default_w.get(grp, {})
        c = calibrated_w.get(grp, {})
        for k in sorted(set(list(d.keys()) + list(c.keys()))):
            all_keys.append(f"{short}.{k}")
            default_vals.append(d.get(k, 0.0))
            cal_vals.append(c.get(k, 0.0))

    fig, ax = plt.subplots(figsize=(max(12, len(all_keys) * 0.4), 5))
    x = np.arange(len(all_keys))
    width = 0.35
    ax.bar(x - width / 2, default_vals, width, label="Default",
           color="#90CAF9", edgecolor="black", linewidth=0.3)
    ax.bar(x + width / 2, cal_vals, width, label="Calibrated",
           color="#EF5350", edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(all_keys, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Weight value", fontsize=12)
    ax.set_title("Phase 4: Default vs Calibrated Sub-weights", fontsize=13)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "figures" / "weight_comparison.png", dpi=150)
    plt.close(fig)
    print("[Phase5] Saved weight_comparison.png")


def plot_convergence(res_history, output_dir: Path) -> None:
    """Plot convergence of objectives over generations (if history available)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    obj_names = ["1 - C-index", "Instability", "1 - Safety AUC", "1 / IQR"]

    for j, ax in enumerate(axes.flat):
        if j >= 4:
            break
        ax.set_xlabel("Generation")
        ax.set_ylabel(obj_names[j])
        ax.set_title(obj_names[j])
        ax.grid(True, alpha=0.3)

    plt.suptitle("Phase 3: NSGA-II Convergence", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "figures" / "convergence.png", dpi=150)
    plt.close(fig)
    print("[Phase5] Saved convergence.png")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate scoring weights via Shapley + Pareto (NSGA-II)")
    parser.add_argument("--survival-data", type=str,
                        default="data/gene_signature/cache/combined_raw_with_survival.csv")
    parser.add_argument("--drug-data", type=str,
                        default="confluencia-2.0-drug/data/breast_cancer_drug_dataset.csv")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--pop-size", type=int, default=100,
                        help="NSGA-II population size")
    parser.add_argument("--n-gen", type=int, default=200,
                        help="NSGA-II number of generations")
    parser.add_argument("--cv-folds", type=int, default=3,
                        help="Cross-validation folds (3 recommended for speed)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shapley-perms", type=int, default=100)
    parser.add_argument("--selection", type=str, default="knee",
                        choices=["knee", "survival", "safety"])
    parser.add_argument("--subsample", type=int, default=500,
                        help="Number of samples to use for NSGA-II (0=all, recommended 300-800)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Shapley + Pareto Weight Calibration for Confluencia 2.0")
    print("=" * 70)

    # ==================================================================
    # Phase 1: Data Preparation
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: DATA PREPARATION")
    print("=" * 70)

    surv_df = load_survival_data(args.survival_data)
    drug_df = load_drug_data(args.drug_data)
    surv_df = normalize_gene_expr(surv_df)
    drug_corrs = estimate_drug_correlations(drug_df)

    sample_inputs = simulate_modality_inputs(surv_df, drug_corrs, rng)
    durations = np.array([s["OS_months"] for s in sample_inputs])
    events = np.array([s["OS_status"] for s in sample_inputs], dtype=int)

    # Filter out NaN/zero durations
    valid_mask = ~np.isnan(durations) & (durations > 0)
    sample_inputs = [s for s, v in zip(sample_inputs, valid_mask) if v]
    durations = durations[valid_mask]
    events = events[valid_mask]
    print(f"[Phase1] Valid samples after filtering: {len(sample_inputs)}")

    # ==================================================================
    # Phase 2: Shapley Value Analysis
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: SHAPLEY VALUE ANALYSIS")
    print("=" * 70)

    shapley = permutation_shapley(
        sample_inputs, durations, events, rng,
        n_perms=args.shapley_perms,
    )

    print("\n[Phase2] Shapley Values (normalized):")
    for m in MODALITIES:
        print(f"  {m:20s}: {shapley[m]:.4f}")

    fusion_from_shapley = {m: shapley[m] for m in MODALITIES}
    print(f"\n[Phase2] Fusion weights from Shapley: {fusion_from_shapley}")

    # ==================================================================
    # Phase 3: NSGA-II Pareto Optimization
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: NSGA-II PARETO OPTIMIZATION")
    print("=" * 70)

    pareto_X, pareto_F = run_nsga2(
        sample_inputs, durations, events, fusion_from_shapley,
        pop_size=args.pop_size, n_gen=args.n_gen,
        cv_folds=args.cv_folds, seed=args.seed,
    )

    print(f"\n[Phase3] Pareto front size: {len(pareto_F)}")
    print(f"[Phase3] C-index range: "
          f"{1 - pareto_F[:, 0].max():.4f} ~ {1 - pareto_F[:, 0].min():.4f}")

    # ==================================================================
    # Phase 4: Knee Point Selection
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 4: KNEE POINT SELECTION")
    print("=" * 70)

    knee_idx = select_knee_point(pareto_F, strategy=args.selection)
    best_x = pareto_X[knee_idx]
    best_F = pareto_F[knee_idx]

    calibrated_decoded = _decode_variable_vector(best_x)

    print(f"\n[Phase4] Selection strategy: {args.selection}")
    print(f"[Phase4] Selected solution index: {knee_idx}")
    print(f"[Phase4] Objectives:")
    print(f"  C-index      = {1 - best_F[0]:.4f}")
    print(f"  Instability  = {best_F[1]:.4f}")
    print(f"  Safety AUC   = {1 - best_F[2]:.4f}")
    print(f"  IQR          = {1 / best_F[3]:.4f}" if best_F[3] > 0 else
          f"  IQR          = N/A")

    # ==================================================================
    # Build output JSON
    # ==================================================================
    calibrated_json = _build_fake_weights(calibrated_decoded, fusion_from_shapley)
    calibrated_json["version"] = 3
    calibrated_json["source"] = "Shapley + NSGA-II Pareto calibration"
    calibrated_json["description"] = (
        "Calibrated weights: fusion from Shapley values, "
        "sub-weights from NSGA-II Pareto optimization"
    )

    # Save
    out_path = output_dir / "calibrated_weights.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(calibrated_json, f, indent=2, ensure_ascii=False)
    print(f"\n[Phase4] Saved calibrated weights to {out_path}")

    # ==================================================================
    # Phase 5: Visualization
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 5: VISUALIZATION")
    print("=" * 70)

    plot_shapley_values(shapley, output_dir)
    plot_pareto_front(pareto_F, knee_idx, output_dir)
    plot_weight_comparison(DEFAULT_WEIGHTS, calibrated_json, output_dir)
    plot_convergence(None, output_dir)

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n--- Fusion Weights (Shapley-derived) ---")
    for m in MODALITIES:
        old = DEFAULT_WEIGHTS["fusion"].get(m, 0.0)
        new = fusion_from_shapley.get(m, 0.0)
        print(f"  {m:20s}: {old:.3f} -> {new:.3f}  (delta = {new - old:+.3f})")

    print("\n--- Sub-weight Changes (selected) ---")
    for grp in ["clinical_sub", "kinetics_sub", "gene_signature_sub"]:
        print(f"\n  {grp}:")
        old_g = DEFAULT_WEIGHTS.get(grp, {})
        new_g = calibrated_json.get(grp, {})
        for k in sorted(set(list(old_g.keys()) + list(new_g.keys()))):
            ov = old_g.get(k, 0.0)
            nv = new_g.get(k, 0.0)
            print(f"    {k:20s}: {ov:.3f} -> {nv:.3f}  (delta = {nv - ov:+.3f})")

    print(f"\n--- Final Performance ---")
    print(f"  C-index:           {1 - best_F[0]:.4f}")
    print(f"  Instability:       {best_F[1]:.4f}")
    print(f"  Safety AUC:        {1 - best_F[2]:.4f}")
    if best_F[3] > 0:
        print(f"  IQR:               {1 / best_F[3]:.4f}")

    print(f"\nOutput files:")
    print(f"  {out_path}")
    print(f"  {fig_dir / 'shapley_values.png'}")
    print(f"  {fig_dir / 'pareto_front.png'}")
    print(f"  {fig_dir / 'weight_comparison.png'}")
    print(f"  {fig_dir / 'convergence.png'}")

    print("\nTo apply calibrated weights:")
    print(f"  cp {out_path} confluencia_shared/scoring_weights.json")
    print("\nDone!")


if __name__ == "__main__":
    main()
