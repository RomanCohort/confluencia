"""
evaluate_full_scoring.py — Complete 5-modality scoring with detailed analysis.

Runs JointScoringEngine.score() on a sample and prints a detailed breakdown
of every sub-score, uncertainty, adaptive weight, and final recommendation.

Usage (AutoDL):
  cd /root/autodl-tmp/confluencia
  PYTHONPATH=/root/autodl-tmp/confluencia:$PYTHONPATH \
  python scripts/evaluate_full_scoring.py \
      --smiles "CC(=O)Oc1ccccc1C(=O)O" \
      --gene-signature TROP2=0.7,NECTIN4=0.5,LIV1=0.3,B7H4=0.6,TMEM65=0.8

  # Use TCGA patient data:
  python scripts/evaluate_full_scoring.py \
      --survival-data data/gene_signature/cache/combined_raw_with_survival.csv \
      --sample-index 0

  # Quick demo with default values:
  python scripts/evaluate_full_scoring.py
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# --- Stub streamlit/plotly ---
def _decorator(*a, **kw):
    if len(a) == 1 and callable(a[0]) and not kw:
        return a[0]
    return lambda f: f

def _make_stub(name):
    mod = types.ModuleType(name)
    mod.__path__ = []
    mod.__getattr__ = lambda attr: _decorator
    return mod

for _mn in ["streamlit", "plotly", "plotly.graph_objects", "plotly.express"]:
    sys.modules.setdefault(_mn, _make_stub(_mn))


def _parse_kv(text: str) -> Dict[str, float]:
    """Parse KEY=VAL,KEY=VAL string into dict."""
    result = {}
    if not text:
        return result
    for item in text.split(","):
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        result[k.strip()] = float(v.strip())
    return result


def build_drug_input(smiles: str, drug_model_path: Optional[str] = None,
                     env_params: Dict[str, float] = {}) -> Dict[str, float]:
    """Build drug pipeline input — either from model prediction or manual values."""
    if drug_model_path:
        import joblib
        from scripts.predict_drug import _MoleculeFeatures, predict_single
        model_path = Path(drug_model_path)
        if model_path.exists():
            bundle = joblib.load(model_path)
            efficacy = predict_single(bundle, smiles, env_params)
            # Use model prediction + reasonable defaults for other fields
            return {
                "efficacy_pred": efficacy,
                "target_binding_pred": np.clip(efficacy * 0.85, 0, 1),
                "immune_activation_pred": np.clip(0.3 + efficacy * 0.4, 0, 1),
                "inflammation_risk_pred": np.clip(0.1 + (1 - efficacy) * 0.2, 0, 1),
                "genotoxicity_risk_pred": np.clip(0.05 + (1 - efficacy) * 0.1, 0, 1),
            }
    # Default demo values (aspirin-like drug)
    return {
        "efficacy_pred": 0.55,
        "target_binding_pred": 0.48,
        "immune_activation_pred": 0.35,
        "inflammation_risk_pred": 0.12,
        "genotoxicity_risk_pred": 0.08,
    }


def build_epitope_input(efficacy: float = 0.5, uncertainty: float = 0.3) -> Dict[str, float]:
    """Build epitope pipeline input."""
    return {
        "efficacy_pred": efficacy,
        "pred_uncertainty": uncertainty,
    }


def build_pk_input(efficacy: float = 0.5) -> Dict[str, float]:
    """Build PK simulation input based on drug efficacy."""
    # Map efficacy to reasonable PK parameters
    hl = np.clip(6 + efficacy * 12, 2, 36)
    auc_c = np.clip(40 + efficacy * 80, 20, 200)
    auc_e = np.clip(25 + efficacy * 60, 10, 150)
    cmax = np.clip(1.5 + efficacy * 5, 0.5, 15)
    tmax = np.clip(3 + np.random.normal(0, 0.5), 0.5, 8)

    return {
        "pkpd_cmax_mg_per_l": float(cmax),
        "pkpd_tmax_h": float(tmax),
        "pkpd_half_life_h": float(hl),
        "pkpd_auc_conc": float(auc_c),
        "pkpd_auc_effect": float(auc_e),
    }


def build_gene_signature_input(gene_vals: Dict[str, float] = {}) -> Dict[str, float]:
    """Build gene signature input from user-specified gene values."""
    from confluencia_shared.gene_signature_enhanced import (
        compute_five_gene_signature_scores,
        predict_immunotherapy_response,
    )

    t = gene_vals.get("TROP2", 0.5)
    n = gene_vals.get("NECTIN4", 0.5)
    l = gene_vals.get("LIV-1", gene_vals.get("LIV1", 0.5))
    b = gene_vals.get("B7-H4", gene_vals.get("B7H4", 0.5))
    m = gene_vals.get("TMEM65", 0.5)

    gs = compute_five_gene_signature_scores(t, n, l, b, m, mode="yang2025")
    imm = predict_immunotherapy_response(gs["risk_score"], gs["immune_score"], m)

    return {
        "trop2": t, "nectin4": n, "liv1": l, "b7h4": b, "tmem65": m,
        "risk_score": gs["risk_score"],
        "efficacy_score": gs["efficacy_score"],
        "proliferation_score": gs["proliferation_score"],
        "immune_score": gs["immune_score"],
        "mito_score": gs["mito_score"],
        "tide_score": imm["tide_score"],
        "ips_estimate": imm["ips_estimate"],
        "predicted_response": imm["predicted_response"],
        "dhe_recommended": m > 0.5 and t > 0.4,
    }


def build_circrna_input(gene_sig: Dict[str, float] = None,
                         efficacy: float = 0.5) -> Dict[str, float]:
    """Build circRNA input derived from gene signature + drug efficacy."""
    if gene_sig is None:
        gene_sig = build_gene_signature_input()

    b = gene_sig.get("b7h4", 0.5)
    risk = gene_sig.get("risk_score", 0.5)
    prolif = gene_sig.get("proliferation_score", 0.5)
    imm_score = gene_sig.get("immune_score", 0.5)
    m = gene_sig.get("tmem65", 0.5)
    tide = gene_sig.get("tide_score", 0.5)
    ips = gene_sig.get("ips_estimate", 0.5)

    return {
        "immunotherapy_score": np.clip(ips * 0.6 + (1 - tide) * 0.4, 0, 1),
        "therapeutic_window": np.clip(0.3 + efficacy * 0.4, 0, 1),
        "tumor_killing_index": np.clip(prolif * 0.5 + efficacy * 0.3, 0, 1),
        "overall_immunogenicity": np.clip(b * 0.3 + (1 - risk) * 0.4, 0, 1),
        "rig_i_score": np.clip(0.3 + m * 0.3, 0, 1),
        "tlr_score": np.clip(0.2 + m * 0.4, 0, 1),
        "pkr_score": np.clip(0.2 + m * 0.3, 0, 1),
        "tide_score": tide,
        "ips": ips * 10.0,
        "predicted_response": "likely_responder" if ips > 0.6 else "likely_non_responder" if tide > 0.7 else "intermediate",
        "immune_cycle_score": np.clip(0.3 + imm_score * 0.5, 0, 1),
        "tme_score": np.clip(0.3 + (1 - risk) * 0.4, 0, 1),
        "trained_model_risk": risk,
    }


def run_scoring(drug_input, epi_input, pk_input, gs_input, cr_input) -> Dict:
    """Run full JointScoringEngine and return detailed results."""
    from confluencia_joint.scoring import JointScoringEngine

    engine = JointScoringEngine()
    result = engine.score(drug_input, epi_input, pk_input, gs_input, cr_input)

    # Convert dataclasses to dicts for JSON output
    output = {
        "composite": result.composite,
        "recommendation": result.recommendation,
        "recommendation_reason": result.recommendation_reason,
        "effective_weights": result.effective_weights,
        "uncertainties": result.uncertainties,
        "clinical": asdict(result.clinical),
        "binding": asdict(result.binding),
        "kinetics": asdict(result.kinetics),
        "gene_signature": asdict(result.gene_signature) if result.gene_signature else None,
        "circrna": asdict(result.circrna) if result.circrna else None,
    }
    return output


def print_detailed_report(output: Dict):
    """Print a detailed, formatted scoring report."""
    print("\n" + "=" * 72)
    print("  CONFLUENCIA 2.0 — FIVE-MODALITY COMPOSITE SCORING REPORT")
    print("=" * 72)

    c = output["clinical"]
    print("\n┌─── CLINICAL SCORE ────────────────────────────────────────────┐")
    print(f"│  Efficacy:          {c['efficacy']:.4f}")
    print(f"│  Target Binding:    {c['target_binding']:.4f}")
    print(f"│  Immune Activation: {c['immune_activation']:.4f}")
    print(f"│  Safety Penalty:    {c['safety_penalty']:.4f}  (toxicity + inflammation)")
    print(f"│  Overall:           {c['overall']:.4f}")
    print(f"│  {c['interpretation']}")
    print("└───────────────────────────────────────────────────────────────┘")

    b = output["binding"]
    print("\n┌─── BINDING (MHC-EPITOPE) SCORE ───────────────────────────────┐")
    print(f"│  Epitope Efficacy:  {b['epitope_efficacy']:.4f}")
    print(f"│  Uncertainty:       {b['uncertainty']:.4f}")
    print(f"│  MHC Affinity:      {b['mhc_affinity_class']}")
    print(f"│  Overall:           {b['overall']:.4f}  (penalized by uncertainty)")
    print(f"│  {b['interpretation']}")
    print("└───────────────────────────────────────────────────────────────┘")

    k = output["kinetics"]
    print("\n┌─── KINETICS (PK/PD) SCORE ────────────────────────────────────┐")
    print(f"│  Cmax:              {k['cmax']:.2f} mg/L")
    print(f"│  Tmax:              {k['tmax']:.2f} h")
    print(f"│  Half-life:         {k['half_life']:.2f} h")
    print(f"│  AUC(conc):         {k['auc_conc']:.2f} mg·h/L")
    print(f"│  AUC(effect):       {k['auc_effect']:.2f}")
    print(f"│  Therapeutic Index: {k['therapeutic_index']:.4f}")
    print(f"│  Overall:           {k['overall']:.4f}")
    print(f"│  {k['interpretation']}")
    print("└───────────────────────────────────────────────────────────────┘")

    gs = output["gene_signature"]
    if gs:
        print("\n┌─── GENE SIGNATURE (5-TARGET) SCORE ──────────────────────────┐")
        print(f"│  TROP2:             {gs['trop2']:.4f}")
        print(f"│  NECTIN4:           {gs['nectin4']:.4f}")
        print(f"│  LIV-1:             {gs['liv1']:.4f}")
        print(f"│  B7-H4:             {gs['b7h4']:.4f}")
        print(f"│  TMEM65:            {gs['tmem65']:.4f}")
        print(f"│  Risk Score:        {gs['risk_score']:.4f}  (higher = worse prognosis)")
        print(f"│  Efficacy Score:    {gs['efficacy_score']:.4f}")
        print(f"│  Proliferation:     {gs['proliferation_score']:.4f}")
        print(f"│  Immune:            {gs['immune_score']:.4f}")
        print(f"│  Mitochondria:      {gs['mito_score']:.4f}")
        print(f"│  TIDE:              {gs['tide_score']:.4f}  (higher = less ICI benefit)")
        print(f"│  IPS:               {gs['ips_estimate']:.4f}  (higher = more immunogenic)")
        print(f"│  Predicted Response: {gs['predicted_response']}")
        print(f"│  DHE Recommended:   {gs['dhe_recommended']}")
        print(f"│  Overall:           {gs['overall']:.4f}")
        print(f"│  {gs['interpretation']}")
        print("└───────────────────────────────────────────────────────────────┘")

    cr = output["circrna"]
    if cr:
        print("\n┌─── circRNA MULTI-OMICS SCORE ────────────────────────────────┐")
        print(f"│  Immunotherapy:     {cr['immunotherapy_score']:.4f}")
        print(f"│  Therapeutic Window: {cr['therapeutic_window']:.4f}")
        print(f"│  Tumor Killing:     {cr['tumor_killing_index']:.4f}")
        print(f"│  Immunogenicity:    {cr['overall_immunogenicity']:.4f}")
        print(f"│  RIG-I:             {cr['rig_i_score']:.4f}")
        print(f"│  TLR7/8:            {cr['tlr_score']:.4f}")
        print(f"│  PKR:               {cr['pkr_score']:.4f}")
        print(f"│  TIDE:              {cr['tide_score']:.4f}")
        print(f"│  IPS:               {cr['ips']:.1f}/10")
        print(f"│  Predicted Response: {cr['predicted_response']}")
        print(f"│  Immune Cycle:      {cr['immune_cycle_score']:.4f}")
        print(f"│  TME:               {cr['tme_score']:.4f}")
        print(f"│  Trained Risk:      {cr['trained_model_risk']:.4f}")
        print(f"│  Overall:           {cr['overall']:.4f}")
        print(f"│  {cr['interpretation']}")
        print("└───────────────────────────────────────────────────────────────┘")

    # --- Uncertainty & Adaptive Weights ---
    print("\n┌─── UNCERTAINTY-ADAPTIVE WEIGHT ANALYSIS ──────────────────────┐")
    unc = output["uncertainties"]
    ew = output["effective_weights"]
    base_w = {
        "clinical": 0.30, "binding": 0.20, "kinetics": 0.15,
        "gene_signature": 0.15, "circrna": 0.20,
    }
    mods = ["clinical", "binding", "kinetics", "gene_signature", "circrna"]
    print(f"│  {'Dimension':<20s} {'Base W':>8s} {'Uncertainty':>12s} {'Effective W':>12s} {'Shift':>8s}")
    print(f"│  {'─'*20} {'─'*8} {'─'*12} {'─'*12} {'─'*8}")
    for dim in mods:
        b_w = base_w.get(dim, 0.0)
        u = unc.get(dim, 0.0)
        e_w = ew.get(dim, 0.0)
        shift = e_w - b_w
        print(f"│  {dim:<20s} {b_w:>8.4f} {u:>12.4f} {e_w:>12.4f} {shift:>+8.4f}")
    print("└───────────────────────────────────────────────────────────────┘")

    # --- Composite & Recommendation ---
    print("\n┌─── FINAL COMPOSITE & RECOMMENDATION ─────────────────────────┐")
    print(f"│  Composite Score:   {output['composite']:.4f}")
    print(f"│  Recommendation:    {output['recommendation']}")
    print(f"│  Reason:            {output['recommendation_reason']}")
    # Thresholds
    from confluencia_shared.weight_loader import get_thresholds
    th = get_thresholds()
    print(f"│  Thresholds:        Go≥{th['go']:.2f}, Conditional≥{th['conditional']:.2f}, Safety floor={th['safety_floor']:.2f}")
    print("└───────────────────────────────────────────────────────────────┘")

    # --- Weight decomposition ---
    print("\n┌─── COMPOSITE DECOMPOSITION ──────────────────────────────────┐")
    contributions = {}
    scores = {
        "clinical": c["overall"],
        "binding": b["overall"],
        "kinetics": k["overall"],
        "gene_signature": gs["overall"] if gs else 0.0,
        "circrna": cr["overall"] if cr else 0.0,
    }
    for dim in mods:
        contrib = ew.get(dim, 0.0) * scores.get(dim, 0.0)
        contributions[dim] = contrib
        pct = contrib / output["composite"] * 100 if output["composite"] > 0 else 0
        print(f"│  {dim:<20s} = {ew.get(dim, 0.0):.4f} × {scores.get(dim, 0.0):.4f} = {contrib:.4f}  ({pct:.1f}%)")
    print(f"│  {'TOTAL':<20s} = {output['composite']:.4f}")
    print("└───────────────────────────────────────────────────────────────┘")

    print("\n" + "=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Full 5-modality scoring with detailed analysis report")
    parser.add_argument("--smiles", type=str, default=None,
                        help="SMILES for drug prediction")
    parser.add_argument("--drug-model", type=str, default=None,
                        help="Drug model bundle path (.joblib)")
    parser.add_argument("--gene-signature", type=str, default=None,
                        help="Gene values: TROP2=0.7,NECTIN4=0.5,LIV1=0.3,B7H4=0.6,TMEM65=0.8")
    parser.add_argument("--epitope-efficacy", type=float, default=0.5,
                        help="Epitope binding efficacy (0-1)")
    parser.add_argument("--epitope-uncertainty", type=float, default=0.3,
                        help="Epitope prediction uncertainty (0-1)")
    parser.add_argument("--survival-data", type=str, default=None,
                        help="TCGA/METABRIC CSV for patient-level gene data")
    parser.add_argument("--sample-index", type=int, default=0,
                        help="Row index in survival data CSV")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save full results as JSON")
    parser.add_argument("--demo", action="store_true",
                        help="Run with default demo values")
    args = parser.parse_args()

    if args.demo or (not args.smiles and not args.survival_data and not args.gene_signature):
        print("[INFO] Running demo with default values. Use --gene-signature or --survival-data for real data.")

    # Build inputs
    gene_vals = _parse_kv(args.gene_signature or "")
    efficacy_base = 0.5

    if args.survival_data:
        import pandas as pd
        surv_path = Path(args.survival_data)
        if not surv_path.exists():
            print(f"ERROR: {args.survival_data} not found"); sys.exit(1)
        df = pd.read_csv(surv_path)
        if args.sample_index >= len(df):
            print(f"ERROR: sample_index {args.sample_index} out of range (max {len(df)-1})"); sys.exit(1)
        row = df.iloc[args.sample_index]
        # Normalize gene expression to 0-1
        for gene in ["TROP2", "NECTIN4", "LIV-1", "B7-H4", "TMEM65"]:
            if gene in row.index:
                val = float(row[gene])
                # Quantile normalize to 0-1 range (assuming log2 expression)
                gene_vals[gene] = np.clip((val - 2) / 10, 0, 1) if val > 0 else 0.5
            else:
                gene_vals[gene] = 0.5
        # Clinical features → efficacy estimate
        grade = float(row.get("grade", 2))
        er = float(row.get("ER_positive", 0.5))
        efficacy_base = np.clip(0.6 - (grade - 2) * 0.1 + er * 0.1, 0.1, 0.9)
        print(f"[INFO] Using patient #{args.sample_index}: genes={gene_vals}, efficacy_base={efficacy_base:.3f}")

    # 1. Drug input
    drug_input = build_drug_input(args.smiles, args.drug_model,
                                   {"efficacy_base": efficacy_base})
    if args.drug_model and args.smiles:
        print(f"[INFO] Drug prediction from model for SMILES: {args.smiles}")
    else:
        print(f"[INFO] Drug input: efficacy={drug_input['efficacy_pred']:.3f}")

    # 2. Epitope input
    epi_input = build_epitope_input(args.epitope_efficacy, args.epitope_uncertainty)

    # 3. PK input
    pk_input = build_pk_input(drug_input["efficacy_pred"])

    # 4. Gene signature input
    gs_input = build_gene_signature_input(gene_vals)

    # 5. circRNA input (derived from gene signature)
    cr_input = build_circrna_input(gs_input, drug_input["efficacy_pred"])

    # Run scoring
    output = run_scoring(drug_input, epi_input, pk_input, gs_input, cr_input)

    # Print detailed report
    print_detailed_report(output)

    # Save JSON if requested
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Results saved to {out_path}")


if __name__ == "__main__":
    main()