"""
circbase_large_scale_validation.py -- Large-scale circRNA immunogenicity validation on circBase data.

Validates the rule-based immune sensing scoring (RIG-I, TLR7/8, PKR) against
circBase pseudo-labels for 5000 sampled circRNA sequences.

Scoring rules (inline, no confluencia imports):
  - RIG-I: 35% blunt_end + 40% motif(GC-rich, 5'-ppp) + 20% GC_content + 5% length
  - TLR7/8: 45% uridine_fraction + 30% AU-rich_content + 20% GU_motif + 5% length
  - PKR: 50% dsRNA (>33bp stem) + 25% length + 20% GC + 5% modification penalty
  - Overall: 0.4*RIG-I + 0.35*TLR + 0.25*PKR

Output: benchmarks/results/circbase_large_scale_validation.json
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths — use relative path from script location for cross-platform compatibility
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "circrna"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SMALL_CSV = DATA_DIR / "circbase_pseudo_labels.csv"
FULL_CSV = DATA_DIR / "circbase_pseudo_labels_full.csv"
LABELS_CSV = DATA_DIR / "labels.csv"
OUTPUT_JSON = RESULTS_DIR / "circbase_large_scale_validation.json"

N_SAMPLE = 5000          # rows to sample for benchmark
MIN_SEQ_LEN = 50         # minimum sequence length for scoring
MAX_SEQ_LEN = 50000      # cap extremely long sequences

# ---------------------------------------------------------------------------
# Inline immune-sensing helpers (no confluencia imports)
# ---------------------------------------------------------------------------

RIG_I_MOTIFS = ["CCUCC", "UCUCC", "ACUCC", "GCUCC"]
TLR_MOTIFS  = ["GUUG", "UUGU", "UGUU", "GUUU", "GUU"]
AU_RICH_RE  = re.compile(r"AUUUA|UUAUUUAU|UAUUUAU|UUAUUUAUU")
M6A_DRACH_RE = re.compile(r"[AGU][AG]AC[ACU]", re.IGNORECASE)  # DRACH motif
BLUNT_END_WINDOW = 20
PKR_MIN_DSRNA = 30


def _to_rna(seq: str) -> str:
    """Convert DNA sequence to RNA notation."""
    return seq.upper().replace("T", "U")


def _gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    s = seq.upper().replace("T", "U")
    gc = sum(1 for c in s if c in "GC")
    return gc / len(s)


def _uridine_fraction(seq: str) -> float:
    """Uridine fraction in RNA sequence."""
    if not seq:
        return 0.0
    s = seq.upper().replace("T", "U")
    u = s.count("U")
    return u / len(s)


def _au_rich_count(seq: str) -> int:
    """Count AU-rich element occurrences."""
    s = seq.upper().replace("T", "U")
    return len(AU_RICH_RE.findall(s))


def _motif_count(seq: str, motifs: list) -> int:
    """Total occurrences of any motif in list."""
    s = seq.upper().replace("T", "U")
    total = 0
    for m in motifs:
        total += s.count(m)
    return total


def _detect_blunt_end(seq: str) -> float:
    """
    Blunt end potential score [0, 1].
    Based on immune_sensing.py _detect_blunt_end: GU-pairs, poly-U penalty, GC at terminus.
    """
    s = _to_rna(seq)
    w = min(BLUNT_END_WINDOW, len(s))
    if w == 0:
        return 0.0
    end5 = s[:w]

    # GU-pair frequency at 5' end
    gu_pairs = end5.count("GU") + end5.count("UG")
    gu_score = min(gu_pairs / max(w / 4, 1), 1.0) * 0.35

    # Poly-U overhang penalty (max 0.30)
    poly_u = len(re.findall(r"UUUU+", end5))
    overhang_penalty = min(poly_u * 0.15, 0.30)

    # GC content at terminus
    gc_count = sum(1 for c in end5 if c in "GC")
    gc_term = (gc_count / w) * 0.25

    # 5' terminal base adjustment
    term_adj = 0.0
    if end5[0] in "GC":
        term_adj = 0.10
    elif end5[0] == "U":
        term_adj = -0.05

    return max(0.0, min(1.0, gu_score - overhang_penalty + gc_term + term_adj))


def _estimate_dsRNA_length(seq: str) -> float:
    """
    Estimate dsRNA (double-stranded) region length as a proxy for structure.

    Strategy: count consecutive GC pairs as proxy for dsRNA stem regions.
    Longer runs of alternating GC indicate more stable dsRNA stems.
    Returns estimated dsRNA length in bp (clamped).
    """
    s = _to_rna(seq)
    if len(s) < PKR_MIN_DSRNA:
        return 0.0

    # Count maximal runs of GC-only subsequences (potential stem regions)
    max_gc_run = 0
    current_run = 0
    for c in s:
        if c in "GC":
            current_run += 1
            max_gc_run = max(max_gc_run, current_run)
        else:
            current_run = 0

    # Also count GC-dinucleotide runs (GC pairs)
    gc_pair_runs = []
    run_len = 0
    for i in range(len(s) - 1):
        if (s[i] in "GC" and s[i+1] in "GC"):
            run_len += 1
        else:
            if run_len > 0:
                gc_pair_runs.append(run_len)
            run_len = 0
    if run_len > 0:
        gc_pair_runs.append(run_len)

    # Sum top-3 longest GC-pair runs as proxy for dsRNA stems
    if gc_pair_runs:
        gc_pair_runs.sort(reverse=True)
        top_runs_sum = sum(gc_pair_runs[:3])
    else:
        top_runs_sum = 0

    # Estimated dsRNA length = weighted combination
    # max_gc_run is bp of pure GC stem; gc_pair_runs captures GC pairs
    estimated = max(max_gc_run, top_runs_sum)
    return min(estimated, len(s) * 0.5)  # cannot exceed half the sequence


def _m6a_site_count(seq: str) -> int:
    """Count m6A DRACH motif occurrences: [AGU][AG]AC[ACU] (RNA form)."""
    s = _to_rna(seq)
    # DRACH in RNA: [AGU][AG]AC[ACU]  -- note we use RNA bases
    pattern = re.compile(r"[AGU][AG]AC[ACU]")
    return len(pattern.findall(s))


# ---------------------------------------------------------------------------
# Core scoring functions (matching immune_sensing.py weights)
# ---------------------------------------------------------------------------

def score_rig_i(seq: str) -> float:
    """
    RIG-I pathway score.
    Weights: 35% blunt_end + 40% motif + 20% GC + 5% length
    """
    s = _to_rna(seq)
    n = len(s)
    if n < MIN_SEQ_LEN:
        return 0.0
    if n > MAX_SEQ_LEN:
        s = s[:MAX_SEQ_LEN]
        n = MAX_SEQ_LEN

    # 1) Blunt end potential (35%)
    blunt = _detect_blunt_end(seq) * 0.35

    # 2) Motif matching (40%)
    mc = _motif_count(s, RIG_I_MOTIFS)
    motif_score = min(mc * 0.10, 0.40)

    # 3) GC content (20%)
    gc = _gc_content(s)
    gc_score = gc * 0.20

    # 4) Length (5%)
    length_score = min(n / 5000 * 0.05, 0.05)

    return min(blunt + motif_score + gc_score + length_score, 1.0)


def score_tlr(seq: str) -> float:
    """
    TLR7/8 pathway score.
    Weights: 45% uridine + 30% AU-rich + 20% GU motif + 5% length
    """
    s = _to_rna(seq)
    n = len(s)
    if n < MIN_SEQ_LEN:
        return 0.0
    if n > MAX_SEQ_LEN:
        s = s[:MAX_SEQ_LEN]
        n = MAX_SEQ_LEN

    # 1) Uridine fraction (45%)
    u_frac = _uridine_fraction(s)
    uridine_score = min(u_frac * 2.25, 0.45)

    # 2) AU-rich elements (30%)
    au_count = _au_rich_count(s)
    au_score = min(au_count * 0.075, 0.30)

    # 3) GU motif (20%)
    tlr_mc = _motif_count(s, TLR_MOTIFS)
    motif_score = min(tlr_mc * 0.05, 0.20)

    # 4) Length (5%)
    length_score = min(n / 6000 * 0.05, 0.05)

    return min(uridine_score + au_score + motif_score + length_score, 1.0)


def score_pkr(seq: str) -> float:
    """
    PKR pathway score.
    Weights: 50% dsRNA(>33bp stem) + 25% length + 20% GC + 5% modification penalty
    """
    s = _to_rna(seq)
    n = len(s)
    if n < MIN_SEQ_LEN:
        return 0.0
    if n > MAX_SEQ_LEN:
        s = s[:MAX_SEQ_LEN]
        n = MAX_SEQ_LEN

    # 1) dsRNA formation potential (50%)
    # Use GC * length_factor as dsRNA proxy (as in immune_sensing.py)
    gc = _gc_content(s)
    length_factor = min(n / 500, 1.0)
    dsrna_potential = min(gc * length_factor, 1.0)
    dsrna_score = dsrna_potential * 0.50

    # 2) Length contribution (25%) -- PKR needs ~30+ bp dsRNA
    if n >= PKR_MIN_DSRNA:
        lf = min((n - PKR_MIN_DSRNA) / 1000, 1.0)
        length_score = lf * 0.25
    else:
        length_score = 0.0

    # 3) GC content (20%)
    gc_score = gc * 0.20

    # 4) Modification penalty (5%) -- m6A reduces PKR activation
    m6a_count = _m6a_site_count(s)
    # Each m6A site contributes ~5% penalty, capped
    mod_penalty = min(m6a_count * 0.01, 0.05)
    mod_factor = 1.0 - mod_penalty

    raw = dsrna_score + length_score + gc_score
    return min(raw * mod_factor, 1.0)


def score_overall(seq: str) -> dict:
    """Compute all immunogenicity scores for one sequence."""
    rig = score_rig_i(seq)
    tlr = score_tlr(seq)
    pkr = score_pkr(seq)
    overall = 0.4 * rig + 0.35 * tlr + 0.25 * pkr
    return {
        "rig_i_score": round(rig, 4),
        "tlr_score":   round(tlr, 4),
        "pkr_score":   round(pkr, 4),
        "overall_immunogenicity": round(overall, 4),
    }


# ---------------------------------------------------------------------------
# Sequence feature extraction
# ---------------------------------------------------------------------------

def extract_features(seq: str) -> dict:
    """Extract all relevant sequence features."""
    s = _to_rna(seq)
    n = len(s)
    gc = _gc_content(s)
    u_frac = _uridine_fraction(s)
    au_rich = _au_rich_count(s)
    m6a_sites = _m6a_site_count(s)
    dsrna_len = _estimate_dsRNA_length(seq)
    rig_motifs = _motif_count(s, RIG_I_MOTIFS)
    tlr_motifs = _motif_count(s, TLR_MOTIFS)
    blunt = _detect_blunt_end(seq)
    return {
        "length": n,
        "gc_content": round(gc, 4),
        "uridine_fraction": round(u_frac, 4),
        "au_rich_count": au_rich,
        "m6a_site_count": m6a_sites,
        "dsrna_estimated_length": round(dsrna_len, 1),
        "rig_i_motif_count": rig_motifs,
        "tlr_motif_count": tlr_motifs,
        "blunt_end_score": round(blunt, 4),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_sequence_column(df: pd.DataFrame) -> str:
    """
    Find the sequence column in a DataFrame, handling various naming conventions.
    Returns column name or raises KeyError.
    """
    candidates = ["sequence", "circrna_sequence", "seq", "rna_sequence", "circ_seq"]
    for c in candidates:
        if c in df.columns:
            return c
    # Fallback: any column containing 'seq' in name (case-insensitive)
    for c in df.columns:
        if "seq" in c.lower():
            return c
    raise KeyError(
        f"Cannot find sequence column in DataFrame. Available columns: {df.columns.tolist()}"
    )


def load_data(n_sample: int = N_SAMPLE) -> pd.DataFrame:
    """
    Load circBase data. Prefer the smaller CSV for speed;
    fall back to sampling from the full CSV.
    """
    # Try small CSV first
    if SMALL_CSV.exists():
        print(f"Loading small CSV: {SMALL_CSV}")
        df = pd.read_csv(SMALL_CSV)
        if len(df) > n_sample:
            df = df.sample(n=n_sample, random_state=42)
        print(f"  Loaded {len(df)} rows (sampled from {SMALL_CSV.name})")
        return df

    # Try full CSV
    if FULL_CSV.exists():
        print(f"Loading full CSV (sampling {n_sample} rows): {FULL_CSV}")
        # Read only needed columns to save memory
        df = pd.read_csv(FULL_CSV, nrows=n_sample)
        print(f"  Loaded {len(df)} rows from {FULL_CSV.name}")
        return df

    raise FileNotFoundError(
        f"No circBase CSV found. Expected {SMALL_CSV} or {FULL_CSV}"
    )


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def spearman_rank_corr(x, y):
    """Compute Spearman rank correlation coefficient."""
    from scipy.stats import spearmanr
    r, p = spearmanr(x, y)
    return r, p


def auc_score(y_true, y_pred_proba):
    """Compute AUC-ROC if labels are binary."""
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        return None


def confusion_matrix_metrics(y_true, y_pred):
    """Compute confusion matrix and derived metrics."""
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return cm, report


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark():
    """Run large-scale circBase immunogenicity validation benchmark."""
    print("=" * 70)
    print("circBase Large-Scale Immunogenicity Validation Benchmark")
    print("=" * 70)
    print()

    # Load data
    df = load_data(N_SAMPLE)

    # Find sequence column
    seq_col = find_sequence_column(df)
    print(f"Sequence column: '{seq_col}'")
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print()

    # Determine label columns
    # Possible label columns: pseudo_immunogenicity, pseudo_immuno_score, pseudo_class, immunogenicity
    label_binary_col = None
    label_score_col = None
    label_class_col = None

    for c in ["pseudo_immunogenicity", "immunogenicity"]:
        if c in df.columns:
            label_binary_col = c
            break
    for c in ["pseudo_immuno_score", "immune_score", "immuno_score"]:
        if c in df.columns:
            label_score_col = c
            break
    for c in ["pseudo_class", "immunogenicity_class"]:
        if c in df.columns:
            label_class_col = c
            break

    print(f"Binary label column: {label_binary_col}")
    print(f"Score label column:  {label_score_col}")
    print(f"Class label column:  {label_class_col}")
    print()

    # Drop rows with missing sequences
    df = df.dropna(subset=[seq_col])
    # Filter by minimum length
    df = df[df[seq_col].str.len() >= MIN_SEQ_LEN].copy()
    print(f"After filtering (len >= {MIN_SEQ_LEN}, non-null seq): {len(df)} rows")
    print()

    # Compute scores for each sequence
    start_time = time.time()
    results_list = []
    features_list = []

    for idx, row in df.iterrows():
        seq = str(row[seq_col])
        scores = score_overall(seq)
        feats = extract_features(seq)
        results_list.append(scores)
        features_list.append(feats)

    elapsed = time.time() - start_time
    print(f"Computed scores for {len(df)} sequences in {elapsed:.2f}s "
          f"({len(df)/elapsed:.1f} seq/s)")
    print()

    # Merge into DataFrame
    scores_df = pd.DataFrame(results_list)
    feats_df  = pd.DataFrame(features_list)
    df = pd.concat([df.reset_index(drop=True), scores_df, feats_df], axis=1)

    # -----------------------------------------------------------------------
    # Analysis: compare predicted vs pseudo-label
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("VALIDATION ANALYSIS")
    print("=" * 70)
    print()

    summary = {}

    # 1. Basic statistics
    print("--- Predicted Score Statistics ---")
    for col in ["rig_i_score", "tlr_score", "pkr_score", "overall_immunogenicity"]:
        vals = df[col].astype(float)
        print(f"  {col}: mean={float(vals.mean()):.4f}, std={float(vals.std()):.4f}, "
              f"min={float(vals.min()):.4f}, max={float(vals.max()):.4f}")
        summary[f"{col}_mean"] = round(float(vals.mean()), 4)
        summary[f"{col}_std"]  = round(float(vals.std()), 4)
        summary[f"{col}_min"]  = round(float(vals.min()), 4)
        summary[f"{col}_max"]  = round(float(vals.max()), 4)

    print()
    print("--- Sequence Feature Statistics ---")
    for col in ["gc_content", "uridine_fraction", "m6a_site_count",
                "dsrna_estimated_length", "au_rich_count"]:
        vals = df[col].astype(float)
        print(f"  {col}: mean={float(vals.mean()):.4f}, std={float(vals.std()):.4f}")
        summary[f"feat_{col}_mean"] = round(float(vals.mean()), 4)
        summary[f"feat_{col}_std"]  = round(float(vals.std()), 4)

    print()

    # 2. Spearman correlation: predicted overall vs pseudo_score
    spearman_results = {}
    if label_score_col and label_score_col in df.columns:
        pred_overall = df["overall_immunogenicity"].values
        pseudo_score = df[label_score_col].values

        # Remove NaN pairs
        mask = ~(np.isnan(pred_overall) | np.isnan(pseudo_score))
        if mask.sum() > 10:
            r, p = spearman_rank_corr(pred_overall[mask], pseudo_score[mask])
            print(f"Spearman r (overall_immunogenicity vs {label_score_col}): "
                  f"r={r:.4f}, p={p:.6f} (n={mask.sum()})")
            spearman_results["overall_vs_pseudo_score"] = {
                "r": round(r, 4), "p": round(p, 6), "n": int(mask.sum())
            }
            summary["spearman_overall_vs_pseudo_score_r"] = round(r, 4)
            summary["spearman_overall_vs_pseudo_score_p"] = round(p, 6)

        # Per-pathway Spearman
        for pathway_col in ["rig_i_score", "tlr_score", "pkr_score"]:
            pred_path = df[pathway_col].values
            mask2 = ~(np.isnan(pred_path) | np.isnan(pseudo_score))
            if mask2.sum() > 10:
                r2, p2 = spearman_rank_corr(pred_path[mask2], pseudo_score[mask2])
                print(f"  Spearman r ({pathway_col} vs {label_score_col}): "
                      f"r={r2:.4f}, p={p2:.6f}")
                spearman_results[f"{pathway_col}_vs_pseudo_score"] = {
                    "r": round(r2, 4), "p": round(p2, 6), "n": int(mask2.sum())
                }

        # Feature-level Spearman
        for feat_col in ["gc_content", "uridine_fraction", "m6a_site_count",
                         "dsrna_estimated_length", "au_rich_count"]:
            feat_vals = df[feat_col].values
            mask3 = ~(np.isnan(feat_vals) | np.isnan(pseudo_score))
            if mask3.sum() > 10:
                r3, p3 = spearman_rank_corr(feat_vals[mask3], pseudo_score[mask3])
                print(f"  Spearman r ({feat_col} vs {label_score_col}): "
                      f"r={r3:.4f}, p={p3:.6f}")
                spearman_results[f"feat_{feat_col}_vs_pseudo_score"] = {
                    "r": round(r3, 4), "p": round(p3, 6), "n": int(mask3.sum())
                }
    else:
        print(f"No score label column found -- skipping Spearman analysis.")

    print()

    # 3. AUC-ROC (if binary label exists)
    auc_results = {}
    if label_binary_col and label_binary_col in df.columns:
        y_true = df[label_binary_col].values.astype(int)
        y_pred_proba = df["overall_immunogenicity"].values

        mask = ~(np.isnan(y_pred_proba))
        if mask.sum() > 20 and len(np.unique(y_true[mask])) >= 2:
            auc = auc_score(y_true[mask], y_pred_proba[mask])
            print(f"AUC-ROC (overall_immunogenicity vs {label_binary_col}): "
                  f"AUC={auc:.4f}")
            auc_results["overall_auc"] = round(auc, 4)
            summary["auc_roc_overall"] = round(auc, 4)

            # Per-pathway AUC
            for pathway_col in ["rig_i_score", "tlr_score", "pkr_score"]:
                pp = df[pathway_col].values
                mask2 = ~(np.isnan(pp))
                if mask2.sum() > 20 and len(np.unique(y_true[mask2])) >= 2:
                    auc2 = auc_score(y_true[mask2], pp[mask2])
                    print(f"  AUC-ROC ({pathway_col} vs {label_binary_col}): "
                          f"AUC={auc2:.4f}")
                    auc_results[f"{pathway_col}_auc"] = round(auc2, 4)
        else:
            print("Not enough binary labels or unique classes for AUC computation.")
    else:
        print("No binary label column found -- skipping AUC analysis.")

    print()

    # 4. Confusion matrix (threshold predicted overall at 0.5)
    cm_results = {}
    if label_binary_col and label_binary_col in df.columns:
        y_true = df[label_binary_col].values.astype(int)
        # Threshold at 0.5 for binary classification
        y_pred_bin = (df["overall_immunogenicity"].values >= 0.5).astype(int)

        mask = ~(np.isnan(df["overall_immunogenicity"].values))
        if mask.sum() > 20 and len(np.unique(y_true[mask])) >= 2:
            cm, report = confusion_matrix_metrics(y_true[mask], y_pred_bin[mask])
            print("Confusion Matrix (threshold=0.5):")
            print(cm)
            print()
            for cls in np.unique(y_true[mask]):
                cls_str = str(cls)
                if cls_str in report:
                    prec = report[cls_str]["precision"]
                    rec  = report[cls_str]["recall"]
                    f1   = report[cls_str]["f1-score"]
                    print(f"  Class {cls}: precision={prec:.3f}, recall={rec:.3f}, f1={f1:.3f}")
                    cm_results[f"class_{cls}_precision"] = round(prec, 3)
                    cm_results[f"class_{cls}_recall"]    = round(rec, 3)
                    cm_results[f"class_{cls}_f1"]        = round(f1, 3)
            cm_results["confusion_matrix"] = cm.tolist()
            cm_results["accuracy"] = round(report["accuracy"], 4)
            print(f"  Overall accuracy: {report['accuracy']:.4f}")

            # Also try optimal threshold via F1 optimization
            from sklearn.metrics import f1_score
            thresholds = np.arange(0.1, 0.9, 0.05)
            best_f1 = 0
            best_thresh = 0.5
            for t in thresholds:
                yp = (df["overall_immunogenicity"].values[mask] >= t).astype(int)
                f1 = f1_score(y_true[mask], yp, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = t
            print(f"  Optimal threshold: {best_thresh:.2f} (F1={best_f1:.4f})")
            cm_results["optimal_threshold"] = round(best_thresh, 2)
            cm_results["optimal_f1"] = round(best_f1, 4)
    else:
        print("No binary label column -- skipping confusion matrix.")

    print()

    # 5. Per-metric analysis: mean scores by pseudo_class group
    group_analysis = {}
    if label_class_col and label_class_col in df.columns:
        print("--- Per-Class Analysis ---")
        for cls_name, grp in df.groupby(label_class_col):
            print(f"  Class '{cls_name}' (n={len(grp)}):")
            for score_col in ["overall_immunogenicity", "rig_i_score",
                              "tlr_score", "pkr_score"]:
                mean = grp[score_col].mean()
                std  = grp[score_col].std()
                print(f"    {score_col}: mean={mean:.4f}, std={std:.4f}")
            group_analysis[str(cls_name)] = {
                "n": len(grp),
                "overall_mean": round(grp["overall_immunogenicity"].mean(), 4),
                "overall_std":  round(grp["overall_immunogenicity"].std(), 4),
                "rig_i_mean":   round(grp["rig_i_score"].mean(), 4),
                "tlr_mean":     round(grp["tlr_score"].mean(), 4),
                "pkr_mean":     round(grp["pkr_score"].mean(), 4),
            }
    elif label_binary_col and label_binary_col in df.columns:
        # Use binary label as group
        print("--- Per-Label Group Analysis ---")
        for cls_val, grp in df.groupby(label_binary_col):
            print(f"  Label={cls_val} (n={len(grp)}):")
            for score_col in ["overall_immunogenicity", "rig_i_score",
                              "tlr_score", "pkr_score"]:
                mean = grp[score_col].mean()
                std  = grp[score_col].std()
                print(f"    {score_col}: mean={mean:.4f}, std={std:.4f}")
            group_analysis[str(cls_val)] = {
                "n": len(grp),
                "overall_mean": round(grp["overall_immunogenicity"].mean(), 4),
                "overall_std":  round(grp["overall_immunogenicity"].std(), 4),
                "rig_i_mean":   round(grp["rig_i_score"].mean(), 4),
                "tlr_mean":     round(grp["tlr_score"].mean(), 4),
                "pkr_mean":     round(grp["pkr_score"].mean(), 4),
            }
    else:
        print("No group label column found -- skipping per-class analysis.")

    print()

    # 6. Feature correlation with overall score
    print("--- Feature-Immunogenicity Correlations ---")
    feat_corr_results = {}
    for feat_col in ["gc_content", "uridine_fraction", "m6a_site_count",
                     "dsrna_estimated_length", "au_rich_count",
                     "rig_i_motif_count", "tlr_motif_count", "blunt_end_score"]:
        fv = df[feat_col].values
        ov = df["overall_immunogenicity"].values
        mask = ~(np.isnan(fv) | np.isnan(ov))
        if mask.sum() > 10:
            r, p = spearman_rank_corr(fv[mask], ov[mask])
            print(f"  {feat_col} vs overall: r={r:.4f}, p={p:.6f}")
            feat_corr_results[feat_col] = {
                "r": round(r, 4), "p": round(p, 6), "n": int(mask.sum())
            }

    print()

    # -----------------------------------------------------------------------
    # Assemble output JSON
    # -----------------------------------------------------------------------
    output = {
        "benchmark": "circbase_large_scale_validation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_sequences": len(df),
        "n_sample_requested": N_SAMPLE,
        "min_seq_length": MIN_SEQ_LEN,
        "data_source": SMALL_CSV.name if SMALL_CSV.exists() else (
            FULL_CSV.name if FULL_CSV.exists() else "unknown"),
        "scoring_method": "rule_based_inline",
        "scoring_weights": {
            "rig_i": {"blunt_end": 0.35, "motif": 0.40, "gc_content": 0.20, "length": 0.05},
            "tlr78": {"uridine": 0.45, "au_rich": 0.30, "gu_motif": 0.20, "length": 0.05},
            "pkr":   {"dsRNA": 0.50, "length": 0.25, "gc_content": 0.20, "mod_penalty": 0.05},
            "overall": {"rig_i": 0.40, "tlr": 0.35, "pkr": 0.25},
        },
        "summary_statistics": summary,
        "spearman_correlations": spearman_results,
        "auc_results": auc_results,
        "confusion_matrix_results": cm_results,
        "group_analysis": group_analysis,
        "feature_correlations": feat_corr_results,
        "compute_time_seconds": round(elapsed, 2),
        "compute_speed_seq_per_sec": round(len(df) / elapsed, 1),
    }

    # Save
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {OUTPUT_JSON}")
    print()

    # Also save per-sequence detail CSV
    detail_csv = RESULTS_DIR / "circbase_large_scale_validation_detail.csv"
    df.to_csv(detail_csv, index=False, encoding="utf-8")
    print(f"Detail CSV saved to: {detail_csv}")
    print()

    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    return output


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Ensure scipy and sklearn are available; install hints if missing
    try:
        from scipy.stats import spearmanr
    except ImportError:
        print("ERROR: scipy not installed. Install with: pip install scipy")
        sys.exit(1)
    try:
        from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
    except ImportError:
        print("ERROR: scikit-learn not installed. Install with: pip install scikit-learn")
        sys.exit(1)

    result = run_benchmark()