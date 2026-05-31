"""
shape_structure_validation.py — Benchmark: validate RNA structure predictions
against SHAPE experimental reactivity data.

Methodology:
1. Load SHAPE experimental data from shape_cache.csv
2. For each RNA sequence, predict structure using:
   a. ViennaRNA RNAfold (if available via subprocess)
   b. Fallback heuristic (GC content, simple stem-loop detection)
3. Correlate predicted pairing probabilities with SHAPE reactivity:
   - Low SHAPE reactivity -> base-paired
   - High SHAPE reactivity -> unpaired
   - Expect negative correlation between predicted pairing and SHAPE reactivity
4. Compute metrics: Pearson r, Spearman r, per-sequence paired/unpaired accuracy
5. Save results to JSON

SHAPE (Selective 2'-Hydroxyl Acylation analyzed by Primer Extension) reactivity
measures nucleotide flexibility: constrained (paired) nucleotides show low SHAPE
values, flexible (unpaired) nucleotides show high SHAPE values.

Reference:
- Wilkinson et al., 2006: SHAPE-directed RNA structure determination
- Deigan et al., 2009: SHAPE pseudo-free energy terms for RNA folding
"""

import csv
import json
import math
import os
import subprocess
import sys
import time
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths — use relative path from script location for cross-platform compatibility
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHAPE_CSV = PROJECT_ROOT / "data" / "structure" / "shape_cache.csv"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
RESULTS_FILE = RESULTS_DIR / "shape_structure_validation.json"

# SHAPE threshold: below this -> classified as "paired", above -> "unpaired"
# Standard threshold from literature (Deigan et al., 2009): ~0.3-0.5
SHAPE_PAIRED_THRESHOLD = 0.4

# Minimum nucleotides for a valid sequence
MIN_SEQ_LENGTH = 10


# ---------------------------------------------------------------------------
# ViennaRNA interface
# ---------------------------------------------------------------------------
def check_viennarna_available() -> bool:
    """Check whether RNAfold is accessible via subprocess."""
    try:
        result = subprocess.run(
            ["RNAfold", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _parse_rnafold_output(stdout: str) -> Optional[Tuple[float, str]]:
    """Parse RNAfold stdout output to extract MFE and dot-bracket structure.

    RNAfold output format:
      Line 0: FASTA header (>benchmark_seq)
      Line 1: Sequence echoed back
      Line 2: Dot-bracket structure + MFE e.g. ".(((...)))  (-34.50)"

    Returns (mfe, dot_bracket) or None on failure.
    """
    lines = stdout.strip().split("\n")
    # Find the structure line (contains dot-bracket chars and MFE in parentheses)
    structure_line = None
    for line in lines:
        if "(" in line and ")" in line and any(c in line for c in ".{}[]"):
            structure_line = line
            break
    # Fallback: try last line or line index 2
    if structure_line is None and len(lines) >= 3:
        structure_line = lines[2]
    elif structure_line is None and len(lines) >= 2:
        structure_line = lines[-1]
    if structure_line is None:
        return None

    parts = structure_line.split()
    if len(parts) < 1:
        return None
    dot_bracket = parts[0]
    # MFE is typically in format (-34.50)
    if len(parts) >= 2:
        mfe_str = parts[1].strip("()")
        try:
            mfe = float(mfe_str)
        except ValueError:
            mfe = 0.0
    else:
        mfe = 0.0
    return (mfe, dot_bracket)


def run_rnafold(sequence: str) -> Optional[Tuple[float, str]]:
    """
    Run ViennaRNA RNAfold on a single sequence.

    Returns (mfe, dot_bracket) or None on failure.
    """
    # Convert DNA -> RNA
    seq = sequence.upper().replace("T", "U")
    seq = "".join(c for c in seq if c in "AUGC")
    if len(seq) < MIN_SEQ_LENGTH:
        return None

    try:
        result = subprocess.run(
            ["RNAfold", "--noPS"],
            input=f">benchmark_seq\n{seq}\n",
            capture_output=True,
            text=True,
            timeout=120,
        )
        parsed = _parse_rnafold_output(result.stdout)
        if parsed is not None:
            return parsed
    except subprocess.TimeoutExpired:
        warnings.warn("RNAfold timeout for sequence (len={})".format(len(seq)))
    except Exception as e:
        warnings.warn("RNAfold error: {}".format(e))

    return None


def run_rnafold_with_shape(sequence: str, shape_values: List[float]) -> Optional[Tuple[float, str]]:
    """
    Run RNAfold with SHAPE constraints (pseudo-free energy terms).

    Uses RNAfold -p --shape=... to incorporate SHAPE reactivity as constraints.
    Deigan et al., 2009: SHAPE pseudo-energy = m * ln(reactivity + 1) + b
    where m ~1.8, b ~-0.6 kcal/mol for paired, and for unpaired it is 0.

    Returns (mfe, dot_bracket) or None on failure.
    """
    seq = sequence.upper().replace("T", "U")
    seq = "".join(c for c in seq if c in "AUGC")
    if len(seq) < MIN_SEQ_LENGTH or len(shape_values) != len(seq):
        return None

    try:
        # Write SHAPE data to a temp file in RNAfold format
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".shape", delete=False) as tmp:
            # RNAfold SHAPE file format: position (1-based), reactivity
            for i, val in enumerate(shape_values):
                tmp.write("{} {:.6f}\n".format(i + 1, val))
            shape_file = tmp.name

        result = subprocess.run(
            ["RNAfold", "--noPS", "--shape", shape_file, "--shapeMethod", "D"],
            input=f">benchmark_seq\n{seq}\n",
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Clean up temp file
        try:
            os.unlink(shape_file)
        except OSError:
            pass

        parsed = _parse_rnafold_output(result.stdout)
        if parsed is not None:
            return parsed
    except subprocess.TimeoutExpired:
        warnings.warn("RNAfold+SHAPE timeout")
    except Exception as e:
        warnings.warn("RNAfold+SHAPE error: {}".format(e))

    return None


# ---------------------------------------------------------------------------
# Fallback heuristic structure prediction
# ---------------------------------------------------------------------------
def compute_gc_content(sequence: str) -> float:
    """Compute GC content of an RNA sequence."""
    seq = sequence.upper()
    gc = sum(1 for c in seq if c in "GC")
    total = sum(1 for c in seq if c in "AUGC")
    return gc / total if total > 0 else 0.0


def estimate_mfe_from_gc(sequence: str) -> float:
    """
    Estimate MFE from GC content.

    Typical RNA MFE: -0.3 to -0.8 kcal/mol per nucleotide.
    Higher GC -> more stable -> more negative MFE.
    """
    seq = sequence.upper().replace("T", "U")
    seq = "".join(c for c in seq if c in "AUGC")
    gc = compute_gc_content(seq)
    mfe_per_nt = -0.3 - 0.5 * gc
    return mfe_per_nt * len(seq)


def fallback_dot_bracket(sequence: str) -> str:
    """
    Generate a heuristic dot-bracket estimate from sequence composition.

    Strategy:
    - Scan for GC-rich windows -> mark as stem (paired)
    - Intervening AU-rich regions -> loop/unpaired
    - Use window-based GC threshold to classify regions
    - Build simplified stem-loop architecture
    """
    seq = sequence.upper().replace("T", "U")
    seq = "".join(c for c in seq if c in "AUGC")
    n = len(seq)
    if n < MIN_SEQ_LENGTH:
        return "." * n

    gc = compute_gc_content(seq)
    # Stem length proportional to GC content
    stem_len = max(3, int(gc * 8) + 2)
    # Window size for GC scanning
    window_size = max(4, stem_len)

    # Step 1: classify each position as GC-rich (potential paired) or AU-rich
    # using a sliding window
    paired_flags = [False] * n
    for i in range(n):
        start = max(0, i - window_size // 2)
        end = min(n, start + window_size)
        window = seq[start:end]
        w_gc = sum(1 for c in window if c in "GC") / len(window) if window else 0
        paired_flags[i] = w_gc > 0.45

    # Step 2: build dot-bracket from paired_flags
    # Simplified model: paired regions form stems with hairpin loops
    result = []
    i = 0
    while i < n:
        # Find the start of a paired run
        if paired_flags[i]:
            # Count consecutive paired positions
            run_start = i
            while i < n and paired_flags[i]:
                i += 1
            run_len = i - run_start

            if run_len >= 3:
                # This is a stem region
                # Open half of stem
                half = min(stem_len, run_len // 2)
                for j in range(half):
                    result.append("(")
                # Internal loop / bulge if run_len > 2*stem_len
                excess = run_len - 2 * half
                for j in range(excess):
                    result.append(".")  # internal bulge
                # Close half of stem
                for j in range(half):
                    result.append(")")
                # Hairpin loop after stem close
                loop_len = max(3, int(half * 0.5))
                for j in range(min(loop_len, n - i)):
                    result.append(".")
                i += min(loop_len, n - i)
            else:
                # Short paired run, treat as unpaired
                for j in range(run_len):
                    result.append(".")
        else:
            result.append(".")
            i += 1

    # Trim/pad to exact sequence length
    db = "".join(result)
    if len(db) > n:
        db = db[:n]
    elif len(db) < n:
        db += "." * (n - len(db))

    # Balance parentheses (simplified: just replace excess parens with dots)
    open_count = db.count("(")
    close_count = db.count(")")
    if open_count > close_count:
        diff = open_count - close_count
        # Replace some "(" with "." from the end
        for _ in range(diff):
            idx = db.rfind("(")
            if idx >= 0:
                db = db[:idx] + "." + db[idx + 1:]
    elif close_count > open_count:
        diff = close_count - open_count
        for _ in range(diff):
            idx = db.find(")")
            if idx >= 0:
                db = db[:idx] + "." + db[idx + 1:]

    return db


def extract_pairing_from_dot_bracket(dot_bracket: str) -> List[float]:
    """
    Extract per-position pairing probability from dot-bracket.

    "(" or ")" -> paired (probability 1.0)
    "." -> unpaired (probability 0.0)

    Returns list of pairing probabilities per position.
    """
    pairing = []
    for c in dot_bracket:
        if c == "(" or c == ")":
            pairing.append(1.0)
        else:
            pairing.append(0.0)
    return pairing


def classify_from_shape(shape_values: List[float],
                        threshold: float = SHAPE_PAIRED_THRESHOLD) -> List[int]:
    """
    Classify each position as paired/unpaired from SHAPE reactivity.

    Low SHAPE (< threshold) -> paired (1)
    High SHAPE (> threshold) -> unpaired (0)

    Returns list of 0/1 classifications.
    """
    return [1 if v < threshold else 0 for v in shape_values]


# ---------------------------------------------------------------------------
# Correlation and accuracy metrics
# ---------------------------------------------------------------------------
def pearson_r(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)
    if var_x == 0 or var_y == 0:
        return 0.0
    return cov_xy / math.sqrt(var_x * var_y)


def spearman_r(x: List[float], y: List[float]) -> float:
    """Compute Spearman rank correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0

    def rank(arr):
        sorted_vals = sorted([(v, i) for i, v in enumerate(arr)])
        ranks = [0.0] * n
        pos = 0
        while pos < n:
            # Handle ties: average rank
            tied_vals = [sorted_vals[pos]]
            j = pos + 1
            while j < n and sorted_vals[j][0] == sorted_vals[pos][0]:
                tied_vals.append(sorted_vals[j])
                j += 1
            avg_rank = (pos + 1 + j) / 2.0  # average of tied positions
            for _, idx in tied_vals:
                ranks[idx] = avg_rank
            pos = j
        return ranks

    rx = rank(x)
    ry = rank(y)
    return pearson_r(rx, ry)


def classification_accuracy(predicted: List[int], observed: List[int]) -> float:
    """Compute accuracy of paired/unpaired classification."""
    if len(predicted) == 0:
        return 0.0
    correct = sum(1 for p, o in zip(predicted, observed) if p == o)
    return correct / len(predicted)


def paired_unpaired_metrics(predicted: List[int], observed: List[int]) -> Dict:
    """
    Compute classification metrics: accuracy, sensitivity, specificity.

    predicted: 1=paired, 0=unpaired (from structure prediction)
    observed: 1=paired, 0=unpaired (from SHAPE data)
    """
    tp = sum(1 for p, o in zip(predicted, observed) if p == 1 and o == 1)
    tn = sum(1 for p, o in zip(predicted, observed) if p == 0 and o == 0)
    fp = sum(1 for p, o in zip(predicted, observed) if p == 1 and o == 0)
    fn = sum(1 for p, o in zip(predicted, observed) if p == 0 and o == 1)
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall for paired
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # recall for unpaired
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Matthews correlation coefficient
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_shape_data(csv_path: Path) -> List[Dict]:
    """
    Load SHAPE data from CSV file.

    Returns list of dicts with keys:
    - sequence_id, sequence, shape_values, error_profile, coverage_profile,
      coverage_score, experimental_method, experimental_conditions, cell_type,
      reference, description
    """
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            shape_values = json.loads(row["reactivity_profile"])
            error_profile = json.loads(row.get("error_profile", "[]"))
            coverage_profile = json.loads(row.get("coverage_profile", "[]"))

            records.append({
                "sequence_id": row["sequence_id"],
                "sequence": row["sequence"],
                "shape_values": shape_values,
                "error_profile": error_profile,
                "coverage_profile": coverage_profile,
                "coverage_score": float(row.get("coverage_score", 0)),
                "experimental_method": row.get("experimental_method", "SHAPE"),
                "experimental_conditions": row.get("experimental_conditions", ""),
                "cell_type": row.get("cell_type", ""),
                "reference": row.get("reference", ""),
                "description": row.get("description", ""),
            })

    return records


# ---------------------------------------------------------------------------
# Main benchmark logic
# ---------------------------------------------------------------------------
def run_benchmark(shape_csv: Path, results_file: Path) -> Dict:
    """
    Run the SHAPE-structure validation benchmark.

    For each sequence in the SHAPE cache:
    1. Predict structure (ViennaRNA if available, else fallback)
    2. Extract pairing probabilities per position
    3. Compare pairing predictions against SHAPE reactivity classification
    4. Compute correlation and classification metrics

    Returns complete results dict.
    """
    print("=" * 70)
    print("SHAPE-Structure Validation Benchmark")
    print("=" * 70)

    # Check ViennaRNA availability
    has_viennarna = check_viennarna_available()
    print("ViennaRNA available: {}".format(has_viennarna))

    # Load data
    print("Loading SHAPE data from: {}".format(shape_csv))
    records = load_shape_data(shape_csv)
    print("Loaded {} sequences".format(len(records)))

    # Per-sequence results
    per_sequence = []
    all_predicted_pairing = []
    all_shape_reactivity = []
    all_predicted_class = []
    all_observed_class = []

    viennarna_count = 0
    fallback_count = 0
    viennarna_shape_count = 0

    start_time = time.time()

    for idx, rec in enumerate(records):
        seq_id = rec["sequence_id"]
        sequence = rec["sequence"]
        shape_values = rec["shape_values"]
        seq_len = len(sequence)

        print("\n[{:2d}/{}] {} (len={})".format(
            idx + 1, len(records), seq_id, seq_len))

        # --- Structure prediction ---
        mfe = None
        dot_bracket = None
        method = "none"

        if has_viennarna:
            # Try RNAfold without SHAPE constraints first
            result = run_rnafold(sequence)
            if result is not None:
                mfe, dot_bracket = result
                method = "viennarna"
                viennarna_count += 1

                # Also try RNAfold with SHAPE constraints (constrained prediction)
                result_shape = run_rnafold_with_shape(sequence, shape_values)
                if result_shape is not None:
                    viennarna_shape_count += 1

        # Fallback if ViennaRNA failed or not available
        if dot_bracket is None:
            mfe = estimate_mfe_from_gc(sequence)
            dot_bracket = fallback_dot_bracket(sequence)
            method = "fallback"
            fallback_count += 1

        # --- Extract pairing from prediction ---
        predicted_pairing = extract_pairing_from_dot_bracket(dot_bracket)

        # Ensure lengths match (pad or trim if needed)
        if len(predicted_pairing) != seq_len:
            if len(predicted_pairing) > seq_len:
                predicted_pairing = predicted_pairing[:seq_len]
                dot_bracket = dot_bracket[:seq_len]
            else:
                predicted_pairing.extend([0.0] * (seq_len - len(predicted_pairing)))
                dot_bracket += "." * (seq_len - len(dot_bracket))

        # --- SHAPE classification (ground truth) ---
        observed_class = classify_from_shape(shape_values)
        predicted_class = [1 if p >= 0.5 else 0 for p in predicted_pairing]

        # --- Correlation metrics ---
        # Pairing probability vs SHAPE reactivity (expect negative correlation)
        pearson = pearson_r(predicted_pairing, shape_values)
        spearman = spearman_r(predicted_pairing, shape_values)

        # --- Classification metrics ---
        class_metrics = paired_unpaired_metrics(predicted_class, observed_class)

        # --- GC content ---
        gc_content = compute_gc_content(sequence)

        # --- SHAPE statistics ---
        mean_shape = sum(shape_values) / len(shape_values) if shape_values else 0
        paired_frac_from_shape = sum(1 for v in shape_values if v < SHAPE_PAIRED_THRESHOLD) / len(shape_values) if shape_values else 0
        paired_frac_from_pred = sum(1 for p in predicted_class if p == 1) / len(predicted_class) if predicted_class else 0

        seq_result = OrderedDict([
            ("sequence_id", seq_id),
            ("sequence_length", seq_len),
            ("gc_content", gc_content),
            ("prediction_method", method),
            ("mfe", mfe),
            ("dot_bracket_preview", dot_bracket[:60] + "..." if len(dot_bracket) > 60 else dot_bracket),
            ("mean_shape_reactivity", mean_shape),
            ("paired_fraction_shape", paired_frac_from_shape),
            ("paired_fraction_predicted", paired_frac_from_pred),
            ("pearson_r", pearson),
            ("spearman_r", spearman),
            ("accuracy", class_metrics["accuracy"]),
            ("sensitivity", class_metrics["sensitivity"]),
            ("specificity", class_metrics["specificity"]),
            ("precision", class_metrics["precision"]),
            ("mcc", class_metrics["mcc"]),
            ("tp", class_metrics["tp"]),
            ("tn", class_metrics["tn"]),
            ("fp", class_metrics["fp"]),
            ("fn", class_metrics["fn"]),
        ])

        per_sequence.append(seq_result)

        # Accumulate for aggregate metrics
        all_predicted_pairing.extend(predicted_pairing)
        all_shape_reactivity.extend(shape_values)
        all_predicted_class.extend(predicted_class)
        all_observed_class.extend(observed_class)

        print("  Method: {}".format(method))
        print("  GC: {:.2%}".format(gc_content))
        print("  MFE: {:.2f}".format(mfe))
        print("  Pearson r (pairing vs SHAPE): {:.4f}".format(pearson))
        print("  Spearman r (pairing vs SHAPE): {:.4f}".format(spearman))
        print("  Accuracy: {:.2%}".format(class_metrics["accuracy"]))
        print("  MCC: {:.4f}".format(class_metrics["mcc"]))
        print("  Paired frac (SHAPE): {:.2%}".format(paired_frac_from_shape))
        print("  Paired frac (pred):  {:.2%}".format(paired_frac_from_pred))

    elapsed = time.time() - start_time

    # --- Aggregate metrics across all sequences ---
    agg_pearson = pearson_r(all_predicted_pairing, all_shape_reactivity)
    agg_spearman = spearman_r(all_predicted_pairing, all_shape_reactivity)
    agg_class = paired_unpaired_metrics(all_predicted_class, all_observed_class)

    # Per-sequence summary statistics
    seq_pearsons = [s["pearson_r"] for s in per_sequence]
    seq_spearmans = [s["spearman_r"] for s in per_sequence]
    seq_accuracies = [s["accuracy"] for s in per_sequence]
    seq_mccs = [s["mcc"] for s in per_sequence]

    def mean_std(vals):
        n = len(vals)
        if n == 0:
            return (0.0, 0.0)
        m = sum(vals) / n
        s = math.sqrt(sum((v - m) ** 2 for v in vals) / n) if n > 1 else 0.0
        return (m, s)

    pearson_mean, pearson_std = mean_std(seq_pearsons)
    spearman_mean, spearman_std = mean_std(seq_spearmans)
    acc_mean, acc_std = mean_std(seq_accuracies)
    mcc_mean, mcc_std = mean_std(seq_mccs)

    # --- Build final results dict ---
    results = OrderedDict([
        ("benchmark", "shape_structure_validation"),
        ("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
        ("shape_csv", str(shape_csv)),
        ("num_sequences", len(records)),
        ("viennarna_available", has_viennarna),
        ("viennarna_predictions", viennarna_count),
        ("viennarna_shape_constrained", viennarna_shape_count),
        ("fallback_predictions", fallback_count),
        ("shape_paired_threshold", SHAPE_PAIRED_THRESHOLD),
        ("elapsed_seconds", round(elapsed, 2)),
        ("aggregate", OrderedDict([
            ("pearson_r", agg_pearson),
            ("spearman_r", agg_spearman),
            ("accuracy", agg_class["accuracy"]),
            ("sensitivity", agg_class["sensitivity"]),
            ("specificity", agg_class["specificity"]),
            ("precision", agg_class["precision"]),
            ("mcc", agg_class["mcc"]),
            ("total_positions", len(all_predicted_class)),
            ("tp", agg_class["tp"]),
            ("tn", agg_class["tn"]),
            ("fp", agg_class["fp"]),
            ("fn", agg_class["fn"]),
        ])),
        ("per_sequence_summary", OrderedDict([
            ("pearson_r_mean", pearson_mean),
            ("pearson_r_std", pearson_std),
            ("spearman_r_mean", spearman_mean),
            ("spearman_r_std", spearman_std),
            ("accuracy_mean", acc_mean),
            ("accuracy_std", acc_std),
            ("mcc_mean", mcc_mean),
            ("mcc_std", mcc_std),
        ])),
        ("per_sequence", per_sequence),
    ])

    # --- Save results ---
    results_dir = results_file.parent
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print("Sequences: {}".format(len(records)))
    print("ViennaRNA predictions: {}".format(viennarna_count))
    print("SHAPE-constrained predictions: {}".format(viennarna_shape_count))
    print("Fallback predictions: {}".format(fallback_count))
    print()
    print("Aggregate correlation (pairing prob vs SHAPE reactivity):")
    print("  Pearson r:  {:.4f}".format(agg_pearson))
    print("  Spearman r: {:.4f}".format(agg_spearman))
    print()
    print("Aggregate classification (paired/unpaired):")
    print("  Accuracy:    {:.2%}".format(agg_class["accuracy"]))
    print("  Sensitivity: {:.2%}".format(agg_class["sensitivity"]))
    print("  Specificity: {:.2%}".format(agg_class["specificity"]))
    print("  Precision:   {:.2%}".format(agg_class["precision"]))
    print("  MCC:         {:.4f}".format(agg_class["mcc"]))
    print()
    print("Per-sequence averages:")
    print("  Pearson r:  {:.4f} +/- {:.4f}".format(pearson_mean, pearson_std))
    print("  Spearman r: {:.4f} +/- {:.4f}".format(spearman_mean, spearman_std))
    print("  Accuracy:   {:.2%} +/- {:.2%}".format(acc_mean, acc_std))
    print("  MCC:        {:.4f} +/- {:.4f}".format(mcc_mean, mcc_std))
    print()
    print("Elapsed: {:.2f}s".format(elapsed))
    print("Results saved to: {}".format(results_file))

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = run_benchmark(SHAPE_CSV, RESULTS_FILE)

    # Print interpretation note
    print()
    print("INTERPRETATION NOTE:")
    print("  Negative Pearson/Spearman r indicates that predicted pairing")
    print("  correlates inversely with SHAPE reactivity (expected: paired")
    print("  nucleotides have LOW SHAPE, so pairing prob should be HIGH when")
    print("  SHAPE is LOW). A strongly negative correlation validates the")
    print("  structure prediction against SHAPE experimental data.")