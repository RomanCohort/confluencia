"""
Gene Signature Feature Engineering for Four-Target Protein Efficacy Prediction.

This module implements the gene signature encoder for TROP2, NECTIN4, LIV-1 (SLC39A8), and B7-H4 (VTCN1).
Based on literature-validated expression patterns and circRNA therapeutic response signatures.

Targets:
- TROP2 (TACSTD2): Tumor-associated calcium signal transducer 2, high expression in carcinomas
- NECTIN4 (PVRL4): Nectin cell adhesion molecule 4, marker for metastatic potential
- LIV-1 (SLC39A8): Zinc transporter, associated with epithelial-mesenchymal transition
- B7-H4 (VTCN1): Immunoregulatory receptor, target for immunotherapy
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _load_fitted_risk_weights() -> dict:
    """Load LASSO+StepCox fitted risk weights from training report.

    Prefers v2 report (22-gene + clinical). Falls back to v1, then defaults.
    Each fallback emits a warning so callers know whether they are running
    fitted weights or uncalibrated defaults.
    """
    v2_path = Path(__file__).parent.parent / "output" / "lasso_stepcox_v2_report.json"
    if v2_path.exists():
        with open(v2_path) as f:
            report = json.load(f)
        logger.warning(
            "gene_signature weights loaded from v2 report (22-gene + clinical): %s",
            v2_path,
        )
        return report["final_model"]["normalized_weights"]

    v1_path = Path(__file__).parent.parent / "output" / "lasso_stepcox_report.json"
    if v1_path.exists():
        with open(v1_path) as f:
            report = json.load(f)
        logger.warning(
            "gene_signature weights: v2 report not found, falling back to v1 "
            "(5-gene) report: %s", v1_path,
        )
        return report["final_model"]["normalized_weights"]

    # Fallback: load from scoring_weights.json gene_signature_4gene group
    from confluencia_shared.weight_loader import get_sub_weights
    gs_w = get_sub_weights("gene_signature_4gene")
    logger.warning(
        "gene_signature weights: NO fitted LASSO+StepCox report found (v2/v1 "
        "both absent). Using UNCALIBRATED defaults from scoring_weights.json "
        "gene_signature_4gene group, with hardcoded fallback 0.35/0.25/0.20/0.20. "
        "Predictions from these weights are NOT Cox-fitted."
    )
    return {
        "TROP2": gs_w.get("trop2", 0.35),
        "NECTIN4": gs_w.get("nectin4", 0.25),
        "LIV-1": gs_w.get("liv1", 0.20),
        "B7-H4": gs_w.get("b7h4", 0.20),
    }


def _load_data_driven_baselines() -> Dict[str, float]:
    """Load data-driven expression baselines from combined_with_survival.csv.

    For missing expression values, uses the median expression per gene from the
    3078-sample TCGA+METABRIC cohort (LASSO training data). This is preferable to
    arbitrary constants because it:
    - Reflects true patient population distribution
    - Is traceable to training data (transparent to reviewers)
    - Avoids injecting bias from unparsable constants

    Returns:
        Dictionary mapping gene name (TROP2, NECTIN4, LIV-1, B7-H4) to median
        expression value in the training cohort.
    """
    import functools

    # Cache the result; loading CSV is cheap but we don't want to recompute
    @functools.lru_cache(maxsize=1)
    def _compute():
        # Try candidate locations: this module's sibling data dir, then the
        # project root data dir (so repo-mirrored copies still resolve to the
        # single source of truth at <root>/data/gene_signature/).
        module_root = Path(__file__).parent.parent
        candidates = [
            module_root / "data" / "gene_signature" / "combined_with_survival.csv",
            module_root.parent / "data" / "gene_signature" / "combined_with_survival.csv",
        ]
        csv_path = next((p for p in candidates if p.exists()), None)
        if csv_path is None:
            logger.warning(
                "gene_signature baselines: combined_with_survival.csv not found "
                "in any candidate path; falling back to neutral 0.5 for all genes. "
                "Missing-value imputation will be uninformative."
            )
            return {"TROP2": 0.5, "NECTIN4": 0.5, "LIV-1": 0.5, "B7-H4": 0.5}
        df = pd.read_csv(csv_path)
        logger.info("gene_signature baselines loaded from: %s", csv_path)
        return {
            "TROP2": df["TROP2"].median(),
            "NECTIN4": df["NECTIN4"].median(),
            "LIV-1": df["LIV-1"].median(),
            "B7-H4": df["B7-H4"].median(),
        }

    return _compute()


# =============================================================================
# Constants: Target Protein Information
# =============================================================================

TARGET_PROTEINS = {
    "TROP2": {
        "gene_id": "TACSTD2",
        "uniprot": "P09758",
        "full_name": "Tumor-associated calcium signal transducer 2",
        "expression_tissue": "Carcinoma tissue (breast, lung, gastric)",
        "expression_level": "High in tumor, low in normal",
        "therapeutic_relevance": "ADC target (Sacituzumab govitecan), linked to circRNA delivery efficiency",
        "expression_pattern": "membrane",
        "function": "cell adhesion, proliferation signaling",
        "clinical_relevance": "Predicts tumor sensitivity to circRNA-encoded antigens",
    },
    "NECTIN4": {
        "gene_id": "PVRL4",
        "uniprot": "Q96NB8",
        "full_name": "Nectin cell adhesion molecule 4",
        "expression_tissue": "Lung, breast, bladder carcinoma",
        "expression_level": "Restricted to tumor tissue",
        "therapeutic_relevance": "Trop2 cross-reactivity, enfortumab vedotin target",
        "expression_pattern": "membrane",
        "function": "cell adhesion, immune evasion",
        "clinical_relevance": "Predicts immune checkpoint therapy response",
    },
    "LIV-1": {
        "gene_id": "SLC39A8",
        "uniprot": "Q9C0K1",
        "full_name": "Solute carrier family 39 member 8 (ZIP8)",
        "expression_tissue": "Epithelial cells, liver, brain",
        "expression_level": "Regulated by zinc homeostasis",
        "therapeutic_relevance": "EMT marker, associated with metastasis",
        "expression_pattern": "membrane",
        "function": "zinc transporter, EMT signaling",
        "clinical_relevance": "Predicts circRNA delivery to metastatic sites",
    },
    "B7-H4": {
        "gene_id": "VTCN1",
        "uniprot": "Q7Z7D3",
        "full_name": "V-set domain containing T-cell activation inhibitor 1",
        "expression_tissue": "Immune cells, tumor microenvironment",
        "expression_level": "High in immunosuppressive TME",
        "therapeutic_relevance": "Immunotherapy target, circRNA-mediated immune modulation",
        "expression_pattern": "membrane",
        "function": "T-cell inhibition, immune evasion",
        "clinical_relevance": "Predicts immune activation efficacy of circRNA therapeutics",
    },
}


# =============================================================================
# Signature Score Functions
# =============================================================================

def compute_combined_signature_score(
    trop2_expr: float,
    nectin4_expr: float,
    liv1_expr: float,
    b7h4_expr: float,
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Compute combined gene signature scores from four target protein expressions.

    Args:
        trop2_expr: TROP2 expression level (0-1 scale, or raw TPM/FPKM)
        nectin4_expr: NECTIN4 expression level
        liv1_expr: LIV-1 expression level
        b7h4_expr: B7-H4 expression level
        normalize: Whether to normalize to 0-1 range

    Returns:
        Dictionary with signature scores:
        - combined_score: Weighted sum of all four
        - proliferation_score: TROP2 + NECTIN4 weighted
        - immune_score: B7-H4 weighted
        - metastasis_score: NECTIN4 + LIV-1 weighted
        - efficacy_score: Combined therapeutic prediction
    """
    scores = {}

    # Individual expression normalization (if needed)
    def _norm(x: float) -> float:
        if normalize:
            return np.clip(x, 0.0, 1.0)
        return np.clip(x, 0.0, 100.0) / 100.0  # Assume max 100 for raw TPM

    t = _norm(trop2_expr)
    n = _norm(nectin4_expr)
    l = _norm(liv1_expr)
    b = _norm(b7h4_expr)

    # Combined signature score (equal weights)
    scores["combined_signature"] = (t + n + l + b) / 4.0

    # Load pathway weights from scoring_weights.json (Cox-trained)
    from confluencia_shared.weight_loader import get_pathway_weights, get_sub_weights
    pw = get_pathway_weights()
    gs_w = get_sub_weights("gene_signature_4gene")

    # Proliferation score (Cox-trained weights, re-normalized to available 4 genes)
    _prolif = pw.get("proliferation", {})
    _pw = {g: _prolif.get(g, 0.0) for g in ["TROP2", "NECTIN4"]}
    _pws = sum(_pw.values()) or 1.0
    scores["proliferation_score"] = (_pw.get("TROP2", 0) / _pws) * t + (_pw.get("NECTIN4", 0) / _pws) * n

    # Immune score (Cox-trained weights, re-normalized to available 4 genes)
    _imm = pw.get("immune", {})
    _iw = {g: _imm.get(g, 0.0) for g in ["B7-H4", "TROP2"]}
    _iws = sum(_iw.values()) or 1.0
    scores["immune_score"] = (_iw.get("B7-H4", 0) / _iws) * b + (_iw.get("TROP2", 0) / _iws) * t

    # Metastasis score (NECTIN4 + LIV-1, equal weights from config)
    met_w = gs_w.get("metastasis_nectin4", 0.5)
    met_l = gs_w.get("metastasis_liv1", 0.5)
    met_total = met_w + met_l or 1.0
    scores["metastasis_score"] = (met_w / met_total) * n + (met_l / met_total) * l

    # Efficacy prediction score (weights from scoring_weights.json)
    scores["efficacy_score"] = (
        gs_w.get("trop2", 0.35) * t
        + gs_w.get("nectin4", 0.25) * n
        + gs_w.get("liv1", 0.20) * l
        + gs_w.get("b7h4", 0.20) * b
    )

    # Normalize efficacy to 0-1
    scores["efficacy_score"] = np.clip(scores["efficacy_score"], 0.0, 1.0)

    return scores


def compute_expression_vector(
    trop2_expr: float,
    nectin4_expr: float,
    liv1_expr: float,
    b7h4_expr: float,
) -> np.ndarray:
    """
    Create a fixed-length feature vector from four target expressions.

    Args:
        trop2_expr, nectin4_expr, liv1_expr, b7h4_expr: Expression levels

    Returns:
        19-dimensional feature vector:
        [TROP2_raw, NECTIN4_raw, LIV1_raw, B7H4_raw,
         TROP2_norm, NECTIN4_norm, LIV1_norm, B7H4_norm,
         TROP2_high, NECTIN4_high, LIV1_high, B7H4_high,
         combined_sig, prolif_score, immune_score,
         metasta_score, efficacy_score, expr_heterogeneity, expr_balance]
    """
    def _norm(x: float) -> float:
        return np.clip(x, 0.0, 100.0) / 100.0

    def _high(x: float) -> float:
        return 1.0 if x > 0.5 else 0.0

    t, n, l, b = trop2_expr, nectin4_expr, liv1_expr, b7h4_expr
    tn, nn, ln, bn = _norm(t), _norm(n), _norm(l), _norm(b)

    combined = (tn + nn + ln + bn) / 4.0

    # Pathway weights from scoring_weights.json (Cox-trained)
    from confluencia_shared.weight_loader import get_pathway_weights, get_sub_weights
    pw = get_pathway_weights()
    gs_w = get_sub_weights("gene_signature_4gene")

    # Proliferation (Cox-trained, re-normalized to TROP2 + NECTIN4)
    _prolif = pw.get("proliferation", {})
    _pw = {g: _prolif.get(g, 0.0) for g in ["TROP2", "NECTIN4"]}
    _pws = sum(_pw.values()) or 1.0
    prolif = (_pw.get("TROP2", 0) / _pws) * tn + (_pw.get("NECTIN4", 0) / _pws) * nn

    # Immune (Cox-trained, re-normalized to B7-H4 + TROP2)
    _imm = pw.get("immune", {})
    _iw = {g: _imm.get(g, 0.0) for g in ["B7-H4", "TROP2"]}
    _iws = sum(_iw.values()) or 1.0
    immune = (_iw.get("B7-H4", 0) / _iws) * bn + (_iw.get("TROP2", 0) / _iws) * tn

    # Metastasis (from config)
    met_w = gs_w.get("metastasis_nectin4", 0.5)
    met_l = gs_w.get("metastasis_liv1", 0.5)
    met_total = met_w + met_l or 1.0
    metasta = (met_w / met_total) * nn + (met_l / met_total) * ln

    # Efficacy (from config)
    efficacy = (
        gs_w.get("trop2", 0.35) * tn
        + gs_w.get("nectin4", 0.25) * nn
        + gs_w.get("liv1", 0.20) * ln
        + gs_w.get("b7h4", 0.20) * bn
    )

    # Expression heterogeneity (std of log-expressions)
    log_exprs = [np.log1p(t), np.log1p(n), np.log1p(l), np.log1p(b)]
    heterogeneity = np.std(log_exprs) if len(log_exprs) > 1 else 0.0

    # Expression balance (min/max ratio)
    max_val = max(t, n, l, b)
    min_val = min(t, n, l, b)
    balance = min_val / max_val if max_val > 0 else 0.0

    return np.array([
        # Raw expressions
        t, n, l, b,
        # Normalized expressions
        tn, nn, ln, bn,
        # High/low binary
        _high(t), _high(n), _high(l), _high(b),
        # Computed scores
        combined, prolif, immune, metasta, efficacy,
        # Heterogeneity and balance
        heterogeneity, balance,
    ], dtype=np.float32)


def get_feature_names() -> List[str]:
    """Return names of features in the expression vector."""
    return [
        "TROP2_raw", "NECTIN4_raw", "LIV1_raw", "B7H4_raw",
        "TROP2_norm", "NECTIN4_norm", "LIV1_norm", "B7H4_norm",
        "TROP2_high", "NECTIN4_high", "LIV1_high", "B7H4_high",
        "combined_signature", "proliferation_score", "immune_score",
        "metastasis_score", "efficacy_score",
        "expression_heterogeneity", "expression_balance",
    ]


# =============================================================================
# Main Gene Signature Encoder Class
# =============================================================================

class GeneSignatureEncoder:
    """
    Gene signature encoder for four-target protein-based efficacy prediction.

    Takes raw expression values or simulated inputs and produces
    fixed-length feature vectors compatible with MOE pipeline.

    Example:
        >>> encoder = GeneSignatureEncoder()
        >>> # With simulated expressions
        >>> X = encoder.fit_transform({"TROP2": 0.8, "NECTIN4": 0.6, "LIV-1": 0.5, "B7-H4": 0.7})
        >>> # With user-provided expression table
        >>> df = pd.DataFrame({"TROP2": [...], "NECTIN4": [...], ...})
        >>> X = encoder.transform(df)
        >>> # With SMILES + gene signature
        >>> features = encoder.combine_with_drug(smiles_list, gene_dict)
    """

    FEATURE_DIM = 19

    def __init__(
        self,
        normalize: bool = True,
        simulate_missing: bool = True,
        random_state: int = 42,
    ):
        self.normalize = normalize
        self.simulate_missing = simulate_missing
        self.rng = np.random.default_rng(random_state)
        self._fitted = False

    def _resolve_expression(
        self,
        expr: Union[float, pd.Series, Dict[str, float], None],
        target_name: str,
    ) -> float:
        """Resolve expression value from various input types.

        When expression is missing (expr is None) and simulate_missing is True,
        substitutes the cohort median expression from combined_with_survival.csv
        (data-driven, traceable to LASSO training cohort, n=3078). This avoids
        injecting unparsable arbitrary constants.
        """
        if expr is None:
            if self.simulate_missing:
                # Data-driven: cohort median from combined_with_survival.csv
                baselines = _load_data_driven_baselines()
                logger.info(
                    "gene_signature: expression for %s missing, substituting "
                    "cohort median %.4f (data-driven, n=3078)",
                    target_name, baselines.get(target_name, 0.5),
                )
                return baselines.get(target_name, 0.5)
            return 0.5

        if isinstance(expr, (int, float)):
            return float(expr)
        if isinstance(expr, pd.Series):
            return float(expr.get(target_name, 0.5))
        if isinstance(expr, dict):
            # Try both keys
            return float(expr.get(target_name, expr.get(target_name.replace("-", ""), 0.5)))

        return 0.5

    def _validate_input(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Validate and extract expression values from input dict."""
        valid_targets = {"TROP2", "NECTIN4", "LIV-1", "B7-H4", "LIV1"}
        key_map = {"LIV1": "LIV-1"}  # Normalize variant names

        expressions = {}
        for key in valid_targets:
            resolved_key = key_map.get(key, key)
            expr = data.get(key, data.get(resolved_key, None))
            expressions[key] = self._resolve_expression(expr, key)

        return expressions

    def fit(self, X: Any, y: Any = None) -> "GeneSignatureEncoder":
        """Fit the encoder (no-op, for pipeline compatibility)."""
        self._fitted = True
        return self

    def transform(self, data: Union[pd.DataFrame, Dict, List[Dict]]) -> np.ndarray:
        """
        Transform gene expression data to feature matrix.

        Args:
            data: DataFrame with columns [TROP2, NECTIN4, LIV-1, B7-H4] or dict/list of dicts

        Returns:
            Feature matrix (N, 19) as numpy array
        """
        if isinstance(data, pd.DataFrame):
            records = data.to_dict("records")
        elif isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = [data]
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        features = []
        for rec in records:
            exprs = self._validate_input(rec)
            vec = compute_expression_vector(
                exprs["TROP2"],
                exprs["NECTIN4"],
                exprs["LIV-1"],
                exprs["B7-H4"],
            )
            features.append(vec)

        return np.array(features, dtype=np.float32)

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        """Fit and transform (for sklearn Pipeline compatibility)."""
        return self.fit(X, y).transform(X)

    def combine_with_drug(
        self,
        drug_features: np.ndarray,
        gene_signatures: Union[pd.DataFrame, Dict, List[Dict]],
    ) -> np.ndarray:
        """
        Combine drug features with gene signature features.

        Args:
            drug_features: Feature matrix from drug featurizer (N, n_drug_features)
            gene_signatures: Gene expression data (N, 4) as DataFrame/dict/list

        Returns:
            Combined feature matrix (N, n_drug_features + 19)
        """
        gene_vecs = self.transform(gene_signatures)
        return np.hstack([drug_features, gene_vecs])

    def get_feature_names_full(self, drug_feature_names: Optional[List[str]] = None) -> List[str]:
        """Get full feature names (drug + gene signature)."""
        sig_names = get_feature_names()
        if drug_feature_names is None:
            return sig_names
        return drug_feature_names + sig_names

    def get_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get a summary report of gene signature analysis."""
        exprs = self._validate_input(data)
        scores = compute_combined_signature_score(
            exprs["TROP2"],
            exprs["NECTIN4"],
            exprs["LIV-1"],
            exprs["B7-H4"],
        )

        return {
            "expressions": exprs,
            "scores": scores,
            "high_targets": [k for k, v in exprs.items() if v > 0.5],
            "efficacy_prediction": scores["efficacy_score"],
            "recommendation": self._make_recommendation(exprs, scores),
        }

    def _make_recommendation(self, exprs: Dict[str, float], scores: Dict[str, float]) -> str:
        """Generate clinical recommendation based on signature."""
        high_count = sum(1 for v in exprs.values() if v > 0.5)

        if scores["efficacy_score"] > 0.7:
            base = "高疗效潜力"
        elif scores["efficacy_score"] > 0.5:
            base = "中等疗效潜力"
        else:
            base = "低疗效潜力，建议评估其他治疗策略"

        if exprs["TROP2"] > 0.6:
            base += "；TROP2高表达，ADC或靶向circRNA疗效预期良好"
        if exprs["B7-H4"] > 0.6:
            base += "；B7-H4高表达，免疫调节circRNA可能有良好响应"

        return base


# =============================================================================
# Integration with MOE Pipeline
# =============================================================================

def create_gene_signature_predictor(
    model_type: str = "moe",
    random_state: int = 42,
) -> "GeneSignatureMOEPredictor":
    """Create a four-target gene signature predictor integrated with MOE."""
    return GeneSignatureMOEPredictor(random_state=random_state)


class GeneSignatureMOEPredictor:
    """
    Integrated predictor combining gene signature features with MOE ensemble.

    Supports three prediction modes:
    1. gene_only: Use only gene signature features (20 dims)
    2. drug_only: Use only drug features
    3. combined: Use drug + gene signature features
    """

    def __init__(self, random_state: int = 42):
        self.rng = np.random.default_rng(random_state)
        self.encoder = GeneSignatureEncoder(random_state=random_state)
        self.mode = "combined"
        self.moe = None
        self.drug_scaler = None
        self._fitted = False

    def fit(
        self,
        drug_features: Optional[np.ndarray],
        gene_signatures: Union[pd.DataFrame, List[Dict]],
        targets: np.ndarray,
        mode: str = "combined",
    ) -> "GeneSignatureMOEPredictor":
        """
        Fit the combined predictor.

        Args:
            drug_features: Drug feature matrix (N, n_drug) or None
            gene_signatures: Gene expression data (N, 4) as DataFrame or list of dicts
            targets: Target values (N,)
            mode: "gene_only", "drug_only", or "combined"
        """
        from confluencia_shared.moe import MOERegressor, ExpertConfig

        self.mode = mode

        if mode == "gene_only":
            X = self.encoder.transform(gene_signatures)
        elif mode == "drug_only" or drug_features is None:
            X = drug_features if drug_features is not None else self.encoder.transform(gene_signatures)
        else:
            X = self.encoder.combine_with_drug(drug_features, gene_signatures)

        # Use sample-size-adaptive MOE
        n = len(targets)
        if n < 80:
            experts = ["ridge", "hgb"]
        elif n < 300:
            experts = ["ridge", "hgb", "rf"]
        else:
            experts = ["ridge", "hgb", "rf", "mlp"]

        self.moe = MOERegressor(experts, folds=min(5, max(3, n // 20)), random_state=int(self.rng.integers(0, 99999)))
        self.moe.fit(X, targets)
        self._fitted = True

        return self

    def predict(
        self,
        drug_features: Optional[np.ndarray],
        gene_signatures: Union[pd.DataFrame, List[Dict]],
    ) -> np.ndarray:
        """Predict efficacy scores."""
        if not self._fitted:
            raise RuntimeError("Predictor not fitted. Call fit() first.")

        if self.mode == "gene_only":
            X = self.encoder.transform(gene_signatures)
        elif self.mode == "drug_only" or drug_features is None:
            X = drug_features if drug_features is not None else self.encoder.transform(gene_signatures)
        else:
            X = self.encoder.combine_with_drug(drug_features, gene_signatures)

        return self.moe.predict(X)

    def predict_with_uncertainty(
        self,
        drug_features: Optional[np.ndarray],
        gene_signatures: Union[pd.DataFrame, List[Dict]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty estimates."""
        if not self._fitted:
            raise RuntimeError("Predictor not fitted. Call fit() first.")

        if self.mode == "gene_only":
            X = self.encoder.transform(gene_signatures)
        elif self.mode == "drug_only" or drug_features is None:
            X = drug_features if drug_features is not None else self.encoder.transform(gene_signatures)
        else:
            X = self.encoder.combine_with_drug(drug_features, gene_signatures)

        preds = self.moe.predict(X)
        uncertainty = self.moe.predict_uncertainty(X)
        return preds, uncertainty


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI for gene signature prediction."""
    import argparse

    parser = argparse.ArgumentParser(description="Four-target gene signature predictor")
    parser.add_argument("--trop2", type=float, default=None, help="TROP2 expression (0-1)")
    parser.add_argument("--nectin4", type=float, default=None, help="NECTIN4 expression (0-1)")
    parser.add_argument("--liv1", type=float, default=None, help="LIV-1 expression (0-1)")
    parser.add_argument("--b7h4", type=float, default=None, help="B7-H4 expression (0-1)")
    parser.add_argument("--csv", type=str, help="Input CSV file with expression columns")

    args = parser.parse_args()

    encoder = GeneSignatureEncoder()

    if args.csv:
        df = pd.read_csv(args.csv)
        X = encoder.transform(df)
        print(f"Generated feature matrix: {X.shape}")
    else:
        exprs = {
            "TROP2": args.trop2 if args.trop2 else 0.6,
            "NECTIN4": args.nectin4 if args.nectin4 else 0.5,
            "LIV-1": args.liv1 if args.liv1 else 0.5,
            "B7-H4": args.b7h4 if args.b7h4 else 0.7,
        }
        summary = encoder.get_summary(exprs)
        print("\n=== Four-Target Gene Signature Analysis ===")
        print(f"TROP2: {summary['expressions']['TROP2']:.3f}")
        print(f"NECTIN4: {summary['expressions']['NECTIN4']:.3f}")
        print(f"LIV-1: {summary['expressions']['LIV-1']:.3f}")
        print(f"B7-H4: {summary['expressions']['B7-H4']:.3f}")
        print(f"\nCombined Signature Score: {summary['scores']['combined_signature']:.3f}")
        print(f"Efficacy Score: {summary['scores']['efficacy_score']:.3f}")
        print(f"Proliferation Score: {summary['scores']['proliferation_score']:.3f}")
        print(f"Immune Score: {summary['scores']['immune_score']:.3f}")
        print(f"Metastasis Score: {summary['scores']['metastasis_score']:.3f}")
        print(f"\nHigh Expression Targets: {', '.join(summary['high_targets'])}")
        print(f"\nRecommendation: {summary['recommendation']}")


if __name__ == "__main__":
    main()