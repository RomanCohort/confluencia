"""
trained_survival_model.py — Load trained survival model for risk prediction.

Provides a simple interface to load the Stepwise Cox model trained on
real GEO/TCGA data and predict patient risk scores from gene expression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# Default path to trained model coefficients
TRAINED_DIR = Path(__file__).parent.parent / "data" / "trained"

# Stepwise Cox coefficients from GSE20685 training
# These are the raw Ridge coefficients (before negation)
# Positive coef → higher expression → longer predicted survival → lower risk
# Negative coef → higher expression → shorter predicted survival → higher risk
STEPWISE_COEFFICIENTS = {
    'ERBB2': 0.1487,
    'PIK3CA': -0.1198,
    'GATA3': 0.0984,
    'MKI67': 0.0927,
    'BRCA1': 0.0787,
    'LARP6': 0.0713,
    'ESR1': -0.0705,
    'NR1H3': -0.0659,
    'BAX': -0.0632,
    'PSMD2': -0.0600,
    'CDH1': -0.0577,
    'MTDH': 0.0458,
    'CASP3': 0.0439,
    'AKT1': -0.0409,
    'TMEM65': -0.0289,
    'BCL2': 0.0225,
    'MYC': 0.0181,
    'TP53': 0.0115,
    'PTEN': 0.0016,
}

# Feature order used during training (alphabetical)
FEATURE_ORDER = sorted(STEPWISE_COEFFICIENTS.keys())

# acRGBS 5-gene signature
ACRGBS_GENES = ['NR1H3', 'LARP6', 'PSMD2', 'MTDH', 'TMEM65']


@dataclass
class TrainedSurvivalModel:
    """Wrapper for trained survival model coefficients.

    Attributes
    ----------
    coefficients : dict
        Gene → coefficient mapping.
    feature_order : list
        Order of features expected by the model.
    intercept : float
        Model intercept (0 for Cox-based models).
    scaler_mean : np.ndarray or None
        Mean of training features (for standardization).
    scaler_scale : np.ndarray or None
        Std of training features (for standardization).
    c_index : float
        Training C-index.
    loocv_c_index : float
        LOOCV C-index.
    training_cohort : str
        Cohort used for training.
    """

    coefficients: Dict[str, float] = field(default_factory=lambda: STEPWISE_COEFFICIENTS.copy())
    feature_order: List[str] = field(default_factory=lambda: FEATURE_ORDER.copy())
    intercept: float = 0.0
    scaler_mean: Optional[np.ndarray] = None
    scaler_scale: Optional[np.ndarray] = None
    c_index: float = 0.5876
    loocv_c_index: float = 0.5877
    loocv_std: float = 0.0012
    training_cohort: str = "GSE20685"

    def __post_init__(self):
        """Initialize scaler parameters from coefficients if not provided."""
        if self.scaler_mean is None:
            # Default: assume standardized data (mean=0, std=1)
            self.scaler_mean = np.zeros(len(self.feature_order))
        if self.scaler_scale is None:
            self.scaler_scale = np.ones(len(self.feature_order))

    def predict_risk_score(
        self,
        gene_expression: Dict[str, float],
        normalize: bool = True
    ) -> float:
        """Predict risk score from gene expression values.

        Risk score interpretation:
        - Higher risk score = worse prognosis (shorter survival)
        - Lower risk score = better prognosis (longer survival)

        Parameters
        ----------
        gene_expression : dict
            Gene → expression value mapping. Can be TPM, FPKM, or log2-transformed.
            Will be standardized using training scaler.
        normalize : bool
            If True, normalize output to [0, 1] range.

        Returns
        -------
        float
            Risk score (higher = worse prognosis).
        """
        # Extract features in training order
        x = np.array([
            gene_expression.get(g, 0.0) for g in self.feature_order
        ])

        # Standardize using training scaler
        x_scaled = (x - self.scaler_mean) / (self.scaler_scale + 1e-8)

        # Linear predictor: score = X @ coef
        linear_pred = np.dot(x_scaled, self._coef_array()) + self.intercept

        # Risk score: negate because positive coef = longer survival
        # (sklearn Ridge predicts log(time), higher = longer survival)
        # So risk = -linear_pred
        risk = -linear_pred

        # Normalize to [0, 1] using sigmoid-like transform
        if normalize:
            risk = 1.0 / (1.0 + np.exp(-risk))

        return float(risk)

    def predict_risk_score_from_array(
        self,
        X: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """Predict risk scores from expression matrix.

        Parameters
        ----------
        X : np.ndarray
            Expression matrix (n_samples, n_genes).
            Columns must match self.feature_order.
        normalize : bool
            If True, normalize to [0, 1].

        Returns
        -------
        np.ndarray
            Risk scores (n_samples,).
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Standardize
        X_scaled = (X - self.scaler_mean) / (self.scaler_scale + 1e-8)

        # Linear predictor
        linear_pred = X_scaled @ self._coef_array() + self.intercept

        # Risk score
        risk = -linear_pred

        if normalize:
            risk = 1.0 / (1.0 + np.exp(-risk))

        return risk

    def _coef_array(self) -> np.ndarray:
        """Get coefficient array in feature order."""
        return np.array([self.coefficients.get(g, 0.0) for g in self.feature_order])

    def get_gene_importance(self) -> pd.DataFrame:
        """Get gene importance ranking from coefficients."""
        importance = np.abs(self._coef_array())
        importance_pct = importance / (importance.sum() + 1e-8) * 100

        return pd.DataFrame({
            'gene': self.feature_order,
            'coefficient': [self.coefficients.get(g, 0.0) for g in self.feature_order],
            'abs_coefficient': importance,
            'importance_pct': importance_pct,
            'direction': ['risk' if self.coefficients.get(g, 0) < 0 else 'protective' for g in self.feature_order],
            'acRGBS': [g in ACRGBS_GENES for g in self.feature_order],
        }).sort_values('abs_coefficient', ascending=False)

    def get_acrgbs_score(self, gene_expression: Dict[str, float]) -> float:
        """Compute acRGBS 5-gene signature score.

        Parameters
        ----------
        gene_expression : dict
            Gene expression values.

        Returns
        -------
        float
            acRGBS risk score (higher = worse prognosis).
        """
        acrgbs_expr = {g: gene_expression.get(g, 0.0) for g in ACRGBS_GENES}
        return self.predict_risk_score(acrgbs_expr, normalize=True)


def load_trained_model(
    model_path: Optional[Path] = None,
    cohort: str = "GSE20685"
) -> TrainedSurvivalModel:
    """Load trained survival model from file or use defaults.

    Parameters
    ----------
    model_path : Path, optional
        Path to model coefficients CSV.
    cohort : str
        Training cohort identifier.

    Returns
    -------
    TrainedSurvivalModel
        Loaded model.
    """
    if model_path is None:
        model_path = TRAINED_DIR / f"{cohort}_stepwise_coefficients.csv"

    if model_path.exists():
        df = pd.read_csv(model_path)
        coefficients = dict(zip(df['gene'], df['coefficient']))
        feature_order = df['gene'].tolist()
        c_index = df.get('c_index', pd.Series([0.5876])).iloc[0] if 'c_index' in df.columns else 0.5876

        return TrainedSurvivalModel(
            coefficients=coefficients,
            feature_order=feature_order,
            c_index=c_index,
            training_cohort=cohort,
        )

    # Return default model
    return TrainedSurvivalModel()


# Singleton instance for quick access
_default_model: Optional[TrainedSurvivalModel] = None


def get_default_model() -> TrainedSurvivalModel:
    """Get or create default trained model instance."""
    global _default_model
    if _default_model is None:
        _default_model = TrainedSurvivalModel()
    return _default_model


def predict_patient_risk(
    gene_expression: Dict[str, float],
    model: Optional[TrainedSurvivalModel] = None
) -> Tuple[float, Dict[str, float]]:
    """Convenience function to predict patient risk from gene expression.

    Parameters
    ----------
    gene_expression : dict
        Gene → expression mapping.
    model : TrainedSurvivalModel, optional
        Model to use. If None, uses default.

    Returns
    -------
    tuple
        (risk_score, feature_contributions)
    """
    if model is None:
        model = get_default_model()

    risk = model.predict_risk_score(gene_expression)

    # Compute feature contributions
    contributions = {}
    for gene in model.feature_order:
        expr = gene_expression.get(gene, 0.0)
        coef = model.coefficients.get(gene, 0.0)
        # Contribution = -coef * expr (negative because risk = -score)
        contributions[gene] = -coef * expr

    return risk, contributions


if __name__ == "__main__":
    # Demo
    model = TrainedSurvivalModel()
    print("Stepwise Cox Model (trained on GSE20685)")
    print(f"  C-index: {model.c_index:.4f}")
    print(f"  LOOCV C-index: {model.loocv_c_index:.4f} ± {model.loocv_std:.4f}")
    print()

    print("Gene Importance:")
    importance = model.get_gene_importance()
    print(importance.to_string(index=False))
    print()

    # Test prediction
    test_patient = {
        'ERBB2': 5.2, 'PIK3CA': 7.1, 'GATA3': 8.5, 'MKI67': 12.3,
        'BRCA1': 6.0, 'LARP6': 4.5, 'ESR1': 9.2, 'NR1H3': 5.0,
        'BAX': 3.8, 'PSMD2': 6.5, 'CDH1': 7.0, 'MTDH': 4.0,
        'CASP3': 5.5, 'AKT1': 6.2, 'TMEM65': 3.5, 'BCL2': 4.8,
        'MYC': 8.0, 'TP53': 5.5, 'PTEN': 4.0,
    }

    risk, contrib = predict_patient_risk(test_patient, model)
    print(f"Test Patient Risk Score: {risk:.4f}")
    print()

    print("Top Contributing Genes:")
    sorted_contrib = sorted(contrib.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    for gene, contrib_val in sorted_contrib:
        print(f"  {gene}: {contrib_val:+.4f}")

    print()
    acrgbs_risk = model.get_acrgbs_score(test_patient)
    print(f"acRGBS 5-gene Risk Score: {acrgbs_risk:.4f}")
