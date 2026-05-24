"""
multimodal_weight_fitting.py - Train and fit multi-modal weights.

Similar to MOERegressor's inverse RMSE weighting approach.

Weights are trained based on:
- Cross-validation performance of each modality
- Correlation with ground truth immunogenicity
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, roc_auc_score

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@dataclass
class ModalityWeights:
    """Trained weights for each modality."""

    sequence: float = 0.35
    innate: float = 0.25
    dose: float = 0.15
    ctm: float = 0.10
    admet: float = 0.10
    gene: float = 0.05

    # Performance metrics
    metrics: Dict = None

    def to_dict(self) -> Dict:
        return {
            'sequence': self.sequence,
            'innate': self.innate,
            'dose': self.dose,
            'ctm': self.ctm,
            'admet': self.admet,
            'gene': self.gene,
        }

    def normalize(self) -> 'ModalityWeights':
        """Normalize weights to sum to 1."""
        total = self.sequence + self.innate + self.dose + self.ctm + self.admet + self.gene
        if total > 0:
            self.sequence /= total
            self.innate /= total
            self.dose /= total
            self.ctm /= total
            self.admet /= total
            self.gene /= total
        return self


def compute_modality_scores(
    sequence: str,
    dose: float = 100.0,
    gene_signature: Dict = None,
) -> Dict:
    """
    Compute individual modality scores for a sequence.

    Returns raw scores from each modality (not weighted).
    """
    try:
        from core.multimodal_predictor import MultiModalCircRNAPredictor
        from core.features import CircRNAFeatureExtractor
        from core.innate_immune import quick_predict
        from core.dose_tox import quick_dose_predict
        from core.circrna_ctm import simulate_circrna_ctm
        from core.admet import quick_admet
    except ImportError:
        from confluencia_circrna_encoder.core.multimodal_predictor import MultiModalCircRNAPredictor
        from confluencia_circrna_encoder.core.features import CircRNAFeatureExtractor
        from confluencia_circrna_encoder.core.innate_immune import quick_predict
        from confluencia_circrna_encoder.core.dose_tox import quick_dose_predict
        from confluencia_circrna_encoder.core.circrna_ctm import simulate_circrna_ctm
        from confluencia_circrna_encoder.core.admet import quick_admet

    # Sequence score (from features)
    extractor = CircRNAFeatureExtractor()
    features = extractor.extract(sequence)
    gc = features[4]  # GC content
    entropy = features[7]  # Entropy
    seq_score = gc * 0.4 + entropy / 2 * 0.3 + np.random.uniform(0, 0.3)

    # Innate immune score
    innate = quick_predict(sequence)
    innate_score = innate['overall_score']

    # Dose response score
    dose_resp = quick_dose_predict(sequence, dose)
    dose_score = dose_resp['therapeutic_window']

    # CTM score
    ctm = simulate_circrna_ctm(sequence, dose)
    ctm_score = ctm['max_effect']

    # ADMET score
    admet = quick_admet(sequence)
    admet_score = 1.0 if admet['pass'] else 0.5

    # Gene score (if provided)
    if gene_signature:
        target_genes = ['TROP2', 'NECTIN4', 'LIV-1', 'B7-H4', 'MKI67']
        gene_scores = [gene_signature.get(g, 0.5) for g in target_genes]
        gene_score = np.mean(gene_scores)
    else:
        gene_score = 0.5

    return {
        'sequence': seq_score,
        'innate': innate_score,
        'dose': dose_score,
        'ctm': ctm_score,
        'admet': admet_score,
        'gene': gene_score,
    }


def fit_modality_weights(
    sequences: List[str],
    labels: np.ndarray,
    doses: Optional[np.ndarray] = None,
    n_folds: int = 5,
    random_state: int = 42,
) -> ModalityWeights:
    """
    Fit multi-modal weights using cross-validation.

    Similar to MOERegressor's inverse RMSE weighting:
    - Compute each modality's prediction on OOF data
    - Weight = 1/RMSE (inverse weighting)
    - Normalize weights

    Args:
        sequences: List of circRNA sequences
        labels: Ground truth immunogenicity labels (0/1 or continuous)
        doses: Dose values for each sequence (optional)
        n_folds: Number of CV folds
        random_state: Random seed

    Returns:
        ModalityWeights with fitted weights
    """
    n = len(sequences)
    doses = doses or np.full(n, 100.0)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Store OOF predictions for each modality
    oof_sequence = np.zeros(n)
    oof_innate = np.zeros(n)
    oof_dose = np.zeros(n)
    oof_ctm = np.zeros(n)
    oof_admet = np.zeros(n)

    print("Fitting multi-modal weights...")

    for train_idx, val_idx in kf.split(sequences):
        # Compute modality scores for validation set
        for i in val_idx:
            scores = compute_modality_scores(sequences[i], doses[i])
            oof_sequence[i] = scores['sequence']
            oof_innate[i] = scores['innate']
            oof_dose[i] = scores['dose']
            oof_ctm[i] = scores['ctm']
            oof_admet[i] = scores['admet']

    # Calculate RMSE for each modality
    rmse_sequence = np.sqrt(mean_squared_error(labels, oof_sequence))
    rmse_innate = np.sqrt(mean_squared_error(labels, oof_innate))
    rmse_dose = np.sqrt(mean_squared_error(labels, oof_dose))
    rmse_ctm = np.sqrt(mean_squared_error(labels, oof_ctm))
    rmse_admet = np.sqrt(mean_squared_error(labels, oof_admet))

    # AUC for binary labels
    if len(np.unique(labels)) == 2:
        auc_sequence = roc_auc_score(labels, oof_sequence)
        auc_innate = roc_auc_score(labels, oof_innate)
        auc_dose = roc_auc_score(labels, oof_dose)
        auc_ctm = roc_auc_score(labels, oof_ctm)
        auc_admet = roc_auc_score(labels, oof_admet)
    else:
        auc_sequence = auc_innate = auc_dose = auc_ctm = auc_admet = 0.5

    print(f"  Sequence RMSE: {rmse_sequence:.4f}, AUC: {auc_sequence:.4f}")
    print(f"  Innate RMSE: {rmse_innate:.4f}, AUC: {auc_innate:.4f}")
    print(f"  Dose RMSE: {rmse_dose:.4f}, AUC: {auc_dose:.4f}")
    print(f"  CTM RMSE: {rmse_ctm:.4f}, AUC: {auc_ctm:.4f}")
    print(f"  ADMET RMSE: {rmse_admet:.4f}, AUC: {auc_admet:.4f}")

    # Inverse weighting (like MOERegressor)
    rmse_dict = {
        'sequence': rmse_sequence,
        'innate': rmse_innate,
        'dose': rmse_dose,
        'ctm': rmse_ctm,
        'admet': rmse_admet,
        'gene': 0.5,  # Fixed for gene (no data to fit)
    }

    inv_weights = {k: 1.0 / max(v, 1e-6) for k, v in rmse_dict.items()}
    total = sum(inv_weights.values())
    weights = {k: v / total for k, v in inv_weights.items()}

    print(f"\n  Fitted weights:")
    for k, v in weights.items():
        print(f"    {k}: {v:.4f}")

    return ModalityWeights(
        sequence=weights['sequence'],
        innate=weights['innate'],
        dose=weights['dose'],
        ctm=weights['ctm'],
        admet=weights['admet'],
        gene=weights['gene'],
        metrics={
            'rmse': rmse_dict,
            'auc': {
                'sequence': auc_sequence,
                'innate': auc_innate,
                'dose': auc_dose,
                'ctm': auc_ctm,
                'admet': auc_admet,
            },
        },
    )


def fit_weights_from_csv(
    csv_path: str,
    n_samples: int = 5000,
    n_folds: int = 5,
) -> ModalityWeights:
    """
    Fit weights from training data CSV.

    Args:
        csv_path: Path to unified_training_data.csv
        n_samples: Number of samples to use (for speed)
        n_folds: CV folds

    Returns:
        Fitted ModalityWeights
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    if n_samples < len(df):
        df = df.sample(n=n_samples, random_state=42)

    sequences = df['sequence'].tolist()
    labels = df['orig_immunogenicity'].values

    print(f"Using {len(sequences)} samples")

    weights = fit_modality_weights(sequences, labels, n_folds=n_folds)

    return weights


def save_weights(weights: ModalityWeights, output_path: str):
    """Save fitted weights to JSON."""
    import json

    data = {
        'weights': weights.to_dict(),
        'metrics': weights.metrics,
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved weights to {output_path}")


def load_weights(weights_path: str) -> ModalityWeights:
    """Load weights from JSON."""
    import json

    with open(weights_path, 'r') as f:
        data = json.load(f)

    return ModalityWeights(
        sequence=data['weights']['sequence'],
        innate=data['weights']['innate'],
        dose=data['weights']['dose'],
        ctm=data['weights']['ctm'],
        admet=data['weights']['admet'],
        gene=data['weights']['gene'],
        metrics=data.get('metrics'),
    )