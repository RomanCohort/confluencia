"""Qualitative Data Integration Module.

Converts wet-lab qualitative results into usable model calibration data.
Supports:
- Threshold calibration (find cutoffs that match qualitative labels)
- Binary classification training
- Parameter weight adjustment
- Model validation against qualitative outcomes
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# ===================================================================
# Qualitative Label Schemas
# ===================================================================

@dataclass
class QualitativeResult:
    """Wet-lab qualitative result format."""
    sample_id: str
    # Drug ADMET qualitative
    hERG_positive: Optional[bool] = None      # True = cardiotoxic
    AMES_positive: Optional[bool] = None      # True = mutagenic
    hepatotoxic: Optional[bool] = None        # True = liver damage
    CYP_inhibitor: Optional[str] = None       # "strong"/"weak"/"none"
    BBB_permeable: Optional[bool] = None      # True = crosses BBB
    druglikeness: Optional[str] = None        # "good"/"moderate"/"poor"
    # circRNA qualitative
    immunogenic: Optional[bool] = None        # True = triggers immune response
    IFN_response: Optional[str] = None        # "strong"/"moderate"/"weak"/"none"
    stable: Optional[bool] = None             # True = half-life > 24h
    translation_active: Optional[bool] = None # True = produces protein
    # Epitope qualitative
    MHC_binding: Optional[str] = None         # "strong"/"moderate"/"weak"/"none"
    # Simulacrum clinical
    tumor_response: Optional[str] = None      # "PR"/"SD"/"PD"
    # Raw notes
    notes: str = ""


# ===================================================================
# Threshold Calibration from Qualitative Data
# ===================================================================

def calibrate_thresholds_from_qualitative(
    predictions: List[float],
    labels: List[str],
    positive_labels: List[str] = ["positive", "toxic", "strong", "yes"],
) -> Dict[str, float]:
    """Find optimal threshold that separates positive vs negative qualitative labels.

    Args:
        predictions: Model continuous predictions (0-1)
        labels: Qualitative labels from wet-lab
        positive_labels: Which labels count as "positive"

    Returns:
        Dict with optimal_threshold, accuracy, confusion_matrix
    """
    # Convert to binary
    binary_labels = [1 if lbl.lower() in [p.lower() for p in positive_labels] else 0
                     for lbl in labels]

    # Find threshold that maximizes accuracy
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_accuracy = 0

    for t in thresholds:
        predicted = [1 if p >= t else 0 for p in predictions]
        accuracy = sum(p == l for p, l in zip(predicted, binary_labels)) / len(labels)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = t

    # Compute confusion matrix at best threshold
    predicted_final = [1 if p >= best_threshold else 0 for p in predictions]
    tp = sum(p == 1 and l == 1 for p, l in zip(predicted_final, binary_labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(predicted_final, binary_labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(predicted_final, binary_labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(predicted_final, binary_labels))

    return {
        "optimal_threshold": best_threshold,
        "accuracy": best_accuracy,
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
    }


def calibrate_multiclass_thresholds(
    predictions: List[float],
    labels: List[str],
    class_order: List[str] = ["none", "weak", "moderate", "strong"],
) -> Dict[str, float]:
    """Find thresholds for multi-class qualitative labels.

    Args:
        predictions: Model continuous predictions (0-1)
        labels: Qualitative labels (e.g., "strong", "moderate", "weak", "none")
        class_order: Ordered classes from lowest to highest

    Returns:
        Dict with thresholds for each class boundary
    """
    # Map labels to numeric levels
    label_to_level = {c.lower(): i for i, c in enumerate(class_order)}
    levels = [label_to_level.get(lbl.lower(), 0) for lbl in labels]

    # Find boundaries
    thresholds = []
    for i in range(1, len(class_order)):
        # Samples at level >= i vs < i
        high_indices = [j for j, l in enumerate(levels) if l >= i]
        low_indices = [j for j, l in enumerate(levels) if l < i]

        if not high_indices or not low_indices:
            thresholds.append(i / len(class_order))
            continue

        high_preds = [predictions[j] for j in high_indices]
        low_preds = [predictions[j] for j in low_indices]

        # Threshold between max(low) and min(high)
        threshold = (max(low_preds) + min(high_preds)) / 2
        thresholds.append(threshold)

    return {
        "class_order": class_order,
        "thresholds": thresholds,
        "class_mapping": {
            class_order[0]: (0, thresholds[0] if thresholds else 0.25),
            class_order[1]: (thresholds[0] if thresholds else 0.25, thresholds[1] if len(thresholds) > 1 else 0.5),
            class_order[2]: (thresholds[1] if len(thresholds) > 1 else 0.5, thresholds[2] if len(thresholds) > 2 else 0.75),
            class_order[3]: (thresholds[2] if len(thresholds) > 2 else 0.75, 1.0),
        }
    }


# ===================================================================
# Parameter Weight Adjustment from Qualitative Patterns
# ===================================================================

def adjust_weights_from_qualitative_patterns(
    current_weights: Dict[str, float],
    samples: List[Dict],
    feature_key: str = "gc_content",
    outcome_key: str = "immunogenic",
    positive_means_higher: bool = True,
) -> Dict[str, float]:
    """Adjust heuristic weights based on qualitative pattern observations.

    Example: If high GC circRNAs are consistently "immunogenic=True",
    increase the GC-related weight parameter.

    Args:
        current_weights: Current weight dict (e.g., {"gc": 0.3, "au": 0.1})
        samples: List of dicts with features and qualitative outcomes
        feature_key: Which feature to analyze
        outcome_key: Which qualitative outcome
        positive_means_higher: True if positive outcome correlates with higher feature

    Returns:
        Adjusted weights
    """
    # Separate positive vs negative samples
    positive_samples = [s for s in samples if s.get(outcome_key, False)]
    negative_samples = [s for s in samples if not s.get(outcome_key, False)]

    if not positive_samples or not negative_samples:
        return current_weights

    # Calculate mean feature values
    pos_mean = np.mean([s.get(feature_key, 0) for s in positive_samples])
    neg_mean = np.mean([s.get(feature_key, 0) for s in negative_samples])

    # Calculate adjustment factor
    if positive_means_higher:
        factor = pos_mean / neg_mean if neg_mean > 0 else 1.5
    else:
        factor = neg_mean / pos_mean if pos_mean > 0 else 1.5

    # Apply adjustment (bounded)
    factor = np.clip(factor, 0.5, 2.0)

    adjusted = {}
    for key, val in current_weights.items():
        if key.lower().startswith(feature_key[:3]):  # Match related weight
            adjusted[key] = val * factor
        else:
            adjusted[key] = val

    return adjusted


# ===================================================================
# Binary Classification Training from Qualitative Labels
# ===================================================================

def train_binary_classifier_from_qualitative(
    features: List[List[float]],
    labels: List[str],
    positive_labels: List[str] = ["positive", "toxic", "yes"],
) -> Dict:
    """Train simple binary classifier using qualitative labels.

    Uses logistic regression - simple and interpretable.

    Args:
        features: Feature vectors (each sample's numeric features)
        labels: Qualitative labels
        positive_labels: Which labels count as positive

    Returns:
        Model coefficients and metadata
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return {"error": "sklearn not available"}

    # Convert labels
    binary_labels = [1 if lbl.lower() in [p.lower() for p in positive_labels] else 0
                     for lbl in labels]

    # Train
    X = np.array(features)
    y = np.array(binary_labels)

    if len(set(y)) < 2:
        return {"error": "Need both positive and negative samples"}

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return {
        "coefficients": model.coef_[0].tolist(),
        "intercept": model.intercept_[0],
        "classes": ["negative", "positive"],
        "feature_importance": {
            f"feature_{i}": coef for i, coef in enumerate(model.coef_[0])
        },
        "accuracy": model.score(X, y),
    }


# ===================================================================
# Model Validation Against Qualitative Outcomes
# ===================================================================

def validate_model_against_qualitative(
    model_predictions: List[float],
    qualitative_labels: List[str],
    threshold: float = 0.5,
    positive_labels: List[str] = ["positive", "toxic", "strong", "yes"],
) -> Dict:
    """Validate model predictions against qualitative wet-lab outcomes.

    Args:
        model_predictions: Continuous model outputs
        qualitative_labels: Wet-lab qualitative results
        threshold: Cutoff for positive prediction

    Returns:
        Validation metrics
    """
    predicted_binary = [1 if p >= threshold else 0 for p in model_predictions]
    actual_binary = [1 if lbl.lower() in [p.lower() for p in positive_labels] else 0
                     for lbl in qualitative_labels]

    # Metrics
    tp = sum(p == 1 and a == 1 for p, a in zip(predicted_binary, actual_binary))
    fp = sum(p == 1 and a == 0 for p, a in zip(predicted_binary, actual_binary))
    tn = sum(p == 0 and a == 0 for p, a in zip(predicted_binary, actual_binary))
    fn = sum(p == 0 and a == 1 for p, a in zip(predicted_binary, actual_binary))

    accuracy = (tp + tn) / len(qualitative_labels)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1,
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "match_rate": accuracy,
        "mismatched_samples": [
            i for i, (p, a) in enumerate(zip(predicted_binary, actual_binary)) if p != a
        ],
    }


# ===================================================================
# Qualitative Data Template for Wet-Lab Team
# ===================================================================

QUALITATIVE_DATA_TEMPLATE = """
# Wet-Lab Qualitative Data Collection Template
# Fill in for each sample tested

Sample_ID: circRNA_001
Sequence: AUGCGCGCGUAUAGCGCGCG (or Drug SMILES)

## Drug ADMET Qualitative Results
hERG_positive: [yes/no]      # Cardiotoxic at 10 μM?
AMES_positive: [yes/no]      # Mutagenic in Salmonella test?
hepatotoxic: [yes/no]       # Liver cell death observed?
CYP_inhibitor: [strong/moderate/none]  # Which CYPs inhibited?
BBB_permeable: [yes/no]     # Detected in brain?
druglikeness: [good/moderate/poor]

## circRNA Qualitative Results
immunogenic: [yes/no]           # IFN-alpha detected (>50 pg/mL)?
IFN_response: [strong/moderate/weak/none]
stable: [yes/no]                # Half-life > 24 hours?
translation_active: [yes/no]    # Luciferase/protein detected?

## Epitope Qualitative Results
MHC_binding: [strong/moderate/weak/none]

## Notes
notes: Free text observations
"""

# ===================================================================
# Example Usage
# ===================================================================

if __name__ == "__main__":
    # Example: calibrate hERG threshold
    print("=== Example: hERG Threshold Calibration ===")

    predictions = [0.1, 0.3, 0.45, 0.6, 0.75, 0.85]
    labels = ["no", "no", "weak", "yes", "yes", "strong"]

    result = calibrate_thresholds_from_qualitative(
        predictions, labels, positive_labels=["yes", "strong"]
    )
    print("Threshold:", result["optimal_threshold"])
    print("Accuracy:", result["accuracy"])
    print("Sensitivity:", result["sensitivity"])

    # Example: multi-class thresholds
    print("\n=== Example: Multi-class Thresholds ===")
    multi_result = calibrate_multiclass_thresholds(predictions, labels)
    print("Class mapping:", multi_result["class_mapping"])

    # Example: validation
    print("\n=== Example: Model Validation ===")
    val_result = validate_model_against_qualitative(predictions, labels, threshold=0.5)
    print("F1 Score:", val_result["f1_score"])
    print("Confusion Matrix:", val_result["confusion_matrix"])