"""
CircRNA Predictor — High-level prediction interface.

Mirrors drug module's predictor.py structure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from .encoder import CircRNAEncoder, CircRNAEncoderConfig

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CircRNAPredictor:
    """
    High-level predictor for circRNA (mirrors drug.DrugPredictor).

    Usage:
        predictor = CircRNAPredictor(model_path="model.pt")
        result = predictor.predict(sequence="AUG...", gene_expr={"TROP2": 7.2})
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        config: Optional[CircRNAEncoderConfig] = None,
    ):
        self.device = device
        self.config = config or CircRNAEncoderConfig()
        self.encoder = CircRNAEncoder(self.config)

        if model_path and Path(model_path).exists():
            self.encoder.load(model_path, device=device)
            print(f"[CircRNAPredictor] Loaded weights from {model_path}")

        try:
            self.encoder.load_backbone(device=device)
        except Exception as e:
            print(f"[CircRNAPredictor] Warning: backbone not loaded: {e}")

    def predict(
        self,
        sequence: str,
        gene_expr: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Single sequence prediction."""
        if gene_expr is None:
            gene_expr = {
                "TROP2": 7.2, "NECTIN4": 5.1, "LIV-1": 3.5,
                "B7-H4": 6.0, "MKI67": 8.0, "MYC": 4.5,
            }
        return self.encoder.predict_single(sequence, gene_expr, self.device)

    def predict_batch(
        self,
        sequences: List[str],
        gene_expr: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Batch prediction."""
        if gene_expr is None:
            default_expr = {g: 0.5 for g in self.config.gene_cols}
            gene_values = [list(default_expr.values()) for _ in sequences]
        else:
            gene_values = gene_expr[self.config.gene_cols].values.tolist()

        gene_tensor = torch.tensor(gene_values, dtype=torch.float32)
        outputs = self.encoder.forward(sequences, gene_tensor, self.device)

        results = []
        for i in range(len(sequences)):
            row = {"sequence_id": i}
            for j, k in enumerate(self.config.composite_keys):
                row[k] = outputs["composite"][i, j].item()
            for j, k in enumerate(self.config.report_keys):
                row[k] = outputs["report"][i, j].item()
            for j, c in enumerate(self.config.response_classes):
                row[f"prob_{c}"] = outputs["response_probs"][i, j].item()
            row["predicted_response"] = self.config.response_classes[
                outputs["response_probs"][i].argmax().item()
            ]
            results.append(row)

        return pd.DataFrame(results)

    def get_summary(self, prediction: Dict) -> Dict[str, str]:
        """Get human-readable summary."""
        imm = prediction.get("overall_immunogenicity", 0.5)
        level = "高" if imm >= 0.6 else ("中" if imm >= 0.4 else "低")

        return {
            "免疫原性水平": level,
            "IPS评分": f"{prediction.get('ips', 0):.1f}/10",
            "预测应答": prediction.get("predicted_response", "intermediate"),
            "肿瘤杀伤指数": f"{prediction.get('tumor_killing_index', 0):.2f}",
            "治疗窗口": f"{prediction.get('therapeutic_window', 0):.2f}",
        }


def quick_predict(sequence: str, **kwargs) -> Dict[str, float]:
    """Quick prediction convenience function."""
    predictor = CircRNAPredictor(**kwargs)
    return predictor.predict(sequence)