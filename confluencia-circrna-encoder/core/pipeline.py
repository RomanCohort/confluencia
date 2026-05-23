"""
CircRNA Pipeline — End-to-end processing pipeline.

Mirrors drug module's pipeline.py structure.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .encoder import CircRNAEncoder, CircRNAEncoderConfig
from .predictor import CircRNAPredictor
from .scoring import CompositeScorer, ReportScorer

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CircRNAPipeline:
    """
    End-to-end circRNA analysis pipeline.

    Steps (mirrors drug pipeline):
    1. Load sequence data
    2. Encode sequences (RNA-FM)
    3. Generate predictions
    4. Apply scoring rules
    5. Generate report
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.predictor = CircRNAPredictor(model_path=model_path, device=device)
        self.composite_scorer = CompositeScorer()
        self.report_scorer = ReportScorer()

    def run(
        self,
        sequences: List[str],
        gene_expr: Optional[Dict] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Run full pipeline."""
        if verbose:
            print(f"[Pipeline] Processing {len(sequences)} sequences...")

        # Step 1: Predict
        t0 = time.time()
        predictions = self.predictor.predict_batch(sequences, gene_expr)
        t1 = time.time()
        if verbose:
            print(f"[Pipeline] Prediction: {t1-t0:.2f}s")

        # Step 2: Apply scoring
        predictions = self.composite_scorer.apply(predictions)
        predictions = self.report_scorer.apply(predictions)

        # Step 3: Summary
        predictions["summary"] = predictions.apply(
            lambda row: self._make_summary(row), axis=1
        )

        if verbose:
            print(f"[Pipeline] Complete: {len(predictions)} results")

        return predictions

    def _make_summary(self, row: pd.Series) -> str:
        """Generate summary string."""
        imm = row.get("overall_immunogenicity", 0.5)
        level = "高" if imm >= 0.6 else ("中" if imm >= 0.4 else "低")
        response = row.get("predicted_response", "intermediate")

        return f"免疫原性: {level} ({imm:.2f}), 应答: {response}"

    def run_from_csv(
        self,
        csv_path: str,
        sequence_col: str = "sequence",
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Run from CSV file."""
        df = pd.read_csv(csv_path)
        sequences = df[sequence_col].tolist()

        gene_cols = [c for c in self.predictor.config.gene_cols if c in df.columns]
        gene_expr = df[gene_cols] if gene_cols else None

        results = self.run(sequences, gene_expr)

        if output_path:
            results.to_csv(output_path, index=False)

        return results