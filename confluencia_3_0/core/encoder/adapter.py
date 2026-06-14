"""
adapter.py — Adapter wrapping CircRNASequenceEncoder for JointScoringEngine.

Provides a from_pretrained() class method that loads a trained encoder
and produces 13-key dicts compatible with _score_circrna(), enabling
seamless integration without modifying the scoring engine.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from confluencia_circrna.encoder.config import CircRNAEncoderConfig
from confluencia_circrna.encoder.model import CircRNASequenceEncoder


class CircRNAEncoderAdapter:
    """Adapter that makes CircRNASequenceEncoder compatible with JointScoringEngine.

    Usage
    -----
    >>> adapter = CircRNAEncoderAdapter.from_pretrained("output/circrna_encoder/best.pt")
    >>> cr_input = adapter.predict(
    ...     sequence="AUCGAUCG...",
    ...     gene_expr={"TROP2": 7.2, "NECTIN4": 5.1, "LIV-1": 3.8, "B7-H4": 6.0},
    ... )
    >>> # cr_input is a 13-key dict → feed to JointScoringEngine.score()
    """

    def __init__(self, model: CircRNASequenceEncoder, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        device: Optional[str] = None,
        config_override: Optional[Dict] = None,
    ) -> "CircRNAEncoderAdapter":
        """Load a trained encoder from checkpoint.

        Parameters
        ----------
        checkpoint_path : str
            Path to the .pt checkpoint file (saved by train_circrna_encoder.py).
        config_override : dict, optional
            Override config parameters.

        Returns
        -------
        CircRNAEncoderAdapter
        """
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint_dir = Path(checkpoint_path).parent

        # Load config
        config_path = checkpoint_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
        else:
            config_dict = {}
            warnings.warn(f"config.json not found at {config_path}, using defaults")

        if config_override:
            config_dict.update(config_override)

        config = CircRNAEncoderConfig(**{
            k: v for k, v in config_dict.items()
            if k in CircRNAEncoderConfig.__dataclass_fields__
        })

        # Build model
        model = CircRNASequenceEncoder(config)

        # Load weights
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)
        elif "state_dict" in state:
            model.load_state_dict(state["state_dict"], strict=False)
        else:
            model.load_state_dict(state, strict=False)

        print(f"[CircRNAEncoder] Loaded from {checkpoint_path}")
        print(f"  Device: {device}")
        print(f"  RNA-FM: {config.rna_fm_model}")

        return cls(model, device)

    def predict(
        self,
        sequence: str,
        gene_expr: Dict[str, float],
    ) -> Dict[str, float]:
        """Predict 13-key scoring dict from circRNA sequence + gene expression.

        Parameters
        ----------
        sequence : str
            circRNA nucleotide sequence (A/U/C/G).
        gene_expr : dict
            {gene_name: expression_value} for TROP2, NECTIN4, LIV-1, B7-H4, MKI67, MYC.

        Returns
        -------
        dict[str, float]
            13-key dict compatible with _score_circrna().
        """
        self.model.eval()
        result = self.model.predict_dict(sequence, gene_expr, device=self.device)

        # Validate output keys
        expected = set(self.model.config.composite_keys + self.model.config.report_keys)
        missing = expected - set(result.keys())
        if missing:
            warnings.warn(f"Missing keys in encoder output: {missing}")

        return result

    def predict_batch(
        self,
        sequences,
        gene_exprs,
    ):
        """Batch prediction for multiple circRNA sequences.

        Parameters
        ----------
        sequences : list[str]
            circRNA nucleotide sequences.
        gene_exprs : list[dict] or pd.DataFrame
            Gene expression for each sequence.

        Returns
        -------
        list[dict]
            List of 13-key scoring dicts.
        """
        import torch
        from confluencia_circrna.encoder.tokenizer import encode_gene_batch

        self.model.eval()
        self.model._ensure_backbone()

        # Gene expression tensor
        if hasattr(gene_exprs, "iloc"):
            # DataFrame
            gene_dicts = [
                gene_exprs.iloc[i].to_dict() for i in range(len(gene_exprs))
            ]
        else:
            gene_dicts = gene_exprs

        gene_arr = encode_gene_batch(gene_dicts, self.model.config.gene_cols)
        gene_tensor = torch.from_numpy(gene_arr).to(self.device)

        # Forward pass
        with torch.no_grad():
            output = self.model.forward(sequences, gene_tensor)

        # Convert batch to list of dicts
        results = []
        from confluencia_circrna.encoder.config import RESPONSE_CLASSES
        for i in range(len(sequences)):
            row = {}
            for key in self.model.config.composite_keys:
                row[key] = float(output[key][i])
            for key in self.model.config.report_keys:
                if key == "predicted_response":
                    idx = int(output["predicted_response_idx"][i])
                    row[key] = RESPONSE_CLASSES[idx]
                else:
                    row[key] = float(output[key][i])
            results.append(row)

        return results
