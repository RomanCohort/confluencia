#!/usr/bin/env python3
"""quick_validation.py - Phase 1快速验证（简单Transformer）"""

import torch
import torch.nn as nn

class QuickValidationModel(nn.Module):
    """简单Transformer用于快速验证数据质量"""
    def __init__(self, d_model=256):
        super().__init__()
        self.seq_embedder = nn.Embedding(4, 64)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8), num_layers=6
        )
        self.coord_decoder = nn.Linear(d_model, 3)
        self.bsj_predictor = nn.Linear(d_model, 1)

    def forward(self, seq, ss_features):
        encoded = self.transformer(self.seq_embedder(seq))
        coords = self.coord_decoder(encoded)
        bsj_dist = self.bsj_predictor(encoded.mean(dim=1))
        return {'coords': coords, 'bsj_distance': bsj_dist}