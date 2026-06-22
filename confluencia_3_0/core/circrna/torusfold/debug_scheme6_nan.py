#!/usr/bin/env python3
"""Diagnose Scheme 6 NaN issue - step by step check."""

import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from confluencia_3_0.core.circrna.torusfold.gnn_latent_diffusion import (
    GNNLatentDiffusionModel, GNNLatentConfig
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

config = GNNLatentConfig(n_diffusion_steps=10, d_node=64)
model = GNNLatentDiffusionModel(config).to(device)

# Fake input
B, L = 2, 50
seq_tokens = torch.randint(0, 5, (B, L), device=device)

print("\n=== Step 1: Encoder ===")
with torch.no_grad():
    latent_cond = model.encoder(seq_tokens)
print(f"  Shape: {latent_cond.shape}")
print(f"  NaN: {torch.isnan(latent_cond).any().item()}")
print(f"  Inf: {torch.isinf(latent_cond).any().item()}")
print(f"  Range: [{latent_cond.min().item():.4f}, {latent_cond.max().item():.4f}]")

print("\n=== Step 2: Diffusion train step ===")
with torch.no_grad():
    diff_out = model.diffusion(latent_cond, mode='train')
print(f"  Type: {type(diff_out)}")
if isinstance(diff_out, dict):
    diff_loss = diff_out['loss']
    latent_pred = diff_out['latent_pred']
    print(f"  Loss: {diff_loss.item():.6f}")
    print(f"  Loss NaN: {torch.isnan(diff_loss).any().item()}")
    print(f"  latent_pred shape: {latent_pred.shape}")
    print(f"  latent_pred NaN: {torch.isnan(latent_pred).any().item()}")
    print(f"  latent_pred range: [{latent_pred.min().item():.4f}, {latent_pred.max().item():.4f}]")

print("\n=== Step 3: Decoder (with clean latent) ===")
with torch.no_grad():
    coords = model.decoder(latent_cond, seq_tokens)
print(f"  Shape: {coords.shape}")
print(f"  NaN: {torch.isnan(coords).any().item()}")
print(f"  Inf: {torch.isinf(coords).any().item()}")
print(f"  Range: [{coords.min().item():.4f}, {coords.max().item():.4f}]")

print("\n=== Step 4: Full forward (train mode) ===")
model.train()
out = model(seq_tokens, mode='train')
print(f"  Keys: {list(out.keys())}")
print(f"  Coords NaN: {torch.isnan(out['coords']).any().item()}")
print(f"  Coords range: [{out['coords'].min().item():.4f}, {out['coords'].max().item():.4f}]")
diff_loss = out.get('diffusion_loss', None)
if diff_loss is not None:
    print(f"  Diff loss: {diff_loss.item():.6f}")
    print(f"  Diff loss NaN: {torch.isnan(diff_loss).any().item()}")

print("\n=== Step 5: Compute training loss ===")
target = torch.randn(B, L, 3, device=device)
target_centered = target - target.mean(dim=1, keepdim=True)
target_scale = torch.norm(target_centered, dim=(1,2), keepdim=True).clamp(min=1.0)
target_norm = target_centered / target_scale

pred = out['coords']
pred_centered = pred - pred.mean(dim=1, keepdim=True)
pred_norm = pred_centered / target_scale

coord_loss = torch.mean((pred_norm - target_norm) ** 2)
print(f"  Coord loss: {coord_loss.item():.6f}")
print(f"  Coord loss NaN: {torch.isnan(coord_loss).any().item()}")

total_loss = diff_loss + 0.1 * coord_loss if diff_loss is not None else coord_loss
print(f"  Total loss: {total_loss.item():.6f}")
print(f"  Total loss NaN: {torch.isnan(total_loss).any().item()}")

print("\n=== Step 6: Backward ===")
total_loss.backward()
n_nan_grads = 0
for name, p in model.named_parameters():
    if p.grad is not None and torch.isnan(p.grad).any():
        n_nan_grads += 1
        print(f"  NaN grad: {name}")
print(f"  Total params with NaN grad: {n_nan_grads}")

print("\n=== DONE ===")
