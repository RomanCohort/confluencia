#!/usr/bin/env python3
"""debug_scheme4_nan.py — Step-by-step NaN diagnosis for Scheme 4."""

import os
import sys
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, PROJECT_ROOT)

from confluencia_3_0.core.circrna.torusfold.train_all_schemes import (
    load_pseudo_labels, CircRNADataset, collate_fn
)


def check_tensor(name, t):
    """Check a tensor for NaN/Inf and print stats."""
    if t is None:
        print(f"  {name}: None")
        return False
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    dtype = t.dtype
    shape = t.shape
    if has_nan or has_inf:
        print(f"  {name}: shape={shape} dtype={dtype} *** NaN={has_nan} Inf={has_inf} ***")
        return True
    if t.numel() > 0 and t.is_floating_point():
        print(f"  {name}: shape={shape} dtype={dtype} range=[{t.min().item():.4f}, {t.max().item():.4f}] mean={t.mean().item():.4f}")
    else:
        print(f"  {name}: shape={shape} dtype={dtype}")
    return False


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    labels_dir = sys.argv[1] if len(sys.argv) > 1 else "data/circrna_3d_merged"
    print(f"Loading from {labels_dir}...")
    sequences, coords_labels, pair_labels, confidence_weights, metadata = load_pseudo_labels(labels_dir)
    print(f"Loaded {len(sequences)} sequences")

    # Create dataset and batch
    ds = CircRNADataset(sequences[:4], coords_labels[:4], pair_labels[:4], confidence_weights[:4])
    batch = collate_fn([ds[i] for i in range(4)])

    seq_ids = batch['seq_ids'].to(device)
    coords_target = batch['coords'].to(device)
    pair_probs = batch.get('pair_probs', None)
    if pair_probs is not None:
        pair_probs = pair_probs.to(device)

    print(f"\n=== Input Data ===")
    check_tensor("seq_ids", seq_ids)
    check_tensor("coords_target", coords_target)
    check_tensor("pair_probs", pair_probs)
    print(f"  lengths: {batch['lengths']}")

    # Normalize
    B, L, _ = coords_target.shape
    coords_centered = coords_target - coords_target.mean(dim=1, keepdim=True)
    coords_scale = torch.norm(coords_centered, dim=(1,2), keepdim=True).clamp(min=1.0)
    coords_norm = coords_centered / coords_scale

    print(f"\n=== Normalized Target ===")
    check_tensor("coords_scale", coords_scale)
    check_tensor("coords_norm", coords_norm)

    # Now build the model and test step by step
    from confluencia_3_0.core.circrna.torusfold.circrna_diffusion import (
        CircRNADiffusionModel, CircDiffusionConfig, CircRNAGraphBuilder,
        CircRNAConditionEncoder, EGNNLayer, SinusoidalEmbedding,
    )

    config = CircDiffusionConfig(
        n_diffusion_steps=50,
        d_node=128,
        d_edge=64,
    )
    model = CircRNADiffusionModel(config).to(device)
    print(f"\nModel created: {sum(p.numel() for p in model.parameters()):,} params")

    # Step 1: Diffusion noise
    t = torch.randint(0, config.n_diffusion_steps, (B,), device=device)
    noise = torch.randn_like(coords_norm)
    alpha_bar = model.alpha_bars[t].view(B, 1, 1)
    coords_noisy = torch.sqrt(alpha_bar) * coords_norm + torch.sqrt(1 - alpha_bar) * noise

    print(f"\n=== Step 1: Diffusion Noise ===")
    check_tensor("alpha_bar", alpha_bar)
    check_tensor("noise", noise)
    check_tensor("coords_noisy", coords_noisy)

    # Step 2: Condition encoder
    cond = model.condition_encoder(seq_ids, None, 310.0, 7.4, 1.0, 150.0)
    print(f"\n=== Step 2: Condition Encoder ===")
    check_tensor("cond", cond)

    # Step 3: Time embedding
    t_emb = model.time_embed(t.float())
    print(f"\n=== Step 3: Time Embedding ===")
    check_tensor("t_emb", t_emb)

    # Step 4: Graph builder
    graph_builder = CircRNAGraphBuilder()
    edge_index, edge_types = graph_builder.build(L, pair_probs)
    edge_index = edge_index.to(device)
    edge_types = edge_types.to(device)
    print(f"\n=== Step 4: Graph Builder ===")
    print(f"  L={L}, E={edge_index.shape[1]}")
    print(f"  edge_index range: [{edge_index.min().item()}, {edge_index.max().item()}]")
    if edge_index.max().item() >= L:
        print(f"  *** ERROR: edge_index out of bounds! max={edge_index.max().item()} >= L={L}")
    check_tensor("edge_types", edge_types)

    # Step 5: Edge features
    E = edge_index.shape[1]
    edge_feat = torch.zeros(B, E, config.d_edge, device=device)
    for et in range(2):
        mask = (edge_types == et)
        if mask.any():
            edge_feat[:, mask, et] = 1.0
    src, dst = edge_index[0], edge_index[1]
    dist = torch.norm(coords_noisy[:, src] - coords_noisy[:, dst], dim=-1, keepdim=True)
    if config.d_edge > 2:
        edge_feat[:, :, 2] = (dist.squeeze(-1) / 20.0).clamp(-5, 5)

    print(f"\n=== Step 5: Edge Features ===")
    check_tensor("dist", dist)
    check_tensor("edge_feat", edge_feat)

    # Step 6: EGNN layers one by one
    node_feat = cond + t_emb.unsqueeze(1)
    coords = coords_noisy.clone()
    print(f"\n=== Step 6: EGNN Layers ===")
    check_tensor("node_feat (input)", node_feat)
    check_tensor("coords (input)", coords)

    for i, layer in enumerate(model.egnn_layers):
        node_feat_new, coords_new = layer(node_feat, coords, edge_index, edge_feat)
        has_nan = check_tensor(f"node_feat (layer {i})", node_feat_new)
        has_nan2 = check_tensor(f"coords (layer {i})", coords_new)
        node_feat = node_feat_new
        coords = coords_new
        if has_nan or has_nan2:
            print(f"  *** NaN detected after EGNN layer {i}! Breaking. ***")
            # Diagnose further
            print(f"\n  Layer {i} detailed diagnosis:")
            src_idx, dst_idx = edge_index[0], edge_index[1]
            h_src = node_feat[:, src_idx]
            h_dst = node_feat[:, dst_idx]
            x_src = coords[:, src_idx]
            x_dst = coords[:, dst_idx]
            rel_coords = x_src - x_dst
            check_tensor("  h_src", h_src)
            check_tensor("  h_dst", h_dst)
            check_tensor("  x_src", x_src)
            check_tensor("  x_dst", x_dst)
            check_tensor("  rel_coords", rel_coords)
            d = torch.norm(rel_coords, dim=-1, keepdim=True).clamp(min=1e-6)
            check_tensor("  dist (clamped)", d)
            msg_input = torch.cat([h_src, h_dst, edge_feat, d], dim=-1)
            check_tensor("  msg_input", msg_input)
            messages = layer.message_mlp(msg_input)
            check_tensor("  messages", messages)
            coord_weight = layer.coord_mlp(messages)
            check_tensor("  coord_weight", coord_weight)
            coord_update = coord_weight * rel_coords
            check_tensor("  coord_update", coord_update)
            break

    # Step 7: Final projection
    if not (torch.isnan(node_feat).any() or torch.isnan(coords).any()):
        displacement = model.coord_proj(node_feat)
        print(f"\n=== Step 7: Displacement ===")
        check_tensor("displacement", displacement)

        noise_pred = displacement
        print(f"\n=== Step 8: Loss ===")
        noise_loss = torch.nn.functional.mse_loss(noise_pred, noise)
        print(f"  noise_loss = {noise_loss.item():.6f}")
        check_tensor("noise_loss", noise_loss)


if __name__ == "__main__":
    main()
