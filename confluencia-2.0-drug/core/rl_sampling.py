from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


class AtomPolicyNet(nn.Module):
    """Simple policy network that scores atoms given node embeddings.

    Input: node embeddings (n, d)
    Output: logits (n,)
    """

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, node_emb: torch.Tensor) -> torch.Tensor:
        # returns (n,) logits
        if node_emb.numel() == 0:
            return torch.zeros((0,), device=node_emb.device)
        logits = self.net(node_emb).squeeze(-1)
        return logits


def sample_atoms(policy: AtomPolicyNet, node_emb: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample k atom indices using policy (categorical over nodes).

    Returns (indices, log_probs)
    """
    logits = policy(node_emb)  # (n,)
    probs = torch.softmax(logits, dim=0)
    n = logits.shape[0]
    if n == 0:
        return torch.tensor([], dtype=torch.long), torch.tensor([])
    m = torch.distributions.Categorical(probs)
    idx = m.sample((k,)) if k > 1 else m.sample()
    if isinstance(idx, torch.Tensor) and idx.dim() == 0:
        idx = idx.unsqueeze(0)
    # compute log_probs for sampled indices
    logp = m.log_prob(idx)
    return idx, logp


def reinforce_update(
    optimizer: optim.Optimizer,
    log_probs: torch.Tensor,
    rewards: torch.Tensor,
    baseline: float = 0.0,
    penalty: torch.Tensor | None = None,
    return_metrics: bool = False,
):
    """Single-step REINFORCE update: maximize expected reward.

    `log_probs` shape: (k,) or (batch, k)
    `rewards` shape: (k,) or (batch, k)
    `penalty` shape: (k,) or (batch, k)
    """
    adv = rewards - baseline
    if penalty is not None:
        adv = adv - penalty
    loss = -(log_probs * adv).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if not return_metrics:
        return float(loss.item())

    metrics = {
        "loss": float(loss.item()),
        "reward_mean": float(rewards.mean().item()) if rewards.numel() else 0.0,
        "penalty_mean": float(penalty.mean().item()) if penalty is not None and penalty.numel() else 0.0,
        "baseline": float(baseline),
        "adv_mean": float(adv.mean().item()) if adv.numel() else 0.0,
    }
    return metrics


__all__ = ["AtomPolicyNet", "sample_atoms", "reinforce_update"]
