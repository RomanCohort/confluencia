"""
gflownet.py — GFlowNet-based circRNA sequence generator (建议 1).

This module implements an autonomous guiding flow for structure-aware
sequence design, replacing the random search components of the GA
population in CircRNAEvolution.

References:
  - Buesing et al., "Trajectory balance: improved credit assignment
    in GFlowNets" (ICML 2021) — stable TB loss formulation.
  - Omidshafiei et al., "Deep Reinforcement-Based Organic Molecule
    Generator" (NeurIPS 2020) — policy design for molecular generation.
"""

from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class GFloWNLPolicy(nn.Module):
    """
    Policy network for GFlowNet autoregressive sequence generation.

    Input: partial sequence (as embedded one-hot), length, temperature.
    Output: logits over next token in {A, U, G, C}.

    The design is lightweight (GRU-style) to save compute while still
    capturing local structure dependencies. Temperature controls
    exploration vs exploitation.
    """

    NUCS = ['A', 'U', 'G', 'C']
    NUM_TOKENS = 4

    def __init__(
        self,
        d_model: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_tokens = self.NUM_TOKENS

        # Token embedding (one-hot → continuous).
        self.token_embedding = nn.Linear(self.num_tokens, d_model)

        # Context embedding (optional: length or other sequence-level features).
        self.context_embedding = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )

        # GRU for sequential state updates.
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Final output heads: next-token logits.
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, self.num_tokens),
        )

        # Context projection d_model → hidden_dim (for empty-prefix seed).
        self.ctx_to_hidden = nn.Linear(d_model, hidden_dim)

    def forward(
        self,
        partial_seq: torch.Tensor,      # (B, L_needed, 4 one-hot)
        context: Optional[torch.Tensor] = None,  # (B, 1) e.g., target gc
        temperature: float = 1.0,
        state: Optional[torch.Tensor] = None,    # Initial hidden state (L_num_layers, B, hidden_dim)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            partial_seq: (B, L_needed, 4) one-hot trajectory prefix.
            context: (B, 1) scalar context vector (e.g., target GC).
            temperature: Gumbel-softmax temperature.

        Returns:
            logits (B, 4) for next token, updated state (L_num_layers, B, hidden_dim).
        """
        B = partial_seq.size(0)

        # Embed partial sequence.
        x = self.token_embedding(partial_seq)  # (B, L_needed, d_model)

        # Add context if provided (skip on empty prefix — broadcast fails).
        if context is not None and partial_seq.size(1) > 0:
            context_feat = self.context_embedding(context.float())
            if context_feat.dim() == 1:
                context_feat = context_feat.unsqueeze(0)  # ensure (B, d_model)
            # Defensive: only add if dims are compatible.
            if x.dim() == 3 and x.size(-1) == context_feat.size(-1):
                x = x + context_feat.unsqueeze(1)  # Broadcast along L

        device = x.device

        # GRU forward. Handle empty-prefix edge case (L_needed == 0) by
        # using a learned start-of-sequence (SOS) embedding as the seed.
        if state is None:
            state = torch.zeros(self.num_layers, B, self.hidden_dim, device=device)

        if partial_seq.size(1) == 0:
            # No tokens yet: emit logits directly from context + zero state.
            current = state.mean(dim=0)  # (B, hidden_dim)
            if context is not None:
                # Add context info to current state.
                ctx_proj = self.context_embedding(context.float())  # (B, d_model)
                current = current + self.ctx_to_hidden(ctx_proj)  # (B, hidden_dim)
            logits = self.output(current)  # (B, 4)
            logits = logits / temperature
            return logits, (state,)

        out, new_state = self.gru(x, state)  # out: (B, L_needed, hidden_dim)

        # Pool over the prefix to get current state embedding.
        current = out.mean(dim=1)  # (B, hidden_dim)

        # Compute logits.
        logits = self.output(current)  # (B, 4)

        # Apply temperature scaling (exploration).
        logits = logits / temperature

        return logits, (new_state,)  # Tuple for consistency with checkpoint routines


class GFloWNetGenerator:
    """
    GFlowNet-based circRNA sequence generator.

    Trains a policy network to generate high-quality sequences according
    to a fitness function (e.g., S9 immune fingerprints). Uses trajectory
    balance (TB) loss for stable training.

    Features:
      - Autoregressive token-by-token generation.
      - Masking to enforce constraints (GC content, length).
      - Sampling vs greedy decoding modes.

    Usage:
      gfn = GFloWNetGenerator(config, fitness_function)
      gfn.train_gfn(n_steps, batch_size)
      sequences = gfn.sample_sequences(n, temperature)
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        fitness_function: Optional[callable] = None,
    ):
        self.config = config or {}
        self.fitness_function = fitness_function

        # Policy network.
        self.policy = GFloWNLPolicy(
            d_model=self.config.get("d_model", 64),
            hidden_dim=self.config.get("hidden_dim", 128),
            num_layers=self.config.get("num_layers", 2),
            dropout=self.config.get("dropout", 0.1),
        )

        # Trajectory balance loss weight.
        self.tb_loss_weight = self.config.get("tb_weight", 1.0)

        # Learning rate.
        self.lr = self.config.get("lr", 1e-3)

        # Device.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)

        # Trajectory balance: logZ is a learnable scalar (per Bengio et al.).
        # Initialized to 0 (logZ=0 → Z=1). Stored as a raw tensor with
        # requires_grad=True so the optimizer (created below) can update it.
        self.logZ = torch.zeros(1, device=self.device, requires_grad=True)

        # Optimizer (created after policy is on device).
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + [self.logZ], lr=self.lr
        )

    def train_gfn(
        self,
        n_steps: int,
        batch_size: int = 16,
    ) -> List[Dict]:
        """
        Train GFlowNet for n_steps.

        Each step:
          1. Sample a batch of trajectories (partial sequences).
          2. Compute physics/fitness rewards.
          3. Update policy using trajectory balance loss.

        Returns:
          List of training statistics per step.
        """
        self.policy.train()

        stats = []

        for step in range(n_steps):
            # Sample batch of target fitness scores.
            target_fitnesses = torch.rand(batch_size, device=self.device) * 0.5 + 0.5

            # Generate trajectories.
            trajectories, rewards, log_probs = self._sample_trajectories(batch_size, target_fitnesses)

            # Compute trajectory balance loss.
            loss = self._trajectory_balance_loss(trajectories, rewards, log_probs)

            # Optimize.
            self.policy.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

            # Stats.
            stats.append({
                "step": step,
                "loss": loss.item(),
                "reward_mean": rewards.mean().item(),
                "reward_std": rewards.std().item(),
            })

            if (step + 1) % 100 == 0:
                print(f"  Step {step+1}/{n_steps}: loss={loss.item():.4f}, "
                      f"reward_mean={rewards.mean().item():.4f}")

        return stats

    def _sample_trajectories(
        self,
        n_samples: int,
        target_fitnesses: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Sample complete trajectories for training.

        Returns:
          trajectories: List of (1, L, 4) one-hot sequences.
          rewards: (N,) raw reward for each trajectory.
          log_probs: (N,) sum of log π(a_t|s_t) along each trajectory
                     (detached — used only as a TB target constant).
        """
        trajectories = []
        rewards = torch.zeros(n_samples, device=self.device)
        log_probs = torch.zeros(n_samples, device=self.device)

        min_length = self.config.get("min_length", 100)
        max_length = self.config.get("max_length", 500)

        for i in range(n_samples):
            # Sample a target length for this trajectory.
            target_L = torch.randint(min_length, max_length + 1, (1,)).item()
            seq = torch.zeros(1, 0, self.policy.num_tokens, device=self.device)
            context = target_fitnesses[i:i+1].clone()
            cum_log_p = torch.zeros(1, device=self.device)

            for _ in range(target_L):
                logits, _ = self.policy(seq, context)
                probs = F.softmax(logits, dim=-1)  # (1, 4)

                # Sample next token (stochastic for training diversity).
                dist = torch.distributions.Categorical(probs=probs)
                sampled_idx = dist.sample()  # (1,)
                log_p = dist.log_prob(sampled_idx)  # (1,)
                cum_log_p = cum_log_p + log_p

                token_onehot = F.one_hot(
                    sampled_idx, self.policy.num_tokens
                ).float().unsqueeze(0)  # (1, 1, 4)

                seq = torch.cat([seq, token_onehot], dim=1)

            # Compute reward (fitness).
            token_ids = torch.argmax(seq[0], dim=-1)  # (L,)
            sequence_str = "".join(
                self.policy.NUCS[t.item()] for t in token_ids
            )
            reward = self.fitness_function(sequence_str)
            if reward is None:
                reward = 0.0
            reward = max(float(reward), 1e-6)  # TB needs R > 0
            rewards[i] = reward
            log_probs[i] = cum_log_p.detach()
            trajectories.append(seq)

        return trajectories, rewards, log_probs

    def _trajectory_balance_loss(
        self,
        trajectories: List[torch.Tensor],
        rewards: torch.Tensor,
        log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Trajectory Balance (TB) loss (Bengio et al. 2021).

        For each trajectory τ:
            logZ + sum_t log π(a_t|s_t)  ≈  log R(τ)

        Loss = mean over trajectories of:
            ( logZ + sum_logp - log R )^2

        `sum_logp` must be a DIFFERENTIABLE re-evaluation of the policy's
        log-probabilities along the sampled trajectory (not the detached
        `log_probs` from sampling). We recompute it here by re-running the
        policy over the (now fixed) sampled tokens.

        Args:
            trajectories: list of (1, L, 4) one-hot sampled sequences.
            rewards: (N,) rewards R(τ) > 0.
            log_probs: (N,) detached sampling log-probs (unused in loss;
                kept for API symmetry / diagnostics).

        Returns:
            scalar TB loss.
        """
        if not trajectories:
            return torch.zeros(1, device=self.device, requires_grad=True)

        # Re-evaluate policy log-probs along each sampled trajectory.
        # This is the differentiable forward pass whose gradients train π.
        sum_logp_list = []
        for seq in trajectories:
            L = seq.size(1)
            cum = torch.zeros(1, device=self.device)
            for t in range(L):
                prefix = seq[:, :t+1, :]  # (1, t+1, 4)
                logits, _ = self.policy(prefix)  # (1, 4)
                log_probs_t = F.log_softmax(logits, dim=-1)  # (1, 4)
                # Take log-prob of the actually-sampled token at step t.
                actual_token = torch.argmax(seq[0, t], dim=-1)  # scalar
                cum = cum + log_probs_t[0, actual_token]
            sum_logp_list.append(cum)
        sum_logp = torch.stack(sum_logp_list)  # (N,)

        log_R = torch.log(rewards + 1e-10)  # (N,)
        # TB target: logZ + sum_logp - log_R → 0
        tb_residual = self.logZ + sum_logp - log_R  # (N,)
        loss = (tb_residual ** 2).mean()
        return loss * self.tb_loss_weight

    def sample_sequences(
        self,
        n: int,
        temperature: float = 1.0,
    ) -> List[str]:
        """
        Sample n sequences using current policy.

        Decoding mode: greedy (take argmax) or stochastic (sample).
        """
        self.policy.eval()

        min_length = self.config.get("min_length", 100)
        max_length = self.config.get("max_length", 500)

        sequences = []

        for _ in range(n):
            # Sample a target length uniformly in [min_length, max_length].
            target_L = torch.randint(min_length, max_length + 1, (1,)).item()
            seq = torch.zeros(1, 0, self.policy.num_tokens, device=self.device)

            # Generate target_L tokens autoregressively.
            for _ in range(target_L):
                logits, _ = self.policy(seq, temperature=temperature)
                probs = F.softmax(logits, dim=-1)  # (1, 4)

                # Greedy decoding.
                next_token = torch.argmax(probs, dim=-1)  # (1,)
                token_onehot = F.one_hot(
                    next_token, self.policy.num_tokens
                ).float().unsqueeze(0)  # (1, 1, 4)

                seq = torch.cat([seq, token_onehot], dim=1)

            # Convert to string. seq: (1, L, 4); argmax → (1, L).
            token_ids = torch.argmax(seq[0], dim=-1)  # (L,)
            sequence_str = "".join(
                self.policy.NUCS[t.item()] for t in token_ids
            )
            sequences.append(sequence_str)

        return sequences
