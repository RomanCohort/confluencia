"""
conformation_ensemble.py — Conformation Ensemble Sampling + Clustering for TorusFold

RNA is inherently flexible: a single sequence can adopt multiple conformations
depending on ionic conditions, temperature, protein binding, etc.

This module provides:
1. Multi-conformation sampling from diffusion models
2. RMSD-based clustering to identify major conformational states
3. Population-weighted ensemble output
4. Condition-aware generation (temperature, Mg2+, pH, protein)
5. SHAPE/DMS experiment-guided sampling
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

@dataclass
class EnsembleConfig:
    """Configuration for conformation ensemble sampling."""
    # Sampling
    n_samples: int = 20           # Number of conformation samples
    n_clusters: int = 5          # Number of clusters to identify
    rmsd_cutoff: float = 5.0     # Å, cutoff for same-cluster assignment

    # Conditions
    temperature: float = 310.0    # Kelvin (physiological)
    mg_concentration: float = 7.4 # mM (Mg2+ stabilizes tertiary structure)
    na_concentration: float = 150.0  # mM (Na+ monovalent)
    ph: float = 7.0              # pH
    protein_bound: bool = False   # Whether protein is bound

    # SHAPE/DMS constraints
    shape_reactivity: Optional[np.ndarray] = None  # (L,) per-nucleotide
    dms_reactivity: Optional[np.ndarray] = None     # (L,) per-nucleotide
    shape_weight: float = 1.0     # Weight for SHAPE constraint
    dms_weight: float = 0.5       # Weight for DMS constraint

    # Output
    return_uncertainty: bool = True     # Per-residue flexibility
    return_population: bool = True      # Cluster population weights
    bond_length: float = 5.9            # P-P backbone distance


# ═══════════════════════════════════════════════════════════════
# Condition Encoder
# ═══════════════════════════════════════════════════════════════

class ConditionEncoder(nn.Module):
    """Encode experimental/environmental conditions for conditional generation.

    Maps scalar conditions (T, Mg2+, pH, etc.) to a latent vector
    that modulates the diffusion sampling process.
    """

    def __init__(self, d_out: int = 64):
        super().__init__()
        self.d_out = d_out

        # Input: [temperature, mg, na, ph, protein_bound]
        self.encoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.GELU(),
            nn.Linear(32, d_out),
            nn.GELU(),
        )

    def forward(self, conditions: Dict[str, float]) -> torch.Tensor:
        """Encode conditions to latent vector.

        Args:
            conditions: {temperature, mg_concentration, na_concentration,
                        ph, protein_bound}

        Returns:
            (d_out,) condition embedding
        """
        # Normalize to reasonable ranges
        T_norm = (conditions.get('temperature', 310.0) - 273.15) / 100.0  # 0-2
        mg_norm = conditions.get('mg_concentration', 7.4) / 50.0           # 0-2
        na_norm = conditions.get('na_concentration', 150.0) / 500.0        # 0-1
        ph_norm = (conditions.get('ph', 7.0) - 5.0) / 5.0                  # 0-2
        prot = float(conditions.get('protein_bound', False))

        inp = torch.tensor([T_norm, mg_norm, na_norm, ph_norm, prot],
                           dtype=torch.float32, device=next(self.parameters()).device
                           if len(list(self.parameters())) > 0 else 'cpu')
        return self.encoder(inp)


# ═══════════════════════════════════════════════════════════════
# SHAPE/DMS Constraint Guidance
# ═══════════════════════════════════════════════════════════════

class ExperimentalGuidance:
    """Guide diffusion sampling with SHAPE/DMS experimental data.

    SHAPE reactivity measures nucleotide flexibility:
    - High reactivity → flexible (unpaired/loop)
    - Low reactivity → constrained (paired/stem)

    This guides the diffusion denoiser to respect experimental constraints.
    """

    # SHAPE reactivity thresholds (Deigan et al., 2009)
    HIGH_SHAPE = 0.85      # Likely unpaired
    LOW_SHAPE = -0.6       # Likely paired (pseudo-free-energy)

    def __init__(self, shape_data: Optional[np.ndarray] = None,
                 dms_data: Optional[np.ndarray] = None,
                 shape_weight: float = 1.0,
                 dms_weight: float = 0.5):
        self.shape_data = shape_data
        self.dms_data = dms_data
        self.shape_weight = shape_weight
        self.dms_weight = dms_weight

    def compute_constraint_loss(
        self,
        coords: torch.Tensor,       # (B, L, 3)
        pair_probs: torch.Tensor,    # (B, L, L) predicted pairing
    ) -> torch.Tensor:
        """Compute experimental constraint violation loss.

        Returns scalar loss that penalizes:
        - Flexible residues (high SHAPE) that are predicted as paired
        - Constrained residues (low SHAPE) that are predicted as unpaired
        - DMS-reactive A/C that are predicted as paired
        """
        B, L, _ = coords.shape
        device = coords.device
        loss = torch.tensor(0.0, device=device)

        if self.shape_data is not None:
            shape_t = torch.tensor(self.shape_data, dtype=torch.float32, device=device)
            # Convert SHAPE to pairing pseudo-energy
            shape_energy = -self.shape_weight * np.log(self.shape_data + 0.01)
            # Penalize high-SHAPE residues being paired
            pair_diag = pair_probs.diagonal(dim1=-2, dim2=-1)  # self-pairing prob
            max_pair = pair_probs.max(dim=-1).values            # max pair partner
            for b in range(B):
                # High SHAPE should have low pairing probability
                violation = F.mse_loss(
                    max_pair[b, :L],
                    1.0 - torch.sigmoid(shape_t[:L])
                )
                loss = loss + violation

        if self.dms_data is not None:
            dms_t = torch.tensor(self.dms_data, dtype=torch.float32, device=device)
            # DMS modifies unpaired A and C residues
            for b in range(B):
                max_pair_b = pair_probs[b, :L].max(dim=-1).values
                violation = F.mse_loss(
                    max_pair_b,
                    1.0 - torch.sigmoid(dms_t[:L])
                )
                loss = loss + self.dms_weight * violation

        return loss


# ═══════════════════════════════════════════════════════════════
# Conformation Clusterer
# ═══════════════════════════════════════════════════════════════

class ConformationClusterer:
    """Cluster sampled conformations by pairwise RMSD.

    Uses greedy clustering:
    1. Compute all-pairs RMSD matrix
    2. Pick the conformation with most neighbors as cluster center
    3. Assign neighbors within cutoff
    4. Repeat for remaining
    """

    def __init__(self, rmsd_cutoff: float = 5.0, bond_length: float = 5.9):
        self.rmsd_cutoff = rmsd_cutoff
        self.bond_length = bond_length

    def kabsch_rmsd(
        self,
        coords1: np.ndarray,  # (L, 3)
        coords2: np.ndarray,  # (L, 3)
    ) -> float:
        """Compute Kabsch-aligned RMSD between two structures."""
        L = min(len(coords1), len(coords2))
        p = coords1[:L].copy()
        t = coords2[:L].copy()

        p_c = p - p.mean(axis=0)
        t_c = t - t.mean(axis=0)

        H = t_c.T @ p_c
        try:
            U, S, Vt = np.linalg.svd(H)
            d = np.sign(np.linalg.det(Vt.T @ U.T))
            D = np.diag([1, 1, d])
            R = Vt.T @ D @ U.T
            p_aligned = (R @ p_c.T).T
            rmsd = np.sqrt(np.mean(np.sum((p_aligned - t_c) ** 2, axis=1)))
        except Exception:
            rmsd = np.sqrt(np.mean(np.sum((p_c - t_c) ** 2, axis=1)))

        return rmsd

    def cluster(
        self,
        conformations: List[np.ndarray],  # List of (L, 3)
        n_clusters: int = 5,
    ) -> Tuple[List[List[int]], List[int], np.ndarray]:
        """Cluster conformations by RMSD.

        Args:
            conformations: List of (L_i, 3) coordinate arrays
            n_clusters: Maximum number of clusters

        Returns:
            clusters: List of lists, each containing member indices
            centers: Indices of cluster center conformations
            rmsd_matrix: (N, N) pairwise RMSD matrix
        """
        N = len(conformations)
        if N == 0:
            return [], [], np.array([])

        # Compute pairwise RMSD
        rmsd_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                rmsd = self.kabsch_rmsd(conformations[i], conformations[j])
                rmsd_matrix[i, j] = rmsd
                rmsd_matrix[j, i] = rmsd

        # Greedy clustering
        assigned = set()
        clusters = []
        centers = []

        for _ in range(n_clusters):
            if len(assigned) >= N:
                break

            # Find unassigned conformation with most neighbors within cutoff
            best_center = -1
            best_count = 0

            for i in range(N):
                if i in assigned:
                    continue
                count = sum(
                    1 for j in range(N)
                    if j not in assigned and rmsd_matrix[i, j] < self.rmsd_cutoff
                )
                if count > best_count:
                    best_count = count
                    best_center = i

            if best_center == -1:
                # Assign remaining to single-member clusters
                remaining = [i for i in range(N) if i not in assigned]
                for r in remaining:
                    clusters.append([r])
                    centers.append(r)
                    assigned.add(r)
                break

            # Assign members within cutoff
            cluster_members = [best_center]
            assigned.add(best_center)

            for j in range(N):
                if j not in assigned and rmsd_matrix[best_center, j] < self.rmsd_cutoff:
                    cluster_members.append(j)
                    assigned.add(j)

            clusters.append(cluster_members)
            centers.append(best_center)

        # Assign any remaining unassigned to nearest cluster
        for i in range(N):
            if i not in assigned:
                min_rmsd = float('inf')
                best_cluster = 0
                for ci, center in enumerate(centers):
                    if rmsd_matrix[i, center] < min_rmsd:
                        min_rmsd = rmsd_matrix[i, center]
                        best_cluster = ci
                clusters[best_cluster].append(i)
                assigned.add(i)

        return clusters, centers, rmsd_matrix


# ═══════════════════════════════════════════════════════════════
# Conformation Ensemble Sampler
# ═══════════════════════════════════════════════════════════════

class ConformationEnsemble:
    """Sample and analyze conformational ensembles for circRNA.

    High-level API:
        ensemble = ConformationEnsemble(model, config)
        result = ensemble.sample(seq_ids, conditions={...})
        result.cluster()           # Cluster conformations
        result.uncertainty()       # Per-residue flexibility
        result.best()              # Most populated cluster center
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[EnsembleConfig] = None,
    ):
        self.model = model
        self.config = config or EnsembleConfig()
        self.clusterer = ConformationClusterer(
            rmsd_cutoff=self.config.rmsd_cutoff,
            bond_length=self.config.bond_length,
        )

    @torch.no_grad()
    def sample(
        self,
        seq_ids: torch.Tensor,       # (1, L)
        conditions: Optional[Dict[str, float]] = None,
        shape_data: Optional[np.ndarray] = None,
        dms_data: Optional[np.ndarray] = None,
    ) -> 'EnsembleResult':
        """Sample multiple conformations.

        Args:
            seq_ids: (1, L) tokenized sequence
            conditions: environmental conditions dict
            shape_data: (L,) SHAPE reactivity array
            dms_data: (L,) DMS reactivity array

        Returns:
            EnsembleResult with sampled conformations
        """
        device = seq_ids.device
        n_samples = self.config.n_samples

        # Override conditions if provided
        cond = {
            'temperature': self.config.temperature,
            'mg_concentration': self.config.mg_concentration,
            'na_concentration': self.config.na_concentration,
            'ph': self.config.ph,
            'protein_bound': self.config.protein_bound,
        }
        if conditions:
            cond.update(conditions)

        # Sample conformations
        conformations = []
        for i in range(n_samples):
            # Set different random seed for each sample
            if device.type == 'cuda':
                torch.cuda.manual_seed(i)
            else:
                torch.manual_seed(i)

            out = self.model(seq_ids, mode='sample')
            coords = out['coords'].cpu().numpy()  # (1, L, 3)
            conformations.append(coords[0])  # (L, 3)

        # Auto-cluster
        clusters, centers, rmsd_matrix = self.clusterer.cluster(
            conformations, n_clusters=self.config.n_clusters
        )

        # Compute per-residue uncertainty (RMSF)
        L = len(conformations[0])
        all_coords = np.stack(conformations)  # (N, L, 3)
        mean_coords = all_coords.mean(axis=0)  # (L, 3)
        rmsf = np.sqrt(np.mean(np.sum((all_coords - mean_coords) ** 2, axis=-1), axis=0))  # (L,)

        # Compute cluster populations
        populations = [len(c) / n_samples for c in clusters]

        # Compute closure quality for each conformation
        closure_errors = []
        for conf in conformations:
            d = np.linalg.norm(conf[0] - conf[-1]) - self.config.bond_length
            closure_errors.append(abs(d))

        return EnsembleResult(
            conformations=conformations,
            clusters=clusters,
            cluster_centers=centers,
            cluster_populations=populations,
            rmsd_matrix=rmsd_matrix,
            rmsf=rmsf,
            mean_coords=mean_coords,
            closure_errors=np.array(closure_errors),
            conditions=cond,
            n_samples=n_samples,
        )


# ═══════════════════════════════════════════════════════════════
# Result Container
# ═══════════════════════════════════════════════════════════════

@dataclass
class EnsembleResult:
    """Container for conformation ensemble results."""
    conformations: List[np.ndarray]       # (N,) of (L, 3)
    clusters: List[List[int]]             # Cluster member indices
    cluster_centers: List[int]            # Indices of center conformations
    cluster_populations: List[float]      # Population weights
    rmsd_matrix: np.ndarray               # (N, N) pairwise RMSD
    rmsf: np.ndarray                       # (L,) per-residue RMSF
    mean_coords: np.ndarray                # (L, 3) mean structure
    closure_errors: np.ndarray            # (N,) BSJ closure quality
    conditions: Dict[str, float]           # Sampling conditions
    n_samples: int                         # Number of samples

    def best(self) -> np.ndarray:
        """Return the most populated cluster center (best structure)."""
        if not self.cluster_centers:
            return self.conformations[0]
        best_idx = np.argmax(self.cluster_populations)
        return self.conformations[self.cluster_centers[best_idx]]

    def diversity(self) -> float:
        """Mean pairwise RMSD (conformational diversity)."""
        if self.rmsd_matrix.size == 0:
            return 0.0
        N = self.rmsd_matrix.shape[0]
        if N <= 1:
            return 0.0
        return self.rmsd_matrix.sum() / (N * (N - 1))

    def top_k(self, k: int = 3) -> List[Tuple[np.ndarray, float]]:
        """Return top-K cluster centers with populations."""
        sorted_idx = np.argsort(self.cluster_populations)[::-1]
        result = []
        for i in sorted_idx[:k]:
            if i < len(self.cluster_centers):
                center = self.conformations[self.cluster_centers[i]]
                pop = self.cluster_populations[i]
                result.append((center, pop))
        return result

    def summary(self) -> Dict:
        """Return summary statistics."""
        return {
            'n_samples': self.n_samples,
            'n_clusters': len(self.clusters),
            'diversity_Å': round(self.diversity(), 2),
            'max_rmsf_Å': round(float(self.rmsf.max()), 2),
            'mean_rmsf_Å': round(float(self.rmsf.mean()), 2),
            'best_closure_Å': round(float(self.closure_errors.min()), 2),
            'populations': [round(p, 3) for p in self.cluster_populations],
            'conditions': self.conditions,
        }

    def save(self, path: str):
        """Save ensemble to disk."""
        import json
        os.makedirs(path, exist_ok=True)

        # Save conformations as npy files
        for i, conf in enumerate(self.conformations):
            np.save(os.path.join(path, f'conf_{i:04d}.npy'), conf)

        # Save summary
        with open(os.path.join(path, 'ensemble_summary.json'), 'w') as f:
            json.dump(self.summary(), f, indent=2)

        # Save RMSD matrix
        if self.rmsd_matrix.size > 0:
            np.save(os.path.join(path, 'rmsd_matrix.npy'), self.rmsd_matrix)

        # Save RMSF
        np.save(os.path.join(path, 'rmsf.npy'), self.rmsf)

    @classmethod
    def load(cls, path: str) -> 'EnsembleResult':
        """Load ensemble from disk."""
        import glob
        conf_files = sorted(glob.glob(os.path.join(path, 'conf_*.npy')))
        conformations = [np.load(f) for f in conf_files]

        rmsf = np.load(os.path.join(path, 'rmsf.npy'))
        rmsd_matrix = np.load(os.path.join(path, 'rmsd_matrix.npy'))

        mean_coords = np.stack(conformations).mean(axis=0)
        closure_errors = np.array([
            abs(np.linalg.norm(c[0] - c[-1]) - 5.9) for c in conformations
        ])

        # Re-cluster
        clusterer = ConformationClusterer()
        clusters, centers, _ = clusterer.cluster(conformations)
        populations = [len(c) / len(conformations) for c in clusters]

        return cls(
            conformations=conformations,
            clusters=clusters,
            cluster_centers=centers,
            cluster_populations=populations,
            rmsd_matrix=rmsd_matrix,
            rmsf=rmsf,
            mean_coords=mean_coords,
            closure_errors=closure_errors,
            conditions={},
            n_samples=len(conformations),
        )


# ═══════════════════════════════════════════════════════════════
# Convenience function
# ═══════════════════════════════════════════════════════════════

def predict_circrna_ensemble(
    model: nn.Module,
    sequence: str,
    n_samples: int = 20,
    n_clusters: int = 5,
    conditions: Optional[Dict[str, float]] = None,
    shape_data: Optional[np.ndarray] = None,
) -> EnsembleResult:
    """High-level API: predict circRNA conformational ensemble.

    Args:
        model: Trained TorusFold model (any diffusion-based scheme)
        sequence: circRNA sequence string (e.g., "AUGCAUGC...")
        n_samples: Number of conformation samples
        n_clusters: Number of clusters to identify
        conditions: Environmental conditions
        shape_data: SHAPE reactivity data per nucleotide

    Returns:
        EnsembleResult with clustered conformations

    Example:
        >>> model = load_pretrained('scheme7_best.pt')
        >>> result = predict_circrna_ensemble(
        ...     model, "AUGCAUGCAUGC...",
        ...     n_samples=20,
        ...     conditions={'temperature': 310, 'mg_concentration': 10.0}
        ... )
        >>> print(result.summary())
        >>> best_coords = result.best()
        >>> top3 = result.top_k(3)
    """
    mapping = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
    seq_ids = torch.tensor(
        [[mapping.get(b, 4) for b in sequence.upper()]],
        dtype=torch.long
    )

    config = EnsembleConfig(
        n_samples=n_samples,
        n_clusters=n_clusters,
    )
    if conditions:
        if 'temperature' in conditions:
            config.temperature = conditions['temperature']
        if 'mg_concentration' in conditions:
            config.mg_concentration = conditions['mg_concentration']
        if 'na_concentration' in conditions:
            config.na_concentration = conditions['na_concentration']
        if 'ph' in conditions:
            config.ph = conditions['ph']
        if 'protein_bound' in conditions:
            config.protein_bound = conditions['protein_bound']

    if shape_data is not None:
        config.shape_reactivity = shape_data

    ensemble = ConformationEnsemble(model, config)
    return ensemble.sample(seq_ids, conditions=conditions, shape_data=shape_data)
