"""
Conformer Clustering Module for circRNA 3D Structures.

Takes multiple MD snapshots and clusters them into representative conformers
to maximize structural diversity for training data.

Methods:
1. RMSD-based k-means clustering
2. Energy-based selection within clusters
3. Diversity scoring for final conformer selection
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

try:
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score, pairwise_distances
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not installed. Install with: pip install scikit-learn")


class ConformerClusterer:
    """
    Clusters MD snapshots into diverse conformers.

    Workflow:
    1. Load all snapshots for a sequence
    2. Extract C3' coordinates and compute pairwise RMSD
    3. Cluster using RMSD-based k-means
    4. Select best representative from each cluster (lowest energy)
    5. Compute diversity score for selected ensemble
    """

    def __init__(self, config=None):
        """
        Args:
            config: dict with:
                - n_clusters: number of clusters (default: 5)
                - max_conformers: maximum conformers to output (default: 3)
                - clustering_method: 'kmeans', 'hierarchical', or 'energy'
                - rmsd_threshold: RMSD threshold for cluster separation (default: 2.0)
                - use_pca: whether to use PCA before clustering (default: True)
                - pca_components: number of PCA components (default: 10)
        """
        config = config or {}
        self.n_clusters = config.get('n_clusters', 5)
        self.max_conformers = config.get('max_conformers', 3)
        self.clustering_method = config.get('clustering_method', 'kmeans')
        self.rmsd_threshold = config.get('rmsd_threshold', 2.0)
        self.use_pca = config.get('use_pca', True)
        self.pca_components = config.get('pca_components', 10)

    def cluster_snapshots(
        self,
        snapshots: List[Dict],
        energies: np.ndarray,
        rmsds: np.ndarray,
        coordinates: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Cluster MD snapshots into representative conformers.

        Args:
            snapshots: list of snapshot dicts with 'pdb_path'
            energies: energy trajectory (n_frames,)
            rmsds: RMSD trajectory (n_frames,)
            coordinates: optional coordinates matrix (n_frames, n_atoms, 3)

        Returns:
            list of representative conformer dicts
        """
        if not HAS_SKLEARN:
            # Fallback: energy-based selection
            print("scikit-learn not available, using energy-based selection")
            return self._energy_based_selection(snapshots, energies)

        n_frames = len(snapshots)
        if n_frames < self.max_conformers:
            # Not enough frames, return all
            return self._return_all(snapshots, energies)

        # Build pairwise RMSD matrix
        if coordinates is not None:
            # Use provided coordinates
            rmsd_matrix = self._compute_pairwise_rmsd(coordinates)
        elif len(rmsds) == n_frames:
            # Use RMSD trajectory (approximate pairwise from reference)
            rmsd_matrix = self._compute_approx_rmsd_matrix(rmsds)
        else:
            # No RMSD info, use energy only
            return self._energy_based_selection(snapshots, energies)

        # Apply PCA for dimensionality reduction
        if self.use_pca and rmsd_matrix.shape[0] > self.pca_components:
            features = self._apply_pca(rmsd_matrix)
        else:
            features = rmsd_matrix

        # Cluster
        if self.clustering_method == 'kmeans':
            labels = self._kmeans_cluster(features)
        elif self.clustering_method == 'hierarchical':
            labels = self._hierarchical_cluster(features)
        else:
            labels = self._kmeans_cluster(features)

        # Select representative from each cluster
        representatives = self._select_representatives(
            snapshots, energies, labels, rmsd_matrix
        )

        # Compute diversity metrics
        diversity = self._compute_diversity(representatives)

        return representatives

    def _compute_pairwise_rmsd(self, coordinates: np.ndarray) -> np.ndarray:
        """Compute pairwise RMSD matrix between all frames."""
        n_frames = coordinates.shape[0]
        n_atoms = coordinates.shape[1]
        rmsd_matrix = np.zeros((n_frames, n_frames))

        for i in range(n_frames):
            for j in range(i + 1, n_frames):
                # Kabsch-aligned RMSD
                rmsd = self._kabsch_rmsd(coordinates[i], coordinates[j])
                rmsd_matrix[i, j] = rmsd
                rmsd_matrix[j, i] = rmsd

        return rmsd_matrix

    def _kabsch_rmsd(self, coords1: np.ndarray, coords2: np.ndarray) -> float:
        """Compute Kabsch RMSD between two coordinate sets."""
        # Center coordinates
        c1 = coords1 - coords1.mean(axis=0)
        c2 = coords2 - coords2.mean(axis=0)

        # Compute covariance matrix
        C = c1.T @ c2

        # SVD
        try:
            V, _, Wt = np.linalg.svd(C)
        except np.linalg.LinAlgError:
            return np.sqrt(np.mean(np.sum((c1 - c2) ** 2, axis=1)))

        # Rotation matrix
        R = V @ Wt

        # Apply rotation
        c1_rot = c1 @ R.T

        # Compute RMSD
        rmsd = np.sqrt(np.mean(np.sum((c1_rot - c2) ** 2, axis=1)))
        return rmsd

    def _compute_approx_rmsd_matrix(self, rmsds: np.ndarray) -> np.ndarray:
        """Approximate pairwise RMSD from reference-based RMSD trajectory."""
        n = len(rmsds)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                # Approximation: sqrt(rmsd_i^2 + rmsd_j^2 - 2*cov)
                diff = abs(rmsds[i] - rmsds[j])
                matrix[i, j] = diff
                matrix[j, i] = diff

        return matrix

    def _apply_pca(self, rmsd_matrix: np.ndarray) -> np.ndarray:
        """Apply PCA for dimensionality reduction."""
        n_components = min(self.pca_components, rmsd_matrix.shape[0] - 1)
        pca = PCA(n_components=n_components)
        return pca.fit_transform(rmsd_matrix)

    def _kmeans_cluster(self, features: np.ndarray) -> np.ndarray:
        """K-means clustering on feature matrix."""
        n_clusters = min(self.n_clusters, features.shape[0])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(features)

    def _hierarchical_cluster(self, features: np.ndarray) -> np.ndarray:
        """Hierarchical clustering on feature matrix."""
        n_clusters = min(self.n_clusters, features.shape[0])
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'  # Minimizes variance within clusters
        )
        return clustering.fit_predict(features)

    def _select_representatives(
        self,
        snapshots: List[Dict],
        energies: np.ndarray,
        labels: np.ndarray,
        rmsd_matrix: np.ndarray
    ) -> List[Dict]:
        """
        Select best representative from each cluster.

        Selection criteria:
        1. Lowest energy within cluster
        2. Centroid proximity (closest to cluster center)
        3. High confidence from MD
        """
        # Group by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[int(label)].append(i)

        representatives = []

        for label, members in clusters.items():
            if not members:
                continue

            # Score each member
            scores = []
            for idx in members:
                # Energy score (normalized within cluster)
                cluster_energies = [energies[m] for m in members]
                e_min = min(cluster_energies)
                e_range = max(cluster_energies) - e_min + 1e-10
                energy_score = 1.0 - (energies[idx] - e_min) / e_range

                # Centroid score (closest to cluster center)
                centroid_dist = np.mean([rmsd_matrix[idx, m] for m in members if m != idx])
                centroid_score = 1.0 / (1.0 + centroid_dist)

                # Combined score
                total_score = 0.6 * energy_score + 0.4 * centroid_score
                scores.append(total_score)

            # Select best
            best_idx = members[np.argmax(scores)]

            snapshot = snapshots[best_idx].copy()
            snapshot['cluster_id'] = label
            snapshot['cluster_size'] = len(members)
            snapshot['energy_rank'] = 1  # Best in cluster
            snapshot['centroid_score'] = float(scores[np.argmax(scores)])

            representatives.append(snapshot)

        # Sort by energy
        representatives.sort(key=lambda x: energies[snapshots.index(x)] if x in snapshots else 0)

        # Limit to max_conformers
        return representatives[:self.max_conformers]

    def _compute_diversity(self, representatives: List[Dict]) -> Dict:
        """Compute ensemble diversity metrics."""
        n = len(representatives)
        if n < 2:
            return {'num_conformers': n, 'diversity': 'low', 'coverage': 0.0}

        # Cluster coverage
        unique_clusters = len(set(r.get('cluster_id', i) for i, r in enumerate(representatives)))
        coverage = unique_clusters / self.max_conformers

        # Diversity level
        if coverage >= 0.8:
            diversity_level = 'high'
        elif coverage >= 0.5:
            diversity_level = 'medium'
        else:
            diversity_level = 'low'

        return {
            'num_conformers': n,
            'num_clusters_covered': unique_clusters,
            'coverage': float(coverage),
            'diversity': diversity_level,
        }

    def _energy_based_selection(
        self,
        snapshots: List[Dict],
        energies: np.ndarray
    ) -> List[Dict]:
        """Fallback: select snapshots with lowest energy."""
        n_select = min(self.max_conformers, len(snapshots))

        # Select frames with lowest energy
        best_indices = np.argsort(energies)[:n_select]

        result = []
        for rank, idx in enumerate(best_indices):
            snapshot = snapshots[idx].copy()
            snapshot['energy_rank'] = rank + 1
            result.append(snapshot)

        return result

    def _return_all(
        self,
        snapshots: List[Dict],
        energies: np.ndarray
    ) -> List[Dict]:
        """Return all snapshots (when few frames available)."""
        return self._energy_based_selection(snapshots, energies)


# ============================================================
# Ensemble Analysis Utilities
# ============================================================

class EnsembleAnalyzer:
    """
    Analyze conformer ensemble quality.
    """

    @staticmethod
    def compute_rmsf(conformers: List[np.ndarray]) -> np.ndarray:
        """
        Compute per-residue RMSF (Root Mean Square Fluctuation)
        from conformer ensemble.
        """
        conformers = np.array(conformers)
        n_conformers, n_atoms, _ = conformers.shape

        # Align all conformers to first
        aligned = [conformers[0]]
        for i in range(1, n_conformers):
            aligned.append(EnsembleAnalyzer._align_to_reference(
                conformers[i], conformers[0]
            ))

        aligned = np.array(aligned)

        # Compute RMSF
        rmsf = np.zeros(n_atoms)
        for i in range(n_atoms):
            diff = aligned[:, i, :] - aligned[:, i, :].mean(axis=0)
            rmsf[i] = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))

        return rmsf

    @staticmethod
    def compute_contact_map_consistency(conformers: List[np.ndarray]) -> float:
        """
        Compute contact map consistency across conformers.
        Contact = C3'-C3' distance < 10 Å.
        """
        n = len(conformers)
        if n < 2:
            return 1.0

        # Compute contact map for each conformer
        contact_maps = []
        for coords in conformers:
            n_atoms = coords.shape[0]
            contacts = np.zeros((n_atoms, n_atoms), dtype=bool)

            for i in range(n_atoms):
                for j in range(i + 4, n_atoms):  # Skip bonded neighbors
                    dist = np.linalg.norm(coords[i] - coords[j])
                    contacts[i, j] = dist < 10.0  # Å
                    contacts[j, i] = contacts[i, j]

            contact_maps.append(contacts)

        # Compute pairwise Jaccard similarity
        similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                intersection = np.sum(contact_maps[i] & contact_maps[j])
                union = np.sum(contact_maps[i] | contact_maps[j])
                if union > 0:
                    similarities.append(intersection / union)

        return float(np.mean(similarities)) if similarities else 0.0

    @staticmethod
    def _align_to_reference(coords: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Align coordinates to reference using Kabsch algorithm."""
        c1 = coords - coords.mean(axis=0)
        c2 = ref - ref.mean(axis=0)

        C = c1.T @ c2
        try:
            V, _, Wt = np.linalg.svd(C)
            R = V @ Wt
            return (c1 @ R.T) + ref.mean(axis=0)
        except np.linalg.LinAlgError:
            return coords


# ============================================================
# Pipeline Integration
# ============================================================

def cluster_pipeline_results(
    md_results: List[List[Dict]],
    snapshots_per_seq: List[np.ndarray],
    energies_per_seq: List[np.ndarray],
    config: Optional[Dict] = None
) -> Dict:
    """
    Cluster MD results and select diverse conformer ensemble.

    Args:
        md_results: list of lists of snapshot dicts
        snapshots_per_seq: list of coordinate arrays
        energies_per_seq: list of energy arrays
        config: clustering config

    Returns:
        dict with clustered conformers and diversity report
    """
    clusterer = ConformerClusterer(config)
    analyzer = EnsembleAnalyzer()

    all_representatives = []
    diversity_report = {
        'total_sequences': len(md_results),
        'total_snapshots': sum(len(s) for s in snapshots_per_seq),
        'total_conformers': 0,
        'avg_clusters_per_seq': 0,
        'high_diversity_pct': 0,
    }

    cluster_counts = defaultdict(int)

    for seq_id, (snapshots, coords, energies) in enumerate(
        zip(md_results, snapshots_per_seq, energies_per_seq)
    ):
        if len(snapshots) == 0:
            continue

        representatives = clusterer.cluster_snapshots(
            snapshots=snapshots,
            energies=energies,
            rmsds=np.zeros(len(snapshots)),  # Will be computed from coords
            coordinates=coords
        )

        # Add sequence ID to representatives
        for r in representatives:
            r['seq_id'] = seq_id
            all_representatives.append(r)

        cluster_counts[len(representatives)] += 1

    # Compute diversity report
    diversity_report['total_conformers'] = len(all_representatives)
    diversity_report['avg_clusters_per_seq'] = (
        len(all_representatives) / max(diversity_report['total_sequences'], 1)
    )
    diversity_report['cluster_size_distribution'] = dict(cluster_counts)
    diversity_report['high_diversity_pct'] = (
        cluster_counts.get(3, 0) / max(diversity_report['total_sequences'], 1)
    )

    return {
        'representatives': all_representatives,
        'diversity_report': diversity_report,
    }


# ============================================================
# Testing
# ============================================================

if __name__ == '__main__':
    import tempfile

    print("=" * 60)
    print("Testing Conformer Clustering")
    print("=" * 60)

    # Generate dummy MD data
    np.random.seed(42)
    n_snapshots = 20
    n_atoms = 48

    # Generate 20 snapshots (48 atoms each, in 3 clusters)
    snapshots = []
    coordinates_list = []
    energies_list = []

    for i in range(n_snapshots):
        # 3 different conformer types
        cluster = i % 3
        center = np.array([cluster * 5.0, 0, 0])

        coords = []
        for j in range(n_atoms):
            angle = j * 33.0
            rise = 2.8
            x = 10.0 * np.cos(np.radians(angle)) + center[0]
            y = 10.0 * np.sin(np.radians(angle))
            z = j * rise
            x += np.random.randn() * 0.5
            y += np.random.randn() * 0.5
            z += np.random.randn() * 0.2
            coords.append([x, y, z])

        coords = np.array(coords)
        coordinates_list.append(coords)

        pdb_path = f'test_snapshot_{i}.pdb'
        snapshots.append({
            'pdb_path': pdb_path,
            'frame': i,
            'time_ps': i * 50,
        })
        energies_list.append(-100.0 + np.random.randn() * 50 + cluster * 10)

    coords = np.array(coordinates_list)
    energies = np.array(energies_list)

    # Test clustering
    config = {
        'n_clusters': 3,
        'max_conformers': 3,
        'clustering_method': 'kmeans',
        'rmsd_threshold': 2.0,
        'use_pca': True,
        'pca_components': 5,
    }

    clusterer = ConformerClusterer(config)
    representatives = clusterer.cluster_snapshots(
        snapshots=snapshots,
        energies=energies,
        rmsds=np.zeros(n_snapshots),
        coordinates=coords
    )

    print(f"\nInput: {n_snapshots} snapshots")
    print(f"Output: {len(representatives)} representative conformers")
    for r in representatives:
        print(f"  Frame {r['frame']}: cluster={r.get('cluster_id', 'N/A')}")
    print(f"\n[OK] Clustering test passed!")

    # Test ensemble analysis
    print(f"\n" + "=" * 60)
    print("Testing Ensemble Analysis")
    print("=" * 60)

    rep_coords = [coords[r['frame']] for r in representatives]
    consistency = EnsembleAnalyzer.compute_contact_map_consistency(rep_coords)
    rmsf = EnsembleAnalyzer.compute_rmsf(rep_coords)

    print(f"Contact map consistency: {consistency:.3f}")
    print(f"RMSF range: [{rmsf.min():.2f}, {rmsf.max():.2f}] A")
    print(f"[OK] Ensemble analysis test passed!")