#!/usr/bin/env python3
"""
pdb_rna_circularize.py -- Extract RNA structures from PDB, circularize them,
and convert to circRNA training dataset format.

Pipeline:
1. Search RCSB PDB for RNA-containing structures (resolution < 3.0 A, RNA length 30-500 nt)
2. Download PDB files, extract C3' atom coordinates and residue sequences
3. Circularize linear fragments via annealing closure or direct end-joining
4. Quality filter: closure distance, steric clashes, bond RMSD
5. Augment each structure 8x (rotation + small noise)
6. Output: sequences.json + coords/*.npy + metadata.json

Target: ~4,000 PDB-derived circularized samples

Usage:
    python pdb_rna_circularize.py --output data/pdb_circrna --target 4000 --augment 8
    python pdb_rna_circularize.py --output data/pdb_circrna --pdb-cache ./pdb_cache --max-workers 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Try importing ViennaRNA; used for secondary structure prediction
# ---------------------------------------------------------------------------
try:
    import RNA
    HAS_VIENNA = True
except ImportError:
    HAS_VIENNA = False

# ---------------------------------------------------------------------------
# Try importing GeometricConstraintSolver from the project; fall back to
# built-in annealing closure if unavailable.
# ---------------------------------------------------------------------------
try:
    from confluencia_3_0.core.circrna.torusfold.constraint_solver import (
        GeometricConstraintSolver,
        SolverConfig,
    )
    _HAS_CONSTRAINT_SOLVER = True
except ImportError:
    _HAS_CONSTRAINT_SOLVER = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download"

# Residue name -> single-letter RNA base
RESIDUE_MAP: Dict[str, str] = {
    "A": "A", "C": "C", "G": "G", "U": "U",
    "DA": "A", "DC": "C", "DG": "G", "DT": "U",
    "I": "A",  # inosine -> A (common substitution)
}


def predict_ss_and_pairs(sequence: str) -> Tuple[str, List[List[int]]]:
    """Predict secondary structure for a circular RNA sequence.

    Uses ViennaRNA circ mode if available, otherwise heuristic pairing.

    Returns:
        (dot_bracket, pair_constraints) where pair_constraints is [[i,j], ...]
    """
    sequence = sequence.upper().replace("T", "U")
    L = len(sequence)

    if HAS_VIENNA:
        try:
            md = RNA.md()
            md.circ = True
            fc = RNA.fold_compound(sequence, md)
            ss, mfe = fc.mfe()

            # Extract pairs from dot-bracket
            pairs = []
            stack = []
            for pos, char in enumerate(ss):
                if char == "(":
                    stack.append(pos)
                elif char == ")" and stack:
                    j_pos = stack.pop()
                    pairs.append([j_pos, pos])
            return ss, pairs
        except Exception:
            pass

    # Heuristic fallback: complement pairing
    complement = {"A": "U", "U": "A", "G": "C", "C": "G"}
    pairs = []
    paired = set()
    for i in range(L):
        if i in paired:
            continue
        for j in range(i + 4, min(i + 20, L)):
            if j in paired:
                continue
            if complement.get(sequence[i]) == sequence[j]:
                pairs.append([i, j])
                paired.add(i)
                paired.add(j)
                break

    ss = "." * L
    for i, j in pairs:
        ss = ss[:i] + "(" + ss[i+1:j] + ")" + ss[j+1:]
    return ss, pairs

# Curated list of well-known RNA PDB IDs (high-quality RNA structures)
CURATED_RNA_PDB_IDS: List[str] = [
    # Ribosomal RNA fragments
    "1FJG", "1FKA", "1GID", "1GIX", "1J5E", "1J7Z", "1JJ2", "1K73",
    "1K8A", "1KX1", "1LC6", "1M5O", "1MMS", "1NBS", "1NJO", "1NKW",
    "1P5I", "1PNU", "1Q29", "1Q7R", "1Q9A", "1QF6", "1RHT", "1S72",
    "1U4D", "1U6B", "1VQ8", "1VRC", "1X8W", "1XJR", "1Y0Q", "1Y26",
    # Riboswitches & ribozymes
    "2A2J", "2AVY", "2B57", "2CKY", "2GDI", "2H0M", "2HOJ", "2I7Z",
    "2JUK", "2JWP", "2K21", "2K95", "2L1V", "2L8V", "2LC8", "2M8K",
    "2N0J", "2QBZ", "2QUS", "2R20", "2RDE", "2RKW", "2UV8", "2VQE",
    # Large RNA structures
    "3D0G", "3D2V", "3F2Q", "3G78", "3HBT", "3IRW", "3J7V", "3J9L",
    "3J9W", "3JAY", "3JBH", "3JCN", "3JCR", "3JCT", "3J5Q", "3J7P",
    "3KC4", "3KCR", "3KIS", "3LWR", "3MGP", "3MWV", "3NDB", "3OAS",
    # Modern high-res RNA structures
    "4ARC", "4B5R", "4CS1", "4DII", "4DR2", "4E6M", "4ENC", "4ERJ",
    "4F8U", "4FN6", "4G6J", "4GMX", "4GXX", "4HHP", "4IG8", "4JRC",
    "4K4W", "4KQY", "4LCK", "4LVV", "4MGN", "4N0M", "4OCB", "4P5J",
    "4QJD", "4R0D", "4RCI", "4RMO", "4S78", "4TNA", "4V4Q", "4V6F",
    # More recent structures
    "5B3P", "5DD0", "5E54", "5FJC", "5GJL", "5JJR", "5K7C", "5KM4",
    "5L4J", "5LJ3", "5MGI", "5MMA", "5NWL", "5O5L", "5T2A", "5U3G",
    "5U64", "5U9B", "5UT6", "5V0F", "5V16", "5W2I", "5W3Q", "5WQJ",
    "6ASM", "6B44", "6C5T", "6DBK", "6E1O", "6FEC", "6GSK", "6H5S",
    "6HVZ", "6IBK", "6J3L", "6JRN", "6K2B", "6L1W", "6M5O", "6ME7",
    "6NE0", "6NVR", "6O0M", "6P1H", "6QNR", "6R1M", "6R3R", "6SAQ",
    "6T1A", "6T7P", "6U42", "6UET", "6V7X", "6VH4", "6W6V", "6WLL",
    "6X3Y", "6XVY", "6YDP", "6YEL", "6Z1M", "6ZMO", "6ZRD", "6ZVQ",
    "7A5V", "7B26", "7C7A", "7K00", "7K4C", "7KPT", "7LHD", "7M5X",
    "7MUK", "7N1F", "7NND", "7O8Z", "7P50", "7PEO", "7QET", "7R6R",
    "7RGA", "7S03", "7S7B", "7T49", "7T9A", "7TL9", "7TOO", "7U6F",
    "7UKW", "7V7Q", "7VDA", "7WME", "7X5C", "7XSL", "7Y92", "7YO4",
    # Telomerase & other complex RNAs
    "6D6V", "7BGD", "7EWE", "7EGE", "7MK8", "7PFO", "7Q5A", "7S1B",
    # ── Known circRNA PDB structures (experimental or validated) ──
    # 8xtp, 8xtq, 8xtr, 8xts — PDB circRNA entries from 2024-2025
    "8XTP", "8XTQ", "8XTR", "8XTS",
    # 9is7 — previously referenced in architecture document
    "9IS7",
    # Additional circRNA-adjacent structures
    "9H8A", "7U7Q", "8G2T", "8G2U", "8G2V",
]

BOND_LENGTH = 5.9  # A, C3'-C3' backbone distance


# ---------------------------------------------------------------------------
# PDB Search
# ---------------------------------------------------------------------------

def search_rcsb_rna(
    max_resolution: float = 3.0,
    min_length: int = 30,
    max_length: int = 500,
    max_results: int = 2000,
    retries: int = 3,
    backoff: float = 2.0,
) -> List[str]:
    """Search RCSB PDB for RNA-containing structures.

    Uses the RCSB Search API to find structures containing RNA chains
    within the specified resolution and length range.

    Returns:
        List of PDB IDs matching the criteria.
    """
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": max_resolution,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_RNA",
                        "operator": "greater",
                        "value": 0,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "return_all_hits": True,
            "results_content_type": ["experimental"],
            "sort": [
                {
                    "sort_by": "rcsb_entry_info.resolution_combined",
                    "direction": "asc",
                }
            ],
        },
    }

    payload = json.dumps(query).encode("utf-8")
    req = urllib.request.Request(
        RCSB_SEARCH_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                pdb_ids = [hit["identifier"] for hit in data.get("result_set", [])]
                print(f"  RCSB search returned {len(pdb_ids)} PDB IDs")
                return pdb_ids
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            print(f"  RCSB search attempt {attempt + 1} failed: {exc}")
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))

    print("  RCSB search failed after all retries; falling back to curated list only")
    return []


def get_pdb_ids(
    max_resolution: float = 3.0,
    min_length: int = 30,
    max_length: int = 500,
    max_results: int = 2000,
) -> List[str]:
    """Combine RCSB search results with curated PDB IDs.

    Returns:
        Deduplicated list of PDB IDs.
    """
    searched = search_rcsb_rna(
        max_resolution=max_resolution,
        min_length=min_length,
        max_length=max_length,
        max_results=max_results,
    )

    seen = set()
    combined: List[str] = []

    for pid in searched + CURATED_RNA_PDB_IDS:
        pid_upper = pid.upper()
        if pid_upper not in seen:
            seen.add(pid_upper)
            combined.append(pid_upper)

    print(f"  Combined PDB ID pool: {len(combined)} unique entries")
    return combined


# ---------------------------------------------------------------------------
# PDB Download
# ---------------------------------------------------------------------------

def download_pdb(
    pdb_id: str,
    cache_dir: str,
    retries: int = 3,
    backoff: float = 2.0,
    cache_only: bool = False,
) -> Optional[str]:
    """Download a PDB file with caching.

    Args:
        cache_only: If True, only return cached files; skip all downloads.

    Returns:
        Path to the cached PDB file, or None on failure.
    """
    os.makedirs(cache_dir, exist_ok=True)
    pdb_id_upper = pdb_id.upper()
    cached_path = os.path.join(cache_dir, f"{pdb_id_upper}.pdb")

    if os.path.exists(cached_path):
        return cached_path

    if cache_only:
        return None

    url = f"{RCSB_DOWNLOAD_URL}/{pdb_id_upper}.pdb"

    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, cached_path)
            return cached_path
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                # PDB ID does not have a plain-text file; skip silently
                return None
            print(f"    Download {pdb_id_upper} attempt {attempt + 1} failed: {exc}")
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
        except Exception as exc:
            print(f"    Download {pdb_id_upper} unexpected error: {exc}")
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))

    return None


# ---------------------------------------------------------------------------
# PDB Parsing
# ---------------------------------------------------------------------------

def parse_pdb_rna(pdb_path: str) -> List[Dict]:
    """Extract RNA chains from a PDB file.

    For each RNA chain, returns a dict with:
    - chain_id: chain identifier
    - sequence: RNA sequence (ACGU)
    - coords: (L, 3) numpy array of C3' atom positions
    - residue_indices: list of residue sequence numbers

    Filters: only residues with a C3' atom and a valid base name are kept.
    """
    chains: Dict[str, Dict] = {}

    with open(pdb_path, "r", errors="replace") as fh:
        for line in fh:
            if not line.startswith("ATOM") and not line.startswith("HETATM"):
                continue

            atom_name = line[12:16].strip()

            # We only need C3' atoms for backbone coordinates
            if atom_name not in ("C3'", "C3*"):
                continue

            chain_id = line[21].strip()
            res_name = line[17:20].strip()
            res_seq = int(line[22:26].strip())

            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            # Map residue name to RNA base
            base = RESIDUE_MAP.get(res_name)
            if base is None:
                continue  # not RNA

            if chain_id not in chains:
                chains[chain_id] = {
                    "chain_id": chain_id,
                    "residues": {},  # res_seq -> (base, [x, y, z])
                }

            # Keep first C3' per residue (in case of alternate conformations)
            if res_seq not in chains[chain_id]["residues"]:
                chains[chain_id]["residues"][res_seq] = (base, [x, y, z])

    results = []
    for chain_id, chain_data in chains.items():
        # Sort by residue sequence number to get correct order
        sorted_resseqs = sorted(chain_data["residues"].keys())
        sequence = "".join(chain_data["residues"][r][0] for r in sorted_resseqs)
        coords = np.array(
            [chain_data["residues"][r][1] for r in sorted_resseqs],
            dtype=np.float64,
        )
        results.append({
            "chain_id": chain_id,
            "sequence": sequence,
            "coords": coords,
            "residue_indices": sorted_resseqs,
        })

    return results


# ---------------------------------------------------------------------------
# Circularization
# ---------------------------------------------------------------------------

def annealing_closure(
    coords: np.ndarray,
    bond_length: float = 5.9,
    n_steps: int = 500,
    lr: float = 0.01,
) -> np.ndarray:
    """Gradually adjust coords to close the circle.

    Moves the first and last C3' atoms toward each other to achieve
    the target bond distance while distributing small adjustments to
    neighboring atoms to preserve local structure.

    Args:
        coords: (L, 3) linear RNA coordinates.
        bond_length: target C3'-C3' distance for closure (A).
        n_steps: maximum number of adjustment steps.
        lr: learning rate for gradient-like adjustment.

    Returns:
        (L, 3) adjusted coordinates with improved closure.
    """
    coords = coords.copy()
    for step in range(n_steps):
        diff = coords[0] - coords[-1]
        dist = np.linalg.norm(diff)
        if dist < bond_length * 1.1:
            break
        correction = lr * (dist - bond_length) * diff / max(dist, 1e-6)
        coords[0] -= correction * 0.5
        coords[-1] += correction * 0.5
        # Distribute small adjustment to neighbors
        coords[1] -= correction * 0.1
        coords[-2] += correction * 0.1
    return coords


def direct_end_joining(
    coords: np.ndarray,
    bond_length: float = 5.9,
    n_steps: int = 300,
    lr: float = 0.02,
) -> np.ndarray:
    """Directly move the last atom toward the first to close the ring.

    A simpler closure approach: gradually pull the terminal atoms together
    while adjusting their neighbors to maintain local bond geometry.

    Args:
        coords: (L, 3) linear RNA coordinates.
        bond_length: target closure distance (A).
        n_steps: number of steps.
        lr: step size.

    Returns:
        (L, 3) coordinates with the ends joined.
    """
    coords = coords.copy()
    L = len(coords)

    for step in range(n_steps):
        # Closure gap
        diff = coords[0] - coords[-1]
        dist = np.linalg.norm(diff)
        if dist < bond_length * 1.05:
            break

        # Move last atom toward first
        direction = diff / max(dist, 1e-6)
        move = lr * (dist - bond_length)
        coords[-1] += move * direction

        # Compensate: adjust the second-to-last to maintain its bond to last
        if L >= 3:
            bond_vec = coords[-1] - coords[-2]
            bond_dist = np.linalg.norm(bond_vec)
            if bond_dist > 1e-6:
                ideal_bond = np.mean(
                    [np.linalg.norm(coords[i + 1] - coords[i]) for i in range(min(L - 1, 10))]
                )
                correction = (bond_dist - ideal_bond) * bond_vec / bond_dist * 0.3
                coords[-2] += correction

        # Similarly adjust near the 5' end
        if L >= 3:
            bond_vec = coords[1] - coords[0]
            bond_dist = np.linalg.norm(bond_vec)
            if bond_dist > 1e-6:
                ideal_bond = np.mean(
                    [np.linalg.norm(coords[i + 1] - coords[i]) for i in range(min(L - 1, 10))]
                )
                correction = (bond_dist - ideal_bond) * bond_vec / bond_dist * 0.3
                coords[1] += correction

    return coords


def circularize_with_solver(coords: np.ndarray) -> Optional[np.ndarray]:
    """Use GeometricConstraintSolver for annealing closure.

    Returns None if the solver is unavailable or fails.
    """
    if not _HAS_CONSTRAINT_SOLVER:
        return None

    try:
        config = SolverConfig(
            bond_length=BOND_LENGTH,
            use_annealing_closure=True,
            annealing_temp_init=500.0,
            annealing_temp_final=300.0,
            annealing_cooling=0.95,
            annealing_steps_per_temp=50,
            closure_tolerance=2.0,
            n_samples=1,
        )
        solver = GeometricConstraintSolver(config)

        # Build a minimal constraint set for the solver
        constraint_set = _MinimalConstraintSet(len(coords))
        conformations = solver.solve(constraint_set)

        if conformations:
            best = conformations[0]
            # Replace only the terminal region of the original coords
            # with solver output to preserve PDB accuracy
            L = len(coords)
            n_zone = max(3, L // 10)
            result = coords.copy()
            # Blend the solver's BSJ zone with the original
            for i in range(min(n_zone, L)):
                alpha = (n_zone - i) / n_zone * 0.5  # fade-out blend
                result[i] = (1 - alpha) * coords[i] + alpha * best[i]
                result[L - 1 - i] = (1 - alpha) * coords[L - 1 - i] + alpha * best[L - 1 - i]
            return result
    except Exception:
        pass

    return None


class _MinimalConstraintSet:
    """Minimal constraint set for GeometricConstraintSolver.

    Provides just the fields the solver needs: seq_len and an empty
    pair_constraints list.
    """

    def __init__(self, seq_len: int):
        self.seq_len = seq_len
        self.pair_constraints: list = []


def circularize(coords: np.ndarray, method: str = "auto") -> np.ndarray:
    """Circularize a linear RNA fragment.

    Strategy order (when method='auto'):
    1. Try GeometricConstraintSolver (if available)
    2. Try annealing_closure
    3. Fall back to direct_end_joining

    Args:
        coords: (L, 3) linear C3' coordinates.
        method: 'auto', 'solver', 'annealing', or 'direct'.

    Returns:
        (L, 3) circularized coordinates.
    """
    if method == "solver":
        result = circularize_with_solver(coords)
        return result if result is not None else annealing_closure(coords)

    if method == "annealing":
        return annealing_closure(coords)

    if method == "direct":
        return direct_end_joining(coords)

    # method == 'auto'
    result = circularize_with_solver(coords)
    if result is not None:
        return result

    # Try annealing closure first
    annealed = annealing_closure(coords)
    closure_dist = np.linalg.norm(annealed[0] - annealed[-1])
    if closure_dist < BOND_LENGTH * 2.0:
        return annealed

    # Fall back to direct end-joining
    return direct_end_joining(coords)


# ---------------------------------------------------------------------------
# Quality Filter
# ---------------------------------------------------------------------------

def check_closure(coords: np.ndarray, max_distance: float = 8.0) -> Tuple[bool, float]:
    """Check if the circularized structure has acceptable closure.

    Args:
        coords: (L, 3) coordinates.
        max_distance: maximum allowed distance between first and last atom.

    Returns:
        (passed, closure_distance)
    """
    dist = float(np.linalg.norm(coords[0] - coords[-1]))
    return dist < max_distance, dist


def check_steric_clashes(coords: np.ndarray, min_distance: float = 2.0) -> Tuple[bool, int]:
    """Check for severe steric clashes.

    A clash is when two non-adjacent atoms are closer than min_distance.

    Args:
        coords: (L, 3) coordinates.
        min_distance: minimum allowed distance for non-bonded atoms.

    Returns:
        (passed, clash_count)
    """
    L = len(coords)
    clash_count = 0

    for i in range(L):
        for j in range(i + 2, L):
            # Skip the closure pair (i=0, j=L-1) since it's the BSJ bond
            if i == 0 and j == L - 1:
                continue
            d = np.linalg.norm(coords[j] - coords[i])
            if d < min_distance:
                clash_count += 1

    return clash_count == 0, clash_count


def check_bond_rmsd(coords: np.ndarray, max_rmsd: float = 3.0) -> Tuple[bool, float]:
    """Check bond length RMSD from the reference bond length.

    Computes the RMS deviation of all consecutive-pair distances from
    BOND_LENGTH.  For a circularized structure this includes the closure
    bond (last -> first).

    Args:
        coords: (L, 3) coordinates.
        max_rmsd: maximum allowed RMSD.

    Returns:
        (passed, bond_rmsd)
    """
    L = len(coords)
    deviations = []
    for i in range(L):
        j = (i + 1) % L
        d = np.linalg.norm(coords[j] - coords[i])
        deviations.append(d - BOND_LENGTH)
    rmsd = float(np.sqrt(np.mean(np.array(deviations) ** 2)))
    return rmsd < max_rmsd, rmsd


def quality_filter(
    coords: np.ndarray,
    max_closure_distance: float = 8.0,
    min_clash_distance: float = 2.0,
    max_bond_rmsd: float = 3.0,
) -> Tuple[bool, Dict]:
    """Run all quality checks on a circularized structure.

    Returns:
        (passed, metrics_dict)
    """
    closure_ok, closure_dist = check_closure(coords, max_closure_distance)
    clash_ok, clash_count = check_steric_clashes(coords, min_clash_distance)
    rmsd_ok, bond_rmsd = check_bond_rmsd(coords, max_bond_rmsd)

    passed = closure_ok and clash_ok and rmsd_ok
    metrics = {
        "closure_distance": round(closure_dist, 3),
        "closure_ok": closure_ok,
        "clash_count": clash_count,
        "clash_ok": clash_ok,
        "bond_rmsd": round(bond_rmsd, 3),
        "rmsd_ok": rmsd_ok,
        "passed": passed,
    }
    return passed, metrics


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def rotation_matrix(axis: str, angle: float) -> np.ndarray:
    """Rotation matrix around a principal axis."""
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    else:
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def augment_structure(
    coords: np.ndarray,
    rng: np.random.RandomState,
    noise_scale: float = 0.3,
) -> np.ndarray:
    """Apply random rotation + small noise to a coordinate array.

    Rotation preserves inter-atomic distances (SO(3) invariant).
    Small Gaussian noise simulates structural uncertainty.
    """
    augmented = coords.copy()

    # Random rotation around all three axes
    angles = rng.uniform(0, 2 * np.pi, 3)
    R = rotation_matrix("x", angles[0]) @ rotation_matrix("y", angles[1]) @ rotation_matrix("z", angles[2])
    augmented = augmented @ R.T

    # Small Gaussian noise
    augmented += rng.normal(0, noise_scale, augmented.shape)

    return augmented


# ---------------------------------------------------------------------------
# Progress display (lightweight -- no external dependency)
# ---------------------------------------------------------------------------

class ProgressBar:
    """Simple text progress bar for the console."""

    def __init__(self, total: int, prefix: str = "", width: int = 40):
        self.total = total
        self.prefix = prefix
        self.width = width
        self.count = 0

    def update(self, n: int = 1):
        self.count += n
        frac = self.count / max(self.total, 1)
        filled = int(self.width * frac)
        bar = "#" * filled + "-" * (self.width - filled)
        pct = frac * 100
        sys.stdout.write(f"\r  {self.prefix} [{bar}] {pct:5.1f}% ({self.count}/{self.total})")
        sys.stdout.flush()
        if self.count >= self.total:
            sys.stdout.write("\n")

    def close(self):
        if self.count < self.total:
            sys.stdout.write("\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    output_dir: str,
    pdb_cache: str,
    target_samples: int = 4000,
    augment_multiplier: int = 8,
    min_length: int = 30,
    max_length: int = 500,
    max_resolution: float = 3.0,
    closure_method: str = "auto",
    max_closure_distance: float = 8.0,
    min_clash_distance: float = 2.0,
    max_bond_rmsd: float = 3.0,
    noise_scale: float = 0.3,
    seed: int = 42,
    cache_only: bool = False,
) -> None:
    """Execute the full PDB RNA circularization pipeline."""
    rng = np.random.RandomState(seed)

    os.makedirs(output_dir, exist_ok=True)
    coords_dir = os.path.join(output_dir, "coords")
    os.makedirs(coords_dir, exist_ok=True)

    all_sequences: List[Dict] = []
    all_coords: List[np.ndarray] = []
    all_metrics: List[Dict] = []

    # ------------------------------------------------------------------
    # Step 1: Collect PDB IDs
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  PDB RNA Circularization Pipeline")
    print("=" * 60)
    print(f"  Target: ~{target_samples} circularized samples")
    print(f"  Augmentation: {augment_multiplier}x")
    print(f"  Length range: {min_length}-{max_length} nt")
    print(f"  Max resolution: {max_resolution} A")
    print()

    print("[1/5] Searching RCSB PDB and building ID pool...")
    if cache_only:
        # Scan cache directory for existing .pdb files instead of searching RCSB
        import glob as _glob
        cached_pdbs = _glob.glob(os.path.join(pdb_cache, "*.pdb"))
        pdb_ids = [os.path.splitext(os.path.basename(p))[0].upper() for p in cached_pdbs]
        print(f"  Cache-only mode: found {len(pdb_ids)} PDB files in cache")
    else:
        pdb_ids = get_pdb_ids(
            max_resolution=max_resolution,
            min_length=min_length,
            max_length=max_length,
        )
    # For efficiency, only download enough PDBs to likely get target_samples
    # Roughly 10-20% of PDBs yield valid RNA chains, so download ~5x target
    max_downloads = min(len(pdb_ids), target_samples * 5 + 50)
    if max_downloads < len(pdb_ids):
        print(f"  Limiting downloads to {max_downloads} (need ~{target_samples} samples)")
    pdb_ids_to_download = pdb_ids[:max_downloads]
    print(f"  Total PDB IDs to process: {len(pdb_ids_to_download)}")
    print()

    # ------------------------------------------------------------------
    # Step 2: Download and parse PDB files
    # ------------------------------------------------------------------
    print("[2/5] Downloading and parsing PDB files...")
    rna_fragments: List[Dict] = []  # {pdb_id, chain_id, sequence, coords}
    dl_bar = ProgressBar(len(pdb_ids_to_download), prefix="Download")

    for pdb_id in pdb_ids_to_download:
        dl_bar.update(1)

        # Early stop if we have enough fragments
        if len(rna_fragments) >= target_samples * 2:
            print(f"\n  Found {len(rna_fragments)} fragments, enough for {target_samples} target")
            break
        pdb_path = download_pdb(pdb_id, pdb_cache, cache_only=cache_only)
        if pdb_path is None:
            continue

        try:
            chains = parse_pdb_rna(pdb_path)
        except Exception as exc:
            print(f"\n    Parse error {pdb_id}: {exc}")
            continue

        for chain in chains:
            seq_len = len(chain["sequence"])
            if seq_len < min_length or seq_len > max_length:
                continue
            rna_fragments.append({
                "pdb_id": pdb_id,
                "chain_id": chain["chain_id"],
                "sequence": chain["sequence"],
                "coords": chain["coords"],
            })

    dl_bar.close()
    print(f"  Extracted {len(rna_fragments)} RNA fragments")
    print()

    if not rna_fragments:
        print("  No RNA fragments found. Exiting.")
        return

    # ------------------------------------------------------------------
    # Step 3: Circularize
    # ------------------------------------------------------------------
    print("[3/5] Circularizing RNA fragments...")
    circ_bar = ProgressBar(len(rna_fragments), prefix="Circularize")
    circularized: List[Dict] = []
    failed_closure = 0

    for frag in rna_fragments:
        circ_bar.update(1)
        try:
            circ_coords = circularize(frag["coords"], method=closure_method)
        except Exception:
            failed_closure += 1
            continue

        circularized.append({
            "pdb_id": frag["pdb_id"],
            "chain_id": frag["chain_id"],
            "sequence": frag["sequence"],
            "coords": circ_coords,
        })

    circ_bar.close()
    print(f"  Circularized: {len(circularized)}, Failed: {failed_closure}")
    print()

    # ------------------------------------------------------------------
    # Step 4: Quality filter
    # ------------------------------------------------------------------
    print("[4/5] Quality filtering...")
    filt_bar = ProgressBar(len(circularized), prefix="Filter")
    passed_fragments: List[Dict] = []
    passed_metrics: List[Dict] = []
    reject_reasons = {"closure": 0, "clash": 0, "rmsd": 0}

    for item in circularized:
        filt_bar.update(1)
        ok, metrics = quality_filter(
            item["coords"],
            max_closure_distance=max_closure_distance,
            min_clash_distance=min_clash_distance,
            max_bond_rmsd=max_bond_rmsd,
        )

        if ok:
            passed_fragments.append(item)
            passed_metrics.append(metrics)
        else:
            if not metrics["closure_ok"]:
                reject_reasons["closure"] += 1
            if not metrics["clash_ok"]:
                reject_reasons["clash"] += 1
            if not metrics["rmsd_ok"]:
                reject_reasons["rmsd"] += 1

    filt_bar.close()
    print(f"  Passed: {len(passed_fragments)}, Rejected: {len(circularized) - len(passed_fragments)}")
    print(f"    Rejection reasons: closure={reject_reasons['closure']}, "
          f"clash={reject_reasons['clash']}, rmsd={reject_reasons['rmsd']}")
    print()

    if not passed_fragments:
        print("  No fragments passed quality filter. Exiting.")
        return

    # ------------------------------------------------------------------
    # Step 5: Augment and assemble dataset
    # ------------------------------------------------------------------
    print(f"[5/5] Augmenting {len(passed_fragments)} structures x{augment_multiplier}...")

    # Compute how many base structures we need to reach the target
    # Handle augment_multiplier=0 case
    effective_augment = max(augment_multiplier, 1)
    needed_base = math.ceil(target_samples / effective_augment)
    if len(passed_fragments) > needed_base:
        # Subsample to avoid exceeding target
        indices = rng.choice(len(passed_fragments), needed_base, replace=False)
        base_fragments = [passed_fragments[i] for i in indices]
        base_metrics = [passed_metrics[i] for i in indices]
    else:
        base_fragments = passed_fragments
        base_metrics = passed_metrics

    # Add original structures (with SS prediction)
    ss_cache: Dict[str, Tuple[str, List]] = {}  # sequence -> (ss, pairs)

    for idx, (frag, metrics) in enumerate(zip(base_fragments, base_metrics)):
        seq_id = f"pdb_{frag['pdb_id']}_{frag['chain_id']}"
        sequence = frag["sequence"]

        # Predict SS (cache to avoid re-predicting for augmentations)
        if sequence not in ss_cache:
            ss_cache[sequence] = predict_ss_and_pairs(sequence)
        ss, pair_constraints = ss_cache[sequence]

        all_sequences.append({
            "id": seq_id,
            "sequence": sequence,
            "secondary_structure": ss,
            "pair_constraints": pair_constraints,
            "length": len(sequence),
            "source": "pdb_circularized",
            "pdb_id": frag["pdb_id"],
            "chain_id": frag["chain_id"],
            "closure_distance": metrics["closure_distance"],
            "bond_rmsd": metrics["bond_rmsd"],
        })
        all_coords.append(frag["coords"])
        all_metrics.append({
            "id": seq_id,
            "source": "original",
            **metrics,
        })

    # Augmented copies
    aug_bar = ProgressBar(len(base_fragments) * (augment_multiplier - 1), prefix="Augment")
    for frag, metrics in zip(base_fragments, base_metrics):
        for aug_idx in range(1, augment_multiplier):
            aug_bar.update(1)
            aug_coords = augment_structure(frag["coords"], rng, noise_scale=noise_scale)

            seq_id = f"pdb_aug{aug_idx}_{frag['pdb_id']}_{frag['chain_id']}"
            sequence = frag["sequence"]
            # Reuse cached SS prediction
            if sequence not in ss_cache:
                ss_cache[sequence] = predict_ss_and_pairs(sequence)
            ss, pair_constraints = ss_cache[sequence]

            all_sequences.append({
                "id": seq_id,
                "sequence": sequence,
                "secondary_structure": ss,
                "pair_constraints": pair_constraints,
                "length": len(sequence),
                "source": "pdb_circularized_aug",
                "pdb_id": frag["pdb_id"],
                "chain_id": frag["chain_id"],
                "augmentation_index": aug_idx,
            })
            all_coords.append(aug_coords)
            all_metrics.append({
                "id": seq_id,
                "source": f"augmented_{aug_idx}",
                "original_closure_distance": metrics["closure_distance"],
                "original_bond_rmsd": metrics["bond_rmsd"],
            })

    aug_bar.close()

    # ------------------------------------------------------------------
    # Save dataset
    # ------------------------------------------------------------------
    print(f"\n  Saving {len(all_sequences)} samples...")

    save_bar = ProgressBar(len(all_coords), prefix="Save")
    for i, (seq_item, coords) in enumerate(zip(all_sequences, all_coords)):
        np.save(os.path.join(coords_dir, f"{seq_item['id']}.npy"), coords)
        save_bar.update(1)
    save_bar.close()

    with open(os.path.join(output_dir, "sequences.json"), "w") as f:
        json.dump(all_sequences, f, indent=2)

    # Build metadata
    pdb_sources = set(s["pdb_id"] for s in all_sequences)
    metadata = {
        "total": len(all_sequences),
        "target": target_samples,
        "length_range": [
            min(s["length"] for s in all_sequences),
            max(s["length"] for s in all_sequences),
        ],
        "augmentation_multiplier": augment_multiplier,
        "sources": {
            "pdb_circularized": sum(1 for s in all_sequences if s["source"] == "pdb_circularized"),
            "pdb_circularized_aug": sum(1 for s in all_sequences if s["source"] == "pdb_circularized_aug"),
        },
        "unique_pdb_entries": len(pdb_sources),
        "quality_thresholds": {
            "max_closure_distance": max_closure_distance,
            "min_clash_distance": min_clash_distance,
            "max_bond_rmsd": max_bond_rmsd,
        },
        "closure_method": closure_method,
        "seed": seed,
        "samples": all_metrics[:200],
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Dataset: {output_dir}/")
    print(f"  Total: {metadata['total']}")
    print(f"  Unique PDB entries: {metadata['unique_pdb_entries']}")
    print(f"  Original (non-augmented): {metadata['sources']['pdb_circularized']}")
    print(f"  Augmented: {metadata['sources']['pdb_circularized_aug']}")
    print(f"  Length range: {metadata['length_range']}")
    print(f"  Closure method: {closure_method}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract RNA from PDB, circularize, and build circRNA training dataset",
    )
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for the dataset")
    parser.add_argument("--pdb-cache", type=str, default="./pdb_cache",
                        help="Directory to cache downloaded PDB files")
    parser.add_argument("--target", type=int, default=4000,
                        help="Target number of circularized samples (default: 4000)")
    parser.add_argument("--augment", type=int, default=8,
                        help="Augmentation multiplier per structure (default: 8)")
    parser.add_argument("--min-length", type=int, default=30,
                        help="Minimum RNA chain length (default: 30)")
    parser.add_argument("--max-length", type=int, default=500,
                        help="Maximum RNA chain length (default: 500)")
    parser.add_argument("--max-resolution", type=float, default=3.0,
                        help="Maximum PDB resolution in Angstroms (default: 3.0)")
    parser.add_argument("--closure-method", type=str, default="auto",
                        choices=["auto", "solver", "annealing", "direct"],
                        help="Circularization method (default: auto)")
    parser.add_argument("--max-closure-distance", type=float, default=8.0,
                        help="Max closure distance for quality filter (default: 8.0)")
    parser.add_argument("--min-clash-distance", type=float, default=2.0,
                        help="Min non-bonded distance for clash filter (default: 2.0)")
    parser.add_argument("--max-bond-rmsd", type=float, default=3.0,
                        help="Max bond RMSD for quality filter (default: 3.0)")
    parser.add_argument("--noise-scale", type=float, default=0.3,
                        help="Gaussian noise scale for augmentation (default: 0.3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--cache-only", action="store_true",
                        help="Only use cached PDB files; skip all downloads")
    args = parser.parse_args()

    run_pipeline(
        output_dir=args.output,
        pdb_cache=args.pdb_cache,
        target_samples=args.target,
        augment_multiplier=args.augment,
        min_length=args.min_length,
        max_length=args.max_length,
        max_resolution=args.max_resolution,
        closure_method=args.closure_method,
        max_closure_distance=args.max_closure_distance,
        min_clash_distance=args.min_clash_distance,
        max_bond_rmsd=args.max_bond_rmsd,
        noise_scale=args.noise_scale,
        seed=args.seed,
        cache_only=args.cache_only,
    )


if __name__ == "__main__":
    main()
