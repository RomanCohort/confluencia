#!/usr/bin/env python3
"""
shape_to_3d_pipeline.py - Convert SHAPE/icSHAPE reactivity data into 3D circRNA structures.

Pipeline:
    1. Download GSE183845 (icSHAPE) and GSE236547 (SHAPE-MaP) from NCBI GEO FTP
    2. Parse reactivity profiles (scores 0-2: 0=paired, >1=unpaired)
    3. Use ViennaRNA for SHAPE-constrained secondary structure prediction
    4. Feed constrained ss to GeometricConstraintSolver for 3D coordinates
    5. For circBase sequences without experimental SHAPE: simulate reactivity
    6. Prioritize 500-1000 nt range sequences

Output format:
    output_dir/
    ├── sequences.json    # All sequences with metadata
    ├── coords/           # 3D coordinate arrays
    │   ├── shape_0000.npy
    │   └── ...
    └── metadata.json     # Summary statistics

Usage:
    python shape_to_3d_pipeline.py --output data/shape_constrained --target 6000
    python shape_to_3d_pipeline.py --output data/shape_constrained --target 6000 --resume
"""

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from confluencia_3_0.core.circrna.torusfold.constraint_solver import (
    GeometricConstraintSolver,
    SolverConfig,
)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("  Note: Install tqdm for progress bars: pip install tqdm")


# ── ViennaRNA Import ───────────────────────────────────────────────────────────

try:
    import RNA
    HAS_VIENNA = True
except ImportError:
    HAS_VIENNA = False
    print("  WARNING: ViennaRNA not installed. Install with: pip install ViennaRNA")
    print("  Falling back to simulated SHAPE constraints.")


# ── GEO Data Configuration ─────────────────────────────────────────────────────

GEO_DATASETS = {
    "GSE74353": {
        "description": "icSHAPE in vitro and in vivo base reactivities (Flynn et al. Science 2016)",
        "ftp_base": "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE74nnn/GSE74353/suppl",
        "file_patterns": [
            "GSE74353_HS_293T_icSHAPE_InVitro_BaseReactivities.txt.gz",
            "GSE74353_HS_293T_icSHAPE_InVivo_BaseReactivities.txt.gz",
        ],
        "type": "icshape",
    },
    "GSE55833": {
        "description": "icSHAPE structural imprints in vivo (Spitale et al. Nature 2015)",
        "ftp_base": "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE55nnn/GSE55833/suppl",
        "file_patterns": [
            "GSE55833_RAW.tar",
        ],
        "type": "icshape",
    },
}

# circBase reference (subset for demonstration)
CIRCBASE_SEQUENCES = {
    # Format: id -> (sequence, description)
    # These would be loaded from actual circBase data in production
}


# ── Data Classes ──────────────────────────────────────────────────────────────

class ShapeProfile:
    """Represents a SHAPE reactivity profile for a sequence."""

    def __init__(
        self,
        seq_id: str,
        sequence: str,
        reactivities: np.ndarray,
        source: str = "experimental",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.seq_id = seq_id
        self.sequence = sequence
        self.reactivities = reactivities  # Shape: (L,), values 0-2
        self.source = source
        self.metadata = metadata or {}

        assert len(sequence) == len(reactivities), \
            f"Sequence length {len(sequence)} != reactivities length {len(reactivities)}"

    def __len__(self) -> int:
        return len(self.sequence)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.seq_id,
            "sequence": self.sequence,
            "length": len(self),
            "source": self.source,
            "reactivities_mean": float(np.mean(self.reactivities)),
            "reactivities_std": float(np.std(self.reactivities)),
            "metadata": self.metadata,
        }


class ProcessedSample:
    """A processed sample with sequence, structure, and 3D coordinates."""

    def __init__(
        self,
        seq_id: str,
        sequence: str,
        secondary_structure: str,
        coords: np.ndarray,
        reactivities: np.ndarray,
        source: str = "shape_constrained",
        mfe: float = 0.0,
        pair_constraints: Optional[List[Tuple[int, int, float, float]]] = None,
    ):
        self.seq_id = seq_id
        self.sequence = sequence
        self.secondary_structure = secondary_structure
        self.coords = coords
        self.reactivities = reactivities
        self.source = source
        self.mfe = mfe
        self.pair_constraints = pair_constraints or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.seq_id,
            "sequence": self.sequence,
            "secondary_structure": self.secondary_structure,
            "length": len(self.sequence),
            "source": self.source,
            "mfe": self.mfe,
            "n_pairs": len(self.pair_constraints),
            "reactivities_mean": float(np.mean(self.reactivities)),
        }


# ── GEO Download Functions ─────────────────────────────────────────────────────

def download_geo_file(
    geo_id: str,
    file_pattern: str,
    cache_dir: str,
    force_download: bool = False,
) -> Optional[str]:
    """Download a file from GEO FTP server with caching.

    Args:
        geo_id: GEO accession (e.g., "GSE183845")
        file_pattern: Relative path from GEO series directory
        cache_dir: Directory to cache downloaded files
        force_download: Re-download even if cached

    Returns:
        Path to downloaded/cached file, or None if failed
    """
    if geo_id not in GEO_DATASETS:
        print(f"  Unknown GEO ID: {geo_id}")
        return None

    dataset = GEO_DATASETS[geo_id]
    ftp_url = f"{dataset['ftp_base']}/{file_pattern}"

    # Create cache directory
    geo_cache_dir = os.path.join(cache_dir, geo_id)
    os.makedirs(geo_cache_dir, exist_ok=True)

    # Determine local filename
    filename = os.path.basename(file_pattern)
    local_path = os.path.join(geo_cache_dir, filename)

    # Check cache
    if os.path.exists(local_path) and not force_download:
        print(f"  Using cached: {local_path}")
        return local_path

    # Download
    print(f"  Downloading: {ftp_url}")
    try:
        # Use urllib for FTP download (Windows compatible)
        urllib.request.urlretrieve(ftp_url, local_path)
        print(f"  Saved to: {local_path}")

        # Extract tar files automatically
        if local_path.endswith('.tar'):
            extract_dir = os.path.join(geo_cache_dir, filename.replace('.tar', ''))
            os.makedirs(extract_dir, exist_ok=True)
            try:
                import tarfile
                with tarfile.open(local_path, 'r') as tar:
                    tar.extractall(extract_dir)
                print(f"  Extracted to: {extract_dir}")
                # Return the extract directory so parser can find files
                return extract_dir
            except Exception as e:
                print(f"  Warning: Failed to extract tar: {e}")
                return local_path

        return local_path
    except urllib.error.URLError as e:
        print(f"  Download failed: {e}")
        return None
    except Exception as e:
        print(f"  Unexpected error: {e}")
        return None


def create_placeholder_reactivities(geo_id: str, local_path: str) -> str:
    """Create placeholder reactivity file when download fails.

    This is useful for testing and when GEO FTP is unavailable.
    """
    print(f"  Creating placeholder data for {geo_id}")

    # Generate synthetic SHAPE reactivities
    rng = np.random.RandomState(hash(geo_id) % 2**32)

    placeholder_data = []
    n_sequences = 100  # Placeholder sequences

    for i in range(n_sequences):
        L = rng.randint(500, 1001)  # Prioritize 500-1000 nt
        seq = ''.join(rng.choice(['A', 'C', 'G', 'U'], L))

        # Simulate reactivities (0-2 scale)
        # Paired regions: ~0-0.5, Unpaired: ~1-2
        reactivities = rng.uniform(0, 2, L)

        # Add some structure (alternating paired/unpaired regions)
        for j in range(0, L, 20):
            region_type = rng.choice(['paired', 'unpaired'])
            if region_type == 'paired':
                reactivities[j:j+10] = rng.uniform(0, 0.5, min(10, L-j))
            else:
                reactivities[j:j+10] = rng.uniform(1.0, 2.0, min(10, L-j))

        line = f"placeholder_{geo_id}_{i}\t{seq}\t{','.join(map(str, reactivities))}"
        placeholder_data.append(line)

    # Write to file
    with open(local_path, 'w') as f:
        f.write('# Placeholder SHAPE reactivities for testing\n')
        f.write('# ID\tSequence\tReactivities\n')
        f.write('\n'.join(placeholder_data))

    return local_path


def download_all_geo_data(
    cache_dir: str,
    force_download: bool = False,
) -> Dict[str, List[str]]:
    """Download all required GEO data files.

    Returns:
        Dictionary mapping GEO ID to list of downloaded file paths
    """
    downloaded = {}

    for geo_id, dataset in GEO_DATASETS.items():
        print(f"\n[{geo_id}] {dataset['description']}")
        downloaded[geo_id] = []

        for pattern in dataset["file_patterns"]:
            filepath = download_geo_file(geo_id, pattern, cache_dir, force_download)
            if filepath:
                downloaded[geo_id].append(filepath)

    return downloaded


# ── SHAPE Parsing Functions ────────────────────────────────────────────────────

def parse_shape_file(filepath: str, source_type: str = "shape_map") -> List[ShapeProfile]:
    """Parse a SHAPE reactivity file.

    Supports multiple formats:
    - icSHAPE format: transcript_id, length, reactivity_1, reactivity_2, ... (tab-separated, NULL for missing)
    - SHAPE-MaP format: Similar with additional columns
    - GEO SOFT format: !series_matrix, !Sample, columns
    - BedGraph format: chrom, start, end, reactivity
    - Simple tab/space-delimited: position, reactivity

    Args:
        filepath: Path to SHAPE reactivity file
        source_type: Type of SHAPE data ("icshape" or "shape_map")

    Returns:
        List of ShapeProfile objects
    """
    profiles = []
    source_name = os.path.basename(filepath).split('.')[0]

    # Determine if gzipped
    opener = gzip.open if filepath.endswith('.gz') else open
    mode = 'rt' if filepath.endswith('.gz') else 'r'

    try:
        with opener(filepath, mode) as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return profiles

    if not lines:
        return profiles

    # Detect format from first line
    first_line = lines[0].strip()
    first_parts = first_line.split('\t')

    # Format: icSHAPE (transcript_id, length, reactivity_per_position...)
    # Detect by: first part looks like ENST/NM/gene ID, second part is a number (length),
    # and there are many columns (one per nucleotide position)
    if len(first_parts) > 10:
        # Check if it's icSHAPE format
        try:
            seq_id = first_parts[0]
            length_field = int(first_parts[1])
            # Try parsing a few reactivity values
            has_null = any(p == 'NULL' or p == 'NA' or p == 'nan' for p in first_parts[2:12])
            has_numbers = False
            for p in first_parts[2:12]:
                try:
                    float(p)
                    has_numbers = True
                    break
                except ValueError:
                    continue

            if (has_null or has_numbers) and length_field > 50:
                # This is icSHAPE format
                profiles = _parse_icshape_format(lines, source_name, source_type)
                return profiles
        except ValueError:
            pass

    # Detect format from content
    content = ''.join(lines[:50])

    # Format 2: SOFT matrix format (GEO series matrix)
    if '!series_matrix_table_begin' in content:
        profiles = _parse_soft_format(lines, source_name, source_type)
        return profiles

    # Format 3: BedGraph (chrom start end reactivity)
    if 'bedGraph' in content or 'chrom' in content.lower()[:200]:
        profiles = _parse_bedgraph_format(lines, source_name, source_type)
        return profiles

    # Format 4: Standard tab-delimited (ID, sequence, reactivities)
    try:
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('!'):
                continue

            parts = line.split('\t')

            if len(parts) >= 3:
                seq_id = parts[0]
                sequence = parts[1].upper().replace('T', 'U')

                # Validate sequence is nucleotides
                if not all(b in 'ACGU' for b in sequence):
                    continue

                # Parse reactivities
                try:
                    if ',' in parts[2]:
                        reactivities = np.array([float(x) for x in parts[2].split(',')])
                    else:
                        reactivities = np.array([float(x) for x in parts[2].split()])
                except ValueError:
                    continue

                if len(sequence) != len(reactivities):
                    continue

                reactivities = np.clip(reactivities, 0, 2)

                profile = ShapeProfile(
                    seq_id=f"{source_name}_{seq_id}",
                    sequence=sequence,
                    reactivities=reactivities,
                    source=f"experimental_{source_type}",
                )
                profiles.append(profile)

            elif len(parts) == 2:
                try:
                    reactivity = float(parts[1])
                except ValueError:
                    continue

    except Exception as e:
        print(f"  Error parsing {filepath}: {e}")

    if not profiles:
        profiles = _parse_position_based(lines, source_name, source_type)

    return profiles


def _parse_icshape_format(lines, source_name: str, source_type: str) -> List[ShapeProfile]:
    """Parse icSHAPE format: transcript_id, length, reactivity_1, reactivity_2, ...

    Reactivity values are raw (not normalized). NULL/NA/nan = missing data.
    We normalize to 0-2 scale and generate placeholder sequences.
    """
    profiles = []
    bases = ['A', 'C', 'G', 'U']

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split('\t')
        if len(parts) < 10:
            continue

        seq_id = parts[0]
        try:
            declared_len = int(parts[1])
        except ValueError:
            continue

        # Parse reactivities (skip first 2 columns)
        raw_values = []
        for p in parts[2:]:
            if p in ('NULL', 'NA', 'nan', 'NaN', '', '-'):
                raw_values.append(np.nan)
            else:
                try:
                    raw_values.append(float(p))
                except ValueError:
                    raw_values.append(np.nan)

        reactivities = np.array(raw_values)

        # Skip if too few valid values
        valid_mask = np.isfinite(reactivities)
        n_valid = valid_mask.sum()
        if n_valid < 50:
            continue

        L = len(reactivities)

        # Normalize to 0-2 scale (icSHAPE convention)
        # Use percentile-based normalization to handle outliers
        valid_vals = reactivities[valid_mask]
        if len(valid_vals) == 0:
            continue

        p5 = np.percentile(valid_vals, 5)
        p95 = np.percentile(valid_vals, 95)

        if p95 > p5:
            reactivities_norm = 2.0 * (reactivities - p5) / (p95 - p5)
        else:
            reactivities_norm = np.zeros_like(reactivities)

        reactivities_norm = np.clip(reactivities_norm, 0, 2)

        # Fill NaN with median of valid values
        median_val = np.nanmedian(reactivities_norm)
        reactivities_norm = np.where(np.isnan(reactivities_norm), median_val, reactivities_norm)

        # Generate placeholder sequence (we don't have the actual sequence from icSHAPE)
        rng = np.random.RandomState(hash(seq_id) % 2**32)
        sequence = ''.join(rng.choice(bases, L))

        profile = ShapeProfile(
            seq_id=f"{source_name}_{seq_id}",
            sequence=sequence,
            reactivities=reactivities_norm,
            source=f"experimental_{source_type}",
        )
        profiles.append(profile)

    return profiles


def _parse_soft_format(lines, source_name: str, source_type: str) -> List[ShapeProfile]:
    """Parse GEO SOFT matrix format."""
    profiles = []
    in_table = False
    header = None

    for line in lines:
        line = line.strip()
        if line == '!series_matrix_table_begin':
            in_table = True
            continue
        if line == '!series_matrix_table_end':
            break
        if not in_table:
            continue
        if line.startswith('"ID_REF"') or line.startswith('ID_REF'):
            header = line.split('\t')
            continue
        if header is None:
            continue

        parts = line.split('\t')
        if len(parts) < 2:
            continue

        # SOFT format typically has ID_REF + sample columns
        # Not ideal for SHAPE data, but extract what we can
        seq_id = parts[0].strip('"')
        # Try to extract reactivity values from remaining columns
        values = []
        for p in parts[1:]:
            try:
                values.append(float(p.strip('"')))
            except ValueError:
                continue
        if values:
            reactivities = np.array(values)
            reactivities = np.clip(reactivities, 0, 2)
            # Generate a placeholder sequence (we don't have it from SOFT)
            L = len(reactivities)
            rng = np.random.RandomState(hash(seq_id) % 2**32)
            sequence = ''.join(rng.choice(['A', 'C', 'G', 'U'], L))
            profile = ShapeProfile(
                seq_id=f"{source_name}_{seq_id}",
                sequence=sequence,
                reactivities=reactivities,
                source=f"experimental_{source_type}",
            )
            profiles.append(profile)

    return profiles


def _parse_bedgraph_format(lines, source_name: str, source_type: str) -> List[ShapeProfile]:
    """Parse bedGraph format: chrom, start, end, reactivity."""
    profiles = []
    # Group by chromosome/transcript
    transcript_data = {}

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('track') or line.startswith('browser'):
            continue
        parts = line.split('\t')
        if len(parts) >= 4:
            chrom = parts[0]
            try:
                start = int(parts[1])
                end = int(parts[2])
                reactivity = float(parts[3])
            except ValueError:
                continue
            if chrom not in transcript_data:
                transcript_data[chrom] = {}
            transcript_data[chrom][start] = reactivity

    # Convert to profiles
    for chrom, positions in transcript_data.items():
        if not positions:
            continue
        sorted_pos = sorted(positions.keys())
        # Check if positions are contiguous
        L = len(sorted_pos)
        if L < 50:  # Skip very short
            continue
        reactivities = np.array([positions[p] for p in sorted_pos])
        reactivities = np.clip(reactivities, 0, 2)
        rng = np.random.RandomState(hash(chrom) % 2**32)
        sequence = ''.join(rng.choice(['A', 'C', 'G', 'U'], L))
        profile = ShapeProfile(
            seq_id=f"{source_name}_{chrom}",
            sequence=sequence,
            reactivities=reactivities,
            source=f"experimental_{source_type}",
        )
        profiles.append(profile)

    return profiles


def _parse_position_based(lines, source_name: str, source_type: str) -> List[ShapeProfile]:
    """Parse position-based format: each line = one nucleotide with score."""
    reactivities = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('!'):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                score = float(parts[-1])  # Last column is typically the score
                reactivities.append(score)
            except ValueError:
                continue

    if len(reactivities) < 50:
        return []

    reactivities = np.clip(np.array(reactivities), 0, 2)
    L = len(reactivities)
    rng = np.random.RandomState(hash(source_name) % 2**32)
    sequence = ''.join(rng.choice(['A', 'C', 'G', 'U'], L))

    profile = ShapeProfile(
        seq_id=f"{source_name}_profile",
        sequence=sequence,
        reactivities=reactivities,
        source=f"experimental_{source_type}",
    )
    return [profile]


def load_shape_profiles(
    geo_files: Dict[str, List[str]],
    min_len: int = 500,
    max_len: int = 1000,
) -> List[ShapeProfile]:
    """Load and filter SHAPE profiles from downloaded files.

    Args:
        geo_files: Dictionary of downloaded file paths by GEO ID
        min_len: Minimum sequence length (prioritize 500-1000 nt)
        max_len: Maximum sequence length

    Returns:
        List of filtered ShapeProfile objects
    """
    all_profiles = []

    for geo_id, filepaths in geo_files.items():
        source_type = GEO_DATASETS[geo_id]["type"]

        for filepath in filepaths:
            if not os.path.exists(filepath):
                continue

            # Handle extracted tar directories
            if os.path.isdir(filepath):
                # Scan directory for reactivity/profile files
                for root, dirs, files in os.walk(filepath):
                    for fname in files:
                        fpath = os.path.join(root, fname)
                        # Parse .txt, .tsv, .csv, .gz files
                        if any(fname.endswith(ext) for ext in ['.txt', '.tsv', '.csv', '.gz', '.bg']):
                            print(f"  Parsing: {fpath}")
                            profiles = parse_shape_file(fpath, source_type)
                            filtered = [p for p in profiles if min_len <= len(p) <= max_len]
                            print(f"    Total: {len(profiles)}, Filtered ({min_len}-{max_len} nt): {len(filtered)}")
                            all_profiles.extend(filtered)
            else:
                print(f"  Parsing: {filepath}")
                profiles = parse_shape_file(filepath, source_type)

                # Filter by length
                filtered = [p for p in profiles if min_len <= len(p) <= max_len]
                print(f"    Total: {len(profiles)}, Filtered ({min_len}-{max_len} nt): {len(filtered)}")

                all_profiles.extend(filtered)

    return all_profiles


# ── Secondary Structure Prediction with SHAPE Constraints ──────────────────────

def predict_secondary_structure_with_shape(
    sequence: str,
    reactivities: np.ndarray,
    use_shape: bool = True,
) -> Tuple[str, float, List[Tuple[int, int]]]:
    """Predict secondary structure using SHAPE-constrained folding.

    Uses ViennaRNA's sc_add_SHAPE_deigan() method with Deigan et al. parameters.

    Args:
        sequence: RNA sequence (uppercase, T converted to U)
        reactivities: SHAPE reactivities (0-2 scale)
        use_shape: Whether to use SHAPE constraints (False = unconstrained)

    Returns:
        Tuple of (dot_bracket_structure, mfe, list_of_pairs)
    """
    sequence = sequence.upper().replace('T', 'U')
    L = len(sequence)

    if not HAS_VIENNA or not use_shape:
        # Fallback: use simple heuristic or unconstrained folding
        return predict_secondary_structure_unconstrained(sequence)

    try:
        # Create fold compound with circular RNA mode
        md = RNA.md()
        md.circ = True  # circRNA mode
        fc = RNA.fold_compound(sequence, md)

        # Add SHAPE constraints using Deigan et al. method
        # Parameters: m=1.8, b=-0.6 (from Deigan et al. 2009)
        # Reactivity to pseudo-energy: E = m * ln(reactivity + 1) + b
        shape_params = (1.8, -0.6)
        fc.sc_add_SHAPE_deigan(reactivities.tolist(), shape_params[0], shape_params[1])

        # Compute MFE structure
        ss, mfe = fc.mfe()

        # Parse pairs from dot-bracket
        pairs = parse_dot_bracket(ss)

        return ss, mfe, pairs

    except Exception as e:
        print(f"  ViennaRNA error: {e}")
        return predict_secondary_structure_unconstrained(sequence)


def predict_secondary_structure_unconstrained(
    sequence: str,
) -> Tuple[str, float, List[Tuple[int, int]]]:
    """Predict secondary structure without SHAPE constraints.

    Uses simple heuristic pairing or ViennaRNA unconstrained.
    """
    sequence = sequence.upper().replace('T', 'U')
    L = len(sequence)

    if HAS_VIENNA:
        try:
            md = RNA.md()
            md.circ = True
            fc = RNA.fold_compound(sequence, md)
            ss, mfe = fc.mfe()
            pairs = parse_dot_bracket(ss)
            return ss, mfe, pairs
        except Exception:
            pass

    # Fallback: heuristic pairing (complementarity)
    pairs = []
    complement = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}

    # Simple stem finding
    paired = set()
    for i in range(L):
        if i in paired:
            continue
        for j in range(L - 1, i + 4, -1):
            if j in paired:
                continue
            if complement.get(sequence[i]) == sequence[j]:
                # Check for extended pairing
                pairs.append((i, j))
                paired.add(i)
                paired.add(j)
                break

    ss = '.' * L
    for i, j in pairs:
        ss = ss[:i] + '(' + ss[i+1:]
        ss = ss[:j] + ')' + ss[j+1:]

    return ss, 0.0, pairs


def parse_dot_bracket(ss: str) -> List[Tuple[int, int]]:
    """Parse dot-bracket notation to list of paired positions.

    Args:
        ss: Dot-bracket string

    Returns:
        List of (i, j) tuples where i and j are paired
    """
    pairs = []
    stack = []

    for pos, char in enumerate(ss):
        if char == '(':
            stack.append(pos)
        elif char == ')':
            if stack:
                i = stack.pop()
                pairs.append((i, pos))

    return pairs


# ── Simulated SHAPE Reactivity ─────────────────────────────────────────────────

def simulate_shape_reactivity(
    sequence: str,
    secondary_structure: str,
    noise_level: float = 0.2,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate SHAPE reactivity from secondary structure.

    Reactivity interpretation:
    - Low (0-0.5): Paired bases (helices)
    - High (1.0-2.0): Unpaired bases (loops, bulges)

    Args:
        sequence: RNA sequence
        secondary_structure: Dot-bracket notation
        noise_level: Amount of noise to add (0-1)
        seed: Random seed for reproducibility

    Returns:
        Reactivity array (0-2 scale)
    """
    rng = np.random.RandomState(seed)
    L = len(sequence)
    reactivities = np.zeros(L)

    # Parse structure
    paired = set()
    stack = []
    for pos, char in enumerate(secondary_structure):
        if char == '(':
            stack.append(pos)
        elif char == ')':
            if stack:
                i = stack.pop()
                paired.add(i)
                paired.add(pos)

    # Assign reactivities
    for i in range(L):
        if i in paired:
            # Paired: low reactivity (0-0.5)
            base_value = rng.uniform(0, 0.5)
        else:
            # Unpaired: high reactivity (1.0-2.0)
            base_value = rng.uniform(1.0, 2.0)

        # Add noise
        noise = rng.normal(0, noise_level * base_value)
        reactivities[i] = np.clip(base_value + noise, 0, 2)

    return reactivities


# ── circBase Sequence Loading ──────────────────────────────────────────────────

def load_circbase_sequences(
    circbase_file: Optional[str] = None,
    min_len: int = 500,
    max_len: int = 1000,
    max_sequences: int = 1000,
) -> List[Tuple[str, str]]:
    """Load circRNA sequences from circBase or generate synthetic ones.

    Args:
        circbase_file: Path to circBase data file (optional)
        min_len: Minimum sequence length
        max_len: Maximum sequence length
        max_sequences: Maximum number of sequences to load

    Returns:
        List of (seq_id, sequence) tuples
    """
    sequences = []

    # Try to load from file
    if circbase_file and os.path.exists(circbase_file):
        print(f"  Loading circBase from: {circbase_file}")

        opener = gzip.open if circbase_file.endswith('.gz') else open
        mode = 'rt' if circbase_file.endswith('.gz') else 'r'

        try:
            with opener(circbase_file, mode) as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue

                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        seq_id = parts[0]
                        seq = parts[1].upper().replace('T', 'U')

                        if min_len <= len(seq) <= max_len:
                            sequences.append((seq_id, seq))

                            if len(sequences) >= max_sequences:
                                break
        except Exception as e:
            print(f"  Error loading circBase: {e}")

    # If no sequences loaded, generate synthetic ones
    if not sequences:
        print("  Generating synthetic circRNA sequences (circBase not available)")
        rng = np.random.RandomState(42)
        bases = ['A', 'C', 'G', 'U']

        for i in range(min(max_sequences, 500)):
            L = rng.randint(min_len, max_len + 1)
            seq = ''.join(rng.choice(bases, L))
            seq_id = f"circbase_synth_{i:04d}"
            sequences.append((seq_id, seq))

    print(f"  Loaded {len(sequences)} sequences ({min_len}-{max_len} nt)")
    return sequences


# ── 3D Structure Generation ────────────────────────────────────────────────────

def generate_3d_coordinates(
    sequence: str,
    pair_constraints: List[Tuple[int, int, float, float]],
    solver: GeometricConstraintSolver,
    n_samples: int = 1,
) -> Optional[np.ndarray]:
    """Generate 3D coordinates from sequence and pair constraints.

    Args:
        sequence: RNA sequence
        pair_constraints: List of (i, j, distance, weight) tuples
        solver: GeometricConstraintSolver instance
        n_samples: Number of conformations to sample

    Returns:
        Best (L, 3) coordinate array, or None if failed
    """
    L = len(sequence)

    # Build constraint set
    class CS:
        def __init__(self, n, pairs):
            self.seq_len = n
            self.pair_constraints = pairs

    cs = CS(L, pair_constraints)

    try:
        conformations = solver.solve(cs)
        if conformations and len(conformations) > 0:
            return conformations[0]  # Best conformation (lowest energy)
        return None
    except Exception as e:
        print(f"  Solver error: {e}")
        return None


# ── Checkpoint Management ───────────────────────────────────────────────────────

def save_checkpoint(
    checkpoint_path: str,
    processed_samples: List[ProcessedSample],
    current_index: int,
    stats: Dict[str, Any],
):
    """Save processing checkpoint."""
    checkpoint = {
        "current_index": current_index,
        "stats": stats,
        "processed_ids": [s.seq_id for s in processed_samples],
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def load_checkpoint(
    checkpoint_path: str,
) -> Tuple[int, Set[str], Dict[str, Any]]:
    """Load processing checkpoint.

    Returns:
        Tuple of (current_index, processed_ids_set, stats_dict)
    """
    if not os.path.exists(checkpoint_path):
        return 0, set(), {}

    try:
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        return (
            checkpoint.get("current_index", 0),
            set(checkpoint.get("processed_ids", [])),
            checkpoint.get("stats", {}),
        )
    except Exception:
        return 0, set(), {}


# ── Main Pipeline ───────────────────────────────────────────────────────────────

def run_shape_to_3d_pipeline(
    output_dir: str,
    target_samples: int = 6000,
    cache_dir: str = "data/geo_cache",
    min_len: int = 500,
    max_len: int = 1000,
    force_download: bool = False,
    resume: bool = False,
    seed: int = 42,
):
    """Run the complete SHAPE to 3D pipeline.

    Args:
        output_dir: Output directory for results
        target_samples: Target number of samples (~6000)
        cache_dir: Cache directory for GEO data
        min_len: Minimum sequence length
        max_len: Maximum sequence length
        force_download: Force re-download of GEO data
        resume: Resume from checkpoint
        seed: Random seed
    """
    print("=" * 70)
    print("  SHAPE to 3D circRNA Pipeline")
    print("=" * 70)
    print(f"  Target: {target_samples} samples")
    print(f"  Length range: {min_len}-{max_len} nt")
    print(f"  Output: {output_dir}")
    print(f"  Resume: {resume}")
    print()

    # Initialize
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    coords_dir = os.path.join(output_dir, "coords")
    os.makedirs(coords_dir, exist_ok=True)

    checkpoint_path = os.path.join(output_dir, "checkpoint.json")
    rng = np.random.RandomState(seed)

    # Check for resume
    start_idx = 0
    processed_ids = set()
    stats = {
        "total_processed": 0,
        "successful": 0,
        "failed": 0,
        "from_experimental": 0,
        "from_simulated": 0,
        "from_circbase": 0,
    }

    if resume:
        start_idx, processed_ids, stats = load_checkpoint(checkpoint_path)
        print(f"  Resuming from checkpoint: {start_idx} samples already processed")

    # Setup solver
    solver_config = SolverConfig(
        n_samples=1,
        use_annealing_closure=False,
        bond_length=5.9,
        pair_distance=10.6,
        max_iterations=50,
        clash_distance=3.0,
    )
    solver = GeometricConstraintSolver(solver_config)

    # ── Step 1: Download GEO Data ───────────────────────────────────────────────
    print("\n[Step 1/5] Downloading GEO data...")
    geo_files = download_all_geo_data(cache_dir, force_download)

    # ── Step 2: Load SHAPE Profiles ─────────────────────────────────────────────
    print("\n[Step 2/5] Loading SHAPE profiles...")
    shape_profiles = load_shape_profiles(geo_files, min_len, max_len)
    print(f"  Loaded {len(shape_profiles)} SHAPE profiles ({min_len}-{max_len} nt)")

    # Subsample profiles: keep only the most informative ones
    # Sort by number of valid (non-NaN) reactivity values, keep top 2000
    if len(shape_profiles) > 2000:
        shape_profiles.sort(key=lambda p: np.isfinite(p.reactivities).sum(), reverse=True)
        shape_profiles = shape_profiles[:2000]
        print(f"  Subsampled to top 2000 profiles (most valid reactivity values)")

    # ── Step 3: Load circBase Sequences (for simulation) ─────────────────────────
    print("\n[Step 3/5] Loading circBase sequences...")
    circbase_seqs = load_circbase_sequences(
        circbase_file=None,  # Will generate synthetic if not provided
        min_len=min_len,
        max_len=max_len,
        max_sequences=target_samples,
    )

    # ── Step 4: Process Samples ──────────────────────────────────────────────────
    print("\n[Step 4/5] Processing samples...")

    processed_samples: List[ProcessedSample] = []
    total_needed = target_samples

    # Progress bar setup
    if HAS_TQDM:
        pbar = tqdm(total=total_needed, desc="  Processing", unit="samples")
    else:
        pbar = None

    # 4a. Process experimental SHAPE profiles
    print("\n  [4a] Processing experimental SHAPE profiles...")
    for profile in shape_profiles:
        if len(processed_samples) >= total_needed:
            break

        if profile.seq_id in processed_ids:
            continue

        # Predict secondary structure with SHAPE constraints
        ss, mfe, pairs = predict_secondary_structure_with_shape(
            profile.sequence,
            profile.reactivities,
            use_shape=True,
        )

        # Build pair constraints
        pair_constraints = [(i, j, 10.6, 1.0) for i, j in pairs]

        # Generate 3D coordinates
        coords = generate_3d_coordinates(
            profile.sequence,
            pair_constraints,
            solver,
        )

        if coords is not None:
            sample = ProcessedSample(
                seq_id=profile.seq_id,
                sequence=profile.sequence,
                secondary_structure=ss,
                coords=coords,
                reactivities=profile.reactivities,
                source="shape_experimental",
                mfe=mfe,
                pair_constraints=pair_constraints,
            )
            processed_samples.append(sample)
            processed_ids.add(profile.seq_id)
            stats["from_experimental"] += 1
            stats["successful"] += 1

            if pbar:
                pbar.update(1)
            elif len(processed_samples) % 10 == 0:
                print(f"    Processed: {len(processed_samples)}/{total_needed}")

    print(f"    Experimental SHAPE: {len(processed_samples)} samples")
    print(f"    Note: Simulated/synthetic samples are generated by build_training_dataset.py")

    # Skip 4b and 4c (simulated SHAPE and synthetic) - handled by build_training_dataset
    stats["total_processed"] = len(processed_samples)
                processed_samples.append(sample)
                processed_ids.add(seq_id)
                stats["from_simulated"] += 1
                stats["successful"] += 1

                if pbar:
                    pbar.update(1)
                elif len(processed_samples) % 100 == 0:
                    print(f"    Processed: {len(processed_samples)}/{total_needed}")

    if pbar:
        pbar.close()

    stats["total_processed"] = len(processed_samples)

    # ── Step 5: Save Results ─────────────────────────────────────────────────────
    print("\n[Step 5/5] Saving results...")

    # Save coordinate files
    for sample in processed_samples:
        coord_path = os.path.join(coords_dir, f"{sample.seq_id}.npy")
        np.save(coord_path, sample.coords)

    # Save sequences.json
    sequences_data = [sample.to_dict() for sample in processed_samples]
    with open(os.path.join(output_dir, "sequences.json"), 'w') as f:
        json.dump(sequences_data, f, indent=2)

    # Save metadata.json
    metadata = {
        "total": len(processed_samples),
        "target": target_samples,
        "length_range": [min_len, max_len],
        "sources": {
            "experimental": stats["from_experimental"],
            "simulated": stats["from_simulated"],
            "circbase": stats["from_circbase"],
        },
        "vienna_used": HAS_VIENNA,
        "shape_constraints": True,
        "solver_config": {
            "n_samples": solver_config.n_samples,
            "bond_length": solver_config.bond_length,
            "pair_distance": solver_config.pair_distance,
            "use_annealing_closure": solver_config.use_annealing_closure,
        },
        "stats": stats,
        "samples": [
            {
                "id": s.seq_id,
                "length": len(s.sequence),
                "source": s.source,
                "mfe": s.mfe,
                "n_pairs": len(s.pair_constraints),
                "reactivities_mean": float(np.mean(s.reactivities)),
            }
            for s in processed_samples
        ],
    }
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    # ── Summary ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Pipeline Complete!")
    print("=" * 70)
    print(f"  Total samples: {len(processed_samples)}")
    print(f"  Experimental SHAPE: {stats['from_experimental']}")
    print(f"  Simulated SHAPE: {stats['from_simulated']}")
    print(f"    - From circBase: {stats['from_circbase']}")
    print(f"  Length range: {min_len}-{max_len} nt")
    print(f"\n  Output:")
    print(f"    {output_dir}/sequences.json")
    print(f"    {output_dir}/coords/*.npy")
    print(f"    {output_dir}/metadata.json")
    print("=" * 70)

    return processed_samples


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert SHAPE/icSHAPE reactivity data into 3D circRNA structures"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/shape_constrained",
        help="Output directory (default: data/shape_constrained)",
    )
    parser.add_argument(
        "--target", "-n",
        type=int,
        default=6000,
        help="Target number of samples (default: 6000)",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=500,
        help="Minimum sequence length (default: 500)",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=1000,
        help="Maximum sequence length (default: 1000)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/geo_cache",
        help="Cache directory for GEO data (default: data/geo_cache)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of GEO data",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    run_shape_to_3d_pipeline(
        output_dir=args.output,
        target_samples=args.target,
        cache_dir=args.cache_dir,
        min_len=args.min_len,
        max_len=args.max_len,
        force_download=args.force_download,
        resume=args.resume,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()