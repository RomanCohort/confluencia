#!/usr/bin/env python3
"""
fetch_real_testset.py — Fetch real circRNA 3D structures and prepare test set.

Data sources:
  1. CircRNA-3DFold (GitHub: RNA-folding-lab/CircRNA-3DFold, Zenodo: 14627860)
     - 500+ circRNA structures with PDB coordinates
  2. DeepFold-RNA (GitHub: robpearc/DeepFold-RNA)
     - bpRNA-1m based, includes circRNA validation set

Pipeline:
  1. Clone/download CircRNA-3DFold dataset
  2. Parse PDB → numpy coords
  3. Select 30 highest-quality samples as held-out test set
  4. Copy remaining as training pool (expand on AutoDL)
  5. Output in TorusFold-compatible format (sequences.json + coords/*.npy)

Usage:
    # On AutoDL (recommended, has network access):
    python fetch_real_testset.py --output /root/data/circrna_real_test --keep 30

    # Local (if network available):
    python fetch_real_testset.py --output data/circrna_real_test --keep 30

    # Skip download, just re-split existing data:
    python fetch_real_testset.py --skip-download --input /root/data/CircRNA-3DFold/data --keep 30
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy required. pip install numpy")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# PDB Parsing
# ═══════════════════════════════════════════════════════════════

def parse_pdb_rna(pdb_path: Path) -> Optional[Dict]:
    """Parse PDB file, extract RNA nucleotide coordinates.

    Returns dict with:
        sequence: str (AUCG)
        coords: np.ndarray (L, 3) — per-nucleotide C3' positions
        n_residues: int
        resolution: float or None
        method: str (X-RAY, NMR, CRYO-EM, etc.)
    """
    sequence = []
    coords = []
    seen_residues = set()
    resolution = None
    method = None

    base_map = {'A': 'A', 'U': 'U', 'G': 'G', 'C': 'C',
                'DA': 'A', 'DU': 'U', 'DG': 'G', 'DC': 'C',
                'I': 'G',  # Inosine → G
                'PSU': 'U',  # Pseudouridine → U
                }

    with open(pdb_path, 'r') as f:
        for line in f:
            # Method
            if line.startswith('EXPDTA'):
                method_str = line[6:].strip()
                if 'X-RAY' in method_str or 'X-RAY DIFFRACTION' in method_str:
                    method = 'X-RAY'
                elif 'NMR' in method_str:
                    method = 'NMR'
                elif 'CRYO-EM' in method_str or 'ELECTRON MICROSCOPY' in method_str:
                    method = 'CRYO-EM'
                else:
                    method = method_str[:20]

            # Resolution
            if line.startswith('REMARK   2 RESOLUTION'):
                try:
                    resolution = float(line.split()[-1])
                except ValueError:
                    pass

            # Atom records
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue

            # Only RNA atoms (C3' for backbone position)
            atom_name = line[12:16].strip()
            if atom_name != "C3'":
                continue

            # Residue info
            chain_id = line[21]
            res_seq = int(line[22:26])
            i_code = line[26]

            # Unique residue key
            res_key = (chain_id, res_seq, i_code)
            if res_key in seen_residues:
                continue
            seen_residues.add(res_key)

            # Base name
            res_name = line[17:20].strip()
            base = base_map.get(res_name, None)
            if base is None:
                continue

            # Coordinates
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            sequence.append(base)
            coords.append([x, y, z])

    if len(coords) < 10:
        return None

    return {
        'sequence': ''.join(sequence),
        'coords': np.array(coords, dtype=np.float32),
        'n_residues': len(coords),
        'resolution': resolution,
        'method': method,
    }


def parse_pdb_simple(pdb_path: Path) -> Optional[Dict]:
    """Fallback: parse PDB extracting all ATOM records as coarse-grained coords.

    Groups by residue, takes centroid per residue.
    """
    residues = {}  # (chain, resseq) → list of [x,y,z]
    method = None

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('EXPDTA'):
                method_str = line[6:].strip()
                if 'X-RAY' in method_str:
                    method = 'X-RAY'
                elif 'NMR' in method_str:
                    method = 'NMR'
                elif 'CRYO-EM' in method_str:
                    method = 'CRYO-EM'

            if not line.startswith('ATOM'):
                continue

            chain = line[21]
            resseq = int(line[22:26])
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            key = (chain, resseq)
            if key not in residues:
                residues[key] = []
            residues[key].append([x, y, z])

    if len(residues) < 10:
        return None

    # Centroid per residue
    coords = []
    for key in sorted(residues.keys()):
        arr = np.array(residues[key])
        coords.append(arr.mean(axis=0))

    return {
        'sequence': 'N' * len(coords),  # Unknown sequence
        'coords': np.array(coords, dtype=np.float32),
        'n_residues': len(coords),
        'resolution': None,
        'method': method,
    }


# ═══════════════════════════════════════════════════════════════
# Data Source: CircRNA-3DFold
# ═══════════════════════════════════════════════════════════════

def download_with_fallback(name: str, zip_url: str, zenodo_url: str,
                            output_dir: Path) -> Optional[Path]:
    """Download a dataset with multiple fallback strategies.

    Strategy order: wget zip → curl zip → zenodo → git clone
    """
    # Determine expected directory name from zip
    repo_slug = zip_url.split('/')[-1].replace('.zip', '')  # e.g. CircRNA-3DFold-main
    repo_dir = output_dir / repo_slug.replace('-main', '')

    if repo_dir.exists() and any(repo_dir.rglob('*')):
        print(f"  {name} already exists at {repo_dir}")
        return repo_dir

    # Strategy 1: wget the GitHub zip archive (no auth needed)
    print(f"  Downloading {name} via wget...")
    zip_path = output_dir / f'{name}.zip'
    try:
        result = subprocess.run(
            ['wget', '-q', '--no-check-certificate', zip_url, '-O', str(zip_path)],
            capture_output=True, timeout=300
        )
        if result.returncode == 0 and zip_path.exists() and zip_path.stat().st_size > 1000:
            print(f"  ✓ Downloaded zip ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")
            subprocess.run(['unzip', '-q', '-o', str(zip_path), '-d', str(output_dir)],
                           capture_output=True)
            zip_path.unlink(missing_ok=True)
            # Find extracted directory
            extracted = output_dir / repo_slug
            if extracted.exists():
                extracted.rename(repo_dir)
            if repo_dir.exists():
                print(f"  ✓ Extracted to {repo_dir}")
                return repo_dir
    except Exception as e:
        print(f"  wget failed: {e}")

    # Strategy 2: curl the GitHub zip archive
    print(f"  Trying curl...")
    try:
        result = subprocess.run(
            ['curl', '-sL', zip_url, '-o', str(zip_path)],
            capture_output=True, timeout=300
        )
        if result.returncode == 0 and zip_path.exists() and zip_path.stat().st_size > 1000:
            print(f"  ✓ Downloaded zip ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")
            subprocess.run(['unzip', '-q', '-o', str(zip_path), '-d', str(output_dir)],
                           capture_output=True)
            zip_path.unlink(missing_ok=True)
            extracted = output_dir / repo_slug
            if extracted.exists():
                extracted.rename(repo_dir)
            if repo_dir.exists():
                print(f"  ✓ Extracted to {repo_dir}")
                return repo_dir
    except Exception as e:
        print(f"  curl failed: {e}")

    # Strategy 3: Zenodo
    if zenodo_url:
        print(f"  Trying Zenodo: {zenodo_url}")
        try:
            result = subprocess.run(
                ['wget', '-q', '--no-check-certificate', zenodo_url, '-O', str(zip_path)],
                capture_output=True, timeout=600
            )
            if result.returncode == 0 and zip_path.exists() and zip_path.stat().st_size > 1000:
                subprocess.run(['unzip', '-q', '-o', str(zip_path), '-d', str(output_dir)],
                               capture_output=True)
                zip_path.unlink(missing_ok=True)
                # Zenodo may extract differently
                for d in output_dir.iterdir():
                    if d.is_dir() and d.name != repo_dir.name and name.lower() in d.name.lower():
                        return d
                if repo_dir.exists():
                    return repo_dir
        except Exception as e:
            print(f"  Zenodo failed: {e}")

    # Strategy 4: git clone (last resort, may need auth)
    print(f"  Trying git clone (may require auth)...")
    git_url = zip_url.replace('/archive/refs/heads/main.zip', '.git')
    try:
        subprocess.run(
            ['git', 'clone', '--depth', '1', git_url, str(repo_dir)],
            capture_output=True, timeout=600
        )
        if repo_dir.exists():
            print(f"  ✓ Cloned to {repo_dir}")
            return repo_dir
    except Exception as e:
        print(f"  git clone failed: {e}")

    print(f"  ✗ All download strategies failed for {name}")
    print(f"  Please manually download from:")
    if zenodo_url:
        print(f"    Zenodo: {zenodo_url}")
    print(f"    GitHub: {zip_url.replace('/archive/refs/heads/main.zip', '')}")
    print(f"  Then: python fetch_real_testset.py --input /path/to/downloaded/data --keep 30")
    return None


def download_circrna_3dfold(output_dir: Path) -> Optional[Path]:
    """Download CircRNA-3DFold dataset."""
    return download_with_fallback(
        name='CircRNA-3DFold',
        zip_url='https://github.com/RNA-folding-lab/CircRNA-3DFold/archive/refs/heads/main.zip',
        zenodo_url='https://zenodo.org/records/14627860/files/dataset.zip',
        output_dir=output_dir,
    )


def download_deepfold_rna(output_dir: Path) -> Optional[Path]:
    """Download DeepFold-RNA dataset."""
    return download_with_fallback(
        name='DeepFold-RNA',
        zip_url='https://github.com/robpearc/DeepFold-RNA/archive/refs/heads/main.zip',
        zenodo_url='',  # No known Zenodo
        output_dir=output_dir,
    )


def scan_pdb_files(data_dir: Path) -> List[Path]:
    """Scan directory for PDB files."""
    pdb_files = []
    for ext in ['*.pdb', '*.PDB', '*.ent', '*.ENT']:
        pdb_files.extend(data_dir.rglob(ext))
    # Also check for .gz compressed PDB
    for ext in ['*.pdb.gz', '*.ent.gz']:
        for gz_path in data_dir.rglob(ext):
            # Decompress
            pdb_path = gz_path.with_suffix('')  # Remove .gz
            if not pdb_path.exists():
                try:
                    import gzip
                    with gzip.open(gz_path, 'rt') as f_in:
                        with open(pdb_path, 'w') as f_out:
                            f_out.write(f_in.read())
                    pdb_files.append(pdb_path)
                except Exception:
                    pass
            else:
                pdb_files.append(pdb_path)
    return sorted(pdb_files)


# ═══════════════════════════════════════════════════════════════
# Data Source: DeepFold-RNA (handled by download_with_fallback)
# ═══════════════════════════════════════════════════════════════

# download_deepfold_rna defined above with download_circrna_3dfold


# ═══════════════════════════════════════════════════════════════
# Quality Scoring & Selection
# ═══════════════════════════════════════════════════════════════

def quality_score(result: Dict) -> float:
    """Score a parsed structure for test set selection.

    Higher = better quality, prefer for test set.
    """
    score = 0.0

    # Method quality
    method_scores = {
        'X-RAY': 3.0,
        'CRYO-EM': 2.5,
        'NMR': 2.0,
    }
    score += method_scores.get(result.get('method'), 1.0)

    # Resolution (lower = better)
    res = result.get('resolution')
    if res is not None:
        if res < 2.0:
            score += 3.0
        elif res < 3.0:
            score += 2.0
        elif res < 5.0:
            score += 1.0
        else:
            score += 0.5

    # Length: prefer moderate lengths (30-500 nt)
    L = result['n_residues']
    if 30 <= L <= 500:
        score += 2.0
    elif 20 <= L <= 800:
        score += 1.0

    # Coordinate quality: check for NaN/Inf
    coords = result['coords']
    if np.isnan(coords).any() or np.isinf(coords).any():
        score -= 10.0
    else:
        # Check coordinate spread (not collapsed)
        spread = np.std(coords)
        if spread > 5.0:
            score += 1.0
        elif spread > 1.0:
            score += 0.5

    # Known sequence (not all N)
    seq = result.get('sequence', '')
    if seq and 'N' not in seq:
        score += 1.0

    return score


# ═══════════════════════════════════════════════════════════════
# Output Formatting
# ═══════════════════════════════════════════════════════════════

def write_torusfold_dataset(
    samples: List[Dict],
    output_dir: Path,
    label: str = "test",
):
    """Write samples in TorusFold-compatible format.

    Creates:
        sequences.json  — list of {id, sequence, source, confidence, ...}
        coords/         — {id}.npy files with (L, 3) float32 arrays
    """
    coords_dir = output_dir / 'coords'
    coords_dir.mkdir(parents=True, exist_ok=True)

    seq_data = []
    for i, sample in enumerate(samples):
        sid = sample['id']
        coords = sample['coords']

        # Save coordinates
        np.save(coords_dir / f'{sid}.npy', coords)

        # Confidence based on method quality
        method = sample.get('method')
        if method == 'X-RAY':
            conf = 1.0
        elif method == 'CRYO-EM':
            res = sample.get('resolution')
            conf = 0.95 if res and res < 3.0 else 0.9
        elif method == 'NMR':
            conf = 0.85
        else:
            conf = 0.7

        # Build sequence entry (compatible with load_pseudo_labels)
        entry = {
            'id': sid,
            'sequence': sample['sequence'],
            'length': len(sample['sequence']),
            'source': sample.get('source', 'unknown'),
            'confidence': conf,
            'method': method or 'unknown',
            'resolution': sample.get('resolution'),
            'quality_score': round(sample.get('quality_score', 0), 2),
        }
        if sample.get('pair_constraints'):
            entry['pair_constraints'] = sample['pair_constraints']
        seq_data.append(entry)

    with open(output_dir / 'sequences.json', 'w') as f:
        json.dump(seq_data, f, indent=2)

    # Metadata
    meta = {
        'label': label,
        'n_samples': len(samples),
        'lengths': [s['n_residues'] for s in samples],
        'sources': list(set(s.get('source', '?') for s in samples)),
        'methods': list(set(s.get('method', '?') for s in samples if s.get('method'))),
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Written: {output_dir}/")
    print(f"    sequences.json: {len(samples)} entries")
    print(f"    coords/: {len(samples)} .npy files")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Fetch real circRNA 3D structures and prepare test set'
    )
    parser.add_argument('--output', type=str, default='data/circrna_real_test',
                        help='Output directory')
    parser.add_argument('--keep', type=int, default=30,
                        help='Number of samples to keep as held-out test set')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip git clone, use existing data')
    parser.add_argument('--input', type=str, default=None,
                        help='Pre-downloaded data directory (skip clone)')
    parser.add_argument('--sources', type=str, nargs='+',
                        default=['circrna-3dfold', 'deepfold-rna'],
                        choices=['circrna-3dfold', 'deepfold-rna'],
                        help='Data sources to fetch')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Fetch Real circRNA 3D Test Set")
    print("=" * 60)
    print(f"  Output: {output_dir}")
    print(f"  Test set size: {args.keep}")
    print(f"  Sources: {args.sources}")

    # ── Step 1: Download data ──
    all_pdb_dirs = []

    if args.input:
        # Use pre-downloaded data
        input_path = Path(args.input)
        if input_path.exists():
            all_pdb_dirs.append(input_path)
            print(f"\n  Using pre-downloaded data: {input_path}")
        else:
            print(f"  ERROR: {args.input} does not exist")
            return
    elif not args.skip_download:
        download_dir = output_dir / '_downloads'
        download_dir.mkdir(exist_ok=True)

        if 'circrna-3dfold' in args.sources:
            print("\n[1/2] Downloading CircRNA-3DFold...")
            repo = download_circrna_3dfold(download_dir)
            if repo:
                data_dir = repo / 'data'
                if data_dir.exists():
                    all_pdb_dirs.append(data_dir)
                    print(f"  Data dir: {data_dir}")
                else:
                    # Try root of repo
                    all_pdb_dirs.append(repo)
                    print(f"  Using repo root: {repo}")

        if 'deepfold-rna' in args.sources:
            print("\n[2/2] Downloading DeepFold-RNA...")
            repo = download_deepfold_rna(download_dir)
            if repo:
                data_dir = repo / 'data'
                if data_dir.exists():
                    all_pdb_dirs.append(data_dir)
                    print(f"  Data dir: {data_dir}")
                else:
                    all_pdb_dirs.append(repo)
    else:
        print("\n  Skipping download (--skip-download)")

    if not all_pdb_dirs:
        print("\n  ERROR: No data directories found!")
        print("  Options:")
        print("    1. Run without --skip-download on a machine with internet")
        print("    2. Use --input /path/to/downloaded/data")
        print("    3. Manually download from:")
        print("       https://github.com/RNA-folding-lab/CircRNA-3DFold")
        print("       https://zenodo.org/records/14627860")
        return

    # ── Step 2: Scan and parse PDB files ──
    print(f"\n{'='*60}")
    print("  Scanning PDB files...")
    print(f"{'='*60}")

    all_pdb_files = []
    for d in all_pdb_dirs:
        found = scan_pdb_files(d)
        print(f"  {d}: {len(found)} PDB files")
        all_pdb_files.extend(found)

    print(f"\n  Total: {len(all_pdb_files)} PDB files")

    if not all_pdb_files:
        print("  No PDB files found! Check data directory structure.")
        return

    # ── Step 3: Parse and score ──
    print(f"\n{'='*60}")
    print("  Parsing structures...")
    print(f"{'='*60}")

    parsed = []
    failed = 0
    for i, pdb_path in enumerate(all_pdb_files):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(all_pdb_files)}] {pdb_path.name}...")

        # Try RNA-specific parsing first, then fallback
        result = parse_pdb_rna(pdb_path)
        if result is None:
            result = parse_pdb_simple(pdb_path)

        if result is None:
            failed += 1
            continue

        # Quality check
        coords = result['coords']
        if np.isnan(coords).any() or np.isinf(coords).any():
            failed += 1
            continue
        if coords.std() < 1.0:
            # Collapsed structure
            failed += 1
            continue

        # Score
        result['quality_score'] = quality_score(result)
        result['id'] = pdb_path.stem
        result['source'] = 'CircRNA-3DFold' if 'CircRNA-3DFold' in str(pdb_path) else 'DeepFold-RNA'

        parsed.append(result)

    print(f"\n  Parsed: {len(parsed)} structures")
    print(f"  Failed: {failed}")

    if not parsed:
        print("  ERROR: No valid structures parsed!")
        return

    # ── Step 4: Sort by quality and split ──
    parsed.sort(key=lambda x: x['quality_score'], reverse=True)

    # Print top 5
    print(f"\n  Top 5 by quality:")
    for i, s in enumerate(parsed[:5]):
        print(f"    {i+1}. {s['id']}: L={s['n_residues']}, "
              f"method={s.get('method','?')}, res={s.get('resolution','?')}, "
              f"score={s['quality_score']:.1f}")

    # Split
    n_test = min(args.keep, len(parsed))
    test_samples = parsed[:n_test]
    train_pool = parsed[n_test:]

    print(f"\n{'='*60}")
    print(f"  Dataset Split")
    print(f"{'='*60}")
    print(f"  Test set: {len(test_samples)} samples (highest quality)")
    print(f"  Training pool: {len(train_pool)} samples (expand on AutoDL)")

    # Length distribution
    test_lengths = [s['n_residues'] for s in test_samples]
    print(f"\n  Test set lengths: min={min(test_lengths)}, max={max(test_lengths)}, "
          f"mean={np.mean(test_lengths):.0f}")

    # ── Step 5: Write output ──
    print(f"\n{'='*60}")
    print("  Writing output...")
    print(f"{'='*60}")

    # Test set
    test_dir = output_dir / 'test_set'
    write_torusfold_dataset(test_samples, test_dir, label="test")

    # Training pool
    if train_pool:
        pool_dir = output_dir / 'training_pool'
        write_torusfold_dataset(train_pool, pool_dir, label="training_pool")

    # Summary
    summary = {
        'n_test': len(test_samples),
        'n_train_pool': len(train_pool),
        'test_ids': [s['id'] for s in test_samples],
        'train_pool_ids': [s['id'] for s in train_pool],
        'test_length_stats': {
            'min': int(min(test_lengths)),
            'max': int(max(test_lengths)),
            'mean': round(float(np.mean(test_lengths)), 1),
        },
        'sources_used': args.sources,
        'note': 'Training pool can be expanded on AutoDL using '
                'generate_pseudo_labels.py or augment_pseudo_labels.py',
    }
    with open(output_dir / 'split_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"{'='*60}")
    print(f"  Test set:     {test_dir}/  ({len(test_samples)} samples)")
    print(f"  Train pool:   {output_dir / 'training_pool'}/  ({len(train_pool)} samples)")
    print(f"  Summary:      {output_dir / 'split_summary.json'}")
    print(f"\n  Next steps on AutoDL:")
    print(f"    1. Training pool is already usable — {len(train_pool)} high-quality samples")
    print(f"    2. Either use directly:")
    print(f"       python train_all_schemes.py --schemes 1 4 6 7 --labels {output_dir / 'training_pool'}")
    print(f"    3. Or expand further:")
    print(f"       python augment_pseudo_labels.py --input {output_dir / 'training_pool'} --output data/expanded --n-aug 5")
    print(f"    4. Eval on held-out test set:")


if __name__ == '__main__':
    main()
