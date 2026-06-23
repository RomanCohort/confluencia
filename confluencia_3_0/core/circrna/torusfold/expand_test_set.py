#!/usr/bin/env python3
"""
expand_test_set.py -- Build expanded test set for TorusFold targeting N >= 30.

Sources:
  1. PDB experimental circRNA structures (CIF format): 9H8A, 8xtp, 8xtq, 8xtr, 8xts, 9is7
  2. IsRNAcirc test set PDB files (~34 circRNA structures)
  3. RCSB PDB RNA search + circularization (supplementary)

Output format: compatible with train_all_schemes.py load_pseudo_labels()
  - sequences.json with id, sequence, length, pair_constraints, source, confidence
  - coords/*.npy with (L, 3) C3' coordinate arrays
  - metadata.json with source breakdown and quality stats

Usage:
    python expand_test_set.py --output data/expanded_test_set --target 30
    python expand_test_set.py --output data/expanded_test_set --target 50 --download-supplementary
"""

import argparse
import json
import math
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BOND_LENGTH = 5.9

RESIDUE_MAP = {
    "A": "A", "C": "C", "G": "G", "U": "U",
    "DA": "A", "DC": "C", "DG": "G", "DT": "U",
    "ADE": "A", "CYT": "C", "GUA": "G", "URI": "U",
    "I": "A",
}

CIF_ATOM_FIELDS = {
    "group_PDB": 0, "id": 1, "type_symbol": 2,
    "label_atom_id": 3, "label_alt_id": 4,
    "label_comp_id": 5, "label_asym_id": 6,
    "label_entity_id": 7, "label_seq_id": 8,
    "pdbx_PDB_ins_code": 9,
    "Cartn_x": 10, "Cartn_y": 11, "Cartn_z": 12,
}

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download"


# ── CIF Parsing ──────────────────────────────────────────────────

def parse_cif_rna(cif_path: str) -> List[Dict]:
    """Extract RNA chains from a CIF file.

    Returns list of {chain_id, sequence, coords: (L,3), residue_indices}.
    """
    chains: Dict[str, Dict] = {}
    in_atom_block = False
    atom_fields = {}
    field_count = 0
    data_lines_seen = 0

    with open(cif_path, "r", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()

            # Detect atom_site loop start
            if stripped.startswith("_atom_site."):
                # Check if this is the first field (starts a loop)
                if not in_atom_block:
                    in_atom_block = True
                    atom_fields = {}
                    field_count = 0

                # Extract field name
                field_name = stripped.split(".")[1].split()[0]
                atom_fields[field_name] = field_count
                field_count += 1
                continue

            # End of loop ONLY when we hit another loop_ or a new _section
            # (NOT on empty lines or # comments, which appear mid-loop in CIF)
            if in_atom_block and stripped.startswith("loop_"):
                in_atom_block = False
                atom_fields = {}
                field_count = 0
                data_lines_seen = 0
                continue

            # End when we see a new category (starts with _ but not _atom_site)
            if in_atom_block and stripped.startswith("_") and not stripped.startswith("_atom_site"):
                in_atom_block = False
                atom_fields = {}
                field_count = 0
                data_lines_seen = 0
                continue

            if not in_atom_block:
                continue

            # Skip comments and empty lines within the loop
            if stripped.startswith("#") or stripped == "":
                continue

            # Parse atom data line
            parts = stripped.split()
            if len(parts) < field_count:
                continue

            data_lines_seen += 1

            # Get required fields
            try:
                group_idx = atom_fields.get("group_PDB", 0)
                atom_name_idx = atom_fields.get("label_atom_id", 3)
                comp_id_idx = atom_fields.get("label_comp_id", 5)
                asym_id_idx = atom_fields.get("label_asym_id", 6)
                seq_id_idx = atom_fields.get("label_seq_id", 8)
                x_idx = atom_fields.get("Cartn_x", 10)
                y_idx = atom_fields.get("Cartn_y", 11)
                z_idx = atom_fields.get("Cartn_z", 12)

                group = parts[group_idx]
                atom_name = parts[atom_name_idx]
                comp_id = parts[comp_id_idx]
                asym_id = parts[asym_id_idx]
                seq_id_str = parts[seq_id_idx]
                x, y, z = float(parts[x_idx]), float(parts[y_idx]), float(parts[z_idx])
            except (ValueError, IndexError, KeyError):
                continue

            if group != "ATOM" and group != "HETATM":
                continue

            # Handle CIF quoted atom names like "C3'" (strip outer quotes only)
            if atom_name.startswith('"') and atom_name.endswith('"'):
                atom_name_clean = atom_name[1:-1]
            else:
                atom_name_clean = atom_name
            if atom_name_clean not in ("C3'", "C3*"):
                continue

            base = RESIDUE_MAP.get(comp_id)
            if base is None:
                continue

            try:
                seq_id = int(seq_id_str)
            except ValueError:
                continue

            if asym_id not in chains:
                chains[asym_id] = {"chain_id": asym_id, "residues": {}}

            if seq_id not in chains[asym_id]["residues"]:
                chains[asym_id]["residues"][seq_id] = (base, [x, y, z])

    results = []
    for chain_id, chain_data in chains.items():
        sorted_resseqs = sorted(chain_data["residues"].keys())
        if len(sorted_resseqs) < 10:
            continue

        sequence = "".join(chain_data["residues"][r][0] for r in sorted_resseqs)
        coords = np.array(
            [chain_data["residues"][r][1] for r in sorted_resseqs],
            dtype=np.float64,
        )

        valid_bases = set("ACGU")
        if not all(b in valid_bases for b in sequence):
            continue

        results.append({
            "chain_id": chain_id,
            "sequence": sequence,
            "coords": coords,
            "residue_indices": sorted_resseqs,
        })

    return results


# ── PDB Parsing ──────────────────────────────────────────────────

def parse_pdb_rna(pdb_path: str) -> List[Dict]:
    """Extract RNA chains from a PDB file."""
    chains: Dict[str, Dict] = {}

    with open(pdb_path, "r", errors="replace") as fh:
        for line in fh:
            if not line.startswith("ATOM") and not line.startswith("HETATM"):
                continue

            atom_name = line[12:16].strip()
            # Handle CIF quoted atom names like "C3'" (strip outer quotes only)
            if atom_name.startswith('"') and atom_name.endswith('"'):
                atom_name_clean = atom_name[1:-1]
            else:
                atom_name_clean = atom_name
            if atom_name_clean not in ("C3'", "C3*"):
                continue

            chain_id = line[21].strip()
            res_name = line[17:20].strip()
            res_seq = int(line[22:26].strip())

            base = RESIDUE_MAP.get(res_name)
            if base is None:
                continue

            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            if chain_id not in chains:
                chains[chain_id] = {"chain_id": chain_id, "residues": {}}

            if res_seq not in chains[chain_id]["residues"]:
                chains[chain_id]["residues"][res_seq] = (base, [x, y, z])

    results = []
    for chain_id, chain_data in chains.items():
        sorted_resseqs = sorted(chain_data["residues"].keys())
        if len(sorted_resseqs) < 10:
            continue

        sequence = "".join(chain_data["residues"][r][0] for r in sorted_resseqs)
        coords = np.array(
            [chain_data["residues"][r][1] for r in sorted_resseqs],
            dtype=np.float64,
        )

        valid_bases = set("ACGU")
        if not all(b in valid_bases for b in sequence):
            continue

        results.append({
            "chain_id": chain_id,
            "sequence": sequence,
            "coords": coords,
            "residue_indices": sorted_resseqs,
        })

    return results


# ── Circularization ──────────────────────────────────────────────

def annealing_closure(
    coords: np.ndarray,
    bond_length: float = 5.9,
    n_steps: int = 500,
    lr: float = 0.01,
) -> np.ndarray:
    """Gradually adjust coords to close the circle."""
    coords = coords.copy()
    L = len(coords)

    for step in range(n_steps):
        diff = coords[0] - coords[-1]
        dist = np.linalg.norm(diff)
        if dist < bond_length * 1.05:
            break

        correction = lr * (dist - bond_length) * diff / max(dist, 1e-6)
        coords[0] -= correction * 0.5
        coords[-1] += correction * 0.5

        n_zone = min(5, L // 2)
        for i in range(1, n_zone + 1):
            alpha = (n_zone - i + 1) / (n_zone + 1)
            coords[i] -= correction * 0.05 * alpha
            coords[-(i + 1)] += correction * 0.05 * alpha

    return coords


# ── Secondary Structure Prediction ──────────────────────────────

def predict_ss_circ(sequence: str) -> Tuple[str, List[List[int]]]:
    """Predict secondary structure for circRNA. Returns (dot_bracket, pairs)."""
    try:
        import RNA
        md = RNA.md()
        md.circ = True
        fc = RNA.fold_compound(sequence, md)
        ss, mfe = fc.mfe()

        pairs = []
        stack = []
        for pos, char in enumerate(ss):
            if char == "(":
                stack.append(pos)
            elif char == ")" and stack:
                j = stack.pop()
                pairs.append([j, pos])
        return ss, pairs
    except ImportError:
        pass

    # Heuristic pairing
    complement = {"A": "U", "U": "A", "G": "C", "C": "G"}
    pairs = []
    paired = set()
    L = len(sequence)
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
    for i, j in sorted(pairs):
        if i < len(ss) and j < len(ss):
            ss = ss[:i] + "(" + ss[i+1:j] + ")" + ss[j+1:]

    return ss, pairs


# ── Quality Metrics ──────────────────────────────────────────────

def compute_quality_metrics(coords: np.ndarray, is_circular: bool = True) -> Dict:
    """Compute quality metrics for a structure."""
    L = len(coords)

    # Bond length statistics
    bond_lengths = []
    for i in range(L - 1):
        d = np.linalg.norm(coords[i + 1] - coords[i])
        bond_lengths.append(d)

    if is_circular:
        d_closure = np.linalg.norm(coords[0] - coords[-1])
        bond_lengths.append(d_closure)
    else:
        d_closure = np.linalg.norm(coords[0] - coords[-1])

    bond_lengths = np.array(bond_lengths)
    bond_rmsd = np.sqrt(np.mean((bond_lengths - BOND_LENGTH) ** 2))
    bond_mean = np.mean(bond_lengths)

    # Steric clashes (non-adjacent atoms < 2.0 A)
    clash_count = 0
    for i in range(L):
        for j in range(i + 3, L):
            if i == 0 and j == L - 1:
                continue  # Skip closure bond
            d = np.linalg.norm(coords[j] - coords[i])
            if d < 2.0:
                clash_count += 1

    # Radius of gyration
    centroid = coords.mean(axis=0)
    rg = np.sqrt(np.mean(np.sum((coords - centroid) ** 2, axis=1)))

    return {
        "closure_distance": round(float(d_closure), 3),
        "bond_mean": round(float(bond_mean), 3),
        "bond_rmsd": round(float(bond_rmsd), 3),
        "clash_count": clash_count,
        "rg": round(float(rg), 3),
    }


# ── IsRNAcirc Test Set Parser ──────────────────────────────────

def parse_isrnacirc_test_set(base_dir: str) -> List[Dict]:
    """Parse IsRNAcirc test set PDB files.

    Directory structure:
      circular_RNA_Data/
        hairpin-RNAs/circXXX/job_IsRNAcirc.pdb
        helix-RNAs/circXXX/job_IsRNAcirc.pdb
        internal-RNAs/circXXX/job_IsRNAcirc.pdb
        junction-RNAs/circXXX/job_IsRNAcirc.pdb
    """
    results = []
    rna_base = os.path.join(base_dir, "circular_RNA_Data")

    if not os.path.exists(rna_base):
        print(f"  IsRNAcirc data not found: {rna_base}")
        return results

    for category in ["hairpin-RNAs", "helix-RNAs", "internal-RNAs", "junction-RNAs"]:
        cat_dir = os.path.join(rna_base, category)
        if not os.path.exists(cat_dir):
            continue

        for circ_name in sorted(os.listdir(cat_dir)):
            circ_dir = os.path.join(cat_dir, circ_name)
            if not os.path.isdir(circ_dir):
                continue

            pdb_path = os.path.join(circ_dir, "job_IsRNAcirc.pdb")
            if not os.path.exists(pdb_path):
                continue

            try:
                chains = parse_pdb_rna(pdb_path)
            except Exception as e:
                print(f"    Parse error {circ_name}: {e}")
                continue

            for chain in chains:
                results.append({
                    "id": f"isrnacirc_{circ_name}_{chain['chain_id']}",
                    "sequence": chain["sequence"],
                    "coords": chain["coords"],
                    "source": "isrnacirc",
                    "category": category.replace("-RNAs", ""),
                    "confidence": 0.7,
                    "circular": True,
                })

    return results


# ── PDB Experimental Parser ──────────────────────────────────

def parse_pdb_experimental(exp_dir: str) -> List[Dict]:
    """Parse experimental PDB CIF files (9H8A, 8xtp, etc.)."""
    results = []

    if not os.path.exists(exp_dir):
        print(f"  Experimental PDB dir not found: {exp_dir}")
        return results

    for fname in sorted(os.listdir(exp_dir)):
        if not (fname.endswith(".cif") or fname.endswith(".pdb")):
            continue

        fpath = os.path.join(exp_dir, fname)
        pdb_id = fname.rsplit(".", 1)[0]

        try:
            if fname.endswith(".cif"):
                chains = parse_cif_rna(fpath)
            else:
                chains = parse_pdb_rna(fpath)
        except Exception as e:
            print(f"    Parse error {pdb_id}: {e}")
            continue

        for chain in chains:
            results.append({
                "id": f"pdb_{pdb_id}_{chain['chain_id']}",
                "sequence": chain["sequence"],
                "coords": chain["coords"],
                "source": "pdb_experimental",
                "pdb_id": pdb_id,
                "confidence": 0.95,
                "circular": pdb_id == "9H8A",  # Only 9H8A is true circRNA
            })

    return results


# ── RCSB PDB Download + Circularization (Supplementary) ──────────

def search_rcsb_rna(
    max_resolution: float = 3.0,
    min_length: int = 30,
    max_length: int = 200,
    max_results: int = 500,
) -> List[str]:
    """Search RCSB PDB for RNA structures suitable for circularization."""
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
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
        },
    }

    payload = json.dumps(query).encode("utf-8")
    req = urllib.request.Request(
        RCSB_SEARCH_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            pdb_ids = [hit["identifier"] for hit in data.get("result_set", [])]
            print(f"  RCSB search returned {len(pdb_ids)} PDB IDs")
            return pdb_ids
    except Exception as e:
        print(f"  RCSB search failed: {e}")
        return []


def download_pdb_file(pdb_id: str, cache_dir: str) -> Optional[str]:
    """Download PDB file with caching."""
    os.makedirs(cache_dir, exist_ok=True)
    pdb_id = pdb_id.upper()

    for ext in [".cif", ".pdb"]:
        cached = os.path.join(cache_dir, f"{pdb_id}{ext}")
        if os.path.exists(cached):
            return cached

    for ext in [".cif", ".pdb"]:
        url = f"{RCSB_DOWNLOAD_URL}/{pdb_id}{ext}"
        cached = os.path.join(cache_dir, f"{pdb_id}{ext}")
        try:
            urllib.request.urlretrieve(url, cached)
            return cached
        except (urllib.error.URLError, urllib.error.HTTPError):
            continue

    return None


def get_supplementary_structures(
    cache_dir: str,
    min_length: int = 30,
    max_length: int = 200,
    max_download: int = 100,
) -> List[Dict]:
    """Download and circularize RNA structures from RCSB PDB."""
    results = []

    pdb_ids = search_rcsb_rna(min_length=min_length, max_length=max_length)
    if not pdb_ids:
        print("  No RCSB results, using curated RNA PDB list")
        pdb_ids = [
            "1FJG", "1GID", "1J7Z", "1K73", "1S72", "1Y0Q",
            "2A2J", "2AVY", "2GDI", "2HOJ", "2K21", "2K95",
            "2L1V", "2L8V", "2QBZ", "2QUS", "2RDE", "2UV8",
            "3D0G", "3IRW", "3J7V", "3J9L", "3KC4", "3KIS",
            "4ARC", "4B5R", "4CS1", "4DR2", "4ENC", "4ERJ",
            "4FN6", "4G6J", "4KQY", "4LVV", "4OCB", "4P5J",
            "5B3P", "5DDP", "5K7C", "5L4J", "5MGI", "5T2A",
            "6ASM", "6B44", "6DBK", "6FEC", "6GSK", "6H5S",
            "7K00", "7K4C", "7M5X", "7N1F", "7O8Z", "7P50",
        ]

    pdb_ids = pdb_ids[:max_download]
    print(f"  Downloading up to {len(pdb_ids)} PDB files...")

    for i, pdb_id in enumerate(pdb_ids):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(pdb_ids)}")

        fpath = download_pdb_file(pdb_id, cache_dir)
        if fpath is None:
            continue

        try:
            if fpath.endswith(".cif"):
                chains = parse_cif_rna(fpath)
            else:
                chains = parse_pdb_rna(fpath)
        except Exception:
            continue

        for chain in chains:
            L = len(chain["sequence"])
            if L < min_length or L > max_length:
                continue

            # Check if ends are close enough to circularize
            closure_dist = np.linalg.norm(chain["coords"][0] - chain["coords"][-1])
            if closure_dist > 50.0:
                continue  # Too far apart

            # Circularize
            circ_coords = annealing_closure(chain["coords"])
            circ_closure = np.linalg.norm(circ_coords[0] - circ_coords[-1])

            if circ_closure > BOND_LENGTH * 2.0:
                continue  # Circularization failed

            results.append({
                "id": f"pdb_circ_{pdb_id}_{chain['chain_id']}",
                "sequence": chain["sequence"],
                "coords": circ_coords,
                "source": "pdb_circularized",
                "pdb_id": pdb_id,
                "confidence": 0.85,
                "circular": True,
                "original_closure": round(float(closure_dist), 3),
                "circularized_closure": round(float(circ_closure), 3),
            })

    return results


# ── Main Pipeline ──────────────────────────────────────────────

def build_expanded_test_set(
    output_dir: str,
    pdb_exp_dir: str,
    isrnacirc_dir: str,
    pdb_cache: str,
    target: int = 30,
    min_length: int = 20,
    max_length: int = 500,
    max_closure: float = 12.0,
    max_bond_rmsd: float = 5.0,
    download_supplementary: bool = False,
) -> None:
    """Build expanded test set targeting N >= target samples."""
    os.makedirs(output_dir, exist_ok=True)
    coords_dir = os.path.join(output_dir, "coords")
    os.makedirs(coords_dir, exist_ok=True)

    print("=" * 60)
    print("  TorusFold Expanded Test Set Builder")
    print("=" * 60)
    print(f"  Target: >= {target} samples")
    print(f"  Length range: {min_length}-{max_length} nt")
    print()

    all_entries: List[Dict] = []

    # ── Source 1: PDB experimental (9H8A, 8xtp, etc.) ──
    print("[1/3] Loading PDB experimental structures...")
    exp_entries = parse_pdb_experimental(pdb_exp_dir)
    print(f"  Found {len(exp_entries)} RNA chains from experimental PDB")

    # Filter and quality-check
    for entry in exp_entries:
        L = len(entry["sequence"])
        if L < min_length or L > max_length:
            continue

        coords = entry["coords"]
        if not entry.get("circular", False):
            # Circularize linear structures
            coords = annealing_closure(coords)

        metrics = compute_quality_metrics(coords)

        # Quality filter
        if metrics["closure_distance"] > max_closure:
            continue
        if metrics["bond_rmsd"] > max_bond_rmsd:
            continue

        # Predict secondary structure
        ss, pairs = predict_ss_circ(entry["sequence"])

        entry["coords"] = coords
        entry["secondary_structure"] = ss
        entry["pair_constraints"] = pairs
        entry["quality"] = metrics
        all_entries.append(entry)

    print(f"  After filtering: {len(all_entries)} experimental entries")

    # ── Source 2: IsRNAcirc test set ──
    print("\n[2/3] Loading IsRNAcirc test set...")
    isrna_entries = parse_isrnacirc_test_set(isrnacirc_dir)
    print(f"  Found {len(isrna_entries)} IsRNAcirc structures")

    for entry in isrna_entries:
        L = len(entry["sequence"])
        if L < min_length or L > max_length:
            continue

        coords = entry["coords"]
        metrics = compute_quality_metrics(coords, is_circular=True)

        if metrics["closure_distance"] > max_closure:
            # Try re-circularizing
            coords = annealing_closure(coords)
            metrics = compute_quality_metrics(coords, is_circular=True)

        if metrics["closure_distance"] > max_closure:
            continue

        ss, pairs = predict_ss_circ(entry["sequence"])

        entry["coords"] = coords
        entry["secondary_structure"] = ss
        entry["pair_constraints"] = pairs
        entry["quality"] = metrics
        all_entries.append(entry)

    print(f"  After filtering: {len(all_entries)} total entries")

    # ── Source 3: RCSB supplementary (optional) ──
    if download_supplementary and len(all_entries) < target:
        print("\n[3/3] Downloading supplementary PDB structures...")
        supp_entries = get_supplementary_structures(
            pdb_cache, min_length, max_length,
            max_download=min(200, (target - len(all_entries)) * 3),
        )

        for entry in supp_entries:
            L = len(entry["sequence"])
            if L < min_length or L > max_length:
                continue

            metrics = compute_quality_metrics(entry["coords"], is_circular=True)

            if metrics["closure_distance"] > max_closure:
                continue

            ss, pairs = predict_ss_circ(entry["sequence"])

            entry["secondary_structure"] = ss
            entry["pair_constraints"] = pairs
            entry["quality"] = metrics
            all_entries.append(entry)

        print(f"  After supplementary: {len(all_entries)} total entries")
    else:
        print(f"\n[3/3] Skipping supplementary download "
              f"(have {len(all_entries)}, target {target})")

    # ── Deduplicate by sequence ──
    seen_seqs = {}
    unique_entries = []
    for entry in all_entries:
        seq = entry["sequence"]
        if seq not in seen_seqs:
            seen_seqs[seq] = len(unique_entries)
            unique_entries.append(entry)

    n_dup = len(all_entries) - len(unique_entries)
    if n_dup > 0:
        print(f"\n  Removed {n_dup} duplicate sequences")

    all_entries = unique_entries

    # ── Save dataset ──
    print(f"\n  Saving {len(all_entries)} test structures...")

    json_entries = []
    source_counts = {}
    length_distribution = {}
    category_counts = {}

    for i, entry in enumerate(all_entries):
        seq_id = entry["id"]

        # Save coords
        np.save(os.path.join(coords_dir, f"{seq_id}.npy"), entry["coords"].astype(np.float32))

        # Build JSON entry
        json_entry = {
            "id": seq_id,
            "sequence": entry["sequence"],
            "secondary_structure": entry.get("secondary_structure", ""),
            "pair_constraints": entry.get("pair_constraints", []),
            "length": len(entry["sequence"]),
            "source": entry.get("source", "unknown"),
            "confidence": entry.get("confidence", 0.5),
            "category": entry.get("category", ""),
        }

        # Add quality metrics
        for k, v in entry.get("quality", {}).items():
            json_entry[f"quality_{k}"] = v

        # Add source-specific fields
        for k in ("pdb_id", "chain_id", "original_closure",
                   "circularized_closure", "circular"):
            if k in entry:
                json_entry[k] = entry[k]

        json_entries.append(json_entry)

        # Stats
        src = entry.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

        cat = entry.get("category", "unknown")
        if cat:
            category_counts[cat] = category_counts.get(cat, 0) + 1

        L = len(entry["sequence"])
        if L <= 50:
            bin_name = "20-50"
        elif L <= 100:
            bin_name = "50-100"
        elif L <= 200:
            bin_name = "100-200"
        elif L <= 500:
            bin_name = "200-500"
        else:
            bin_name = "500+"
        length_distribution[bin_name] = length_distribution.get(bin_name, 0) + 1

    # Save sequences.json
    with open(os.path.join(output_dir, "sequences.json"), "w") as f:
        json.dump(json_entries, f, indent=2)

    # Compute aggregate quality stats
    closure_dists = [e.get("quality_closure_distance", 0) for e in json_entries]
    bond_rmsds = [e.get("quality_bond_rmsd", 0) for e in json_entries]

    # Save metadata
    metadata = {
        "total": len(json_entries),
        "target": target,
        "meets_target": len(json_entries) >= target,
        "length_range": [
            min(e["length"] for e in json_entries) if json_entries else 0,
            max(e["length"] for e in json_entries) if json_entries else 0,
        ],
        "mean_length": round(float(np.mean([e["length"] for e in json_entries])), 1) if json_entries else 0,
        "mean_closure": round(float(np.mean(closure_dists)), 3) if closure_dists else 0,
        "mean_bond_rmsd": round(float(np.mean(bond_rmsds)), 3) if bond_rmsds else 0,
        "sources": source_counts,
        "categories": category_counts,
        "length_distribution": length_distribution,
        "quality_thresholds": {
            "max_closure": max_closure,
            "max_bond_rmsd": max_bond_rmsd,
        },
        "min_length": min_length,
        "max_length": max_length,
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"  Expanded Test Set: {output_dir}/")
    print(f"  Total samples: {metadata['total']}")
    print(f"  Target met: {'YES' if metadata['meets_target'] else 'NO'} "
          f"(target was {target})")
    print(f"  Length range: {metadata['length_range'][0]}-{metadata['length_range'][1]} nt")
    print(f"  Mean length: {metadata['mean_length']:.1f} nt")
    print(f"  Mean closure: {metadata['mean_closure']:.3f} A")
    print(f"  Mean bond RMSD: {metadata['mean_bond_rmsd']:.3f} A")
    print(f"\n  Source breakdown:")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"    {src}: {count}")
    if category_counts:
        print(f"\n  Category breakdown (IsRNAcirc):")
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            print(f"    {cat}: {count}")
    print(f"\n  Length distribution:")
    for bin_name in ["20-50", "50-100", "100-200", "200-500", "500+"]:
        count = length_distribution.get(bin_name, 0)
        if count > 0:
            print(f"    {bin_name} nt: {count}")
    print(f"{'=' * 60}")
    print(f"\n  Next: Run evaluation on expanded test set")
    print(f"  python evaluate_scheme.py --test-set {output_dir} --device cuda")


# ── CLI ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build expanded test set for TorusFold (N >= 30)"
    )
    parser.add_argument("--output", type=str, default="data/expanded_test_set",
                        help="Output directory")
    parser.add_argument("--pdb-exp-dir", type=str,
                        default="",
                        help="Directory with experimental PDB CIF files")
    parser.add_argument("--isrnacirc-dir", type=str,
                        default="",
                        help="IsRNAcirc test set directory")
    parser.add_argument("--pdb-cache", type=str, default="data/pdb_cache",
                        help="PDB download cache directory")
    parser.add_argument("--target", type=int, default=30,
                        help="Minimum target number of samples (default: 30)")
    parser.add_argument("--min-length", type=int, default=20,
                        help="Minimum sequence length (default: 20)")
    parser.add_argument("--max-length", type=int, default=500,
                        help="Maximum sequence length (default: 500)")
    parser.add_argument("--max-closure", type=float, default=12.0,
                        help="Max closure distance for quality filter (default: 12.0)")
    parser.add_argument("--max-bond-rmsd", type=float, default=5.0,
                        help="Max bond RMSD for quality filter (default: 5.0)")
    parser.add_argument("--download-supplementary", action="store_true",
                        help="Download supplementary PDB structures if needed")
    args = parser.parse_args()

    # Find project data directory relative to this script
    script_dir = Path(__file__).resolve().parent

    # Default paths based on known project structure
    # Script is at: core/circrna/torusfold/expand_test_set.py
    # Data is at: data/circrna_3d/
    project_root = script_dir.parents[3]  # confluencia_3_0/

    pdb_exp_dir = args.pdb_exp_dir
    if not pdb_exp_dir:
        pdb_exp_dir = str(project_root / "data" / "circrna_3d" / "pdb_experimental")

    isrnacirc_dir = args.isrnacirc_dir
    if not isrnacirc_dir:
        isrnacirc_dir = str(project_root / "data" / "circrna_3d" / "isrnacirc_test_set")

    build_expanded_test_set(
        output_dir=args.output,
        pdb_exp_dir=pdb_exp_dir,
        isrnacirc_dir=isrnacirc_dir,
        pdb_cache=args.pdb_cache,
        target=args.target,
        min_length=args.min_length,
        max_length=args.max_length,
        max_closure=args.max_closure,
        max_bond_rmsd=args.max_bond_rmsd,
        download_supplementary=args.download_supplementary,
    )


if __name__ == "__main__":
    main()
