#!/usr/bin/env python3
"""
fetch_pdb_rna.py — Download real RNA structures from RCSB PDB and prepare
circRNA-relevant test/training split for TorusFold.

Pipeline:
  1. Search RCSB PDB for RNA-related entries (multiple queries)
  2. Filter for actual RNA structures (not DNA/protein-only)
  3. Download PDB files
  4. Parse C3' coordinates, score by quality
  5. Split: top-30 → test_set, rest → training_pool
  6. Output in TorusFold format (sequences.json + coords/*.npy)

Usage:
    python fetch_pdb_rna.py --output data/pdb_rna_real --keep 30
    python fetch_pdb_rna.py --skip-download --input data/pdb_rna_raw --keep 30
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy required. pip install numpy")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# RCSB PDB Search API
# ═══════════════════════════════════════════════════════════════

def search_rcsb(query_text: str, max_results: int = 200) -> List[str]:
    """Search RCSB PDB using full-text search.

    Returns list of PDB IDs.
    """
    url = "https://search.rcsb.org/rcsbsearch/v2/query"

    payload = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {"value": query_text}
        },
        "return_type": "entry",
        "request_options": {
            "return_all_hits": True
        }
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            hits = result.get("result_set", [])
            ids = [h["identifier"] for h in hits[:max_results]]
            print(f"    Query '{query_text}': {result.get('total_count', len(hits))} hits, using {len(ids)}")
            return ids
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        print(f"    Query '{query_text}': HTTP {e.code} — {body[:200]}")
        return []
    except Exception as e:
        print(f"    Query '{query_text}': {e}")
        return []


def search_rcsb_rna_polymer(max_results: int = 500) -> List[str]:
    """Search for entries that contain RNA polymer entities.

    Uses full_text search for "RNA" — the attribute-based search is not
    available on the public API, so we rely on metadata filtering later.
    """
    return search_rcsb("RNA", max_results=max_results)


def get_entry_metadata(pdb_id: str) -> Optional[Dict]:
    """Get metadata for a PDB entry from RCSB REST API."""
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TorusFold/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        title = data.get("struct", {}).get("title", "")
        method = ""
        exptl = data.get("exptl", [])
        if exptl:
            method = exptl[0].get("method", "")

        entry_info = data.get("rcsb_entry_info", {})
        resolution = entry_info.get("resolution_combined", [None])
        if isinstance(resolution, list) and resolution:
            resolution = resolution[0]

        rna_count = entry_info.get("polymer_entity_count_RNA", 0)
        dna_count = entry_info.get("polymer_entity_count_DNA", 0)
        protein_count = entry_info.get("polymer_entity_count_protein", 0)

        return {
            "pdb_id": pdb_id,
            "title": title,
            "method": method,
            "resolution": resolution,
            "rna_count": rna_count,
            "dna_count": dna_count,
            "protein_count": protein_count,
        }
    except Exception as e:
        return {"pdb_id": pdb_id, "error": str(e)}


# ═══════════════════════════════════════════════════════════════
# PDB Download
# ═══════════════════════════════════════════════════════════════

def download_pdb_file(pdb_id: str, output_dir: Path) -> Optional[Path]:
    """Download a PDB file from RCSB."""
    save_path = output_dir / f"{pdb_id}.pdb"

    if save_path.exists() and save_path.stat().st_size > 100:
        return save_path

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8", errors="ignore")

        if len(content) < 50:
            return None

        with open(save_path, "w") as f:
            f.write(content)

        size_kb = save_path.stat().st_size / 1024
        return save_path
    except Exception as e:
        # Try .cif format as fallback
        cif_url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        cif_path = output_dir / f"{pdb_id}.cif"

        try:
            req = urllib.request.Request(cif_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read().decode("utf-8", errors="ignore")

            if len(content) < 50:
                return None

            with open(cif_path, "w") as f:
                f.write(content)

            return cif_path
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════
# PDB Parsing
# ═══════════════════════════════════════════════════════════════

def parse_pdb_rna(pdb_path: Path) -> Optional[Dict]:
    """Parse PDB file, extract RNA nucleotide C3' coordinates."""
    sequence = []
    coords = []
    seen_residues = set()
    resolution = None
    method = None

    base_map = {
        "A": "A", "U": "U", "G": "G", "C": "C",
        "DA": "A", "DU": "U", "DG": "G", "DC": "C",
        "I": "G", "PSU": "U",
    }

    with open(pdb_path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("EXPDTA"):
                method_str = line[6:].strip()
                if "X-RAY" in method_str:
                    method = "X-RAY"
                elif "NMR" in method_str:
                    method = "NMR"
                elif "CRYO-EM" in method_str or "ELECTRON MICROSCOPY" in method_str:
                    method = "CRYO-EM"
                else:
                    method = method_str[:20]

            if line.startswith("REMARK   2 RESOLUTION"):
                try:
                    resolution = float(line.split()[-1])
                except ValueError:
                    pass

            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            if len(line) < 54:
                continue

            atom_name = line[12:16].strip()
            if atom_name != "C3'":
                continue

            chain_id = line[21]
            try:
                res_seq = int(line[22:26].strip())
            except ValueError:
                continue
            i_code = line[26]

            res_key = (chain_id, res_seq, i_code)
            if res_key in seen_residues:
                continue
            seen_residues.add(res_key)

            res_name = line[17:20].strip()
            base = base_map.get(res_name, None)
            if base is None:
                continue

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
        "sequence": "".join(sequence),
        "coords": np.array(coords, dtype=np.float32),
        "n_residues": len(coords),
        "resolution": resolution,
        "method": method,
    }


def parse_pdb_all_residues(pdb_path: Path) -> Optional[Dict]:
    """Fallback: parse PDB extracting all residue centroids (coarse-grained)."""
    residues = {}
    method = None

    with open(pdb_path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("EXPDTA"):
                method_str = line[6:].strip()
                if "X-RAY" in method_str:
                    method = "X-RAY"
                elif "NMR" in method_str:
                    method = "NMR"
                elif "CRYO-EM" in method_str:
                    method = "CRYO-EM"

            if not line.startswith("ATOM"):
                continue

            if len(line) < 54:
                continue

            chain = line[21]
            try:
                resseq = int(line[22:26].strip())
            except ValueError:
                continue
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

    coords = []
    for key in sorted(residues.keys()):
        arr = np.array(residues[key])
        coords.append(arr.mean(axis=0))

    return {
        "sequence": "N" * len(coords),
        "coords": np.array(coords, dtype=np.float32),
        "n_residues": len(coords),
        "resolution": None,
        "method": method,
    }


# ═══════════════════════════════════════════════════════════════
# Quality Scoring
# ═══════════════════════════════════════════════════════════════

def quality_score(result: Dict, pdb_id: str = "") -> float:
    """Score a parsed structure for test set selection. Higher = better."""
    score = 0.0

    # BONUS: circRNA/lariat-specific entries (most relevant for our use case)
    # These are the most valuable structures for circRNA-specific testing
    circrna_priority = {
        # Confirmed circular RNA structures
        "9H83": 20.0,  # Small circular RNA dimer (cryo-EM)
        "9H86": 20.0,  # Small circular RNA dimer
        "9H8A": 20.0,  # Small circular RNA dimer
        "6G7Z": 15.0,  # Lariat-capping ribozyme
        "4P9R": 15.0,  # Lariat capping ribozyme
        "4P8Z": 12.0,  # Lariat capping ribozyme
        # RNA-only structures (highly relevant)
        "7VKI": 10.0,  # ESRP1 qRRM2-RNA complex
    }
    if pdb_id in circrna_priority:
        score += circrna_priority[pdb_id]

    # Method quality
    method_scores = {"X-RAY": 3.0, "CRYO-EM": 2.5, "NMR": 2.0}
    score += method_scores.get(result.get("method"), 1.0)

    # Resolution
    res = result.get("resolution")
    if res is not None:
        if res < 2.0:
            score += 3.0
        elif res < 3.0:
            score += 2.0
        elif res < 5.0:
            score += 1.0
        else:
            score += 0.5

    # Length: prefer moderate (30-500 nt)
    L = result["n_residues"]
    if 30 <= L <= 500:
        score += 2.0
    elif 20 <= L <= 800:
        score += 1.0

    # Coordinate quality
    coords = result["coords"]
    if np.isnan(coords).any() or np.isinf(coords).any():
        score -= 10.0
    else:
        spread = np.std(coords)
        if spread > 5.0:
            score += 1.0
        elif spread > 1.0:
            score += 0.5

    # Known sequence
    seq = result.get("sequence", "")
    if seq and "N" not in seq:
        score += 1.0

    return score


# ═══════════════════════════════════════════════════════════════
# Output Formatting (TorusFold-compatible)
# ═══════════════════════════════════════════════════════════════

def write_torusfold_dataset(
    samples: List[Dict],
    output_dir: Path,
    label: str = "test",
):
    """Write samples in TorusFold format (sequences.json + coords/*.npy)."""
    coords_dir = output_dir / "coords"

    # Clean existing coords directory to avoid stale files
    if coords_dir.exists():
        import shutil
        shutil.rmtree(coords_dir)

    coords_dir.mkdir(parents=True, exist_ok=True)

    seq_data = []
    for sample in samples:
        sid = sample["id"]
        coords = sample["coords"]

        np.save(coords_dir / f"{sid}.npy", coords)

        method = sample.get("method")
        if method == "X-RAY":
            conf = 1.0
        elif method == "CRYO-EM":
            res = sample.get("resolution")
            conf = 0.95 if res and res < 3.0 else 0.9
        elif method == "NMR":
            conf = 0.85
        else:
            conf = 0.7

        entry = {
            "id": sid,
            "sequence": sample["sequence"],
            "length": len(sample["sequence"]),
            "source": sample.get("source", "PDB"),
            "confidence": conf,
            "method": method or "unknown",
            "resolution": sample.get("resolution"),
            "quality_score": round(sample.get("quality_score", 0), 2),
        }
        seq_data.append(entry)

    with open(output_dir / "sequences.json", "w") as f:
        json.dump(seq_data, f, indent=2)

    meta = {
        "label": label,
        "n_samples": len(samples),
        "lengths": [s["n_residues"] for s in samples],
        "methods": list(set(s.get("method", "?") for s in samples if s.get("method"))),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Written: {output_dir}/")
    print(f"    sequences.json: {len(samples)} entries")
    print(f"    coords/: {len(samples)} .npy files")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Download real RNA structures from PDB for TorusFold"
    )
    parser.add_argument("--output", type=str, default="data/pdb_rna_real",
                        help="Output directory")
    parser.add_argument("--keep", type=int, default=30,
                        help="Number of top-quality samples for test set")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, parse existing PDB files")
    parser.add_argument("--input", type=str, default=None,
                        help="Directory with pre-downloaded PDB files")
    parser.add_argument("--max-download", type=int, default=200,
                        help="Max PDB entries to download")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir = output_dir / "pdb_raw"
    pdb_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Fetch Real RNA Structures from PDB")
    print("=" * 60)
    print(f"  Output: {output_dir}")
    print(f"  Test set size: {args.keep}")

    # ── Step 1: Discover PDB IDs ──
    all_ids: List[str] = []

    if args.input:
        # Use pre-downloaded PDB files
        input_path = Path(args.input)
        existing = list(input_path.glob("*.pdb")) + list(input_path.glob("*.cif"))
        print(f"\n  Using pre-downloaded: {input_path} ({len(existing)} files)")
        all_ids = [p.stem for p in existing]
    elif args.skip_download:
        # Check pdb_dir for existing files
        existing = list(pdb_dir.glob("*.pdb")) + list(pdb_dir.glob("*.cif"))
        print(f"\n  Skip download. Found {len(existing)} existing files in {pdb_dir}")
        all_ids = [p.stem for p in existing]
    else:
        print("\n[1/4] Searching RCSB PDB for RNA structures...")

        # Priority: circRNA/lariat/ribozyme-specific searches (more relevant)
        priority_terms = [
            "circular RNA",   # 343 hits — most relevant
            "lariat RNA",     # 38 hits — back-splice junction
            "circRNA",        # 3 hits — direct
            "ribozyme",       # 406 hits — catalytic RNA, often circular
            "RNA hairpin",    # common RNA motif
            "RNA junction",   # multi-way junctions
            "back-splice",    # circRNA biogenesis
        ]
        for term in priority_terms:
            extra = search_rcsb(term, max_results=100)
            for pid in extra:
                if pid not in all_ids:
                    all_ids.append(pid)

        # Also add known circRNA-related PDB IDs from literature
        known_ids = [
            "9H83", "9H86", "9H8A",  # Lariat RNA structures (2024)
            "6AHW", "6R9R", "9KGH", "9KGG",  # RNA-protein complexes
            "7VKI", "7VKJ", "7WRN",  # circRNA-specific
            "6G7Z", "5UKI", "4P9R",  # Lariat
        ]
        for pid in known_ids:
            if pid not in all_ids:
                all_ids.append(pid)

        # Deduplicate
        seen = set()
        unique_ids = []
        for pid in all_ids:
            if pid not in seen:
                seen.add(pid)
                unique_ids.append(pid)
        all_ids = unique_ids

        print(f"\n  Total unique PDB IDs: {len(all_ids)}")

        # ── Step 2: Filter for RNA content ──
        print(f"\n[2/4] Filtering for RNA structures (checking metadata)...")

        rna_ids_filtered = []
        for i, pdb_id in enumerate(all_ids[:args.max_download]):
            if (i + 1) % 20 == 0 or i == 0:
                print(f"  Checking {i+1}/{min(len(all_ids), args.max_download)}...")

            meta = get_entry_metadata(pdb_id)
            if meta is None or "error" in meta:
                continue

            # Must have RNA polymer entities
            rna_count = meta.get("rna_count", 0)
            if rna_count and rna_count > 0:
                rna_ids_filtered.append(pdb_id)
                title = meta.get("title", "")[:60]
                method = meta.get("method", "?")
                print(f"    + {pdb_id}: RNA={rna_count}, {method}, {title}...")

            time.sleep(0.1)  # Rate limit

        print(f"\n  RNA-containing entries: {len(rna_ids_filtered)}")
        all_ids = rna_ids_filtered

        if not all_ids:
            print("  No RNA entries found! Trying download without filter...")
            all_ids = unique_ids[:args.max_download]

        # ── Step 3: Download PDB files ──
        print(f"\n[3/4] Downloading {len(all_ids)} PDB files...")

        downloaded = 0
        failed = 0
        for i, pdb_id in enumerate(all_ids):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(all_ids)}]")

            path = download_pdb_file(pdb_id, pdb_dir)
            if path is not None:
                downloaded += 1
            else:
                failed += 1

            time.sleep(0.05)  # Rate limit

        print(f"\n  Downloaded: {downloaded}, Failed: {failed}")

    # ── Step 4: Parse and score ──
    print(f"\n[4/4] Parsing structures and scoring...")

    pdb_files = list(pdb_dir.glob("*.pdb")) + list(pdb_dir.glob("*.cif"))
    if args.input:
        input_path = Path(args.input)
        pdb_files = list(input_path.glob("*.pdb")) + list(input_path.glob("*.cif"))

    print(f"  PDB files to parse: {len(pdb_files)}")

    parsed = []
    failed = 0
    for i, pdb_path in enumerate(pdb_files):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(pdb_files)}] {pdb_path.name}...")

        # Try RNA-specific parsing first
        result = parse_pdb_rna(pdb_path)
        if result is None:
            result = parse_pdb_all_residues(pdb_path)

        if result is None:
            failed += 1
            continue

        # Quality checks
        coords = result["coords"]
        if np.isnan(coords).any() or np.isinf(coords).any():
            failed += 1
            continue
        if coords.std() < 1.0:
            failed += 1
            continue

        # Skip entries with unknown sequence (all N) for test set quality
        seq = result.get("sequence", "")
        if "N" in seq and len(seq) == seq.count("N"):
            # Still add to parsed but mark as low priority
            result["quality_score"] = 0.0
            result["id"] = pdb_path.stem
            result["source"] = "PDB"
            parsed.append(result)
            continue

        # Score and tag
        result["id"] = pdb_path.stem
        result["quality_score"] = quality_score(result, pdb_id=result["id"])
        result["source"] = "PDB"

        parsed.append(result)

    print(f"\n  Parsed successfully: {len(parsed)}")
    print(f"  Failed: {failed}")

    if not parsed:
        print("  ERROR: No valid structures parsed!")
        return

    # ── Sort by quality and split ──
    parsed.sort(key=lambda x: x["quality_score"], reverse=True)

    print(f"\n  Top 10 by quality:")
    for i, s in enumerate(parsed[:10]):
        print(f"    {i+1}. {s['id']}: L={s['n_residues']}, "
              f"method={s.get('method','?')}, res={s.get('resolution','?')}, "
              f"score={s['quality_score']:.1f}")

    n_test = min(args.keep, len(parsed))
    test_samples = parsed[:n_test]
    train_pool = parsed[n_test:]

    print(f"\n{'='*60}")
    print(f"  Dataset Split")
    print(f"{'='*60}")
    print(f"  Test set: {len(test_samples)} samples (highest quality)")
    print(f"  Training pool: {len(train_pool)} samples")

    test_lengths = [s["n_residues"] for s in test_samples]
    if test_lengths:
        print(f"  Test lengths: min={min(test_lengths)}, max={max(test_lengths)}, "
              f"mean={np.mean(test_lengths):.0f}")

    # ── Write output ──
    test_dir = output_dir / "test_set"
    write_torusfold_dataset(test_samples, test_dir, label="test")

    if train_pool:
        pool_dir = output_dir / "training_pool"
        write_torusfold_dataset(train_pool, pool_dir, label="training_pool")

    # Summary
    summary = {
        "n_test": len(test_samples),
        "n_train_pool": len(train_pool),
        "test_ids": [s["id"] for s in test_samples],
        "train_pool_ids": [s["id"] for s in train_pool],
        "test_length_stats": {
            "min": int(min(test_lengths)) if test_lengths else 0,
            "max": int(max(test_lengths)) if test_lengths else 0,
            "mean": round(float(np.mean(test_lengths)), 1) if test_lengths else 0,
        },
        "source": "RCSB PDB (real experimental structures)",
        "note": "These are real experimentally determined RNA structures from PDB. "
                "Training pool can be expanded on AutoDL.",
    }
    with open(output_dir / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"{'='*60}")
    print(f"  Test set:     {test_dir}/  ({len(test_samples)} samples)")
    print(f"  Train pool:   {output_dir / 'training_pool'}/  ({len(train_pool)} samples)")
    print(f"  Summary:      {output_dir / 'split_summary.json'}")
    print(f"  Raw PDB:      {pdb_dir}/")


if __name__ == "__main__":
    main()
