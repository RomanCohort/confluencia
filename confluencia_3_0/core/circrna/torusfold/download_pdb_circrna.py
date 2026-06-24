#!/usr/bin/env python3
"""
download_pdb_circrna.py — Download PDB structures related to circRNA.

Searches RCSB PDB for "circular RNA" and downloads relevant structures.
"""

import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent.parent.parent.parent / "data" / "pdb_circrna"


def search_pdb(query: str, max_results: int = 50) -> list:
    """Search RCSB PDB for entries matching query."""
    url = "https://search.rcsb.org/rcsbsearch/v2/query"

    payload = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {"value": query}
        },
        "return_type": "entry",
        "request_options": {
            "results_content_type": "experimental",
            "return_all_hits": False
        }
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
            return [e['identifier'] for e in data.get('result_set', [])[:max_results]]
    except Exception as e:
        print(f"Search failed: {e}")
        return []


def get_entry_info(pdb_id: str) -> dict:
    """Get metadata for a PDB entry."""
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"

    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            data = json.loads(response.read().decode('utf-8'))

            # Extract relevant fields
            title = data.get('struct', {}).get('title', '')
            method = data.get('exptl', [{}])[0].get('method', 'Unknown')
            resolution = data.get('rcsb_entry_info', {}).get('resolution_combined', [None])
            if isinstance(resolution, list) and resolution:
                resolution = resolution[0]

            # Check if RNA
            polymer_count = data.get('rcsb_entry_info', {}).get('polymer_entity_count', 0)

            return {
                'pdb_id': pdb_id,
                'title': title,
                'method': method,
                'resolution': resolution,
                'polymer_count': polymer_count,
            }
    except Exception as e:
        return {'pdb_id': pdb_id, 'error': str(e)}


def download_pdb(pdb_id: str, output_dir: Path) -> bool:
    """Download PDB file."""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    output_path = output_dir / f"{pdb_id}.pdb"

    if output_path.exists():
        print(f"  ✓ Already exists: {pdb_id}")
        return True

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read().decode('utf-8')

        with open(output_path, 'w') as f:
            f.write(content)

        print(f"  ✓ Downloaded: {pdb_id}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {pdb_id} - {e}")
        return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Download PDB circRNA Structures")
    print("=" * 60)
    print(f"  Output: {OUTPUT_DIR}")

    # Search for circRNA-related structures
    print("\n[1/3] Searching PDB for 'circular RNA'...")
    pdb_ids = search_pdb("circular RNA", max_results=100)
    print(f"  Found {len(pdb_ids)} entries")

    # Also search for specific terms
    for term in ["circRNA", "back-splice", "lariat RNA"]:
        extra = search_pdb(term, max_results=30)
        for pid in extra:
            if pid not in pdb_ids:
                pdb_ids.append(pid)

    print(f"  Total unique: {len(pdb_ids)}")

    if not pdb_ids:
        print("  No PDB entries found!")
        return

    # Get info and filter
    print("\n[2/3] Filtering for RNA structures...")
    rna_entries = []

    for i, pdb_id in enumerate(pdb_ids):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Checking {i+1}/{len(pdb_ids)}...")

        info = get_entry_info(pdb_id)

        if 'error' in info:
            continue

        title = info.get('title', '').lower()
        method = info.get('method', '')

        # Filter: must be RNA-related
        is_rna = any(kw in title for kw in ['rna', 'ribonucleic', 'circrna', 'circular rna', 'lariat'])

        if is_rna:
            rna_entries.append(info)
            print(f"    ✓ {pdb_id}: {info['title'][:50]}... ({method})")

    print(f"\n  RNA structures found: {len(rna_entries)}")

    if not rna_entries:
        print("  No RNA structures found!")
        # Fallback: download all results anyway
        print("  Downloading all search results as fallback...")
        rna_entries = [{'pdb_id': pid} for pid in pdb_ids[:50]]

    # Download PDB files
    print("\n[3/3] Downloading PDB files...")
    downloaded = 0

    for entry in rna_entries[:50]:  # Limit to 50
        pdb_id = entry['pdb_id']
        if download_pdb(pdb_id, OUTPUT_DIR):
            downloaded += 1

    print(f"\n{'='*60}")
    print(f"  Downloaded: {downloaded} PDB files")
    print(f"  Location: {OUTPUT_DIR}")
    print(f"{'='*60}")

    # Save metadata
    meta_path = OUTPUT_DIR / "pdb_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(rna_entries, f, indent=2)
    print(f"  Metadata: {meta_path}")


if __name__ == '__main__':
    main()
