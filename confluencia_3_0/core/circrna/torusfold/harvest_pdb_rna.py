"""Harvest pure-nucleic-acid PDB structures (no protein), filter to RNA-only chains.

Pipeline (NAKB ID list -> RCSB download -> local RNA filter):
1. Fetch pure-NA entry IDs from NAKB Solr API (NAKBnaList filled, NAKBprotList empty)
   — ~5000 entries, pre-filtered (no RNA-protein complexes)
2. Download each PDB file from RCSB (cached on disk)
3. Parse with Biopython, keep RNA-only chains 20-500nt with C3' atoms (drop DNA chains)
4. Save C3' coords as .npy per chain (aligned with circrna_3d_all format)

Cyclization (BSJ linker) happens in a separate stage.

Resumable: downloaded PDBs cached in data/pdb_raw/, filtered coords in data/pdb_rna_c3prime/.
"""
import os, sys, time, json
import requests
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent
DEPLOY_ROOT = BASE.parents[3]
RAW_DIR = DEPLOY_ROOT / 'data' / 'pdb_raw'
OUT_DIR = DEPLOY_ROOT / 'data' / 'pdb_rna_c3prime'
PURE_NA_IDS = DEPLOY_ROOT / 'data' / 'pdb_rna_ids_pure_na.json'
RNASOLO_IDS = DEPLOY_ROOT / 'data' / 'pdb_rna_ids_rnasolo.json'
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

NAKB_SOLR = ('https://nakb.org/node/solr/nakb/select?fl=id,pdbid,NAKBnaList,'
             'NAKBprotList,NAKBligList&q=NAKBna:*%20OR%20NAKBprot:*%20OR%20'
             'NAKBlig:*&wt=csv&rows=200000')
RCSB_FILE = 'https://files.rcsb.org/download/{id}.pdb'


def fetch_rna_ids(source='rnasolo'):
    """Fetch RNA PDB IDs. source='rnasolo' (8471 cleaned RNA-only, preferred)
    or 'nakb' (5097 pure-NA incl DNA, needs local DNA filtering)."""
    id_file = RNASOLO_IDS if source == 'rnasolo' else PURE_NA_IDS
    if id_file.exists():
        ids = json.loads(id_file.read_text())
        print(f'  cached {source} ID list: {len(ids)} entries')
        return ids
    if source == 'rnasolo':
        print('  ERROR: RNAsolo ID list not cached. Run the CSV mapping step first.')
        print('         Falling back to NAKB.')
        source = 'nakb'
        id_file = PURE_NA_IDS
    if id_file.exists():
        return json.loads(id_file.read_text())
    print('  fetching pure-NA IDs from NAKB Solr...')
    import csv, io
    r = requests.get(NAKB_SOLR, timeout=90)
    reader = csv.DictReader(io.StringIO(r.text))
    rows = list(reader)
    pure = [row['pdbid'] for row in rows
            if row['NAKBnaList'] and not row['NAKBprotList']]
    PURE_NA_IDS.write_text(json.dumps(pure))
    print(f'  pure-NA entries (no protein): {len(pure)}')
    return pure


def download_pdb(pdb_id, retry=2):
    cached = RAW_DIR / f'{pdb_id}.pdb'
    if cached.exists() and cached.stat().st_size > 100:
        return cached.read_text(encoding='utf-8', errors='ignore')
    url = RCSB_FILE.format(id=pdb_id)
    for attempt in range(retry + 1):
        try:
            r = requests.get(url, timeout=40)
            if r.status_code == 200 and r.text:
                cached.write_text(r.text, encoding='utf-8')
                return r.text
            if r.status_code == 404:
                return None
        except Exception:
            time.sleep(1)
    return None


def filter_rna_chains(pdb_text, pdb_id, min_len=20, max_len=500):
    """Parse PDB, return list of (chain_id, L, coords[L,3]) for pure RNA chains
    with C3' atoms, length in [min_len, max_len]. Pure = no protein/DNA atoms."""
    from Bio.PDB import PDBParser
    from io import StringIO
    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    try:
        struct = parser.get_structure(pdb_id, StringIO(pdb_text))
    except Exception:
        return []
    model = next(iter(struct))
    results = []
    for chain in model:
        has_protein = False
        has_dna = False
        c3 = []
        for res in chain:
            if res.id[0] != ' ':
                continue
            resname = res.get_resname().strip()
            # DNA residues: DA/DC/DG/DT
            if resname in ('DA', 'DC', 'DG', 'DT', 'DU'):
                has_dna = True
            # protein: standard amino acids (3-letter)
            if resname in ('ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY',
                           'HIS','ILE','LEU','LYS','MET','PHE','PRO','SER',
                           'THR','TRP','TYR','VAL','MSE'):
                has_protein = True
            # RNA: A/C/G/U (or RA/RC/RG/RU in some files)
            if resname in ('A','C','G','U','RA','RC','RG','RU','PSU','5MC','OMC','H2U'):
                if "C3'" in res:
                    c3.append(res["C3'"].get_coord())
                elif 'C3*' in res:
                    c3.append(res['C3*'].get_coord())
        if has_protein or has_dna:
            continue  # not pure RNA
        if min_len <= len(c3) <= max_len:
            results.append((chain.id, len(c3), np.array(c3, dtype=np.float64)))
    return results


def main(limit=None, id_filter=None, source='rnasolo'):
    import logging
    log_file = DEPLOY_ROOT / 'data' / f'harvest_{source}.log'
    logging.basicConfig(filename=str(log_file), level=logging.INFO,
                        format='%(asctime)s %(message)s', force=True)
    log = logging.getLogger()
    print('=' * 60)
    print(f'  PDB RNA harvest (source={source} -> RCSB download -> RNA filter)')
    print(f'  log: {log_file}')
    print('=' * 60)
    ids = fetch_rna_ids(source=source)
    if id_filter:
        ids = [i for i in ids if id_filter(i)]
    if limit:
        ids = ids[:limit]
    print(f'  processing {len(ids)} entries')
    log.info(f'start source={source} n_ids={len(ids)}')

    n_ok = n_skip = n_fail = 0
    for i, pid in enumerate(ids):
        try:
            txt = download_pdb(pid)
            if txt is None:
                n_fail += 1; continue
            chains = filter_rna_chains(txt, pid)
            if not chains:
                n_skip += 1; continue
            for cid, L, coords in chains:
                out = OUT_DIR / f'{pid}_{cid}.npy'
                np.save(out, coords)
                n_ok += 1
        except Exception as e:
            log.error(f'{pid} FAILED: {e}')
            n_fail += 1
        if (i + 1) % 50 == 0:
            msg = f'[{i+1}/{len(ids)}] kept {n_ok} chains, skipped {n_skip}, failed {n_fail}'
            print(msg); log.info(msg)
    done_msg = f'DONE source={source}: {n_ok} chains kept ({n_skip} skip, {n_fail} fail)'
    print(done_msg); log.info(done_msg)
    print(f'  output: {OUT_DIR}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None, help='max entries to process')
    ap.add_argument('--filter', type=str, default=None, help='substring filter on PDB id')
    ap.add_argument('--source', type=str, default='rnasolo',
                    choices=['rnasolo', 'nakb'], help='ID list source (rnasolo=8471 cleaned, nakb=5097 pure-NA)')
    args = ap.parse_args()
    main(limit=args.limit, id_filter=(lambda i: args.filter in i) if args.filter else None,
         source=args.source)
