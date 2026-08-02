"""consolidate_pdb_cyclized.py — Pack PDB-cyclized RNA C3' coords into consolidated.npz.

Phase-0 (目标1) 预训练数据打包。对齐 circrna_3d_all_consolidated.npz 格式
(ids/lengths/coords)，额外加 `seqs`：CG 管线的序列来自 circBase FASTA，但 PDB
环化数据没有 FASTA 源，序列从 pdb_raw/ 缓存重读 PDB 提取（collate 需要 tokenize）。

两个关键点：
1. 序列提取必须只保留「有 C3'/C3* 原子的 RNA 残基」——harvest 只把这类残基的
   坐标写进 .npy，序列若多/少一个残基，tokenize 后长度就和坐标对不上。
2. 用 ATOM 行纯文本解析（不用 Biopython Structure）：一个 PDB 只读一遍、提取
   全部 RNA chain 的序列，避免对每个 npy 重复解析大结构（核糖体级会慢到不可用）。

Usage:
    python consolidate_pdb_cyclized.py
    # Reads:  ../data/pdb_cyclized/*.npy  + ../data/pdb_raw/*.pdb
    # Writes: ../data/pdb_cyclized/consolidated.npz
"""
import os, sys, time
from collections import defaultdict
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

BASE = Path(__file__).resolve().parent
DEPLOY_ROOT = BASE.parents[3]
IN_DIR = DEPLOY_ROOT / 'data' / 'pdb_cyclized'
RAW_DIR = DEPLOY_ROOT / 'data' / 'pdb_raw'
OUTPUT = IN_DIR / 'consolidated.npz'

MIN_LEN, MAX_LEN = 20, 500
LINEAR_DIR = DEPLOY_ROOT / 'data' / 'pdb_rna_c3prime'  # 线性 RNA C3'（未环化）

# Same RNA residue set as harvest_pdb_rna.filter_rna_chains.
RNA_RES = {
    'A': 'A', 'RA': 'A',
    'U': 'U', 'RU': 'U', 'PSU': 'U', 'H2U': 'U',
    'G': 'G', 'RG': 'G',
    'C': 'C', 'RC': 'C', '5MC': 'C', 'OMC': 'C',
}


def extract_chains_seqs(pdb_id: str) -> dict:
    """Parse one cached PDB (single pass) → {chain_id: rna_seq}.

    Mirrors harvest's coordinate extraction so sequence length == coordinate
    length: only ATOM-record RNA residues with a C3'/C3* atom count. HETATM
    modified bases (PSU/H2U/5MC/OMC) are EXCLUDED — harvest's `res.id[0] != ' '`
    drops them (Biopython flags HETATM residues 'H_<resname>'), so coords never
    contain them. Text-based ATOM parsing (no Biopython Structure) keeps big
    ribosome PDBs fast.
    """
    pdb_path = RAW_DIR / f'{pdb_id}.pdb'
    seqs: dict = defaultdict(list)
    seen: set = set()  # (chain, resSeq) dedup — one C3' per residue
    try:
        with open(pdb_path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line[:6] != 'ATOM  ':
                    continue  # HETATM modified bases excluded (matches harvest)
                atom = line[12:16].strip()
                if atom not in ("C3'", 'C3*'):
                    continue
                resname = line[17:20].strip()
                base = RNA_RES.get(resname)
                if base is None:
                    continue
                chain = line[21]
                resseq = line[22:26].strip()
                key = (chain, resseq)
                if key in seen:
                    continue
                seen.add(key)
                seqs[chain].append(base)
    except OSError:
        return {}
    return {c: ''.join(s) for c, s in seqs.items()}


def extract_chain_seq_bio(pdb_id: str, chain_id: str) -> str | None:
    """Biopython fallback for the few text/coords mismatches — identical logic
    to harvest's filter_rna_chains, so the result is guaranteed length-aligned
    with the coordinates. Slow (full Structure build), only used as fallback."""
    pdb_path = RAW_DIR / f'{pdb_id}.pdb'
    if not pdb_path.exists():
        return None
    try:
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True, PERMISSIVE=True)
        struct = parser.get_structure(pdb_id, pdb_path)
    except Exception:
        return None
    model = next(iter(struct))
    for chain in model:
        if chain.id != chain_id:
            continue
        chars = []
        for res in chain:
            if res.id[0] != ' ':
                continue
            base = RNA_RES.get(res.get_resname().strip())
            if base is None:
                continue
            if "C3'" in res or 'C3*' in res:
                chars.append(base)
        return ''.join(chars) if chars else None
    return None


def _pack_dir(files, pdb_cache, circ_label: int):
    """Pack one directory's .npy files. Returns (ids, lengths, coords, seqs, is_circular).
    circ_label: 1 for cyclized (pseudo-circRNA), 0 for linear RNA."""
    ids, lengths, coords_list, seqs, circs = [], [], [], [], []
    bad = n_fallback = 0
    t1 = time.time()
    for i, f in enumerate(files):
        stem = f.stem
        pdb_id = stem.rsplit('_', 1)[0]
        chain = stem.rsplit('_', 1)[1] if '_' in stem else ''
        try:
            arr = np.load(f)
            L = arr.shape[0]
            if L < MIN_LEN or L > MAX_LEN or np.isnan(arr).any() or np.isinf(arr).any():
                bad += 1; continue
            s = pdb_cache.get(pdb_id, {}).get(chain)
            if s is None or len(s) != L:
                # Text parse mismatch → Biopython fallback (harvest-identical).
                s = extract_chain_seq_bio(pdb_id, chain)
            if s is None or len(s) != L:
                # Still no match → ACGU fallback (matches the CG pipeline's
                # FASTA-fallback degradation).
                n_fallback += 1
                s = ('ACGU' * (L // 4 + 1))[:L]
            ids.append(stem)
            lengths.append(L)
            coords_list.append(arr.astype(np.float32))
            seqs.append(s)
            circs.append(circ_label)
        except Exception:
            bad += 1
            continue
        if (i + 1) % 1000 == 0:
            print(f'  [{i+1}/{len(files)}] ok={len(coords_list)} bad={bad} seq_fallback={n_fallback} ({time.time()-t1:.1f}s)')
    return ids, lengths, coords_list, seqs, circs, bad, n_fallback


def main(include_linear: bool = True):
    print('=' * 60)
    print('  Consolidate PDB RNA (cyclized + linear) to consolidated.npz')
    print('=' * 60)
    print(f'Input:  {IN_DIR} (cyclized)  +  {LINEAR_DIR} (linear, {"ON" if include_linear else "OFF"})')

    files = sorted(IN_DIR.glob('*.npy'))
    files_lin = sorted(LINEAR_DIR.glob('*.npy')) if include_linear else []
    print(f'Found {len(files)} cyclized + {len(files_lin)} linear .npy files')

    # Cache: {pdbid: {chain: seq}} — each PDB read exactly once.
    t0 = time.time()
    pdb_cache: dict = {}
    for f in files + files_lin:
        pdb_id = f.stem.rsplit('_', 1)[0]
        if pdb_id not in pdb_cache:
            pdb_cache[pdb_id] = extract_chains_seqs(pdb_id)
    print(f'Cached {len(pdb_cache)} unique PDBs in {time.time()-t0:.1f}s')

    ids, lengths, coords_list, seqs, circs, bad, n_fb = [], [], [], [], [], 0, 0
    r = _pack_dir(files, pdb_cache, circ_label=1)
    ids += r[0]; lengths += r[1]; coords_list += r[2]; seqs += r[3]; circs += r[4]
    bad += r[5]; n_fb += r[6]
    if files_lin:
        r2 = _pack_dir(files_lin, pdb_cache, circ_label=0)
        ids += r2[0]; lengths += r2[1]; coords_list += r2[2]; seqs += r2[3]; circs += r2[4]
        bad += r2[5]; n_fb += r2[6]

    n = len(coords_list)
    n_circ = sum(1 for c in circs if c == 1)
    print(f'Packed {n} samples ({n_circ} circular, {n-n_circ} linear), {bad} skipped, {n_fb} seq fallback')

    max_L = max(lengths)
    coords_padded = np.zeros((n, max_L, 3), dtype=np.float32)
    for i, c in enumerate(coords_list):
        L = lengths[i]
        coords_padded[i, :L, :] = c[:L, :]
    print(f'Coords array: [{n}, {max_L}, 3] = {coords_padded.nbytes/1e6:.1f} MB')

    np.savez_compressed(OUTPUT,
        ids=np.array(ids, dtype=object),
        lengths=np.array(lengths, dtype=np.int32),
        coords=coords_padded,
        seqs=np.array(seqs, dtype=object),
        is_circular=np.array(circs, dtype=np.int8),   # 1=环化, 0=线性
    )
    print(f'Saved: {OUTPUT} ({OUTPUT.stat().st_size/1e6:.1f} MB)')

    print('\nLength buckets (PDB pretrain):')
    for lo, hi, name in [(20, 100, 'short'), (101, 200, 'medium'), (201, 500, 'long')]:
        cnt = sum(1 for L in lengths if lo <= L <= hi)
        print(f'  {name} ({lo}-{hi}): {cnt}')
    print('\nDONE')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-linear', action='store_true', help='exclude linear RNA (cyclized only)')
    args = ap.parse_args()
    main(include_linear=not args.no_linear)
