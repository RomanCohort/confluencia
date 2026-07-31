"""consolidate_npy_to_npz.py — Merge 82k .npy files into a single consolidated npz.

Usage:
    python consolidate_npy_to_npz.py
    # Reads: ../data/circrna_3d_all/*.npy
    # Writes: ../data/circrna_3d_all_consolidated.npz

Loads individual .npy in <2 min vs per-sample np.load in training script.
"""
import os, sys, time, json
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = os.path.normpath(os.path.join(os.path.abspath('.'), '..', '..', '..', '..', 'data', 'circrna_3d_all'))
OUTPUT = os.path.normpath(os.path.join(os.path.abspath('.'), '..', '..', '..', '..', 'data', 'circrna_3d_all_consolidated.npz'))

print('=' * 60)
print('  Consolidating .npy files to single npz')
print('=' * 60)
print(f'Input: {DATA_DIR}')
print(f'Output: {OUTPUT}')

t0 = time.time()
npy_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.npy')])
print(f'Found {len(npy_files)} .npy files')

ids = []
lengths = []
coords_list = []
meta = []

bad = 0
for i, fn in enumerate(npy_files):
    path = os.path.join(DATA_DIR, fn)
    try:
        arr = np.load(path)
        L = arr.shape[0]
        if L < 10 or np.isnan(arr).any() or np.isinf(arr).any():
            bad += 1
            continue
        ids.append(fn.split('.')[0])
        lengths.append(L)
        coords_list.append(arr.astype(np.float32))
    except Exception as e:
        bad += 1
        continue
    if (i + 1) % 10000 == 0:
        print(f'  Loaded {i+1}/{len(npy_files)} files ({time.time()-t0:.1f}s)')

n = len(coords_list)
print(f'Loaded {n} valid samples, {bad} skipped in {time.time()-t0:.1f}s')

# Build coords array [N, max_L, 3] padded
max_L = max(lengths)
print(f'Max L: {max_L}, total padding overhead: ~{max_L*3*n/1e9:.1f} GB')

coords_padded = np.zeros((n, max_L, 3), dtype=np.float32)
for i, c in enumerate(coords_list):
    L = lengths[i]
    coords_padded[i, :L, :] = c[:L, :]

print(f'Coords array ready, {time.time()-t0:.1f}s')

# Save
print('Saving consolidated npz...')
t1 = time.time()
np.savez_compressed(OUTPUT,
    ids=np.array(ids, dtype=object),
    lengths=np.array(lengths, dtype=np.int32),
    coords=coords_padded,
)
print(f'Consolidated npz saved in {time.time()-t1:.1f}s')

# Stats
total_size = os.path.getsize(OUTPUT) / 1e9
print(f'File size: {total_size:.2f} GB')
print()
print('Buckets:')
for lo, hi, name in [(151, 200, 'short'), (201, 500, 'medium'), (501, 1000, 'long'), (1001, 5000, 'xlong')]:
    cnt = np.sum((np.array(lengths) >= lo) & (np.array(lengths) <= hi))
    print(f'  {name} ({lo}-{hi}): {cnt}')

print()
print(f'Total time: {time.time()-t0:.1f}s')
print('DONE')
