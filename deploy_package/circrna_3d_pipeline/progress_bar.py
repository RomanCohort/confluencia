#!/usr/bin/env python3
"""
Progress bar wrapper for pipeline execution.

Usage: Already integrated in pipeline.py
"""

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: tqdm not installed. Install with: pip install tqdm")


def create_progress_bar(total, desc="Processing"):
    """Create a progress bar."""
    if HAS_TQDM:
        return tqdm(total=total, desc=desc, unit="seq",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    else:
        return None


def update_progress_bar(pbar, n=1):
    """Update progress bar."""
    if pbar:
        pbar.update(n)


def close_progress_bar(pbar):
    """Close progress bar."""
    if pbar:
        pbar.close()


def print_stage_progress(stage_name, seq_id, total_seqs):
    """Print stage progress without tqdm."""
    print(f"\n{'='*70}")
    print(f"[{stage_name}] Processing sequence {seq_id+1}/{total_seqs}")
    print(f"{'='*70}")
    print(f"  Progress: {(seq_id+1)/total_seqs*100:.1f}%")
    print(f"  Time: {time.strftime('%H:%M:%S')}")


import time