#!/usr/bin/env python3
"""Inspect table regions in a Markdown file and print separator/header candidates.
"""
import re
from typing import List

SEP_RE = re.compile(r'^\s*\|?(?:\s*:?-+:?\s*\|)+\s*$')


def split_cells(ln: str) -> List[str]:
    parts = [p.strip() for p in ln.split('|')]
    if parts and parts[0] == '':
        parts = parts[1:]
    if parts and parts[-1] == '':
        parts = parts[:-1]
    return parts


def inspect(path: str):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    in_fence = False
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        if ln.lstrip().startswith('```') or ln.lstrip().startswith('~~~'):
            in_fence = not in_fence
            i += 1
            continue
        if in_fence:
            i += 1
            continue
        if '|' in ln:
            start = i
            j = i
            while j < n and lines[j].strip() != '' and '|' in lines[j]:
                j += 1
            region = lines[start:j]
            print('Region lines', start+1, '-', j)
            sep_indices = [idx for idx, r in enumerate(region) if SEP_RE.match(r)]
            print('  sep_indices:', sep_indices)
            for sep in sep_indices:
                if sep == 0:
                    print('   sep at 0 -> no header')
                    continue
                header_ln = region[sep - 1]
                cells = split_cells(header_ln)
                non_empty = sum(1 for c in cells if c)
                print(f'   candidate sep={sep}, header_cols={len(cells)}, non_empty={non_empty}, header="{header_ln.strip()}"')
            print('  region preview:')
            for k, r in enumerate(region[:10]):
                print(f'    {start+k+1}: {r.rstrip()}')
            print('')
            i = j
        else:
            i += 1


if __name__ == '__main__':
    inspect('readme/TOTALREADME_katex_fixed.md')
