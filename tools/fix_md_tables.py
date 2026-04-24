#!/usr/bin/env python3
"""
Normalize Markdown tables in .md files: trim stray empty trailing columns,
ensure header separator rows are correct, and pad/trim rows to a consistent
column count. Creates backups with suffix `.bak.mdtable` when writing.

This script is conservative:
- Only processes contiguous blocks of lines containing '|' that include a
  Markdown separator row (e.g. '| --- | :---: |') -- avoids directory trees.
- Skips regions inside fenced code blocks.
"""
import argparse
import os
import re
from typing import List, Tuple


SEP_RE = re.compile(r'^\s*\|?(?:\s*:?-+:?\s*\|)+\s*$')


def is_fence(line: str) -> bool:
    return line.lstrip().startswith('```') or line.lstrip().startswith('~~~')


def normalize_table_region(region: List[str]) -> Tuple[List[str], bool]:
    """Normalize a contiguous region of pipe-containing lines if it has a separator row.
    Returns (new_lines, changed_flag).
    """
    # find all separator row indices and use the last separator as the data anchor
    sep_indices = [idx for idx, ln in enumerate(region) if SEP_RE.match(ln)]
    if not sep_indices:
        return region, False
    last_sep = max(sep_indices)
    if last_sep == 0:
        return region, False
    # header is the line immediately before the last separator
    header_line = region[last_sep - 1]
    header_cells = [p.strip() for p in header_line.split('|')]
    if header_cells and header_cells[0] == '':
        header_cells = header_cells[1:]
    if header_cells and header_cells[-1] == '':
        header_cells = header_cells[:-1]
    if not header_cells:
        return region, False

    def split_cells(ln: str) -> List[str]:
        parts = [p.strip() for p in ln.split('|')]
        if parts and parts[0] == '':
            parts = parts[1:]
        if parts and parts[-1] == '':
            parts = parts[:-1]
        return parts

    header_cells = split_cells(header_line)
    if not header_cells:
        return region, False
    col_count = len(header_cells)

    # Build normalized header and separator
    norm_header = '| ' + ' | '.join(header_cells) + ' |\n'
    norm_sep = '| ' + ' | '.join(['---'] * col_count) + ' |\n'

    # Data rows start after the last separator in the region (skip repeated header/separator blocks)
    data_rows = region[last_sep + 1:]
    norm_data = []
    for row in data_rows:
        cells = split_cells(row)
        # if there are more cells than header, merge extras into the last column
        if len(cells) > col_count:
            merged = ' '.join(cells[col_count - 1:]).strip()
            cells = cells[:col_count - 1] + [merged]
        elif len(cells) < col_count:
            cells = cells + [''] * (col_count - len(cells))
        norm_data.append('| ' + ' | '.join(cells) + ' |\n')

    new_region = [norm_header, norm_sep] + norm_data
    changed = ''.join(new_region) != ''.join(region)
    return new_region, changed


def process_file(path: str, write: bool = True) -> bool:
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    out: List[str] = []
    i = 0
    in_fence = False
    changed_any = False
    n = len(lines)
    while i < n:
        ln = lines[i]
        if is_fence(ln):
            in_fence = not in_fence
            out.append(ln)
            i += 1
            continue
        if in_fence:
            out.append(ln)
            i += 1
            continue

        if '|' in ln:
            # collect contiguous non-blank lines that contain '|'
            start = i
            j = i
            while j < n and lines[j].strip() != '' and '|' in lines[j]:
                j += 1
            region = lines[start:j]
            new_region, changed = normalize_table_region(region)
            out.extend(new_region)
            if changed:
                changed_any = True
            i = j
        else:
            out.append(ln)
            i += 1

    if changed_any and write:
        bak = path + '.bak.mdtable'
        with open(bak, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(out)
    return changed_any


def find_md_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        # skip some common large/third-party folders
        if any(p in dirpath for p in ['.git', 'build', 'dist', '__pycache__', 'node_modules']):
            continue
        for fn in filenames:
            if fn.lower().endswith('.md'):
                yield os.path.join(dirpath, fn)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='.', help='workspace root')
    p.add_argument('--yes', action='store_true', help='write changes (create backups)')
    args = p.parse_args()

    fixed = []
    for f in find_md_files(args.root):
        try:
            with open(f, 'r', encoding='utf-8', errors='replace') as fh:
                txt = fh.read()
        except Exception:
            continue
        # quick heuristic: process only if file contains a separator-like row
        if '|' not in txt:
            continue
        if not re.search(r':?-+:?\s*\|', txt) and '--- |' not in txt:
            # no obvious separator pattern
            continue
        ok = process_file(f, write=args.yes)
        if ok:
            fixed.append(f)
            print('Fixed:', f)

    print('Done. Files fixed:', len(fixed))
    for f in fixed:
        print(f)


if __name__ == '__main__':
    main()
