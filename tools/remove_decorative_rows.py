#!/usr/bin/env python3
"""
Remove decorative rows from Markdown table regions.

Behavior:
- Only considers lines containing '|' (table-like rows) and not inside fenced code blocks.
- Keeps legitimate Markdown separator rows like '| --- | :---: |'.
- Removes rows where every non-empty cell contains no alphanumeric characters
  (i.e., composed of decorative dashes/box-drawing/arrow symbols),
  merging extra cells is NOT performed here — this script only deletes clearly decorative rows.

Creates a backup for each modified file with suffix '.bak.decor'.
"""
import argparse
import os
import re
from typing import List

SEP_RE = re.compile(r"^\s*\|?(?:\s*:?-+:?\s*\|)+\s*$")


def split_cells(ln: str) -> List[str]:
    parts = [p.strip() for p in ln.split('|')]
    if parts and parts[0] == '':
        parts = parts[1:]
    if parts and parts[-1] == '':
        parts = parts[:-1]
    return parts


def is_decorative_cell(cell: str) -> bool:
    s = cell.strip()
    if s == '':
        return False
    # If contains any alphanumeric (including CJK/letters/digits) it's useful
    if any(ch.isalnum() for ch in s):
        return False
    # If contains letters like punctuation-only (arrows, box-drawing, repeated dashes), treat as decorative
    return True


def should_remove_line(line: str) -> bool:
    # keep separator rows
    if SEP_RE.match(line):
        return False
    if '|' not in line:
        return False
    cells = split_cells(line)
    if not cells:
        return False
    # find non-empty cells
    non_empty = [c for c in cells if c.strip() != '']
    if not non_empty:
        return False
    # if any non-empty cell contains alnum -> keep
    for c in non_empty:
        if not is_decorative_cell(c):
            return False
    # All non-empty cells are decorative -> remove
    return True


def process_file(path: str, write: bool = True) -> int:
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    out: List[str] = []
    i = 0
    n = len(lines)
    in_fence = False
    removed = 0
    while i < n:
        ln = lines[i]
        if ln.lstrip().startswith('```') or ln.lstrip().startswith('~~~'):
            in_fence = not in_fence
            out.append(ln)
            i += 1
            continue
        if in_fence:
            out.append(ln)
            i += 1
            continue
        if should_remove_line(ln):
            removed += 1
            i += 1
            continue
        out.append(ln)
        i += 1

    if removed and write:
        bak = path + '.bak.decor'
        with open(bak, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(out)
    return removed


def find_md_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        if any(p in dirpath for p in ['.git', 'build', 'dist', '__pycache__', 'node_modules']):
            continue
        for fn in filenames:
            if fn.lower().endswith('.md'):
                yield os.path.join(dirpath, fn)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='.', help='workspace root')
    p.add_argument('--yes', action='store_true', help='write changes (create backups)')
    args = p.parse_args()

    total_removed = 0
    files_changed = []
    for f in find_md_files(args.root):
        try:
            with open(f, 'r', encoding='utf-8', errors='replace') as fh:
                txt = fh.read()
        except Exception:
            continue
        # quick heuristic: skip files without likely decorative glyphs
        if not re.search(r'[─━—─┄┈━─]+', txt):
            continue
        rem = process_file(f, write=args.yes)
        if rem:
            files_changed.append((f, rem))
            total_removed += rem
            print(f'Removed {rem} decorative rows: {f}')

    print(f'Done. Total decorative rows removed: {total_removed}; files changed: {len(files_changed)}')
    for f, r in files_changed:
        print(f, r)


if __name__ == '__main__':
    main()
