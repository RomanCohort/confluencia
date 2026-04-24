#!/usr/bin/env python3
"""
Fix ASCII/box-drawing tables in Markdown files by converting them to standard Markdown tables.
Creates a backup for each modified file with suffix `.bak.asciifix`.
"""
import argparse
import os
import re
from typing import List

BOX_RE = re.compile(r"[┌┐└┘├┤┬┴─│┼╭╮╯╰]")


def is_box_line(s: str) -> bool:
    return bool(BOX_RE.search(s))


def is_table_like(s: str) -> bool:
    return bool(BOX_RE.search(s) or '|' in s)


def fix_region(region_lines: List[str]) -> List[str]:
    rows: List[List[str]] = []
    for ln in region_lines:
        s = ln.rstrip('\n').replace('│', '|')
        s = re.sub(r'[┌┐└┘├┤┬┴┼╭╮╯╰]', '', s)
        if '|' in s:
            parts = [p.strip() for p in s.split('|')]
            if parts and parts[0] == '':
                parts = parts[1:]
            if parts and parts[-1] == '':
                parts = parts[:-1]
            rows.append(parts)
        else:
            if s.strip() == '':
                continue
            rows.append([s.strip()])
    if not rows:
        return region_lines
    ncols = max(len(r) for r in rows)
    norm_rows = [r + [''] * (ncols - len(r)) for r in rows]

    def is_sep_row(r: List[str]) -> bool:
        return all(re.match(r'^:?-+:?$', cell or '') for cell in r)

    md: List[str] = []
    header = norm_rows[0]
    md.append('| ' + ' | '.join(header) + ' |\n')
    if len(norm_rows) > 1 and is_sep_row(norm_rows[1]):
        sep = [cell if re.match(r'^:?-+:?$', cell or '') else '---' for cell in norm_rows[1]]
        md.append('| ' + ' | '.join(sep) + ' |\n')
        start = 2
    else:
        md.append('| ' + ' | '.join(['---'] * ncols) + ' |\n')
        start = 1
    for r in norm_rows[start:]:
        md.append('| ' + ' | '.join(r) + ' |\n')
    return md


def process_file(path: str, write: bool = True) -> bool:
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    changed = False
    out: List[str] = []
    i = 0
    in_code = False
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith('```') or line.lstrip().startswith('~~~'):
            in_code = not in_code
            out.append(line)
            i += 1
            continue
        if in_code:
            out.append(line)
            i += 1
            continue
        if is_box_line(line):
            start = i
            while start - 1 >= 0 and lines[start - 1].strip() != '' and is_table_like(lines[start - 1]):
                start -= 1
            end = i
            while end + 1 < len(lines) and lines[end + 1].strip() != '' and is_table_like(lines[end + 1]):
                end += 1
            region = lines[start:end + 1]
            fixed = fix_region(region)
            if ''.join(fixed) != ''.join(region):
                changed = True
                out.extend(fixed)
            else:
                out.extend(region)
            i = end + 1
        else:
            out.append(line)
            i += 1
    if changed and write:
        bak = path + '.bak.asciifix'
        with open(bak, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(out)
    return changed


def find_files(root: str, exts: List[str]):
    for dirpath, dirnames, filenames in os.walk(root):
        if any(p in dirpath for p in ['.git', 'build', 'dist', '__pycache__', 'node_modules']):
            continue
        for fn in filenames:
            if any(fn.lower().endswith(e) for e in exts):
                yield os.path.join(dirpath, fn)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='.', help='workspace root')
    p.add_argument('--ext', action='append', default=['.md'], help='file extensions to process')
    p.add_argument('--yes', action='store_true', help='write changes (create backups)')
    args = p.parse_args()
    files = list(find_files(args.root, args.ext))
    total = 0
    fixed_files: List[str] = []
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8', errors='replace') as fh:
                txt = fh.read()
        except Exception:
            continue
        if not BOX_RE.search(txt):
            continue
        ok = process_file(f, write=args.yes)
        if ok:
            fixed_files.append(f)
            total += 1
            print('Fixed:', f)
    print('Done. Files fixed:', total)
    if fixed_files:
        print('\n'.join(fixed_files))


if __name__ == '__main__':
    main()
