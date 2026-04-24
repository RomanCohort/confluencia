#!/usr/bin/env python3
"""Scan Markdown files for table rows that don't match header column counts.
Prints file path and offending line numbers for manual inspection.
"""
import os
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


def find_issues_in_file(path: str):
    issues = []
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
            # collect contiguous pipe lines
            start = i
            j = i
            while j < n and lines[j].strip() != '' and '|' in lines[j]:
                j += 1
            region = lines[start:j]
            # find separator
            sep_idx = None
            for idx, r in enumerate(region):
                if SEP_RE.match(r):
                    sep_idx = idx
                    break
            if sep_idx is not None and sep_idx > 0:
                header_cells = split_cells(region[sep_idx - 1])
                col_count = len(header_cells)
                # check rows
                for k in range(sep_idx + 1, len(region)):
                    cells = split_cells(region[k])
                    if len(cells) != col_count:
                        issues.append((start + k + 1, col_count, len(cells), region[k].rstrip('\n')))
            i = j
        else:
            i += 1
    return issues


def find_md_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        if any(p in dirpath for p in ['.git', 'build', 'dist', '__pycache__', 'node_modules']):
            continue
        for fn in filenames:
            if fn.lower().endswith('.md'):
                yield os.path.join(dirpath, fn)


def main():
    root = '.'
    any_issues = False
    for f in find_md_files(root):
        issues = find_issues_in_file(f)
        if issues:
            any_issues = True
            print('File:', f)
            for line_no, expected, found, text in issues[:20]:
                print(f'  Line {line_no}: expected {expected} cols but found {found}: {text}')
            print('')
    if not any_issues:
        print('No table column-count issues found.')


if __name__ == '__main__':
    main()
