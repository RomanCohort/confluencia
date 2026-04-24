#!/usr/bin/env python3
import runpy

mod = runpy.run_path('tools/remove_decorative_rows.py')
should_remove_line = mod['should_remove_line']

path = 'readme/TOTALREADME_katex_fixed.md'
print('Checking', path)
with open(path, 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

for i, l in enumerate(lines, 1):
    if '|' in l:
        mark = 'REMOVE' if should_remove_line(l) else 'KEEP'
        print(f'{i:4d}: {mark} | {l.rstrip()}')

print('Done')
