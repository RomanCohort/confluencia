#!/usr/bin/env python3
path = 'readme/TOTALREADME_katex_fixed.md'
print('Scanning', path)
with open(path, 'r', encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f, 1):
        if '─' in line or '━' in line or '—' in line:
            if '|' in line:
                print(f'Line {i}: {line.rstrip()}')
                # show repr and ords of first 60 chars
                print(' repr:', repr(line[:120]))
                print(' ords:', [ord(ch) for ch in line[:60]])
                print('')
