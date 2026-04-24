#!/usr/bin/env python3
import sys
from pathlib import Path

def fix_file(p: Path):
    text = p.read_text(encoding='utf-8')
    # backup
    bak = p.with_suffix(p.suffix + '.bak')
    bak.write_text(text, encoding='utf-8')
    n_open = text.count('\\(')
    n_close = text.count('\\)')
    new = text.replace('\\(', '$').replace('\\)', '$')
    p.write_text(new, encoding='utf-8')
    print(f"Wrote {p}\nreplaced \\({n_open} times and \\){n_close} times")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: fix_math_delimiters.py <file>')
        sys.exit(2)
    path = Path(sys.argv[1])
    if not path.exists():
        print('File not found:', path)
        sys.exit(1)
    fix_file(path)
