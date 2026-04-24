#!/usr/bin/env python3
"""
Convert a PNG into one or more ICO files (multi-size).

Usage:
    python tools/make_icons.py src.png dst1.ico [dst2.ico ...]

This script uses Pillow. Install it with:
    python -m pip install pillow
"""
from PIL import Image
import sys
import os

def convert(src, dst):
    img = Image.open(src).convert("RGBA")
    sizes = [(256,256),(128,128),(64,64),(48,48),(32,32),(16,16)]
    img.save(dst, format='ICO', sizes=sizes)
    print(f"Saved {dst}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python tools/make_icons.py src.png dst1.ico [dst2.ico ...]")
        sys.exit(2)
    src = sys.argv[1]
    if not os.path.exists(src):
        print(f"Source not found: {src}")
        sys.exit(1)
    for dst in sys.argv[2:]:
        ddir = os.path.dirname(dst)
        if ddir and not os.path.exists(ddir):
            os.makedirs(ddir, exist_ok=True)
        convert(src, dst)

if __name__ == '__main__':
    main()
