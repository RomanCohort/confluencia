#!/usr/bin/env python3
"""
trRosettaRNA2包装脚本 - 从顶层调用trRNA2/predict.py

这个脚本将参数转发到真正的predict.py在trRNA2目录
"""

import os
import sys
import subprocess
from pathlib import Path

# 真正的predict.py位置
TRRNA2_DIR = Path(__file__).parent / 'trRNA2'
REAL_PREDICT = TRRNA2_DIR / 'predict.py'

if not REAL_PREDICT.exists():
    print(f"ERROR: predict.py not found at {REAL_PREDICT}")
    print(f"trRosettaRNA2 directory structure:")
    for item in Path(__file__).parent.iterdir():
        print(f"  {item}")
    sys.exit(1)

# 转发所有参数到真正的predict.py
cmd = [sys.executable, str(REAL_PREDICT)] + sys.argv[1:]

print(f"Calling: {cmd}")
result = subprocess.run(cmd, cwd=str(TRRNA2_DIR))
sys.exit(result.returncode)