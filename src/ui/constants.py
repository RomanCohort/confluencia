"""src.ui.constants -- 全局常量与项目路径。

从 frontend.py 提取，保持单一来源。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

# ----------------------------------------------------------------------
# _PROJECT_ROOT: 项目根目录 (IGEM集成方案/)
# ----------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()          # ui/constants.py
_PROJECT_ROOT = _THIS_FILE.parents[1]          # IGEM集成方案/

# 确保项目根目录在 sys.path，以便 `from src...` 导入正常工作
for _p in (_PROJECT_ROOT, _PROJECT_ROOT.parent):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

# ----------------------------------------------------------------------
# 公开常量
# ----------------------------------------------------------------------
APP_TITLE = "confluencia:IGEM-FBH 虚拟筛选前端"
IGEM_FBH_URL = os.getenv("IGEM_FBH_URL", "https://igem.org")

# 支持的文件上传类型（导出为公开名称）
UPLOAD_TYPES: List[str] = [
    "csv", "tsv", "txt", "xlsx", "xls",
    "json", "jsonl", "ndjson",
    "parquet", "pq",
    "feather", "arrow",
]