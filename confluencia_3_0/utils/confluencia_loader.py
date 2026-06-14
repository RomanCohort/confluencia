"""Confluencia 懒加载工具"""
import importlib.util
import os
from typing import Any, Optional


def lazy_import_module(module_path: str, module_name: str) -> Optional[Any]:
    """懒加载Python模块

    Args:
        module_path: 模块文件路径
        module_name: 模块名

    Returns:
        模块对象，失败返回None
    """
    if not os.path.exists(module_path):
        return None

    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    except Exception:
        pass

    return None


def find_confluencia(base_path: str) -> Optional[str]:
    """查找Confluencia项目路径

    Args:
        base_path: 搜索根路径

    Returns:
        Confluencia路径，未找到返回None
    """
    candidates = [
        os.path.join(base_path, "confluencia-2.0-drug"),
        os.path.join(base_path, "confluencia_joint"),
        os.path.join(base_path, "confluencia-2.0"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None