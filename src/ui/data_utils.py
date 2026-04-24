"""src.ui.data_utils -- 数据加载、缓存与预处理工具。

从 frontend.py 提取。提供通用的数据加载、缓存和预处理函数。
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------
def _strip_leading_comments(text: str) -> Tuple[str, List[str]]:
    """剥离以 #, //, ;, % 开头的注释行，返回 (cleaned_text, removed_lines)。"""
    lines = text.splitlines(keepends=True)
    removed: List[str] = []
    kept: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and (stripped.startswith("#") or stripped.startswith("//") or stripped.startswith(";") or stripped.startswith("%")):
            removed.append(line)
        else:
            kept.append(line)
    return "".join(kept), removed


# ----------------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------------
def _load_table_from_bytes(data: bytes, name: str) -> pd.DataFrame:
    """根据文件扩展名加载表格数据。"""
    ext = Path(name).suffix.lower().lstrip(".")
    buf = io.BytesIO(data)

    if ext in {"csv", "txt"}:
        text = buf.read().decode("utf-8", errors="ignore")
        cleaned, _ = _strip_leading_comments(text)
        return pd.read_csv(io.StringIO(cleaned), sep=None, engine="python")
    if ext == "tsv":
        text = buf.read().decode("utf-8", errors="ignore")
        cleaned, _ = _strip_leading_comments(text)
        return pd.read_csv(io.StringIO(cleaned), sep="\t")
    if ext in {"xlsx", "xls"}:
        return pd.read_excel(buf)
    if ext == "json":
        return pd.read_json(buf)
    if ext in {"jsonl", "ndjson"}:
        return pd.read_json(buf, lines=True)
    if ext in {"parquet", "pq"}:
        return pd.read_parquet(buf)
    if ext in {"feather", "arrow"}:
        return pd.read_feather(buf)
    raise ValueError(f"不支持的文件类型: .{ext}")


@st.cache_data(show_spinner=False, max_entries=50)
def _cached_load_table_from_bytes(data: bytes, name: str) -> pd.DataFrame:
    """缓存版数据加载（按文件内容缓存）。"""
    return _load_table_from_bytes(data, name)


@st.cache_data(show_spinner=False, max_entries=20)
def _cached_concat_tables(files: List[Tuple[str, bytes]], add_source_col: str) -> pd.DataFrame:
    """加载并合并多个表格文件。"""
    frames = []
    for name, data in files:
        df = _cached_load_table_from_bytes(data, name)
        if add_source_col:
            df[add_source_col] = name
        frames.append(df)
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, axis=0, ignore_index=True)


def _load_tables_from_uploads(
    uploads: List[Any], *, add_source_col: str = "_source"
) -> pd.DataFrame:
    """从上传文件列表加载并合并表格。"""
    if not uploads:
        raise ValueError("未选择文件")
    files = []
    for up in uploads:
        name = getattr(up, "name", "uploaded")
        files.append((str(name), up.getvalue()))
    return _cached_concat_tables(files, str(add_source_col))


# ----------------------------------------------------------------------
# 列推断
# ----------------------------------------------------------------------
def _guess_numeric_columns(df: pd.DataFrame, *, max_cols: int = 20) -> List[str]:
    """推断数值型列（80%以上可转数值的列）。"""
    cols = []
    for c in df.columns:
        ser = pd.to_numeric(df[c], errors="coerce")
        ratio = float(ser.notna().mean()) if len(ser) else 0.0
        if ratio >= 0.8:
            cols.append(str(c))
        if len(cols) >= max_cols:
            break
    return cols


def _guess_categorical_columns(df: pd.DataFrame, *, exclude: List[str], max_cols: int = 20) -> List[str]:
    """推断分类型列（唯一值数量在 2-30 之间且不在排除列表中）。"""
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        ser = df[c]
        n_unique = int(ser.nunique(dropna=True)) if len(ser) else 0
        if 1 < n_unique <= 30:
            cols.append(str(c))
        if len(cols) >= max_cols:
            break
    return cols


# ----------------------------------------------------------------------
# 预处理
# ----------------------------------------------------------------------
def _apply_preprocess(
    df: pd.DataFrame,
    *,
    trim_cols: bool,
    lower_cols: bool,
    strip_cells: bool,
    drop_empty_rows: bool,
    drop_empty_cols: bool,
    drop_duplicates: bool,
    numeric_cols: List[str],
    categorical_cols: List[str],
    fill_numeric: str,
    fill_categorical: str,
) -> pd.DataFrame:
    """应用数据预处理操作。"""
    out = df.copy()

    if trim_cols:
        out.columns = [str(c).strip() for c in out.columns]
    if lower_cols:
        out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]

    if strip_cells:
        for c in out.columns:
            if out[c].dtype == object:
                out[c] = out[c].astype("string").str.strip()

    if drop_empty_rows:
        out = out.dropna(axis=0, how="all")
    if drop_empty_cols:
        out = out.dropna(axis=1, how="all")
    if drop_duplicates:
        out = out.drop_duplicates()

    # 强制转换数值列
    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if fill_numeric != "不填充":
        for c in numeric_cols:
            if c not in out.columns:
                continue
            if fill_numeric == "中位数":
                out[c] = out[c].fillna(out[c].median())
            elif fill_numeric == "均值":
                out[c] = out[c].fillna(out[c].mean())
            elif fill_numeric == "0":
                out[c] = out[c].fillna(0)

    if fill_categorical != "不填充":
        for c in categorical_cols:
            if c not in out.columns:
                continue
            if fill_categorical == "众数":
                mode = out[c].mode(dropna=True)
                if len(mode) > 0:
                    out[c] = out[c].fillna(mode.iloc[0])
            elif fill_categorical == "unknown":
                out[c] = out[c].fillna("unknown")

    return out


def _render_preprocess_panel(df: pd.DataFrame, key_prefix: str) -> Dict[str, Any]:
    """渲染数据预处理面板，返回配置字典。"""
    with st.expander("导入预处理", expanded=False):
        trim_cols = st.checkbox("列名去空格", value=True, key=f"{key_prefix}_trim_cols")
        lower_cols = st.checkbox("列名小写并用下划线", value=False, key=f"{key_prefix}_lower_cols")
        strip_cells = st.checkbox("字符串去空格", value=True, key=f"{key_prefix}_strip_cells")
        drop_empty_rows = st.checkbox("删除全空行", value=True, key=f"{key_prefix}_drop_empty_rows")
        drop_empty_cols = st.checkbox("删除全空列", value=True, key=f"{key_prefix}_drop_empty_cols")
        drop_duplicates = st.checkbox("删除重复行", value=False, key=f"{key_prefix}_drop_duplicates")

        auto_numeric = _guess_numeric_columns(df)
        numeric_cols = st.multiselect(
            "数值列（用于类型转换/缺失填充）",
            options=list(df.columns),
            default=[c for c in auto_numeric if c in df.columns],
            key=f"{key_prefix}_numeric_cols",
        )
        auto_cat = _guess_categorical_columns(df, exclude=list(numeric_cols))
        categorical_cols = st.multiselect(
            "分类型列（用于缺失填充）",
            options=[c for c in df.columns if c not in set(numeric_cols)],
            default=[c for c in auto_cat if c in df.columns],
            key=f"{key_prefix}_categorical_cols",
        )
        fill_numeric = st.selectbox(
            "数值缺失填充",
            options=["不填充", "中位数", "均值", "0"],
            index=1,
            key=f"{key_prefix}_fill_numeric",
        )
        fill_categorical = st.selectbox(
            "分类型缺失填充",
            options=["不填充", "众数", "unknown"],
            index=0,
            key=f"{key_prefix}_fill_categorical",
        )

    return {
        "trim_cols": trim_cols,
        "lower_cols": lower_cols,
        "strip_cells": strip_cells,
        "drop_empty_rows": drop_empty_rows,
        "drop_empty_cols": drop_empty_cols,
        "drop_duplicates": drop_duplicates,
        "numeric_cols": list(numeric_cols),
        "categorical_cols": list(categorical_cols),
        "fill_numeric": fill_numeric,
        "fill_categorical": fill_categorical,
    }


def _parse_kv_lines(text: str) -> Dict[str, float]:
    """解析 key=value 或 key: value 格式的参数文本。"""
    params: Dict[str, float] = {}
    if not text:
        return params
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
        elif ":" in line:
            k, v = line.split(":", 1)
        else:
            raise ValueError(f"参数行格式错误: '{line}'，应为 key=value 或 key: value")
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"参数行 key 为空: '{line}'")
        try:
            params[k] = float(v)
        except Exception:
            raise ValueError(f"参数值需要是数值: '{line}'")
    return params