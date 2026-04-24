"""src.ui.model_utils -- 模型加载、缓存与工具函数。

从 frontend.py 提取。提供模型加载、缓存、特征化器缓存等函数。
"""
from __future__ import annotations

import io
import joblib
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# 类型提示前向声明
try:
    from src.epitope.predictor import EpitopeModelBundle
except Exception:
    EpitopeModelBundle = Any  # type: ignore


# ----------------------------------------------------------------------
# 模型列表
# ----------------------------------------------------------------------
def _list_local_models() -> List[str]:
    """列出本地 models/ 目录下的 .joblib 模型文件。"""
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    items = [str(p) for p in models_dir.glob("*.joblib")]
    pretrained_dir = models_dir / "pretrained"
    if pretrained_dir.exists():
        items.extend([str(p) for p in pretrained_dir.glob("*.joblib")])
    return sorted(items)


def _list_local_models_drug() -> List[str]:
    """列出药物模型 (drug_*.joblib)。"""
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    items = [str(p) for p in models_dir.glob("drug_*.joblib")]
    pretrained_dir = models_dir / "pretrained"
    if pretrained_dir.exists():
        items.extend([str(p) for p in pretrained_dir.glob("drug_*.joblib")])
        items.extend([str(p) for p in pretrained_dir.glob("drug_pretrained.joblib")])
    return sorted(items)


def _list_local_models_drug_torch() -> List[str]:
    """列出 Torch 药物模型 (.pt)。"""
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    items = [str(p) for p in models_dir.glob("drug_torch*.pt")]
    pretrained_dir = models_dir / "pretrained"
    if pretrained_dir.exists():
        items.extend([str(p) for p in pretrained_dir.glob("*.pt")])
    return sorted(items)


def _list_local_models_drug_transformer() -> List[str]:
    """列出 Transformer 药物模型。"""
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    items = [str(p) for p in models_dir.glob("drug_transformer*.pt")]
    pretrained_dir = models_dir / "pretrained"
    if pretrained_dir.exists():
        items.extend([str(p) for p in pretrained_dir.glob("*transformer*.pt")])
    return sorted(items)


def _list_local_models_docking() -> List[str]:
    """列出对接模型 (.pt 文件)。"""
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    items = [str(p) for p in models_dir.glob("docking_*.pt")]
    items.extend([str(p) for p in models_dir.glob("drug_docking*.pt")])
    pretrained_dir = models_dir / "pretrained"
    if pretrained_dir.exists():
        items.extend([str(p) for p in pretrained_dir.glob("*docking*.pt")])
    return sorted(set(items))


def _parse_hidden_sizes(text: str) -> List[int]:
    """解析隐藏层大小文本，如 '64,32,16' -> [64, 32, 16]。"""
    if not text.strip():
        return []
    parts = [x.strip() for x in text.split(",")]
    return [int(p) for p in parts if p]


# ----------------------------------------------------------------------
# 文件工具
# ----------------------------------------------------------------------
def _get_file_mtime(path: str) -> float:
    """获取文件修改时间。"""
    try:
        return float(Path(path).stat().st_mtime)
    except Exception:
        return 0.0


# ----------------------------------------------------------------------
# 缓存加载器
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False, max_entries=20)
def _cached_joblib_from_bytes(data: bytes):
    """从字节数据缓存加载 joblib 模型。"""
    return joblib.load(io.BytesIO(data))


@st.cache_resource(show_spinner=False, max_entries=20)
def _cached_joblib_from_path(path: str, mtime: float):
    """从文件路径缓存加载 joblib 模型。"""
    _ = mtime  # 用于缓存失效检测
    return joblib.load(path)


@st.cache_resource(show_spinner=False, max_entries=10)
def _cached_torch_bundle_from_bytes(data: bytes):
    """从字节数据缓存加载 Torch 模型包。"""
    from src.drug.torch_predictor import load_torch_bundle_from_bytes  # type: ignore
    return load_torch_bundle_from_bytes(data)


@st.cache_resource(show_spinner=False, max_entries=10)
def _cached_torch_bundle_from_path(path: str, mtime: float):
    """从文件路径缓存加载 Torch 模型包。"""
    _ = mtime
    from src.drug.torch_predictor import load_torch_bundle  # type: ignore
    return load_torch_bundle(path)


@st.cache_resource(show_spinner=False, max_entries=10)
def _cached_docking_bundle_from_bytes(data: bytes):
    """从字节数据缓存加载对接模型包。"""
    from src.drug.docking_cross_attention import load_docking_bundle_from_bytes
    return load_docking_bundle_from_bytes(data)


@st.cache_resource(show_spinner=False, max_entries=10)
def _cached_docking_bundle_from_path(path: str, mtime: float):
    """从文件路径缓存加载对接模型包。"""
    _ = mtime
    from src.drug.docking_cross_attention import load_docking_bundle
    return load_docking_bundle(path)


@st.cache_resource(show_spinner=False, max_entries=10)
def _cached_transformer_bundle_from_bytes(data: bytes):
    """从字节数据缓存加载 Transformer 模型包。"""
    from src.drug.transformer_predictor import load_transformer_bundle_from_bytes  # type: ignore
    return load_transformer_bundle_from_bytes(data)


@st.cache_resource(show_spinner=False, max_entries=10)
def _cached_transformer_bundle_from_path(path: str, mtime: float):
    """从文件路径缓存加载 Transformer 模型包。"""
    _ = mtime
    from src.drug.transformer_predictor import load_transformer_bundle  # type: ignore
    return load_transformer_bundle(path)


@st.cache_resource(show_spinner=False, max_entries=10)
def _cached_sequence_featurizer(version: int):
    """缓存序列特征化器。"""
    from src.epitope.featurizer import SequenceFeatures
    return SequenceFeatures(version=int(version))


@st.cache_resource(show_spinner=False, max_entries=10)
def _cached_molecule_featurizer(version: int, radius: int, n_bits: int):
    """缓存分子特征化器。"""
    from src.drug.featurizer import MoleculeFeatures  # type: ignore
    return MoleculeFeatures(version=int(version), radius=int(radius), n_bits=int(n_bits))


# ----------------------------------------------------------------------
# 统一加载接口
# ----------------------------------------------------------------------
def _load_bundle(uploaded_file, local_path: Optional[str]) -> Any:
    """加载表位模型包（上传文件或本地路径二选一）。"""
    if uploaded_file is not None:
        return _cached_joblib_from_bytes(uploaded_file.getvalue())
    if local_path:
        return _cached_joblib_from_path(local_path, _get_file_mtime(local_path))
    raise ValueError("请上传模型文件或选择本地模型")


def _load_drug_bundle(uploaded_file, local_path: Optional[str]) -> Any:
    """加载药物模型包。"""
    if uploaded_file is not None:
        return _cached_joblib_from_bytes(uploaded_file.getvalue())
    if local_path:
        return _cached_joblib_from_path(local_path, _get_file_mtime(local_path))
    raise ValueError("请上传模型文件或选择本地模型")


def _load_torch_bundle(uploaded_file, local_path: Optional[str]) -> Any:
    """加载 Torch 模型包。"""
    if uploaded_file is not None:
        return _cached_torch_bundle_from_bytes(uploaded_file.getvalue())
    if local_path:
        return _cached_torch_bundle_from_path(local_path, _get_file_mtime(local_path))
    raise ValueError("请上传模型文件或选择本地模型")


# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------
def _render_plot_images(out_dir: Path, prefix: str) -> None:
    """渲染回归诊断图（散点图、残差直方图、残差vs预测）。"""
    p1 = out_dir / f"{prefix}_scatter.png"
    p2 = out_dir / f"{prefix}_residual_hist.png"
    p3 = out_dir / f"{prefix}_residual_vs_pred.png"

    cols = st.columns(3)
    if p1.exists():
        cols[0].image(str(p1), caption=p1.name, use_container_width=True)
    if p2.exists():
        cols[1].image(str(p2), caption=p2.name, use_container_width=True)
    if p3.exists():
        cols[2].image(str(p3), caption=p3.name, use_container_width=True)


def _bundle_input_vector(bundle: Any, sequence: str, env: Dict[str, float]) -> np.ndarray:
    """根据模型包构建输入向量。"""
    feat_v = int(getattr(bundle, "featurizer_version", 1) or 1)
    featurizer = _cached_sequence_featurizer(feat_v)
    seq_x = featurizer.transform_one(sequence).astype(np.float32)

    env_vec = []
    for c in bundle.env_cols:
        if c in env:
            env_vec.append(float(env[c]))
        else:
            env_vec.append(float(bundle.env_medians.get(c, 0.0)))

    env_x = np.asarray(env_vec, dtype=np.float32)
    x = np.concatenate([seq_x, env_x], axis=0)
    return x.astype(np.float32)


def _make_x_only_epitope(
    df: pd.DataFrame,
    *,
    sequence_col: str,
    env_cols: List[str],
    env_medians: Dict[str, float],
    featurizer_version: int,
) -> np.ndarray:
    """为表位预测构建特征矩阵（仅 X，无目标）。"""
    from src.epitope.featurizer import SequenceFeatures
    featurizer = SequenceFeatures(version=int(featurizer_version))
    seq_x = featurizer.transform_many(df[sequence_col].astype(str).tolist())

    if env_cols:
        env_df = df[env_cols].copy()
        for c in env_cols:
            env_df[c] = pd.to_numeric(env_df[c], errors="coerce").astype(float)
            env_df[c] = env_df[c].fillna(float(env_medians.get(c, float(env_df[c].median()))))
        env_x = env_df.to_numpy(dtype=np.float32)
    else:
        env_x = np.zeros((len(df), 0), dtype=np.float32)

    return np.concatenate([seq_x, env_x], axis=1).astype(np.float32)