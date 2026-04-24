from __future__ import annotations

import hashlib
import os
import json
import joblib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from .mamba3 import Mamba3Config, Mamba3LiteEncoder

# Import canonical AA constants from shared
from confluencia_shared.features import (
    AA_ORDER,
    AA_TO_IDX,
    HYDROPHOBIC,
    POLAR,
    ACIDIC,
    BASIC,
)

# Lazy ESM-2 import
_esm2_encoder = None
_pca_transformer = None


def _get_esm2_encoder(model_size: str = "650M", cache_dir: str = ""):
    """懒加载 ESM-2 编码器 (单例)"""
    global _esm2_encoder
    if _esm2_encoder is None:
        from .esm2_encoder import ESM2Encoder

        _esm2_encoder = ESM2Encoder(
            model_size=model_size,
            cache_dir=cache_dir if cache_dir else None,
        )
    return _esm2_encoder


def _get_pca_path(cache_dir: str, model_size: str, pca_dim: int) -> Optional[str]:
    """PCA 模型缓存路径"""
    if not cache_dir:
        return None
    from pathlib import Path
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return str(d / f"esm2_pca_{model_size}_to{pca_dim}.joblib")


def _apply_pca(esm2_features: np.ndarray, pca_dim: int, cache_dir: str, model_size: str, fit: bool = True) -> np.ndarray:
    """对 ESM-2 特征进行 PCA 降维"""
    global _pca_transformer
    from sklearn.decomposition import IncrementalPCA

    pca_path = _get_pca_path(cache_dir, model_size, pca_dim)

    if _pca_transformer is None:
        # 尝试从缓存加载
        if pca_path and os.path.exists(pca_path) and not fit:
            _pca_transformer = joblib.load(pca_path)
            print(f"[PCA] 加载缓存 PCA 模型: {pca_path}")
        elif fit:
            # 拟合新 PCA
            actual_dim = min(pca_dim, esm2_features.shape[1], esm2_features.shape[0])
            print(f"[PCA] 拟合 PCA: {esm2_features.shape[1]} -> {actual_dim} 维")
            _pca_transformer = IncrementalPCA(n_components=actual_dim, batch_size=4096)
            _pca_transformer.fit(esm2_features)
            explained = _pca_transformer.explained_variance_ratio_.sum()
            print(f"[PCA] 解释方差比: {explained:.4f}")
            # 缓存 PCA 模型
            if pca_path:
                joblib.dump(_pca_transformer, pca_path)
                print(f"[PCA] 已缓存: {pca_path}")
        else:
            raise RuntimeError("PCA not fitted and no cache found")

    result = _pca_transformer.transform(esm2_features)
    print(f"[PCA] 降维: {esm2_features.shape} -> {result.shape}")
    return result.astype(np.float32)

KMER_HASH_SALT = b"cf2-kmer-v1"
KMER_HASH_VERSION = "blake2b-64-person-cf2-kmer-v1"
FEATURE_SCHEMA_VERSION = "epitope-feature-schema-v2"


@dataclass(frozen=True)
class FeatureSpec:
    mamba: Mamba3Config = Mamba3Config()
    kmer_hash_dim: int = 64
    env_candidates: Tuple[str, ...] = ("dose", "freq", "treatment_time", "circ_expr", "ifn_score")
    use_esm2: bool = False
    esm2_model_size: str = "650M"
    esm2_cache_dir: str = ""
    esm2_pca_dim: int = 0  # PCA降维维度，0表示不做PCA
    use_mhc: bool = False
    mhc_allele_col: str = "mhc_allele"


# ESM-2 嵌入维度映射 (用于 feature names 生成)
_ESM2_EMBED_DIMS = {
    "8M": 320,
    "35M": 480,
    "150M": 640,
    "650M": 1280,
}


def feature_schema_id(spec: FeatureSpec | None = None) -> str:
    spec = spec or FeatureSpec()
    env = ",".join(spec.env_candidates)
    return (
        f"{FEATURE_SCHEMA_VERSION};"
        f"mamba_d={spec.mamba.d_model};"
        f"kmer_dim={spec.kmer_hash_dim};"
        f"env={env};"
        f"kmer_hash={KMER_HASH_VERSION};"
        f"esm2={spec.use_esm2}:{spec.esm2_model_size};"
        f"mhc={spec.use_mhc}"
    )


def _clean_seq(seq: str) -> str:
    return str(seq or "").strip().upper().replace(" ", "")


def _stable_hash_u64(text: str) -> int:
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=8, person=KMER_HASH_SALT)
    return int.from_bytes(h.digest(), byteorder="little", signed=False)


def _hash_kmer(seq: str, k: int, dim: int) -> np.ndarray:
    out = np.zeros((dim,), dtype=np.float32)
    s = _clean_seq(seq)
    if len(s) < k:
        return out
    for i in range(len(s) - k + 1):
        token = s[i : i + k]
        idx = _stable_hash_u64(f"{token}|{i % 13}|{k}") % dim
        out[idx] += 1.0
    n = np.linalg.norm(out)
    if n > 0:
        out = out / n
    return out


def _biochem_stats(seq: str) -> np.ndarray:
    s = _clean_seq(seq)
    l = float(len(s))
    if l <= 0:
        return np.zeros((16,), dtype=np.float32)

    aa_counts = np.zeros((len(AA_ORDER),), dtype=np.float32)
    hydro = 0.0
    polar = 0.0
    acidic = 0.0
    basic = 0.0

    n_terminus = s[: max(1, len(s) // 3)]
    c_terminus = s[-max(1, len(s) // 3) :]

    for ch in s:
        idx = AA_TO_IDX.get(ch)
        if idx is not None:
            aa_counts[idx] += 1.0
        if ch in HYDROPHOBIC:
            hydro += 1.0
        if ch in POLAR:
            polar += 1.0
        if ch in ACIDIC:
            acidic += 1.0
        if ch in BASIC:
            basic += 1.0

    aa_entropy = 0.0
    p = aa_counts / max(l, 1.0)
    for v in p:
        if v > 0:
            aa_entropy -= float(v * np.log(v + 1e-8))

    n_hydro = sum(1 for ch in n_terminus if ch in HYDROPHOBIC) / float(len(n_terminus))
    c_hydro = sum(1 for ch in c_terminus if ch in HYDROPHOBIC) / float(len(c_terminus))

    return np.array(
        [
            l,
            hydro / l,
            polar / l,
            acidic / l,
            basic / l,
            aa_entropy,
            n_hydro,
            c_hydro,
            float(s.count("P")) / l,
            float(s.count("G")) / l,
            float(s.count("W") + s.count("F") + s.count("Y")) / l,
            float(s.count("R") + s.count("K")) / l,
            float(s.count("D") + s.count("E")) / l,
            float(s.count("N") + s.count("Q")) / l,
            float(len(set(s))) / float(len(AA_ORDER)),
            float(sum(1 for ch in s if ch not in AA_TO_IDX)) / l,
        ],
        dtype=np.float32,
    )


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "epitope_seq" not in out.columns:
        out["epitope_seq"] = ""
    for c in ["dose", "freq", "treatment_time", "circ_expr", "ifn_score"]:
        if c not in out.columns:
            out[c] = 0.0
    return out


def build_feature_matrix(df: pd.DataFrame, spec: FeatureSpec | None = None) -> Tuple[np.ndarray, List[str], List[str]]:
    """构建特征矩阵，支持 ESM-2 和 MHC 增强特征

    特征顺序:
        1. Mamba3Lite 编码 (~168维)
        2. ESM-2 嵌入 (320/640/1280维，当 use_esm2=True)
        3. MHC 特征 (979维，当 use_mhc=True)
        4. k-mer hash (64*2=128维)
        5. 生化统计 (16维)
        6. 环境变量 (0-5维)

    总维度 (use_esm2=True, use_mhc=True, 650M): ~2604维
    总维度 (use_mhc=True): ~1272维
    总维度 (use_esm2=False, use_mhc=False): ~317维
    """
    spec = spec or FeatureSpec()
    encoder = Mamba3LiteEncoder(spec.mamba)

    work = ensure_columns(df)
    env_cols = [c for c in spec.env_candidates if c in work.columns]

    # Vectorize: extract sequences and env values outside the loop
    sequences = work["epitope_seq"].astype(str).tolist()
    if env_cols:
        env_matrix = work[env_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    else:
        env_matrix = np.zeros((len(work), 0), dtype=np.float32)

    # ESM-2 特征 (可选)
    esm2_features = None
    esm2_dim = 0
    if spec.use_esm2:
        try:
            esm2_encoder = _get_esm2_encoder(
                model_size=spec.esm2_model_size,
                cache_dir=spec.esm2_cache_dir,
            )
            esm2_features = esm2_encoder.encode(sequences)
            esm2_dim = esm2_features.shape[1]

            # PCA 降维 (如果配置)
            if spec.esm2_pca_dim > 0 and esm2_dim > spec.esm2_pca_dim:
                esm2_features = _apply_pca(
                    esm2_features,
                    pca_dim=spec.esm2_pca_dim,
                    cache_dir=spec.esm2_cache_dir or "",
                    model_size=spec.esm2_model_size,
                    fit=True,  # 首次调用时拟合
                )
                esm2_dim = esm2_features.shape[1]

        except Exception as e:
            print(f"[Warning] ESM-2 编码失败，跳过: {e}")
            spec = FeatureSpec(
                mamba=spec.mamba,
                kmer_hash_dim=spec.kmer_hash_dim,
                env_candidates=spec.env_candidates,
                use_esm2=False,
                esm2_pca_dim=spec.esm2_pca_dim,
                use_mhc=spec.use_mhc,
                mhc_allele_col=spec.mhc_allele_col,
            )

    # MHC 特征 (可选)
    mhc_features = None
    mhc_dim = 0
    if spec.use_mhc:
        try:
            from .mhc_features import MHCFeatureEncoder
            mhc_encoder = MHCFeatureEncoder()
            mhc_dim = mhc_encoder.feature_dim  # 979

            # 获取 alleles
            if spec.mhc_allele_col in work.columns:
                alleles = work[spec.mhc_allele_col].fillna("HLA-A*02:01").astype(str).tolist()
            else:
                # 无 allele 信息时使用默认值
                alleles = ["HLA-A*02:01"] * len(work)

            mhc_features = mhc_encoder.encode_batch(sequences, alleles)
        except Exception as e:
            print(f"[Warning] MHC 特征编码失败，跳过: {e}")
            spec = FeatureSpec(
                mamba=spec.mamba,
                kmer_hash_dim=spec.kmer_hash_dim,
                env_candidates=spec.env_candidates,
                use_esm2=spec.use_esm2,
                esm2_model_size=spec.esm2_model_size,
                esm2_cache_dir=spec.esm2_cache_dir,
                esm2_pca_dim=spec.esm2_pca_dim,
                use_mhc=False,
            )

    xs: List[np.ndarray] = []
    for i, seq in enumerate(sequences):
        m = encoder.encode(seq)
        k2 = _hash_kmer(seq, k=2, dim=spec.kmer_hash_dim)
        k3 = _hash_kmer(seq, k=3, dim=spec.kmer_hash_dim)
        bio = _biochem_stats(seq)
        env = env_matrix[i]

        # 拼接特征
        parts = [m["summary"], m["local_pool"], m["meso_pool"], m["global_pool"]]
        if esm2_features is not None:
            parts.append(esm2_features[i])
        if mhc_features is not None:
            parts.append(mhc_features[i])
        parts.extend([k2, k3, bio, env])

        x = np.concatenate(parts, axis=0)
        xs.append(x.astype(np.float32))

    # Feature names
    names = encoder.feature_names()

    # ESM-2 feature names
    if spec.use_esm2 and esm2_dim > 0:
        names += [f"esm2_{i}" for i in range(esm2_dim)]

    # MHC feature names
    if spec.use_mhc and mhc_dim > 0:
        names += [f"mhc_{i}" for i in range(mhc_dim)]

    names += [f"kmer2_{i}" for i in range(spec.kmer_hash_dim)]
    names += [f"kmer3_{i}" for i in range(spec.kmer_hash_dim)]
    names += [
        "bio_length",
        "bio_hydrophobic_frac",
        "bio_polar_frac",
        "bio_acidic_frac",
        "bio_basic_frac",
        "bio_entropy",
        "bio_n_hydrophobic",
        "bio_c_hydrophobic",
        "bio_proline_frac",
        "bio_glycine_frac",
        "bio_aromatic_frac",
        "bio_basic2_frac",
        "bio_acidic2_frac",
        "bio_amide_frac",
        "bio_unique_residue_ratio",
        "bio_unknown_ratio",
    ]
    names += [f"env_{c}" for c in env_cols]

    if not xs:
        return np.zeros((0, len(names)), dtype=np.float32), names, env_cols
    return np.stack(xs, axis=0), names, env_cols
