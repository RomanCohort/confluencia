from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


AMINO_ACIDS = list("ARNDCQEGHILKMFPSTWYV")


def sequence_to_aaindex(sequence: str, aaindex: Dict[str, np.ndarray],
                         unknown_value: Optional[float] = 0.0) -> np.ndarray:
    """
    将氨基酸序列转换为 AAIndex 表征矩阵。

    Parameters
    - sequence: 单个字母氨基酸序列（字符串）。
    - aaindex: 字典，键为单字母氨基酸（A, R, N...），值为 numpy 数组（长度 k），表示每个氨基酸的性质向量。
    - unknown_value: 当遇到未知/缺失氨基酸时，填充值（标量或向量将被广播）。

    Returns
    - np.ndarray: 形状为 (L, k) 的数组，L 为序列长度，k 为 AAIndex 向量维度。
    """
    seq = sequence.strip().upper()
    if not seq:
        return np.zeros((0, 0), dtype=float)

    # Determine vector length
    sample = None
    for v in aaindex.values():
        sample = np.asarray(v)
        break
    if sample is None:
        raise ValueError("aaindex mapping is empty")
    k = sample.shape[-1]

    # Prepare output
    out = np.zeros((len(seq), k), dtype=float)
    for i, aa in enumerate(seq):
        vec = aaindex.get(aa)
        if vec is None:
            # broadcast scalar unknown_value to vector if necessary
            if np.isscalar(unknown_value):
                out[i, :] = float(unknown_value)
            else:
                v = np.asarray(unknown_value)
                if v.shape == (k,):
                    out[i, :] = v
                else:
                    out[i, :] = 0.0
        else:
            arr = np.asarray(vec)
            if arr.shape[-1] != k:
                raise ValueError("Inconsistent AAIndex vector lengths")
            out[i, :] = arr
    return out


def one_hot_encode(sequence: str, aa_order: Optional[List[str]] = None) -> np.ndarray:
    """
    将氨基酸序列编码为单热（one-hot）矩阵。

    Parameters
    - sequence: 氨基酸序列字符串。
    - aa_order: 氨基酸顺序列表，默认使用 ARNDCQEGHILKMFPSTWYV。

    Returns
    - np.ndarray: 形状为 (L, 20) 的 0/1 矩阵。
    """
    if aa_order is None:
        aa_order = AMINO_ACIDS
    idx = {aa: i for i, aa in enumerate(aa_order)}
    seq = sequence.strip().upper()
    L = len(seq)
    M = len(aa_order)
    out = np.zeros((L, M), dtype=np.float32)
    for i, aa in enumerate(seq):
        j = idx.get(aa)
        if j is not None:
            out[i, j] = 1.0
    return out


def continuous_onehot_encode(sequence: str, aaindex: Dict[str, np.ndarray],
                              aa_order: Optional[List[str]] = None,
                              unknown_value: Optional[float] = 0.0) -> np.ndarray:
    """
    将氨基酸序列编码为单热与连续 AAIndex 向量的拼接（每残基为 [one-hot | properties]）。

    Returns
    - np.ndarray: 形状为 (L, 20 + k) 的数组。
    """
    oh = one_hot_encode(sequence, aa_order=aa_order)
    idx_mat = sequence_to_aaindex(sequence, aaindex, unknown_value=unknown_value)
    if oh.shape[0] != idx_mat.shape[0]:
        raise ValueError("Sequence length mismatch between encodings")
    return np.concatenate([oh, idx_mat.astype(np.float32)], axis=1)


def load_aaindex_from_csv(path: str, aa_col: str = "AA") -> Dict[str, np.ndarray]:
    """
    从 CSV 文件加载 AAIndex 映射表。CSV 需要包含一列氨基酸单字母（如 'AA'），以及若干数值列。
    返回字典：{ 'A': np.array([...]), ... }
    """
    df = pd.read_csv(path)
    if aa_col not in df.columns:
        raise ValueError(f"CSV must contain column '{aa_col}'")
    # numeric columns except the aa_col
    num_cols = [c for c in df.columns if c != aa_col and is_numeric_dtype(df[c])]
    mapping: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        aa = str(row[aa_col]).strip().upper()
        if not aa:
            continue
        mapping[aa] = np.asarray(row[num_cols], dtype=float)
    return mapping
