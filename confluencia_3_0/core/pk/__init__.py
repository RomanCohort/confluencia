"""PK 模块 — circRNA 药代动力学模型

内化自 confluencia-2.0-drug/core/ctm.py，消除对 2.0 的运行时依赖。
"""

from .rnactm import (
    RNACTMParams,
    infer_rna_ctm_params,
    simulate_rna_ctm,
    summarize_rna_ctm_curve,
)

from .legacy_ctm import (
    CTMParams,
    params_from_micro_scores,
    simulate_ctm,
    summarize_curve,
)

__all__ = [
    # RNACTM 六室模型 (circRNA 专用)
    "RNACTMParams",
    "infer_rna_ctm_params",
    "simulate_rna_ctm",
    "summarize_rna_ctm_curve",
    # Legacy 四室模型 (通用药物)
    "CTMParams",
    "params_from_micro_scores",
    "simulate_ctm",
    "summarize_curve",
]
