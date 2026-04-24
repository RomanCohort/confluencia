"""Helper suggestions used by the predictor UI (small compatibility layer)."""
from __future__ import annotations

from typing import Dict, List, Any


def make_metric_suggestions(metrics: Dict[str, Any]) -> List[str]:
    suggestions: List[str] = []
    try:
        r2 = float(metrics.get("r2", 0.0))
    except Exception:
        r2 = 0.0
    try:
        rmse = float(metrics.get("rmse", 0.0))
    except Exception:
        rmse = 0.0

    if r2 < 0.2:
        suggestions.append("低 R2：考虑增加训练数据或使用更简单的模型。")
    if rmse > 0 and rmse > 1.0:
        suggestions.append("较高的 RMSE：检查数据标准化与特征工程。")
    if not suggestions:
        suggestions.append("模型表现合理，考虑微调超参数以进一步提升。")
    return suggestions


def make_training_suggestions(history: Dict[str, List[float]]) -> List[str]:
    suggestions: List[str] = []
    train_loss = history.get("train_loss") or []
    val_loss = history.get("val_loss") or []
    if train_loss and val_loss:
        if val_loss[-1] > train_loss[-1] * 1.1:
            suggestions.append("可能过拟合：尝试增加正则化或早停。")
        elif val_loss[-1] < train_loss[-1] * 0.9:
            suggestions.append("验证损失明显低于训练损失：检查数据泄露或评估逻辑。")
    if not suggestions:
        suggestions.append("无明显训练问题。")
    return suggestions
