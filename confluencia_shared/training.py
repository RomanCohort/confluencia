from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class EarlyStopping:
    patience: int = 10
    min_delta: float = 0.0
    mode: str = "min"
    best: Optional[float] = None
    num_bad_epochs: int = 0

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = float(value)
            self.num_bad_epochs = 0
            return False

        improved = (value < self.best - self.min_delta) if self.mode == "min" else (value > self.best + self.min_delta)
        if improved:
            self.best = float(value)
            self.num_bad_epochs = 0
            return False

        self.num_bad_epochs += 1
        return self.num_bad_epochs >= int(self.patience)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    schedule: str,
    *,
    epochs: int,
    step_size: int = 10,
    gamma: float = 0.5,
    min_lr: float = 1e-6,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    schedule = str(schedule or "none").lower()
    if schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(epochs)), eta_min=float(min_lr))
    if schedule == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(step_size)), gamma=float(gamma))
    return None


def make_training_suggestions(history: Dict[str, List[float]]) -> List[str]:
    tips: List[str] = []
    train = history.get("train_loss", [])
    val = history.get("val_loss", [])
    if len(train) >= 3 and len(val) >= 3:
        if val[-1] > min(val[:-1]) * 1.05:
            tips.append("验证损失回升，建议开启早停或提高 dropout / weight_decay。")
        if train[-1] < val[-1] * 0.7:
            tips.append("训练-验证差距偏大，可能过拟合，建议加大正则或减少模型容量。")
        if val[-1] > val[-2] > val[-3]:
            tips.append("验证损失连续上升，可尝试降低学习率或启用余弦退火。")
    if not tips:
        tips.append("训练曲线稳定，可尝试微调学习率或延长训练轮次。")
    return tips


def make_metric_suggestions(metrics: Dict[str, float]) -> List[str]:
    tips: List[str] = []
    r2 = float(metrics.get("r2", 0.0)) if metrics.get("r2") is not None else None
    rmse = float(metrics.get("rmse", 0.0)) if metrics.get("rmse") is not None else None
    if r2 is not None:
        if r2 < 0.2:
            tips.append("R2 偏低，建议增加样本量或切换更强模型（如 hgb/mlp/transformer）。")
        elif r2 < 0.6:
            tips.append("R2 中等，可尝试调参（正则/深度/学习率）提升性能。")
    if rmse is not None and rmse > 0:
        tips.append("如 RMSE 偏高，可考虑特征工程或加入更多条件变量。")
    if not tips:
        tips.append("指标表现良好，可进行交叉验证或小幅调参巩固结果。")
    return tips
