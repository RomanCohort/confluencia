"""confluencia_3_0/core/circrna/confidence_metrics.py

动态置信度计算模块（V3 改进）

核心改进：
  - 置信度不再是硬编码常量，而是动态计算
  - 四维度分解：模型状态、结构完整性、物理约束、时间效率
  - 支持用户自定义权重（YAML 配置文件）
  - 降级事件记录（fallback_log）

设计理念：
  方案2：在 StructurePredictionResult 内部添加 compute_confidence() 方法
  方案3：两个字段并存（向后兼容）
  方案C：动态参数传入（运行时可调）

作者：Confluencia Team
日期：2026-06-30
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING
import numpy as np
import yaml
import os

if TYPE_CHECKING:
    from confluencia_3_0.core.circrna.structure_backend import StructurePredictionResult


# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class ConfidenceBreakdown:
    """置信度分解结果（可解释）。

    四维度评分：
      1. model_score    - 模型状态（训练程度、可用性）
      2. structure_score - 结构完整性（坐标、配对信息）
      3. physics_score   - 物理约束满足（BSJ closure、键长）
      4. time_score      - 时间效率

    综合：加权平均（权重可自定义）
    """
    # === 综合评分 ===
    overall: float  # [0.0, 1.0]

    # === 四维度 ===
    model_score: float
    structure_score: float
    physics_score: float
    time_score: float

    # === 元信息 ===
    method: str
    model_status: str  # "trained", "training", "unavailable", "physical"
    sequence_length: int

    # === 警告和建议 ===
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # === 辅助字段 ===
    elapsed_time: float = 0.0
    available: bool = True

    # === 降级记录 ===
    fallback_events: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """转换为字典（用于日志/API）。"""
        return {
            "overall": self.overall,
            "model": self.model_score,
            "structure": self.structure_score,
            "physics": self.physics_score,
            "time": self.time_score,
            "method": self.method,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }


# ═══════════════════════════════════════════════════════════════
# 默认配置（可被配置文件覆盖）
# ═══════════════════════════════════════════════════════════════

DEFAULT_WEIGHTS = {
    "model": 0.30,
    "structure": 0.35,
    "physics": 0.25,
    "time": 0.10,
}

THRESHOLDS = {
    "bsj_closure_good": 10.0,  # Å
    "bond_rmsd_good": 0.5,    # Å
    "clash_count_max": 5,
    "timeout_threshold": 30.0,  # seconds
}


def validate_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """校验并修正权重配置。

    Args:
        weights: 用户提供的权重字典

    Returns:
        校验后的权重字典（总和=1.0，每个值∈[0.1, 0.5])
    """
    # 必须包含四个维度
    required_keys = ["model", "structure", "physics", "time"]
    for key in required_keys:
        if key not in weights:
            weights[key] = DEFAULT_WEIGHTS[key]

    # 每个权重 ∈ [0.1, 0.5]（防止极端配置）
    for key in weights:
        weights[key] = float(np.clip(weights[key], 0.1, 0.5))

    # 总和归一化到 1.0
    total = sum(weights.values())
    if total > 0:
        for key in weights:
            weights[key] /= total

    return weights


# ═══════════════════════════════════════════════════════════════
# 核心计算函数
# ═══════════════════════════════════════════════════════════════

def compute_confidence_breakdown(
    result: "StructurePredictionResult",
    weights: Optional[Dict[str, float]] = None,
) -> ConfidenceBreakdown:
    """计算动态置信度分解。

    Args:
        result: StructurePredictionResult 对象
        weights: 自定义权重，None 则使用默认值

    Returns:
        ConfidenceBreakdown: 分解结果
    """
    # 校验权重
    w = validate_weights(weights or DEFAULT_WEIGHTS.copy())

    # === 1. 模型状态评分 ===
    model_score, model_status = _compute_model_score(result)

    # === 2. 结构完整性评分 ===
    structure_score = _compute_structure_score(result)

    # === 3. 物理约束评分 ===
    physics_score = _compute_physics_score(result)

    # === 4. 时间效率评分 ===
    time_score = _compute_time_score(result)

    # === 5. 综合 ===
    overall = (
        w["model"] * model_score +
        w["structure"] * structure_score +
        w["physics"] * physics_score +
        w["time"] * time_score
    )

    # === 6. 收集警告 ===
    warnings = []
    recommendations = []

    if not result.available:
        warnings.append("⚠️ 无有效预测结果")

    if result.timed_out:
        warnings.append(f"⚠️ 预测超时 ({result.elapsed_time:.1f}s)")

    if result.coords is None:
        warnings.append("⚠️ 缺少 3D 坐标，仅有二级结构")

    if result.bsj_closure > THRESHOLDS["bsj_closure_good"]:
        warnings.append(f"⚠️ BSJ closure 过大: {result.bsj_closure:.1f}Å")

    if result.bond_rmsd > THRESHOLDS["bond_rmsd_good"]:
        warnings.append(f"⚠️ Bond RMSD 过大: {result.bond_rmsd:.2f}Å")

    if hasattr(result, "clash_count") and result.clash_count > THRESHOLDS["clash_count_max"]:
        warnings.append(f"⚠️ 冲突数过多: {result.clash_count}")

    # === 7. 生成建议 ===
    if overall < 0.5:
        recommendations.append("💡 置信度过低，建议切换后端或验证实验")
    elif overall < 0.7:
        recommendations.append("💡 中等置信度，建议结合多个预测结果")
    else:
        recommendations.append("✅ 高置信度预测，结果可信")

    if model_status == "training":
        recommendations.append("💡 TorusFold 模型正在训练中，精度会逐步提升")

    # === 8. 降级记录 ===
    fallback_events = result.fallback_log if hasattr(result, "fallback_log") else []
    if fallback_events:
        warnings.append(f"⚠️ 发生 {len(fallback_events)} 次降级")

    # === 9. 序列长度 ===
    seq_len = 0
    if result.coords is not None:
        seq_len = len(result.coords)
    elif result.pair_probs is not None:
        seq_len = result.pair_probs.shape[0]
    elif result.pair_list is not None:
        seq_len = max(max(i, j) for i, j in result.pair_list) + 1

    return ConfidenceBreakdown(
        overall=float(np.clip(overall, 0.0, 1.0)),
        model_score=model_score,
        structure_score=structure_score,
        physics_score=physics_score,
        time_score=time_score,
        method=result.method,
        model_status=model_status,
        sequence_length=seq_len,
        warnings=warnings,
        recommendations=recommendations,
        elapsed_time=result.elapsed_time,
        available=result.available,
        fallback_events=fallback_events,
    )


def _compute_model_score(result: "StructurePredictionResult") -> tuple:
    """计算模型状态评分。

    Returns:
        (score, status)
    """
    if result.method == "torusfold":
        if result.timed_out or result.coords is None:
            return 0.30, "unavailable"
        return 0.90, "trained"  # 假设已训练

    elif result.method == "pipeline":
        return 0.75, "physical"

    else:  # heuristic
        return 0.50, "heuristic"


def _compute_structure_score(result: "StructurePredictionResult") -> float:
    """计算结构完整性评分。"""
    base = 0.5

    if result.coords is not None and len(result.coords) > 0:
        base = 1.0
    elif result.pair_probs is not None or result.pair_list:
        base = 0.7
    else:
        base = 0.3

    if result.timed_out:
        base *= 0.6

    return float(np.clip(base, 0.0, 1.0))


def _compute_physics_score(result: "StructurePredictionResult") -> float:
    """计算物理约束评分。"""
    if result.coords is None:
        return 0.5

    score = 0.5

    # BSJ closure
    if result.bsj_closure <= THRESHOLDS["bsj_closure_good"]:
        score += 0.25
    else:
        score -= 0.10

    # Bond RMSD
    if result.bond_rmsd <= THRESHOLDS["bond_rmsd_good"]:
        score += 0.25
    else:
        score -= 0.10

    # Clash count
    if hasattr(result, "clash_count"):
        if result.clash_count <= THRESHOLDS["clash_count_max"]:
            score += 0.20
        else:
            score -= 0.15

    return float(np.clip(score, 0.0, 1.0))


def _compute_time_score(result: "StructurePredictionResult") -> float:
    """计算时间效率评分。"""
    if result.timed_out:
        return 0.6

    elapsed = result.elapsed_time

    if elapsed > THRESHOLDS["timeout_threshold"]:
        return 0.8
    elif elapsed > 5.0:
        return 0.9
    else:
        return 1.0


# ═══════════════════════════════════════════════════════════════
# 等级分类和格式化
# ═══════════════════════════════════════════════════════════════

CONFIDENCE_LEVELS = {
    "very_low": (0.0, 0.3),
    "low": (0.3, 0.5),
    "medium": (0.5, 0.7),
    "high": (0.7, 0.85),
    "very_high": (0.85, 1.0),
}


def classify_confidence(breakdown: ConfidenceBreakdown) -> str:
    """分类置信度等级。"""
    for level, (lo, hi) in CONFIDENCE_LEVELS.items():
        if lo <= breakdown.overall < hi:
            return level
    return "unknown"


def format_breakdown(breakdown: ConfidenceBreakdown) -> str:
    """格式化输出（用于日志）。"""
    lines = []
    lines.append("=" * 60)
    lines.append("置信度分解报告")
    lines.append("=" * 60)

    level = classify_confidence(breakdown)
    lines.append(f"\n综合置信度: {breakdown.overall:.2f} [{level}]")
    lines.append(f"  方法: {breakdown.method}")
    lines.append(f"  模型状态: {breakdown.model_status}")
    lines.append(f"  序列长度: {breakdown.sequence_length} nt")
    lines.append(f"  耗时: {breakdown.elapsed_time:.1f}s")

    lines.append("\n四维度评分:")
    lines.append("-" * 60)
    lines.append(f"  模型状态: {breakdown.model_score:.2f}")
    lines.append(f"  结构完整性: {breakdown.structure_score:.2f}")
    lines.append(f"  物理约束: {breakdown.physics_score:.2f}")
    lines.append(f"  时间效率: {breakdown.time_score:.2f}")

    if breakdown.warnings:
        lines.append("\n⚠️ 警告:")
        for w in breakdown.warnings:
            lines.append(f"  {w}")

    if breakdown.recommendations:
        lines.append("\n💡 建议:")
        for r in breakdown.recommendations:
            lines.append(f"  {r}")

    if breakdown.fallback_events:
        lines.append("\n降级记录:")
        for event in breakdown.fallback_events:
            lines.append(f"  - {event}")

    lines.append("=" * 60)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 配置文件加载
# ═══════════════════════════════════════════════════════════════

def load_config(config_path: Optional[str] = None) -> Dict:
    """从 YAML 文件加载配置。

    Args:
        config_path: 配置文件路径，None 则使用默认路径

    Returns:
        配置字典（weights + thresholds + presets）
    """
    # 默认路径
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__),
            "confidence_config.yaml"
        )

    if not os.path.exists(config_path):
        return {
            "weights": DEFAULT_WEIGHTS,
            "thresholds": THRESHOLDS,
        }

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 校验权重
    if "weights" in config:
        config["weights"] = validate_weights(config["weights"])

    # 合并阈值
    if "thresholds" in config:
        thresholds = THRESHOLDS.copy()
        thresholds.update(config["thresholds"])
        config["thresholds"] = thresholds

    return config


def get_preset_weights(preset_name: str) -> Dict[str, float]:
    """获取预设权重配置。

    Args:
        preset_name: "fast_screening", "high_accuracy", "balanced"

    Returns:
        权重字典
    """
    config = load_config()

    if "presets" in config and preset_name in config["presets"]:
        return validate_weights(config["presets"][preset_name])

    return DEFAULT_WEIGHTS.copy()


# ═══════════════════════════════════════════════════════════════
# 降级记录辅助函数
# ═══════════════════════════════════════════════════════════════

FALLBACK_EVENT_TYPES = [
    "torusfold_timeout",
    "torusfold_unavailable",
    "pipeline_timeout",
    "pipeline_partial",
    "heuristic_fallback",
]


def log_fallback_event(
    result: "StructurePredictionResult",
    event_type: str,
    details: Optional[str] = None,
) -> None:
    """记录降级事件到 result.fallback_log。

    Args:
        result: StructurePredictionResult 对象
        event_type: 事件类型（见 FALLBACK_EVENT_TYPES）
        details: 详细说明（可选）
    """
    if event_type not in FALLBACK_EVENT_TYPES:
        return

    event_msg = f"[{event_type}]"
    if details:
        event_msg += f" {details}"

    if hasattr(result, "fallback_log"):
        result.fallback_log.append(event_msg)
        # 限制最大事件数
        if len(result.fallback_log) > 10:
            result.fallback_log = result.fallback_log[-10:]


if __name__ == "__main__":
    # 测试示例
    print("=" * 60)
    print("置信度计算模块测试")
    print("=" * 60)

    # 测试配置加载
    config = load_config()
    print("\n默认配置:")
    print(f"  weights: {config['weights']}")
    print(f"  thresholds: {config['thresholds']}")

    # 测试预设权重
    for preset_name in ["fast_screening", "high_accuracy", "balanced"]:
        weights = get_preset_weights(preset_name)
        print(f"\n预设 '{preset_name}': {weights}")

    # 测试校验函数
    bad_weights = {"model": 0.9, "structure": 0.9, "physics": 0.9, "time": 0.9}
    good_weights = validate_weights(bad_weights)
    print(f"\n校验极端权重:")
    print(f"  输入: {bad_weights}")
    print(f"  输出: {good_weights}")