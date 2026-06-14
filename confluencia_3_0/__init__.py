"""Confluencia 3.0 — circRNA + TNBC Simulacrum 统一计算平台。

面向 circRNA 药物发现的多任务计算平台，整合 TNBC 模拟环境和 circRNA 免疫/结构/进化子系统，
采用 EventBus-first 事件驱动架构，Backend 统一调度 heuristic/vienna/esm2 三档精度。
"""

__version__ = "3.0.0"

from confluencia_3_0.core.config import Confluencia3Config, CircRNAConfig
