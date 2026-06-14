"""进化优化模块

内化自 confluencia-2.0-drug/core/evolution.py + generative.py，消除对 2.0 的运行时依赖。

两个进化系统:
1. 分子进化 (molecule_evolution) — REINFORCE + Pareto 多目标优化
2. circRNA 进化 (cirrna_evolution) — circRNA 序列设计优化
"""

from .molecule_evolution import (
    EvolutionConfig,
    EvolutionArtifacts,
    evolve_molecules_with_reflection,
)

from .cirrna_evolution import (
    CircRNAEvolutionConfig,
    CircRNAEvolutionArtifacts,
    evolve_cirrna,
    run_cirrna_evolution,
    optimize_for_translation,
    optimize_for_stability,
    optimize_for_immune_safety,
)

from .actions import MOLECULE_ACTIONS, CIRCRNA_ACTIONS

__all__ = [
    # 分子进化
    "EvolutionConfig",
    "EvolutionArtifacts",
    "evolve_molecules_with_reflection",
    # circRNA 进化
    "CircRNAEvolutionConfig",
    "CircRNAEvolutionArtifacts",
    "evolve_cirrna",
    "run_cirrna_evolution",
    "optimize_for_translation",
    "optimize_for_stability",
    "optimize_for_immune_safety",
    # 常量
    "MOLECULE_ACTIONS",
    "CIRCRNA_ACTIONS",
]
