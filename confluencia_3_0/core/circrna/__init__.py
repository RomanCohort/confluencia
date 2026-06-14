"""
Confluencia 3.0 — circRNA + TNBC Simulacrum 统一平台

circRNA 子系统：免疫感知、结构预测、折叠动力学、序列进化、疫苗评估等。

TorusFold (v2): 环形拓扑感知的 circRNA 深度学习框架
- TPE + CircPairformer + Diffusion 结构预测
- BSJ 跨接配对 + 闭合约束
- 物理约束求解器 (Plan B: 几何约束, Plan A: CG MD, 零训练数据补充)
- 三级相互作用模块 (未来启用)
"""

from .torusfold import TorusFold, TorusFoldConfig
from .torusfold_scorer import TorusFoldScorer, TorusFoldSignals, quick_score
from .torusfold.physics_structure_head import PhysicsStructureHead
