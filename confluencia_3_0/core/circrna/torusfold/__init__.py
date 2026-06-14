"""
TorusFold — circRNA 结构预测的环形拓扑感知深度学习框架

核心创新：
1. Torus Positional Encoding (TPE): 周期性位置编码，消除 5'/3' 末端偏差
2. CircPairformer: AF3 风格的三角更新 + 环形距离 bias
3. BSJ 跨接配对: 反向剪接位点配对自然涌现
4. 闭合约束: Diffusion 采样中的 x[0]≈x[-1] 几何约束

模块：
- tpe: TorusPositionalEncoding + CircularRelativeBias
- triangle_update: CircPairformer (三角更新 + 三角注意力)
- diffusion_structure: 3D 结构预测 (Simple + Diffusion + Flexible)
- torusfold: 主模型整合
- tertiary_interaction: RNA 三级相互作用 (假结/kissing loops, 未来启用)
- irs_pair: BSJ 配对分析器
"""

from .tpe import TorusPositionalEncoding, TorusPositionalEncoding2D, CircularRelativeBias
from .triangle_update import (
    TriangleMultiplicativeUpdate,
    TriangleAttention,
    PairTransition,
    CircPairformerBlock,
    CircPairformerStack,
)
from .diffusion_structure import (
    CircDiffusionStructure,
    SimpleStructureHead,
    FlexibleStructureHead,
    ClosureConstrainedDiffusion,
)
from .torusfold import TorusFold, TorusFoldConfig, PairInitialization, PairPredictionHead
from .tertiary_interaction import (
    TertiaryInteractionModule,
    circ_contact_from_linear,
)
from .irs_pair import BSJPairAnalyzer, circular_distance_matrix
from .equivariant_backbone import CircEquivariantBackbone, TorusTransformerLayer

__all__ = [
    # Core innovations
    "TorusFold",
    "TorusFoldConfig",
    "TorusPositionalEncoding",
    "CircularRelativeBias",
    "CircPairformerStack",
    # Structure prediction
    "SimpleStructureHead",
    "CircDiffusionStructure",
    "FlexibleStructureHead",
    "ClosureConstrainedDiffusion",
    # Pair analysis
    "PairInitialization",
    "PairPredictionHead",
    "BSJPairAnalyzer",
    "circular_distance_matrix",
    # Future modules
    "TertiaryInteractionModule",
    "circ_contact_from_linear",
    # Backbone
    "CircEquivariantBackbone",
    "TorusTransformerLayer",
]
