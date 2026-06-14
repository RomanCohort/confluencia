"""
Confluencia 3.0 核心模块

四大支柱:
1. RNACTM — circRNA 六室药代动力学 (core/pk/rnactm)
2. Simulacrum — TNBC 全流程模拟 (core/tumor, tme, treatment, clinical)
3. TorusFold — 环形拓扑感知深度学习结构预测 (core/circrna/torusfold)
4. ViennaRNA — 物理约束二级结构 (core/circrna/structure_prediction)

评分架构:
  序列 → [ViennaRNA] → MFE/dsRNA/stability
       → [ImmuneSensing] → RIG-I/TLR7/TLR8/PKR
       → [TorusFold] → 3D pairmap/BSJ配对/柔性
       → [RNACTM] → PK曲线/半衰期/AUC
  四路汇聚 → 四维目标 [stability, translation, immune_evasion, delivery]
           → 加权 reward → IPS / therapeutic_window / joint_score
"""

# 四大支柱导出
from .pk.rnactm import RNACTMParams, infer_rna_ctm_params, simulate_rna_ctm, summarize_rna_ctm_curve
from .circrna.torusfold import TorusFold, TorusFoldConfig
from .circrna.torusfold import TorusPositionalEncoding, CircularRelativeBias
from .circrna.immune_sensing import predict_circrna_immunogenicity, ImmuneSensingConfig
from .circrna.structure_prediction import StructurePredictor, StructureFeatures

# Simulacrum 子系统管理器
from .subsystem_managers import (
    SubsystemManager,
    TumorManager,
    TMEManager,
    TreatmentManager,
    BiomarkerManager,
    ClinicalManager,
    CircRNAManager,
)

# 配置
from .config import Confluencia3Config, CircRNAConfig

__all__ = [
    # Pillar 1: RNACTM
    "RNACTMParams",
    "infer_rna_ctm_params",
    "simulate_rna_ctm",
    "summarize_rna_ctm_curve",
    # Pillar 2: Simulacrum managers
    "SubsystemManager",
    "TumorManager",
    "TMEManager",
    "TreatmentManager",
    "BiomarkerManager",
    "ClinicalManager",
    "CircRNAManager",
    # Pillar 3: TorusFold
    "TorusFold",
    "TorusFoldConfig",
    "TorusPositionalEncoding",
    "CircularRelativeBias",
    # Pillar 4: ViennaRNA + Immune
    "StructurePredictor",
    "StructureFeatures",
    "predict_circrna_immunogenicity",
    "ImmuneSensingConfig",
    # Config
    "Confluencia3Config",
    "CircRNAConfig",
]
