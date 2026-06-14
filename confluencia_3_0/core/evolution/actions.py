"""进化动作空间常量"""

# 药物分子进化动作
MOLECULE_ACTIONS = ["ed2mol", "mutate_light", "mutate_heavy"]

# circRNA 序列进化动作
# NOTE: "shuffle_ires_flanking" 替代旧版 "shuffle_utr" — circRNA 没有传统 UTR
CIRCRNA_ACTIONS = ["mutate_backbone", "optimize_ires", "shuffle_ires_flanking", "add_modification"]
