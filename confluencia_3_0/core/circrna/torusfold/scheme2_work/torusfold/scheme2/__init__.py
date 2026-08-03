"""
TorusFold-Scheme2 — circRNA 3D 结构预测（零训练几何求解 + 全原子精修）。

核心模块:
- GeometricConstraintSolver: 正多边形初始化 + 配对约束 + 闭合校正
- openmm_refine: OpenMM 粗粒度能量最小化 (每核苷酸 1 粒子)
- allatom_reconstruct: 粗粒度 → 全原子 (amber14 RNA.OL3 模板)
- amber_refine: Amber14 OL3 + OBC1 隐式溶剂约束最小化
- predict_3d: 端到端序列 → 粗粒度 3D 坐标
- predict_3d_allatom: 端到端序列 → 全原子 3D 结构 (含 H)

默认 RL 权重 (2026-08-02): use_rl=True 且未显式指定时自动用训练好的策略.
- BC prior (bc_policy_big.pt): 20x 扩展数据训练, 学"选哪些 far 块拉拢".
- DPO 打分器 (dpo_policy_v3_compat.pt): 学"保近程配对 + far 拉近"构象价值.
"""
import numpy as np
from pathlib import Path

# ── RL 默认权重路径 (use_rl=True 时自动加载) ──
_DEFAULT_RL_POLICY_PATH = str(Path(__file__).resolve().parents[3]
                              / "data" / "bc_policy_big.pt")
_DEFAULT_DPO_POLICY_PATH = str(Path(__file__).resolve().parents[3]
                               / "data" / "dpo_policy_v3_compat.pt")

from .constraint_solver import (
    GeometricConstraintSolver,
    SolverConfig,
    circular_distance_matrix_np,
    is_bsj_crossing_np,
)
from .refine import (
    vienna_pair_probs,
    scheme2_initial_coords,
    openmm_refine,
    predict_3d,
    BOND_LEN,
    PAIR_DIST,
    CLASH_DIST,
)
from .allatom_reconstruct import (
    reconstruct_all_atom,
    AllAtomStructure,
    get_atom_xyzs,
)
from .amber_refine import amber_refine
from .rl_optimizer import optimize_far_pairs
from .rl_optimizer import (
    ConformationSample,
    ConformationDistribution,
    PolicyNetwork,
    MCTS,
    BlockState,
    RLOptimizerState,
    build_rl_state,
    compute_reward,
    apply_action,
)
from .pair_graph import (
    parse_case_annotation,
    build_full_pair_graph,
    extract_stem_blocks,
    complementarity_scan,
    build_pair_graph,
    far_end_pairs,
)

__all__ = [
    "GeometricConstraintSolver",
    "SolverConfig",
    "circular_distance_matrix_np",
    "is_bsj_crossing_np",
    "vienna_pair_probs",
    "scheme2_initial_coords",
    "openmm_refine",
    "predict_3d",
    "reconstruct_all_atom",
    "AllAtomStructure",
    "get_atom_xyzs",
    "amber_refine",
    "optimize_far_pairs",
    "predict_3d_allatom",
    "BOND_LEN",
    "PAIR_DIST",
    "CLASH_DIST",
    "complementarity_scan",
    "build_pair_graph",
    "far_end_pairs",
]

__version__ = "0.2.0"


def predict_3d_allatom(
    sequence: str,
    *,
    pair_threshold: float = 0.5,
    platform_name: str = "CPU",
    max_iterations: int = 3000,
    use_rl: bool = False,
    rl_policy_path: str = None,
    rl_n_simulations: int = 50,
    coding_mask=None,
    rl_dpo_weight: float = 0.0,
    rl_dpo_policy_path: str = None,
    rl_dpo_rollout: bool = False,
    rl_dpo_simulate: bool = False,
    rl_use_defaults: bool = True,
    use_3bead: bool = True,
):
    """端到端: 序列 → 全原子 RNA 结构 (amber14 OL3 精修)。

    链路: ViennaRNA 配对 → CG 几何求解 → CG OpenMM 精修 → [RL 优化远端配对]
          → 1EHZ 全原子重建 → Amber14 OL3 + OBC1 约束最小化。

    pair_threshold=0.5 比 predict_3d 的 0.3 严: 全原子链路要把配对喂给
    amber 力场做约束, 弱配对 (0.3-0.5) 喂进去会引入虚假约束导致力场
    在错的茎上拉扯, 所以这里只留高置信配对 (有意分档, 非笔误)。

    use_rl: 是否启用 RL 远端配对优化 (默认 False, 不破现有链路)。
        ViennaRNA 在长序列 (1000nt+) 远端配对 DP 爆炸会漏, RL 补这块:
        pair_graph 扫描补长程配对 + MCTS 策略拉拢远端块。RL 是搜索信号
        不是评估结论, 优化后必回 amber 收尾 (最终结构由 amber 定)。
    rl_policy_path: 策略网络权重路径 (None 且 rl_use_defaults=True 时自动用
        bc_policy_big.pt; None 且 rl_use_defaults=False 用随机策略 baseline)。
    rl_use_defaults: True (默认) 时 use_rl 自动加载训练好的 BC prior + DPO 打分器
        + dpo_simulate (220x 加速). False 完全关默认 (纯随机策略, 对照用).
    use_3bead: True (默认) 用 3-bead 分段折叠 (二级结构初始化 + 茎区螺旋/环区松散),
        立体折叠 (PC3>0.05). False 用旧 1-bead (scheme2_initial_coords 平面圆).
    rl_n_simulations: MCTS 模拟次数。
    coding_mask: (L,) bool 数组, True=coding 区残基。来自序列大小写
        (parse_case_annotation) 或 ORF 预测。RL 全序列可动, amber 精修时
    rl_dpo_weight: DPO 构象质量打分器权重 (0=关闭, >0 启用, 见 rl_optimizer)。
        DPO policy (train_dpo.py 训练) 学"保近程配对 + far 拉近", 引导 MCTS
        探索时偏好这类高质量构象。
    rl_dpo_policy_path: DPO policy 权重路径 (None 时若 rl_dpo_weight>0 则跳过)。
    rl_dpo_rollout: 用 DPO V (3ms) 替代完整 compute_reward (1295ms) 做 MCTS rollout
        估值. DPO V 与 compute_reward 排序对齐 (Spearman 0.73), 探索方向正确.
        实测: 同样 n_sim 快 46%, 或 3x 模拟下质量更高且仍快 37% (2026-08-02).
        (L,) bool, True=coding 区残基。来自序列大小写
        coding 区 P 用高 k (10000) 钉回 RL 之前的 CG 原坐标, 保真实结构;
        非 coding 区 P 用低 k (1000) 钉 amber 输入坐标, 接受 RL 优化+物理收敛。
        None 时 amber 全部按非 coding 处理 (不钉死任何区)。

    Returns:
        dict {coords_cg, pairs, e0_cg, e1_cg, atoms, coords_aa,
              e0_aa, e1_aa, amber_info, rl_info}
    """
    pairs, bpp = vienna_pair_probs(sequence, pair_threshold)
    if use_3bead:
        # 3-bead 分段折叠管线 (2026-08-02): 二级结构初始化 + 茎区螺旋/环区松散.
        # 替代旧 1-bead (scheme2_initial_coords 平面圆 + openmm_refine 单粒子),
        # 立体折叠 (PC3>0.05), 输出仍是 (L,3) P 坐标, 下游全原子重建兼容.
        from .segmented_folding import (
            init_from_secondary_structure, refine_segmented_3bead,
        )
        L = len(sequence)
        p_init = init_from_secondary_structure(L, pairs)
        cg_coords, e0_cg, e1_cg = refine_segmented_3bead(
            p_init, pairs, platform_name, n_anneal=60)
    else:
        # 旧 1-bead 管线 (向后兼容 / 对照)
        init = scheme2_initial_coords(sequence, pairs, n_samples=8)
        if init is None:
            raise RuntimeError(f"Scheme2 CG 几何求解失败 (L={len(sequence)})")
        cg_coords, e0_cg, e1_cg = openmm_refine(init, pairs, platform_name)

    rl_info = None
    cg_coords_for_amber = cg_coords  # amber 钉死 coding 区用的 CG 原坐标
    if use_rl:
        # 自动应用训练好的默认策略 (use_rl=True 且未显式指定时).
        # rl_use_defaults=True: 补全 BC prior + DPO 打分器 + 廉价 rollout.
        #   实测 (2026-08-02): BC prior pair +4~15% clash -5~13%, dpo_simulate 220x 快.
        # 显式传参覆盖对应项; rl_use_defaults=False 完全关默认 (纯随机策略 baseline).
        if rl_use_defaults:
            if rl_policy_path is None:
                rl_policy_path = _DEFAULT_RL_POLICY_PATH
            if rl_dpo_policy_path is None and rl_dpo_weight <= 0:
                rl_dpo_policy_path = _DEFAULT_DPO_POLICY_PATH
                rl_dpo_weight = 5.0
            if not rl_dpo_rollout and not rl_dpo_simulate:
                rl_dpo_simulate = True  # 全树 DPO reward, 220x 加速 (质量持平)
        # pair_graph: 补 ViennaRNA 漏掉的长程配对 + 标记远端配对
        _, scan_pairs, far_pairs = build_full_pair_graph(
            sequence, pairs, do_scan=True,
        )
        stem_blocks = extract_stem_blocks(pairs, scan_pairs)
        if far_pairs:
            opt_p, cg_orig, rl_info = optimize_far_pairs(
                cg_coords, sequence, far_pairs, stem_blocks,
                policy_path=rl_policy_path,
                n_simulations=rl_n_simulations,
                coding_mask=coding_mask,
                dpo_weight=rl_dpo_weight,
                dpo_policy_path=rl_dpo_policy_path,
                dpo_rollout=rl_dpo_rollout,
                dpo_simulate=rl_dpo_simulate,
            )
            cg_coords = opt_p               # RL 优化后的 CG P 喂给 amber 重建 (Å)
            cg_coords_for_amber = cg_orig    # RL 之前的 CG 原坐标, 给 amber 钉死 coding (Å)
        else:
            rl_info = {"skipped": True, "reason": "no_far_pairs"}

    # 用 1EHZ 晶体模板重建 (替代旧手算 allatom_reconstruct, 解决 amber_field 恒正)
    from .aform_from_template import reconstruct_all_atom as reconstruct_from_template
    structure = reconstruct_from_template(cg_coords, sequence)
    # amber_refine 的 cg_coords 期望 nm (跟 structure.atoms.xyz/10.0 同单位)。
    # cg_coords_for_amber 是 Å (openmm_refine 输出), 转 nm。
    cg_coords_nm = None
    if use_rl and cg_coords_for_amber is not None:
        cg_coords_nm = np.asarray(cg_coords_for_amber, dtype=np.float64) / 10.0
    coords_aa, e0_aa, e1_aa, amber_info = amber_refine(
        structure, pairs,
        platform_name=platform_name,
        max_iterations=max_iterations,
        coding_mask=coding_mask,
        cg_coords=cg_coords_nm,
    )

    # 可计算结构指纹 + 结构信号 (对齐原生 TorusFold 契约, 纯计算无 DL)
    from .immune_heuristic import (
        compute_immune_fingerprints, compute_structure_signals,
    )
    bsj_dist = float(np.linalg.norm(cg_coords[0] - cg_coords[-1]))
    immune_fingerprints = compute_immune_fingerprints(
        coords_aa, structure, pairs, sequence,
    )
    structure_signals = compute_structure_signals(
        coords_aa, structure, pairs, bpp, sequence,
        e1_aa=e1_aa, bsj_dist=bsj_dist, cg_coords=cg_coords,
    )

    return {
        "coords_cg": cg_coords,
        "pairs": pairs,
        "pair_probs": bpp,
        "e0_cg": e0_cg,
        "e1_cg": e1_cg,
        "atoms": structure,
        "coords_aa": coords_aa,
        "e0_aa": e0_aa,
        "e1_aa": e1_aa,
        "amber_info": amber_info,
        "rl_info": rl_info,
        # 结构方法标识 (对齐 TrunkOutput.structure_method)
        "structure_method": "scheme2_allatom",
        "available": True,
        # 5 个可计算免疫指纹 (per-residue + scalar, 对齐 ImmuneFingerprintHeads)
        "immune_fingerprints": immune_fingerprints,
        # 结构信号 (对齐 TorusFoldSignals + ImmuneSensingResultV3 透传字段)
        "structure_signals": structure_signals,
    }
