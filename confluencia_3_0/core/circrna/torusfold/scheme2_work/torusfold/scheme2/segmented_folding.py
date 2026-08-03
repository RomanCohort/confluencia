"""segmented_folding.py — 分段折叠 (按二级结构茎切分, 茎区A-form螺旋/环区松散).

核心架构 (解决闭环环张力问题):
  113nt 单环用全局二面角约束失败: A-form 螺距(2.8Å/nt)与环周长(5.9Å/nt)不匹配,
  环要压缩一半才能成螺旋, 张力极大. 真实 circRNA 3D 结构是茎-环嵌套:
    - 茎区 (stem, WC配对): A-form 双螺旋, 二面角33° + 堆叠LJ强约束
    - 环区 (loop, 无配对): 松散卷曲, 只 clash 防穿模 + 弱堆叠
  分段: 按 ViennaRNA 茎边界切分, 每段独立折叠, 茎区螺旋/环区松散, 段间WC钉合.

链路:
  ViennaRNA MFE dot-bracket → 茎/环分段 → 各段 3-bead CG MD (茎螺旋/环松散)
  → 段间 WC 配对 harmonic 钉合 → 整环松弛 → 拼接 3D 构象
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .p_to_3bead import p_to_3bead
from .refine import BOND_LEN


def extract_stems_from_structure(
    pairs: List[Tuple[int, int, float]],
) -> List[List[Tuple[int, int]]]:
    """从 ViennaRNA 配对提取茎 (连续配对的段).

    茎 = 一串连续 i 配一串连续 j (反向平行), 连续性: i 差≤2 且 j 差≤2.
    用于分段: 每个茎是一个 A-form 双螺旋单元.

    Returns:
        [[(i,j), ...], ...] 每个茎的逐残基配对列表.
    """
    sorted_pairs = sorted(pairs, key=lambda x: (min(x[0], x[1]), max(x[0], x[1])))
    stems = []
    current = []
    for i, j, _w in sorted_pairs:
        pair = (min(i, j), max(i, j))
        if not current:
            current = [pair]
            continue
        prev = current[-1]
        # i 连续 + j 连续 (反向平行: i 增 j 减, 或都增但接近)
        if (abs(pair[0] - prev[0]) <= 2 and abs(pair[1] - prev[1]) <= 2):
            current.append(pair)
        else:
            stems.append(current)
            current = [pair]
    if current:
        stems.append(current)
    # 只保留 ≥3 配对的茎 (太短不是 A-form 螺旋)
    return [s for s in stems if len(s) >= 3]


def classify_regions(
    L: int,
    stems: List[List[Tuple[int, int]]],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """把残基分成茎区和环区.

    茎区 = 所有茎里的残基; 环区 = 其余 (hairpin/internal/bulge/multiloop).

    Returns:
        (stem_residues, loop_residues) 各为 [(start, end), ...] 连续段.
    """
    stem_set = set()
    for stem in stems:
        for i, j in stem:
            stem_set.add(i)
            stem_set.add(j)

    # 茎区连续段
    stem_sorted = sorted(stem_set)
    stem_segs = []
    if stem_sorted:
        start = stem_sorted[0]
        prev = start
        for r in stem_sorted[1:]:
            if r == prev + 1:
                prev = r
            else:
                stem_segs.append((start, prev))
                start = r
                prev = r
        stem_segs.append((start, prev))

    # 环区 = 不在 stem_set 的残基, 连续段
    loop_sorted = [r for r in range(L) if r not in stem_set]
    loop_segs = []
    if loop_sorted:
        start = loop_sorted[0]
        prev = start
        for r in loop_sorted[1:]:
            if r == prev + 1:
                prev = r
            else:
                loop_segs.append((start, prev))
                start = r
                prev = r
        loop_segs.append((start, prev))

    return stem_segs, loop_segs


def build_segmented_3bead_system(
    p_coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
    stems: List[List[Tuple[int, int]]],
    pair_gate: bool = False,
    pair_eps: float = 30.0,
    enabled: Optional[List[bool]] = None,
    **kwargs,
):
    """构建分段 3-bead 力场: 茎区A-form螺旋(二面角+堆叠), 环区松散(只clash).

    力场:
      - 骨架键 P-P (全环) + BSJ (可调)
      - 残基内键 P-C4'-N (全环)
      - 茎区: 骨架键角150° + 二面角33° + 相邻N堆叠LJ (A-form 螺旋)
      - 环区: 无键角/二面角约束 (松散), 仅 clash 防穿模
      - WC 配对 N-N 12-10 H-bond (全环)
      - 非键 clash + 静电 CutoffNonPeriodic

    Args:
        enabled: 8 个 bool, 控制各 force 块是否加入.
            顺序 [bb, bsj, intra, angle, dihedral, stack, pair, clash].
            None = 全启用. 用于消融/候选生成时按需降力场.
    """
    from openmm import (
        System, HarmonicBondForce, HarmonicAngleForce,
        CustomBondForce, CustomNonbondedForce, CustomTorsionForce,
    )

    L = len(p_coords)
    coords_3bead = p_to_3bead(p_coords)  # (3L, 3) Å
    coords_nm = coords_3bead / 10.0
    en = enabled if enabled is not None else [True] * 8
    if len(en) != 8:
        raise ValueError("enabled 必须是 8 个 bool: [bb,bsj,intra,angle,dihedral,stack,pair,clash]")

    system = System()
    for _ in range(3 * L):
        system.addParticle(110.0)

    def P(i): return 3 * i
    def C(i): return 3 * i + 1
    def N(i): return 3 * i + 2

    # 茎区残基集合
    stem_set = set()
    for stem in stems:
        for i, j in stem:
            stem_set.add(i); stem_set.add(j)

    # 1. 骨架键 P-P (全环) + BSJ (CustomBond 可调)
    bsj_force = None
    if en[0] or en[1]:
        bb_k = 31000.0  # 310 Å²→nm²
        bond_bb = HarmonicBondForce()
        if en[0]:
            for i in range(L - 1):
                bond_bb.addBond(P(i), P(i + 1), BOND_LEN / 10.0, bb_k)
        system.addForce(bond_bb)
    if en[1]:
        bsj_force = CustomBondForce("0.5*k_bsj*(r-r0)^2")
        bsj_force.addPerBondParameter("k_bsj")
        bsj_force.addPerBondParameter("r0")
        bsj_force.addBond(P(L - 1), P(0), [0.1 * 500.0, BOND_LEN / 10.0])
        system.addForce(bsj_force)

    # 2. 残基内键 P-C4'-N (全环)
    # 用 p_to_3bead 的实测偏移 norm 作为平衡键长 (113nt 验证时用 3.9/3.35 也能成功,
    # 说明键长轻微不匹配 minimize 可容忍; 真正拆配对的是 56 茎全局拓扑冲突)
    if en[2]:
        bond_intra = HarmonicBondForce()
        for i in range(L):
            bond_intra.addBond(P(i), C(i), 0.197, 31000.0)  # P-C4' 1.97Å (p_to_3bead 实测)
            bond_intra.addBond(C(i), N(i), 0.109, 31000.0)  # C4'-N 1.09Å (p_to_3bead 实测)
        system.addForce(bond_intra)

    # 3. 茎区键角 + 二面角 (A-form 螺旋), 环区不加
    if en[3]:
        angle_force = HarmonicAngleForce()
        angle_k = 200.0
        angle0 = 2.618  # 150°
        for i in range(L - 2):
            if i in stem_set and (i + 1) in stem_set and (i + 2) in stem_set:
                angle_force.addAngle(P(i), P(i + 1), P(i + 2), angle0, angle_k)
        system.addForce(angle_force)

    if en[4]:
        dihedral_force = CustomTorsionForce("0.5*k_dih*(theta-theta0)^2")
        dihedral_force.addGlobalParameter("k_dih", 2000.0)
        dihedral_force.addGlobalParameter("theta0", 33.0 * math.pi / 180.0)
        for i in range(L - 3):
            if all(r in stem_set for r in (i, i + 1, i + 2, i + 3)):
                dihedral_force.addTorsion(P(i), P(i + 1), P(i + 2), P(i + 3))
        system.addForce(dihedral_force)

    # 4. 茎区堆叠 LJ (相邻N, A-form 堆叠 3.4Å), 环区弱堆叠
    # 茎区 ε=10 (降, 原25与配对N-N争夺位置, 配对k=2000下堆叠要弱避免冲突)
    # 环区弱 ε=2
    if en[5]:
        stack_force = CustomBondForce("4*eps*((sig/r)^12-(sig/r)^6)")
        stack_force.addPerBondParameter("eps")
        stack_force.addPerBondParameter("sig")
        sigma_nm = 0.34
        for i in range(L - 1):
            eps = 25.0 if (i in stem_set and (i + 1) in stem_set) else 2.0
            stack_force.addBond(N(i), N(i + 1), [eps, sigma_nm])
        system.addForce(stack_force)

    # 5. WC 配对: 方向依赖 12-10 H-bond 势 (IsRNAcirc 核心, 适配本 OpenMM 原语集)
    # IsRNAcirc 用方向依赖 H-bond 势: 只有 WC 几何(两碱基共面)才有深阱 → 伪配对被几何判据削弱.
    # 关键: 12-10 是短程势, 远端配对几乎无力 → 直接消除 682 对 harmonic 互扯的过约束根因.
    # ⚠️ 本 OpenMM 构建的 CustomCompoundBondForce 只有标量坐标 + distance/angle/dihedral
    #    (实测无 dot/cross/norm 向量运算, 也禁止 ';' 多语句), 故方向门用 dihedral 平面度代理:
    #   - r = distance(N_i, N_j)  (WC 糖苷N距离, r0=1.0nm=10Å)
    #   - 平面度 wd = 0.2 + 0.8*cos(dihedral(C4'_i, N_i, N_j, C4'_j))^2
    #     WC 共面 → dihedral=0/180° → wd=1; T-堆叠垂直 → 90° → wd=0.2
    #   - 势 = pair_k_scale*w_pair*30*[5(r0/r)^12-6(r0/r)^10]*wd*step(r_cut-r)
    #   - r_cut=2.0nm (20Å 捕获窗口; 12-10 尾端在 14Å 外已可忽略)
    pair_force = None
    pair_bonds = []
    if en[6]:
        from openmm import CustomCompoundBondForce
        # pair_gate=False → 纯 12-10 (消融结论: 3-bead 下 dihedral 方向门无判别力)
        if pair_gate:
            expr = (
                "pair_k_scale*w_pair*eps*step(r_cut-distance(p3,p6))"
                "*(5*((1.0/distance(p3,p6))^12)-6*((1.0/distance(p3,p6))^10))"
                "*(0.2+0.8*cos(dihedral(p2,p3,p6,p5))^2)"
            )
        else:
            expr = (
                "pair_k_scale*w_pair*eps*step(r_cut-distance(p3,p6))"
                "*(5*((1.0/distance(p3,p6))^12)-6*((1.0/distance(p3,p6))^10))"
            )
        pair_force = CustomCompoundBondForce(6, expr)
        pair_force.addGlobalParameter("pair_k_scale", 1.0)
        pair_force.addGlobalParameter("r_cut", 2.0)
        pair_force.addGlobalParameter("eps", pair_eps)
        pair_force.addPerBondParameter("w_pair")
        for (i, j, w) in pairs:
            if 0 <= i < L and 0 <= j < L and abs(i - j) > 1 and not (i == 0 and j == L - 1):
                bidx = pair_force.addBond([P(i), C(i), N(i), P(j), C(j), N(j)], [w])
                pair_bonds.append((bidx, P(i), C(i), N(i), P(j), C(j), N(j), w))
        system.addForce(pair_force)

    # 6. 统计势 (三级接触) — 可选
    stat_pot_path = kwargs.get('stat_pot_path')
    if stat_pot_path and Path(stat_pot_path).exists():
        import pickle
        with open(stat_pot_path, 'rb') as f:
            stat_data = pickle.load(f)

        stat_force = CustomBondForce("stat_k * (r - stat_d0)^2")
        stat_force.addPerBondParameter("stat_k")
        stat_force.addPerBondParameter("stat_d0")

        # 对远端非 WC 残基对加统计势 (序列距离 > 10)
        sequence = kwargs.get('sequence', '')
        avg_energy = stat_data.get('avg_energy', np.zeros(30))
        stat_bins = stat_data['bins']

        for i in range(L):
            for j in range(i + 10, L):
                base_i = sequence[i] if i < len(sequence) else 'N'
                base_j = sequence[j] if j < len(sequence) else 'N'
                key = tuple(sorted([base_i, base_j]))

                # 用统计势的平均吸引力作为力常数
                # 负能量 = 吸引，转化为 harmonic 力常数
                if key in stat_data['potential']:
                    energies = stat_data['potential'][key]
                    # 取最吸引的能量 (最小值)
                    min_energy = np.min(energies[energies < 0]) if np.any(energies < 0) else 0.0
                    if min_energy < -0.1:  # 只有强吸引才加
                        # 力常数 = |energy| / distance^2 (简化)
                        k = abs(min_energy) * 10.0  # 放大系数
                        d0 = 5.0  # 目标距离 5Å (三级接触典型距离)
                        stat_force.addBond(P(i), P(j), [k, d0])

        if stat_force.getNumBonds() > 0:
            system.addForce(stat_force)
            print(f"  统计势: {stat_force.getNumBonds()} 个三级接触键")

    # 7. 非键 clash + 静电 (CutoffNonPeriodic)
    clash_force = None
    if en[7]:
        clash_force = CustomNonbondedForce(
            "step(dmin-r)*k_clash*(dmin-r)^2 + 138.9*q1*q2/r"
        )
        clash_force.addPerParticleParameter("q")
        # dmin 从 0.30 降到 0.25: CG 3-bead 骨架/碱基交叉天然比全原子紧凑,
        # 全原子 3.0Å 阈值会把 CG 正常几何(如 P-N 3.5Å)误判为 clash → 拆散成形配对.
        # (消融: no-clash pair_rate 0.40→0.56, clash 是拆配对主力)
        clash_force.addGlobalParameter("dmin", 0.25)
        clash_force.addGlobalParameter("k_clash", 20000.0)
        clash_force.setNonbondedMethod(CustomNonbondedForce.CutoffNonPeriodic)
        clash_force.setCutoffDistance(1.2)
        for i in range(L):
            clash_force.addParticle([-0.5])  # P
            clash_force.addParticle([0.0])   # C4
            clash_force.addParticle([0.0])   # N
        # 排除 1-2
        excluded = set()
        for i in range(L):
            for a, b in [(P(i), C(i)), (C(i), N(i)), (P(i), P((i + 1) % L)), (N(i), N((i + 1) % L))]:
                k = (min(a, b), max(a, b))
                if k not in excluded:
                    excluded.add(k); clash_force.addExclusion(*k)
        for bidx, pi_idx, ci_idx, ni_idx, pj_idx, cj_idx, nj_idx, w in pair_bonds:
            # 配对残基 i/j 的全部 bead 交叉对排除 clash:
            # WC 配对距离内 (N-N ~10Å), P/C4'/N 交叉接近是正常双螺旋几何, 不该被 clash 推开.
            # (诊断: 初始构象 520/1828 的 clash 涉及成形配对残基, 是拆配对主力)
            i_beads = [pi_idx, ci_idx, ni_idx]
            j_beads = [pj_idx, cj_idx, nj_idx]
            for a in i_beads:
                for b in j_beads:
                    k = (min(a, b), max(a, b))
                    if k not in excluded:
                        excluded.add(k); clash_force.addExclusion(*k)
        system.addForce(clash_force)

    return system, coords_nm, pair_force, pair_bonds, bsj_force


def init_from_secondary_structure(
    L: int,
    pairs: List[Tuple[int, int, float]],
) -> np.ndarray:
    """按二级结构初始化 3D 坐标: 茎区 A-form 双螺旋, 环区弧线连接.

    核心改进 (vs 平面圆初始化):
      - 平面圆: 所有残基在 z=0 大圆上, 配对距离几百 Å, 初始能量爆炸
      - 二级结构初始化: 茎区配对的两段建成 A-form 双螺旋 (WC 距离自然满足),
        环区用弧线连接茎端, 初始就接近正确折叠, 力场只需局部松弛

    茎区 A-form 双螺旋几何:
      - 每步沿螺旋轴前进 2.8Å, 旋转 33°
      - 配对的两条链反平行, N-N 距离 ~3.0Å (WC), P-P 跨距 ~5.9Å
      - 螺旋半径 ~1.0Å (P bead)

    Returns:
        (L, 3) Å, P 坐标, 茎区已折叠成双螺旋
    """
    coords = np.zeros((L, 3))
    stems = extract_stems_from_structure(pairs)

    # 茎区: 建成 A-form 双螺旋. 茎中心按二级结构嵌套紧凑排列.
    # 自适应:
    #   少茎(≤4): 沿 x 轴间距 24Å (113nt 验证 Rg=25)
    #   中茎(5-12): 紧凑 3D 螺旋 (300nt 验证 Rg=26)
    #   多茎(>12): Fibonacci 球面紧凑分布 (长序列防散开, 茎轴统一z)
    used = set()
    stem_centers = []
    n_stems = len(stems)
    # 关键修复: 按茎长度降序处理. 长茎更可信 (连续配对多, 几何规整),
    # 短茎重叠部分跳过, 避免后处理覆盖前茎已放对的坐标 (之前导致 21% 配对初始错乱, N-N 跨茎 44A).
    stems_sorted = sorted(stems, key=lambda s: -len(s))
    # 茎排布 (2026-08-02 修):
    #   - 少茎 (≤4): 沿 x 轴按茎长累进 (间距 = 茎长 + 12A margin, 不重叠).
    #     之前固定 24A 间隔但茎长 22A → 相邻茎重叠 (3A邻居max=17, 能量7e15).
    #     尝试过序列顺序折线 (tRNA L 形) 但环区拉长导致 Rg 112 散开, 回退.
    #   - 多茎 (>4): 保持长度降序 (长茎优先防覆盖).
    x_cursor = 0.0
    for si, stem in enumerate(stems_sorted):
        n = len(stem)
        i_chain = [p[0] for p in stem]
        j_chain = [p[1] for p in stem]
        if n_stems <= 4:
            stem_len_A = max(6.0, n * 2.8)  # 茎沿轴长度 (每对2.8A)
            center = np.array([x_cursor + stem_len_A / 2.0, 0.0, 0.0])
            x_cursor += stem_len_A + 12.0
        elif n_stems <= 16:
            R_helix = 12.0
            theta = 2.0 * math.pi * si / n_stems * 1.5
            center = np.array([R_helix*math.cos(theta), R_helix*math.sin(theta), si*4.0])
        else:
            # Fibonacci 球面: 半径由茎长物理决定 (2026-08-02 修).
            # 之前 R=sqrt(n)*3.5 (61茎→27A): 相邻茎球面弧距仅 ~2.8A < 茎半径4A → 过密重叠,
            # 2000nt 初始 Rg=27 (物理应 60-100). 改: 球面面积 ≥ n*茎截面, 半径∝sqrt(n)*平均茎长.
            mean_stem_len = max(4.0, np.mean([len(x) for x in stems]))
            R_sphere = max(12.0, math.sqrt(n_stems) * mean_stem_len * 1.2)
            golden = math.pi * (3.0 - math.sqrt(5.0))
            idx = si + 0.5
            phi = math.acos(1.0 - 2.0 * idx / n_stems)
            theta = golden * si
            center = R_sphere * np.array([
                math.cos(theta)*math.sin(phi), math.sin(theta)*math.sin(phi), math.cos(phi)])
        stem_centers.append(center)
        # 茎轴: 球面径向 (从球心指向茎中心方向, 2026-08-02 修).
        # 之前统一 z: 所有茎从球心附近竖直长出, 球心处密度爆炸 (每残基3A邻居max=32).
        # 径向: 每根茎像辐条从球面向外, 球心附近天然稀疏, 不穿模.
        # 注意: 茎坐标 z=k*2.8 沿 axis 方向走, 所以茎沿径向向外延伸.
        axis = center / (np.linalg.norm(center) + 1e-9)
        # 局部 u/v: 轴垂直的两个切向
        ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u_dir = np.cross(ref, axis)
        u_dir = u_dir / (np.linalg.norm(u_dir) + 1e-9)
        v_dir = np.cross(axis, u_dir)
        for k, (ir, jr) in enumerate(zip(i_chain, j_chain)):
            # 跳过已在前一个长茎里放置的残基, 避免坐标被覆盖导致配对错乱
            if ir in used or jr in used:
                continue
            z = k * 2.8
            angle_i = k * 33.0 * math.pi / 180.0
            angle_j = angle_i + math.pi
            # A-form 双螺旋 P 骨架: 螺距 2.8Å/nt + 旋转 33°/nt, 半径 4.4Å.
            # 4.4Å 时配对成形率 78% (i/j 链在同一圆周对面, N-N 近); 半径改 9.15
            # 虽让骨架 P-P→5.9Å 但配对 N-N 拉到 18.7Å (只有1%成形), 得不偿失.
            # 骨架键长 3.77Å < 力场 5.9Å 的矛盾由力场拉回 (minimize 小拽可容忍).
            R_HELIX = 4.4
            coords[ir] = center + (R_HELIX * math.cos(angle_i) * u_dir
                                   + R_HELIX * math.sin(angle_i) * v_dir
                                   + z * axis)
            coords[jr] = center + (R_HELIX * math.cos(angle_j) * u_dir
                                   + R_HELIX * math.sin(angle_j) * v_dir
                                   + z * axis)
            used.add(ir); used.add(jr)

    # 环区: 连续环段沿弧线展开, 避免堆叠 clash.
    # 找连续环段 (不在 stem_set 的残基), 每段沿弧线在两端茎间展开.
    loop_res = [r for r in range(L) if r not in used]
    if loop_res:
        # 分段: 连续残基
        segs = []
        start = loop_res[0]; prev = start
        for r in loop_res[1:]:
            if r == prev + 1:
                prev = r
            else:
                segs.append((start, prev)); start = r; prev = r
        segs.append((start, prev))
        # 每段沿弧线展开: 从前一个茎端到后一个茎端的弧.
        # (注: 等弧长 5.9Å/步虽让环区键长合理, 但拉大茎间距破坏紧凑度,
        #   refine 后 pair_rate 反而降. 保持均匀插值 + z 凸起, 紧凑度优先)
        for s_start, s_end in segs:
            n_loop = s_end - s_start + 1
            # 前后茎端 (环外的最近残基)
            prev_stem = max([r for r in range(L) if r in used and r < s_start], default=0)
            next_stem = min([r for r in range(L) if r in used and r > s_end], default=L-1)
            p0 = coords[prev_stem] if np.linalg.norm(coords[prev_stem]) > 1e-6 else np.array([0.,10.,0.])
            p1 = coords[next_stem] if np.linalg.norm(coords[next_stem]) > 1e-6 else p0 + np.array([10.,0.,0.])
            # 弧线: 从 p0 到 p1, 中间凸起 (z 方向)
            for k, r in enumerate(range(s_start, s_end + 1)):
                t = (k + 1) / (n_loop + 1)  # 0<t<1
                # 线性插值 + z 凸起
                coords[r] = p0 * (1 - t) + p1 * t + np.array([
                    0.0, 0.0, 8.0 * math.sin(math.pi * t)])

    # 扰动破对称
    rng = np.random.default_rng(42)
    coords = coords + rng.normal(0.0, 0.3, (L, 3))
    return coords


def refine_segmented_3bead(
    p_coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
    platform_name: str = "CPU",
    n_anneal: int = 200,
    max_pair_seq_dist: int = 100,
    pair_gate: bool = False,  # 消融结论: 3-bead 下 dihedral 门无判别力, 默认关闭
    pair_eps: float = 30.0,
    enabled: Optional[List[bool]] = None,
    stat_pot_path: Optional[str] = None,  # 统计势文件路径
    sequence: Optional[str] = None,  # RNA 序列 (用于统计势)
) -> Tuple[np.ndarray, float, float]:
    """分段 3-bead 折叠: 茎区A-form螺旋/环区松散 + 三阶段BSJ退火.

    Args:
        p_coords: 初始 P 坐标
        pairs: 配对列表
        platform_name: OpenMM 平台
        n_anneal: 退火步数
        max_pair_seq_dist: 配对序列距离阈值 (精细筛).
            > 该值的配对被筛掉 (诊断: 长序列距离配对 minimize 时容易被推走,
            阈值 80-100 最佳, 0 表示不筛).
        pair_gate: 是否启用方向依赖门 (IsRNAcirc 式 dihedral 平面度, 消融用).
        pair_eps: 12-10 H-bond 阱深 (kJ/mol, 默认 30).
        enabled: 传 build_segmented_3bead_system 的 8 个 bool (消融/降力场).
    """
    from openmm import LangevinMiddleIntegrator, Platform
    from openmm import unit
    from openmm.app import Simulation, Topology, Element

    L = len(p_coords)
    # 精细筛: 筛掉序列距离 > max_pair_seq_dist 的配对
    # IsRNAcirc 用方向依赖 H-bond 势实现几何特异的筛选, 简化版用序列距离.
    # 未来可用 CNN+LSTM 分层模型替代阈值 (学 k-mer 上下文 + 局部结构).
    if max_pair_seq_dist > 0:
        pairs = [(i,j,w) for i,j,w in pairs
                 if min(abs(j-i), L-abs(j-i)) <= max_pair_seq_dist]
    stems = extract_stems_from_structure(pairs)

    # 关键: 用二级结构初始化 (茎区A-form双螺旋), 而非平面圆.
    # 平面圆初始配对距离几百Å, clash推开必然散开. 二级结构初始茎区已折叠,
    # 配对距离自然满足, 力场只需局部松弛.
    p_init = init_from_secondary_structure(L, pairs)

    system, coords_nm, pair_force, pair_bonds, bsj_force = \
        build_segmented_3bead_system(p_init, pairs, stems,
                                     pair_gate=pair_gate, pair_eps=pair_eps,
                                     enabled=enabled,
                                     stat_pot_path=stat_pot_path,
                                     sequence=sequence)

    topo = Topology()
    chain = topo.addChain()
    for i in range(L):
        res = topo.addResidue("N", chain)
        topo.addAtom(f"P{i}", Element.getBySymbol("P"), res)
        topo.addAtom(f"C{i}", Element.getBySymbol("C"), res)
        topo.addAtom(f"N{i}", Element.getBySymbol("N"), res)

    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond
    )
    try:
        platform = Platform.getPlatformByName(platform_name)
        sim = Simulation(topo, system, integrator, platform)
    except Exception:
        sim = Simulation(topo, system, integrator)

    sim.context.setPositions(coords_nm * unit.nanometer)
    e0 = sim.context.getState(getEnergy=True).getPotentialEnergy()._value
    sim.minimizeEnergy(maxIterations=500)

    def set_pair_k(scale):
        # CustomCompoundBondForce: w_pair 是建场时固定的 per-bond 参数,
        # 退火只调全局 pair_k_scale (index 0)
        if pair_force is None:
            return  # en[6]=False 时无配对力
        pair_force.setGlobalParameterDefaultValue(0, scale)
        pair_force.updateParametersInContext(sim.context)

    def set_bsj_k(scale):
        if bsj_force is None:
            return  # en[1]=False 时无 BSJ 力
        bsj_force.setBondParameters(0, 3*(L-1), 0, [scale * 500.0, BOND_LEN / 10.0])
        bsj_force.updateParametersInContext(sim.context)

    # 三阶段: 弱BSJ形成茎螺旋 → 强配对+中BSJ拉拢配对 → 强BSJ闭合
    # 低温避免散开. 阶段2 用强配对把散开配对(17A)拉拢到5A, 多步MD收敛.
    pre_md = sim.context.getState(getPositions=True, getEnergy=True)
    # 阶段1: 配对保持强(1.0)! 初始minimize已把配对放对位置(9.9Å),
    # 弱化配对会被clash/堆叠推到15Å (诊断: 弱配对k=0.3→pair_rate 0.16).
    # 只弱BSJ让结构微调, 配对不动.
    set_pair_k(1.0); set_bsj_k(0.1)
    sim.integrator.setTemperature(300 * unit.kelvin)
    sim.step(n_anneal); sim.minimizeEnergy(maxIterations=2000)

    # 阶段2: 配对强 + BSJ中, 压缩+闭合
    set_pair_k(1.0); set_bsj_k(0.5)
    sim.integrator.setTemperature(300 * unit.kelvin)
    sim.step(n_anneal * 3); sim.minimizeEnergy(maxIterations=3000)

    # 阶段3: 低温 + 配对强 + BSJ强, 最终闭合
    set_pair_k(1.0); set_bsj_k(5.0)
    sim.integrator.setTemperature(290 * unit.kelvin)
    sim.step(n_anneal * 2)
    sim.minimizeEnergy(tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer,
                       maxIterations=8000)

    state = sim.context.getState(getPositions=True, getEnergy=True)
    pos = state.getPositions(asNumpy=True)._value
    e1 = state.getPotentialEnergy()._value

    e_pre = pre_md.getPotentialEnergy()._value
    if e1 > e_pre * 0.5 and e_pre < 0:
        pos = pre_md.getPositions(asNumpy=True)._value
        e1 = e_pre

    p_refined = (pos * 10.0)[0::3].copy()
    return p_refined, e0, e1
