"""cg_forcefield.py — 3-bead/5-bead CG 力场 (IsRNAcirc 式, 真堆叠 LJ 核心)。

替代 refine.py 的 1-bead openmm_refine 力场。3-bead: P/C4'(S)/N1-N9(B1)。
核心改进: 真碱基堆叠 LJ (相邻 N bead 间), 让平面环产生立体螺旋。

力场项:
  1. 骨架键 (inter-nt): P[i]-P[i+1] harmonic r0=5.9Å + BSJ
  2. 残基内键 (intra-nt): P-C4' r0=3.9Å, C4'-N r0=3.35Å
  3. 骨架键角: P[i]-P[i+1]-P[i+2] (A-form 弯曲倾向)
  4. 真碱基堆叠 LJ: 相邻 N bead 间 12-6 LJ, σ=3.4Å ε=0.5 kcal/mol
  5. WC 配对: N[i]-N[j] harmonic r0≈5.0Å (P-N 距离, WC 几何)
  6. 非键 clash + 静电: CutoffNonPeriodic cutoff 12Å (策略1: O(N²)→O(N))

单位: 内部用 Å (跟 p_to_3bead.py / constraint_solver_amber.py 一致)。
OpenMM 期望 nm, 在 build_system 时转。
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from .p_to_3bead import p_to_3bead, split_3bead_coords

# ── 力场参数 (Å, kJ/mol) ──
# 实测自 1EHZ 模板 (aform_template.npz)
BOND_P_C4 = 3.90      # P-C4' 残基内键长
BOND_C4_N = 3.35      # C4'-N 残基内键长
BOND_P_NEXT = 5.90    # P[i]-P[i+1] 骨架键长 (CG, P-O3'-P 跨距)

# 力常数 (kJ/mol/Å², 借 amber14 OL3)
K_BOND_BACKBONE = 310.0   # P-P 骨架 (借 C-C)
K_BOND_INTRA = 310.0      # P-C4', C4'-N (借 C-C)
K_BOND_PAIR = 250.0        # WC 配对 N-N

# 真碱基堆叠 LJ (核心新项, 出 3D 的关键)
# 之前 ε=2.1 太弱, 被配对 harmonic 拉开。提到 8.0 (约 3kT, 强于热运动)。
# 同时配对改 flat-bottom 只排斥 (见下), 让堆叠主导形成局部螺旋。
STACK_SIGMA = 3.4         # Å, 碱基堆叠距离
STACK_EPSILON = 20.0      # kJ/mol (强, ~8kT, 才能压过环张力拉拢相邻N到3.4Å)

# 骨架键角 (A-form 弯曲倾向, ~3 点 P 角)
ANGLE_PPP = 2.618         # rad (150°, A-form 螺旋 3 点 P 弯曲角)
K_ANGLE = 200.0           # kJ/mol/rad² (提强, 让骨架有螺旋倾向)

# 骨架二面角 (A-form delta ~84° / 螺旋扭转 ~33°)
# 直接约束 P-P-P-P 二面角到 A-form 扭转, 让链形成右手螺旋
DIH_PPPP = 33.0 * math.pi / 180.0  # rad (A-form 扭转角)
K_DIHEDRAL = 2000.0       # kJ/mol/rad² (强, 才能驱动环扭转, 克服环张力)

# 非键
CLASH_DIST = 3.0          # Å, bead 间最小距离
CUTOFF = 12.0             # Å, 非键 cutoff (策略1)
K_CLASH = 200.0           # clash 力常数

# 静电 (P bead 负电, 互斥)
Q_P = -0.5                # e

# WC 配对 N-N 目标距离 (P-N 实测 ~5.0-5.4, 配对时 N 对 N 略近)
PAIR_N_N = 5.0            # Å


def p_coords_to_3bead(p_coords: np.ndarray) -> np.ndarray:
    """(L,3) P → (3L,3) [P,C4',N] per nt。复用 p_to_3bead。"""
    return p_to_3bead(p_coords)


def build_3bead_system(
    p_coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
    *,
    pair_scale: float = 1.0,
    bsj_k_scale: float = 1.0,
):
    """构建 3-bead CG OpenMM system + 初始坐标。

    Args:
        p_coords: (L,3) P 坐标 (Å)
        pairs: [(i,j,w)] ViennaRNA 配对
        pair_scale: 配对力常数缩放 (退火用: 弱化/恢复)
        bsj_k_scale: BSJ 闭合力常数缩放 (退火用: 初始弱让螺旋形成, 后期强闭合)

    Returns:
        (system, coords_3bead_nm, bond_force, pair_force, clash_force, stack_force)
        coords 单位 nm (OpenMM 用)。各 force 返回供退火调参。
    """
    from openmm import (
        System, HarmonicBondForce, HarmonicAngleForce,
        CustomBondForce, CustomNonbondedForce, CustomTorsionForce,
    )
    from openmm import unit

    L = len(p_coords)
    coords_3bead = p_coords_to_3bead(p_coords)  # (3L, 3) Å
    N_total = 3 * L
    coords_nm = coords_3bead / 10.0  # Å → nm

    system = System()
    for _ in range(N_total):
        system.addParticle(330.0 / 3.0)  # 每 bead ~110 Da (核苷酸 330 分 3 bead)

    # bead 索引: 残基 i 的 P=3i, C4=3i+1, N=3i+2
    def P(i): return 3 * i
    def C4(i): return 3 * i + 1
    def N(i): return 3 * i + 2

    # 1. 骨架键 P[i]-P[i+1] (inter-nt, 非BSJ)
    bond_backbone = HarmonicBondForce()
    bb_k = K_BOND_BACKBONE * 100.0  # Å²→nm² 转换: k_kJ/mol/Å² = k*100 kJ/mol/nm²
    for i in range(L - 1):
        bond_backbone.addBond(P(i), P(i + 1), BOND_P_NEXT / 10.0, bb_k)
    system.addForce(bond_backbone)

    # 1b. BSJ (首末 P), 用 CustomBondForce per-bond k 可调, 退火时动态加强闭合
    # 初始弱 (让二面角+堆叠形成螺旋, 不被环张力压扁), 退火后期强 (闭合环).
    bsj_force = CustomBondForce("0.5*k_bsj*(r-r0)^2")
    bsj_force.addPerBondParameter("k_bsj")
    bsj_force.addPerBondParameter("r0")
    bsj_force.addBond(P(L - 1), P(0), [bsj_k_scale * 500.0, BOND_P_NEXT / 10.0])
    system.addForce(bsj_force)

    # 2. 残基内键 P-C4', C4'-N (intra-nt)
    bond_intra = HarmonicBondForce()
    ik = K_BOND_INTRA * 100.0
    for i in range(L):
        bond_intra.addBond(P(i), C4(i), BOND_P_C4 / 10.0, ik)
        bond_intra.addBond(C4(i), N(i), BOND_C4_N / 10.0, ik)
    system.addForce(bond_intra)

    # 3. 骨架键角 P[i]-P[i+1]-P[i+2] (A-form 弯曲倾向)
    angle_force = HarmonicAngleForce()
    for i in range(L - 2):
        angle_force.addAngle(P(i), P(i + 1), P(i + 2), ANGLE_PPP, K_ANGLE)
    angle_force.addAngle(P(L - 2), P(L - 1), P(0), ANGLE_PPP, K_ANGLE)
    angle_force.addAngle(P(L - 1), P(0), P(1), ANGLE_PPP, K_ANGLE)
    system.addForce(angle_force)

    # 3.5 骨架二面角 P[i]-P[i+1]-P[i+2]-P[i+3] (A-form 右手螺旋扭转 ~33°)
    # 这是让链产生立体螺旋的关键约束: 二面角定义绕轴扭转, 形成螺旋而非平面.
    # CustomTorsionForce.theta 返回弧度, theta0 用弧度.
    dihedral_force = CustomTorsionForce("0.5*k_dih*(theta-theta0)^2")
    dihedral_force.addGlobalParameter("k_dih", K_DIHEDRAL)
    dihedral_force.addGlobalParameter("theta0", DIH_PPPP)
    for i in range(L - 3):
        dihedral_force.addTorsion(P(i), P(i + 1), P(i + 2), P(i + 3))
    # 环形闭合: 跨 BSJ 的二面角
    dihedral_force.addTorsion(P(L - 3), P(L - 2), P(L - 1), P(0))
    dihedral_force.addTorsion(P(L - 2), P(L - 1), P(0), P(1))
    dihedral_force.addTorsion(P(L - 1), P(0), P(1), P(2))
    system.addForce(dihedral_force)

    # 4. WC 配对 N[i]-N[j] — flat-bottom 只排斥 (过近才推, 远了不拉)
    # 之前用 harmonic 把结构压扁成2D。改 flat-bottom: 配对不主导折叠方向,
    # 只防止配对过近穿模, 让堆叠 LJ 主导形成局部螺旋。退火时 pair_scale 调排斥强度。
    pair_force = CustomBondForce("0.5*k_pair*step(r0-r)*(r-r0)^2")
    pair_force.addPerBondParameter("k_pair")
    pair_force.addPerBondParameter("r0")
    pair_bonds = []
    for (i, j, w) in pairs:
        if 0 <= i < L and 0 <= j < L and abs(i - j) > 1 and not (i == 0 and j == L - 1):
            bidx = pair_force.addBond(N(i), N(j),
                                      [K_BOND_PAIR * w * pair_scale, PAIR_N_N / 10.0])
            pair_bonds.append((bidx, N(i), N(j), w))
    system.addForce(pair_force)

    # 5. 真碱基堆叠 LJ (核心!) — 相邻 nt 的 N bead 间
    # σ=3.4Å, ε=2.1 kJ/mol。LJ 在 OpenMM 用 nm: σ_nm=0.34, ε 不变
    stack_force = CustomBondForce(
        "4*eps*((sig/r)^12 - (sig/r)^6)"
    )
    stack_force.addPerBondParameter("eps")
    stack_force.addPerBondParameter("sig")
    stack_bonds = []
    sigma_nm = STACK_SIGMA / 10.0
    for i in range(L - 1):
        bidx = stack_force.addBond(N(i), N(i + 1), [STACK_EPSILON, sigma_nm])
        stack_bonds.append((bidx, N(i), N(i + 1)))
    stack_force.addBond(N(L - 1), N(0), [STACK_EPSILON, sigma_nm])  # BSJ 堆叠
    stack_bonds.append((None, N(L - 1), N(0)))
    system.addForce(stack_force)

    # 6. 非键 clash + 静电 (CutoffNonPeriodic, 策略1: O(N²)→O(N))
    clash_force = CustomNonbondedForce(
        "step(dmin-r)*k_clash*(dmin-r)^2 + Coul*q1*q2/r"
    )
    clash_force.addPerParticleParameter("q")
    clash_force.addGlobalParameter("dmin", CLASH_DIST / 10.0)
    clash_force.addGlobalParameter("k_clash", K_CLASH * 100.0)
    clash_force.addGlobalParameter("Coul", 138.935456)  # kJ·nm/mol/e²
    clash_force.setNonbondedMethod(CustomNonbondedForce.CutoffNonPeriodic)
    clash_force.setCutoffDistance(CUTOFF / 10.0)
    # P bead 带负电, C4'/N 中性
    for i in range(L):
        clash_force.addParticle([Q_P])      # P
        clash_force.addParticle([0.0])       # C4'
        clash_force.addParticle([0.0])       # N
    # 排除 1-2 (残基内 + 骨架 + BSJ + 堆叠 + 配对)
    excluded = set()
    for i in range(L):
        for (a, b) in [(P(i), C4(i)), (C4(i), N(i))]:  # 残基内
            k = (min(a, b), max(a, b))
            if k not in excluded:
                excluded.add(k); clash_force.addExclusion(*k)
    for i in range(L - 1):
        k = (P(i), P(i + 1))
        if k not in excluded:
            excluded.add(k); clash_force.addExclusion(*k)
    k = (P(L - 1), P(0))
    if k not in excluded:
        excluded.add(k); clash_force.addExclusion(*k)
    for i in range(L - 1):
        k = (N(i), N(i + 1))
        if k not in excluded:
            excluded.add(k); clash_force.addExclusion(*k)
    k = (N(L - 1), N(0))
    if k not in excluded:
        excluded.add(k); clash_force.addExclusion(*k)
    for (bidx, ni, nj, w) in pair_bonds:
        k = (min(ni, nj), max(ni, nj))
        if k not in excluded:
            excluded.add(k); clash_force.addExclusion(*k)
    system.addForce(clash_force)

    return system, coords_nm, pair_force, pair_bonds, stack_force, stack_bonds, bsj_force


def refine_3bead(
    p_coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
    platform_name: str = "CPU",
    n_anneal: int = 200,
) -> Tuple[np.ndarray, float, float]:
    """3-bead CG MD unfolding-refolding 退火。

    Args:
        p_coords: (L,3) P 坐标 (Å)
        pairs: 配对
        platform_name: OpenMM 平台
        n_anneal: 退火步数 (每阶段)

    Returns:
        (refined_p_coords (L,3) Å, e0, e1)
        返回 P bead 坐标 (其它 bead 丢弃, 下游用 p_to_3bead 重建)
    """
    from openmm import LangevinMiddleIntegrator, Platform
    from openmm import unit
    from openmm.app import Simulation, Topology, Element

    L = len(p_coords)
    # 关键: 初始构象必须有足够大的 3D 扰动破对称。
    # 完美平面/直线上, 二面角约束的力矩为零 (三点共线), 约束无法启动扭转。
    # 必须初始就有非零二面角, 才能让二面角约束 + 堆叠 LJ 产生立体螺旋。
    # p_coords 若来自 solver (平面圆), 这里加 3Å 各向同性扰动破对称。
    p_init = p_coords.copy()
    rng = np.random.default_rng(42)
    p_init = p_init + rng.normal(0.0, 0.3, (L, 3))  # σ=3Å 破平面对称

    system, coords_nm, pair_force, pair_bonds, stack_force, stack_bonds, bsj_force = \
        build_3bead_system(p_init, pairs, pair_scale=1.0, bsj_k_scale=0.1)  # 初始弱BSJ

    # 拓扑 (3 bead/nt, 通用残基)
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

    sim.minimizeEnergy(tolerance=50.0 * unit.kilojoules_per_mole / unit.nanometer,
                       maxIterations=500)

    def set_pair_k(scale):
        for bidx, ni, nj, w in pair_bonds:
            pair_force.setBondParameters(
                bidx, ni, nj, [K_BOND_PAIR * w * scale, PAIR_N_N / 10.0])
        pair_force.updateParametersInContext(sim.context)

    def set_bsj_k(scale):
        """BSJ 闭合力常数缩放: 初始弱(让螺旋形成), 后期强(闭合环)."""
        bsj_force.setBondParameters(0, 3*(L-1), 0,
                                    [scale * 500.0, BOND_P_NEXT / 10.0])
        bsj_force.updateParametersInContext(sim.context)

    # ── 三阶段退火 (核心: 弱BSJ形成螺旋 → 中BSJ压缩 → 强BSJ闭合) ──
    # 关键: 不能用高温500K (会打散环, Rg暴增). 用中温 + 强二面角 + 强堆叠
    # 让螺旋逐步形成, BSJ 从弱→强逐步闭合.
    pre_md = sim.context.getState(getPositions=True, getEnergy=True)

    # 阶段1: 中温 + 弱配对 + 弱BSJ, 二面角+堆叠形成 A-form 螺旋 (不散开)
    set_pair_k(0.1); set_bsj_k(0.1)
    sim.integrator.setTemperature(350 * unit.kelvin)
    sim.step(n_anneal)
    sim.minimizeEnergy(maxIterations=1000)

    # 阶段2: 中温 + 强配对(拉拢WC) + 中BSJ, 配对拉拢+堆叠压缩环成紧凑3D
    set_pair_k(1.0); set_bsj_k(0.5)
    sim.integrator.setTemperature(320 * unit.kelvin)
    sim.step(n_anneal)
    sim.minimizeEnergy(maxIterations=1000)

    # 阶段3: 低温 + 强配对 + 强BSJ, 最终闭合环
    set_pair_k(1.0); set_bsj_k(3.0)
    sim.integrator.setTemperature(300 * unit.kelvin)
    sim.step(n_anneal)
    sim.minimizeEnergy(tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer,
                       maxIterations=3000)

    state = sim.context.getState(getPositions=True, getEnergy=True)
    pos = state.getPositions(asNumpy=True)._value  # nm
    e1 = state.getPotentialEnergy()._value

    # 安全网: MD 暴走回退
    e_pre = pre_md.getPotentialEnergy()._value
    if e1 > e_pre * 0.5 and e_pre < 0:
        pos = pre_md.getPositions(asNumpy=True)._value
        e1 = e_pre

    pos_ang = pos * 10.0  # nm → Å
    # 取 P bead (索引 0,3,6,...)
    p_refined = pos_ang[0::3].copy()
    return p_refined, e0, e1
