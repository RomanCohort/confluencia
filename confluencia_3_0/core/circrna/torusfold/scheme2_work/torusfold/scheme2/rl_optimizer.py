"""
rl_optimizer.py - RL 远端配对优化器 (MCTS + 策略网络)。

RL 不替代物理管线, 只补 circRNA 长程配对的局部最优盲区。在 CG 粒度
(P 坐标) 上用 MCTS 探索跳出局部解, 把远端配对拉拢到 WC 几何 (C1'-C1'
~10.5 Å), 再交给已有物理管线 (1EHZ 重建 + amber 精修) 收敛局部几何。

架构 (见 docs/scheme2_rl_design.md):
  - 状态: 远端配对块小图 (节点=茎块, 边=块间拓扑距离)
  - 策略网络: 3 层 GNN + 动作头 (π_block, π_dir, π_step)
  - 动作: (块索引, 6 方向, 3 步长) 离散
  - reward: Σ exp(-|d_C1'C1' - 20.0| / 2) over 远端配对
  - MCTS: policy 先验 + rollout 跑短 CG 精修评估

训练: PPO + GAE (training/ 单独脚本)。
推理: 加载权重, MCTS 搜索输出优化后 P 坐标。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import math

# torch 惰性导入 (rl_optimizer 可能在不训练时被 import, 避免强依赖)
_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


# ---------- 常量 ----------
# 动作空间 (离散)
N_DIRECTIONS = 6   # ±x ±y ±z
N_STEPS = 3        # 步长档: 0.5, 2.0, 5.0 Å
STEP_SIZES = (0.5, 2.0, 5.0)
DIRECTIONS = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1],
], dtype=np.float64)

# 旋转动作 (方案 B: 扩动作空间)
# dir_idx >= N_DIRECTIONS 时, 动作类型从"平移"切换为"旋转"。
# 旋转轴 = x / -x / y / -y / z / -z (对应 dir_idx=6,7,8,9,10,11)。
# 旋转角度: step_idx → (30°, 60°, 90°)
N_ROT_AXES = 6  # ±x ±y ±z 作为旋转轴
ROT_STEP_DEG = (30.0, 60.0, 90.0)

# dir_idx 含义:
#   0..5   -> 平移 (方向 = DIRECTIONS[dir_idx], 步长 = STEP_SIZES[step_idx])
#   6..11  -> 旋转 (轴   = ROT_AXES 方向, 角度 = ROT_STEP_DEG[step_idx])

WC_TARGET_DIST = 20.0  # Å, Watson-Crick C1'-C1' 目标距离 (L>300nt 远端配对 P-P 经验值)

# [v2] Per-pair target distance 映射: 序列距离 → 3D 目标距离
NEAR_SEQ_DIST = 100    # ≤100nt 算近程 (stem 内/局部茎-环)
FAR_SEQ_DIST = 100     # >100nt 算远端 (跨茎)
NEAR_TARGET_30 = 11.0  # 序列距离 ≤30nt: 目标 10-12Å (A-form P-P)
NEAR_TARGET_100 = 20.0 # 序列距离 30-100nt: 目标 15-25Å
FAR_PULL_FRAC = 0.6    # 远端: 目标 = 初始距离 × 0.6 (拉拢 40%)
R_NEAR_STAY_SIGMA = 5.0  # Å, 近程保持高斯 σ
R_FAR_PULL_WEIGHT = 1.0  # 远端拉拢权重


def compute_pair_targets(p_init: np.ndarray, far_pairs, L: int):
    """从初始构象估算每对远端配对的 3D 目标距离.

    映射规则:
      - 序列距离 ≤ 30nt (stem 内): 目标 10-12Å (A-form P-P)
      - 序列距离 30-100nt: 目标 15-25Å (局部茎-环)
      - 序列距离 > 100nt: 目标 = 初始距离 × 0.6 (拉拢 40%)
    Returns:
        targets: List[float], 与 far_pairs 一一对应
    """
    targets = []
    for (i, j) in far_pairs:
        seq_dist = min(abs(j - i), L - abs(j - i))
        d_init = float(np.linalg.norm(p_init[i] - p_init[j]))
        if seq_dist <= 30:
            targets.append(NEAR_TARGET_30)
        elif seq_dist <= NEAR_SEQ_DIST:
            targets.append(NEAR_TARGET_100)
        else:
            targets.append(d_init * FAR_PULL_FRAC)
    return targets


# ---------- 构象分布数据结构 (SamplingDesign 式概率空间输出) ----------
@dataclass
class ConformationSample:
    """单个 CG 构象采样 (MCTS 树节点)。

    SamplingDesign 等价: 从构象概率分布中采得的一个具体构象。
    与单点搜索不同, 多个 sample 共同定义整个构象分布。
    """
    p_coords: np.ndarray
    reward: float                    # 即时 reward (expansion 阶段评估)
    value: float = 0.0               # MCTS value (rollout/leaf 估值)
    visits: int = 1                  # MCTS visit count (概率权重的代理)
    action_path: List[Tuple[int, int, int]] = field(default_factory=list)
    # 两层 mutation 详细信息: [(bidx, type_id, dir_or_axis_id, sidx), ...]
    # type_id: 0=平移, 1=旋转
    action_detail: List[Tuple[int, int, int, int]] = field(default_factory=list)
    depth: int = 0                   # 在 MCTS 树中的深度 (0=根)

    def __post_init__(self):
        if not isinstance(self.action_path, list):
            self.action_path = []
        if not isinstance(self.action_detail, list):
            self.action_detail = []


@dataclass
class ConformationDistribution:
    """构象概率分布 (SamplingDesign 式的概率空间输出)。

    核心思想 (对标 SamplingDesign Nature Comms 2026):
      - 不用单一结构表征构象, 而是用一组 sample + 概率权重
      - 概率权重来自 MCTS visit count (类比: π(s) ∝ visit(s))
      - temperature 控制采样多样性: T→0 集中到最优构象, T→∞ 均匀采样
      - 支持从分布中采样 (MC 近似期望) 和加权汇总

    与 SamplingDesign 的对应:
      - "耦合变量分布"  →  块级耦合: 每个块的 (bidx, dir, step) 联合概率
      - "梯度下降塑造分布" → MCTS backprop 更新 visit (等效于 update π(s))
      - "MC 采样估期望"  → self.sample() + self.weighted_mean()
    """
    samples: List[ConformationSample] = field(default_factory=list)
    temperature: float = 1.0         # softmax 温度 (1.0=原始 visit, <1=集中, >1=分散)
    metadata: Dict = field(default_factory=dict)

    # ── 概率计算 ──
    @property
    def probabilities(self) -> np.ndarray:
        """从 visit counts 归一化得到采样概率 (temperature softmax)。"""
        if not self.samples:
            return np.array([])
        visits = np.array([s.visits for s in self.samples], dtype=np.float64)
        logits = np.log(visits + 1e-6) / max(self.temperature, 1e-6)
        logits = logits - logits.max()
        probs = np.exp(logits)
        total = probs.sum()
        if total > 0:
            probs = probs / total
        # 归一化后必须严格和为 1 (rng.choice 对浮点敏感, 1e-6 误差会崩).
        # clamp 到 [0,1] 后按比例微调最后一项, 保证 sum == 1.
        probs = np.clip(probs, 0.0, 1.0)
        probs = probs / probs.sum()
        return probs

    @property
    def entropy(self) -> float:
        """Shannon 熵 (nats), 衡量分布的分散程度。

        高熵 = 构象空间分散, MCTS 探索充分
        低熵 = 分布集中, MCTS 收敛到少数构象
        """
        probs = self.probabilities
        if len(probs) == 0:
            return 0.0
        mask = probs > 1e-8
        return float(-np.sum(probs[mask] * np.log(probs[mask])))

    @property
    def concentration(self) -> float:
        """分布集中度 (0~1): 最可能的样本概率。

        1.0 = 完全集中 (单一构象主导)
        ~0 = 均匀分散 (构象空间充分探索)
        """
        probs = self.probabilities
        if len(probs) == 0:
            return 0.0
        return float(probs.max())

    # ── 汇总统计 ──
    @property
    def mean_reward(self) -> float:
        """加权平均 reward (E[r] 的 MC 估计)。"""
        probs = self.probabilities
        if len(probs) == 0:
            return 0.0
        rewards = np.array([s.reward for s in self.samples])
        return float(np.sum(probs * rewards))

    @property
    def mean_value(self) -> float:
        """加权平均 MCTS value。"""
        probs = self.probabilities
        if len(probs) == 0:
            return 0.0
        values = np.array([s.value for s in self.samples])
        return float(np.sum(probs * values))

    @property
    def mode(self) -> Optional[ConformationSample]:
        """最可能的构象 (visit 最多的样本)。"""
        if not self.samples:
            return None
        return max(self.samples, key=lambda s: s.visits)

    @property
    def mode_coords(self) -> Optional[np.ndarray]:
        """最可能构象的 P 坐标 (向后兼容: 等价于旧版 MCTS 输出)。"""
        m = self.mode
        return m.p_coords if m is not None else None

    @property
    def best_coords(self) -> Optional[np.ndarray]:
        """reward 最高的构象 P 坐标.

        2026-08-02: dpo_simulate 下 mode (visit 最多) 会被 UCB 探索偏置污染
        (访问多但质量差). 用 reward 最高替代 — 这才是 MCTS 该输出的最优构象.
        """
        if not self.samples:
            return None
        best = max(self.samples, key=lambda s: s.reward)
        return best.p_coords

    # ── 采样 ──
    def sample(self, k: int = 1, rng: Optional[np.random.RandomState] = None) -> List[ConformationSample]:
        """从分布中采样 k 个构象 (MC 近似 E[...])。

        这是 SamplingDesign 核心思路的关键:
        "精确计算分布的期望不可行 → 用 MC 采样近似"。
        """
        if not self.samples:
            return []
        rng = rng or np.random.default_rng()
        probs = self.probabilities
        indices = rng.choice(len(self.samples), size=k, p=probs, replace=True)
        return [self.samples[int(i)] for i in indices]

    def weighted_mean_reward(self, reward_fn, *, max_samples: int = 1000,
                              rng: Optional[np.random.RandomState] = None) -> float:
        """用 MC 采样估计 reward_fn(self.p_coords) 的期望值。

        等价于 SamplingDesign 的:
        E[r] ≈ (1/k) Σ_{i=1}^{k} r(s_i), where s_i ~ π(s)
        """
        rng = rng or np.random.default_rng()
        samples = self.sample(max_samples, rng=rng)
        if not samples:
            return 0.0
        rewards = [reward_fn(s.p_coords) for s in samples]
        return float(np.mean(rewards))

    # ── 分布操作 ──
    def merge(self, other: "ConformationDistribution") -> "ConformationDistribution":
        """合并两个构象分布 (来自不同 MCTS run, 或不同 tier 的优化结果)。

        合并后 visit 相加 (等效于联合采样的累积权重)。
        """
        all_samples = list(self.samples) + list(other.samples)
        return ConformationDistribution(
            samples=all_samples,
            temperature=self.temperature,
            metadata={**self.metadata, **other.metadata},
        )

    def top_k(self, k: int) -> "ConformationDistribution":
        """按 visit 排序, 返回 top-k 构象分布。"""
        if not self.samples:
            return self
        sorted_samples = sorted(self.samples, key=lambda s: s.visits, reverse=True)
        return ConformationDistribution(
            samples=sorted_samples[:k],
            temperature=self.temperature,
            metadata={**self.metadata, "top_k": k},
        )

    def summary(self) -> str:
        """可读的分布摘要。"""
        if not self.samples:
            return "Empty distribution"
        return (f"ConformationDistribution(n={len(self.samples)}, "
                f"H={self.entropy:.2f}nats, "
                f"concentration={self.concentration:.3f}, "
                f"T={self.temperature}, "
                f"mean_reward={self.mean_reward:.4f})")

    # ── 块级耦合变量分布 (对标 SamplingDesign 的耦合变量) ──
    def block_action_distribution(self) -> Dict[int, np.ndarray]:
        """提取每个 block 的 (bidx, dir, step) 联合概率分布。

        等价于 SamplingDesign 的"每对碱基的耦合变量分布":
        - 每个 block 对应一对"碱基对"(位置 i 和 j)
        - 动作空间 (12 dir × 3 step = 36) 是块级"有效构象空间"
        - visit 加权给出每个动作的概率

        返回: {block_idx: (36,) probability array}
        """
        block_dist: Dict[int, np.ndarray] = {}
        if not self.samples:
            return block_dist
        probs = self.probabilities
        for si, s in enumerate(self.samples):
            # 两层 mutation: 优先用 action_detail (含 type_id),
            # 退化到扁平 action_path (从 didx 推断 type_id)
            if s.action_detail:
                actions = s.action_detail  # (bidx, type_id, dir_or_axis_id, sidx)
            else:
                actions = []
                for (b, didx, s2) in s.action_path:
                    # didx: 0-5=平移, 6-11=旋转
                    if didx < N_DIRECTIONS:
                        actions.append((b, 0, didx, s2))
                    else:
                        actions.append((b, 1, didx - N_DIRECTIONS, s2))
            for (bidx, type_id, dir_or_axis_id, sidx) in actions:
                # 两层 mutation 编码:
                # 平移 (type 0): action_idx = dir_or_axis_id * 3 + sidx  (0-17)
                # 旋转 (type 1): action_idx = 18 + dir_or_axis_id * 3 + sidx (18-35)
                if type_id == 0:
                    action_idx = dir_or_axis_id * N_STEPS + sidx
                else:
                    action_idx = (N_DIRECTIONS * N_STEPS) + dir_or_axis_id * N_STEPS + sidx
                if bidx not in block_dist:
                    block_dist[bidx] = np.zeros(
                        (N_DIRECTIONS + N_ROT_AXES) * N_STEPS, dtype=np.float64)
                block_dist[bidx][action_idx] += probs[si] * s.visits
        for bidx in block_dist:
            dist = block_dist[bidx]
            if dist.sum() > 0:
                block_dist[bidx] = dist / dist.sum()
        return block_dist

    def to_dict(self) -> dict:
        """序列化 (用于日志/可视化)。"""
        return {
            "n_samples": len(self.samples),
            "temperature": self.temperature,
            "entropy_nats": float(self.entropy),
            "concentration": float(self.concentration),
            "mean_reward": float(self.mean_reward),
            "mean_value": float(self.mean_value),
            "mode_reward": self.mode.reward if self.mode else None,
            "metadata": self.metadata,
            "sample_rewards": [s.reward for s in self.samples[:20]],
            "sample_visits": [s.visits for s in self.samples[:20]],
        }


# ---------- 状态表示 ----------
@dataclass
class BlockState:
    """一个远端配对茎块的状态。"""
    block_idx: int                 # 块在远端列表中的索引
    residues_i: List[int]          # 块内 i 侧残基索引
    residues_j: List[int]          # 块内 j 侧残基索引
    centroid_i: np.ndarray         # i 侧质心 P 坐标 (3,)
    centroid_j: np.ndarray         # j 侧质心 P 坐标 (3,)
    current_deviation: float       # 当前 C1'-C1' 平均偏差 (Å)
    tier: int = 2                  # DRAG 分层: 0=浅(<10Å), 1=中(10-20Å), 2=深(>20Å)


@dataclass
class RLOptimizerState:
    """RL 优化的完整状态。"""
    p_coords: np.ndarray            # (L, 3) CG P 坐标
    sequence: str
    far_blocks: List[BlockState]   # 远端配对块列表
    far_pairs: List[Tuple[int, int]]  # 远端配对 (i, j) 列表
    # 块间邻接 (稀疏): [(block_a, block_b, topo_dist), ...]
    block_edges: List[Tuple[int, int, float]] = field(default_factory=list)
    # coding mask: 透传给下游 amber 精修, coding 区残基钉死
    coding_mask: Optional[np.ndarray] = None  # shape (L,) bool, True=coding
    # DRAG 式拓扑区域: block_idx -> region_id (0, 1, 2)
    # 按环上质心位置分 3 段, 每个 region 是一组拓扑上"靠近"的远端块
    regions: Optional[Dict[int, int]] = None


def build_rl_state(
    p_coords: np.ndarray,
    sequence: str,
    far_pairs: List[Tuple[int, int]],
    stem_blocks: List[List[Tuple[int, int]]],
    coding_mask: Optional[np.ndarray] = None,
) -> RLOptimizerState:
    """从 CG P 坐标 + 远端配对 + 茎块构建 RL 状态。

    Args:
        p_coords: (L, 3) CG 求解输出的 P 坐标
        sequence: ACGU 字符串
        far_pairs: 远端配对 [(i, j), ...] (来自 pair_graph.far_end_pairs)
        stem_blocks: 茎块 [[(i, j), ...], ...] (来自 pair_graph.extract_stem_blocks)
        coding_mask: 可选 coding 标注 (来自 pair_graph.parse_case_annotation)
            透传给下游, 不影响 RL 动作空间 (RL 全序列可动)
    """
    # 筛出远端茎块 (块内配对都在 far_pairs 里)
    far_set = set((min(i, j), max(i, j)) for i, j in far_pairs)
    far_blocks: List[BlockState] = []
    for bidx, block in enumerate(stem_blocks):
        # 块内配对是否都在远端集
        in_far = all((min(i, j), max(i, j)) in far_set for i, j in block)
        if not in_far:
            continue
        res_i = [i for i, _ in block]
        res_j = [j for _, j in block]
        ci = p_coords[res_i].mean(axis=0)
        cj = p_coords[res_j].mean(axis=0)
        dev = float(np.mean([
            np.linalg.norm(p_coords[i] - p_coords[j]) for i, j in block
        ]))
        tier = 0 if dev < 10.0 else (1 if dev < 20.0 else 2)
        far_blocks.append(BlockState(
            block_idx=bidx, residues_i=res_i, residues_j=res_j,
            centroid_i=ci, centroid_j=cj, current_deviation=dev,
            tier=tier,
        ))

    # 外部 stem_blocks 可能混有 local pair (extract_stem_blocks 合并了所有配对),
    # 导致全部 in_far=False → n_blocks=0。退化: 直接从 far_pairs 聚类构建远端块。
    if not far_blocks and far_pairs:
        # 按 i 排序, 找连续 i 的段作为 block (与 extract_stem_blocks 相同的聚类逻辑,
        # 但只用 far_pairs, 不混 local pair)
        sorted_fp = sorted(far_pairs, key=lambda x: (min(x), max(x)))
        blocks_from_far: List[List[Tuple[int, int]]] = []
        current: List[Tuple[int, int]] = []
        for i, j in sorted_fp:
            pair = (i, j) if i < j else (j, i)
            if not current:
                current = [pair]
                continue
            # i 连续 (差 ≤ 2) 且 j 连续 → 同一 block
            prev = current[-1]
            if (abs(pair[0] - prev[0]) <= 2 and abs(pair[1] - prev[1]) <= 2) or \
               (abs(pair[0] - prev[1]) <= 2 and abs(pair[1] - prev[0]) <= 2):
                current.append(pair)
            else:
                blocks_from_far.append(current)
                current = [pair]
        if current:
            blocks_from_far.append(current)

        for bidx, block in enumerate(blocks_from_far):
            # 确保 block 内配对 i→j 方向一致 (i 递增, j 递减, 反向平行)
            sorted_block = sorted(block, key=lambda x: x[0])
            res_i = [b[0] for b in sorted_block]
            res_j = [b[1] for b in sorted_block]
            ci = p_coords[res_i].mean(axis=0)
            cj = p_coords[res_j].mean(axis=0)
            dev = float(np.mean([
                np.linalg.norm(p_coords[i] - p_coords[j]) for i, j in sorted_block
            ]))
            tier = 0 if dev < 10.0 else (1 if dev < 20.0 else 2)
            far_blocks.append(BlockState(
                block_idx=bidx, residues_i=res_i, residues_j=res_j,
                centroid_i=ci, centroid_j=cj, current_deviation=dev,
                tier=tier,
            ))

    # 块间邻接: 拓扑距离 < 100 的块对 (稀疏边)
    block_edges: List[Tuple[int, int, float]] = []
    for a in range(len(far_blocks)):
        for b in range(a + 1, len(far_blocks)):
            d_ij = np.linalg.norm(far_blocks[a].centroid_i - far_blocks[b].centroid_j)
            d_ji = np.linalg.norm(far_blocks[a].centroid_j - far_blocks[b].centroid_i)
            d = min(d_ij, d_ji)
            if d < 100.0:
                block_edges.append((a, b, float(d)))

    # DRAG 式拓扑区域: 按块内残基的环上中点位置分 3 段
    # 每个 region 是一组环上拓扑"靠近"的远端块, 降低长序列 RL 的 block 选择熵
    L = len(p_coords)
    n_blocks = len(far_blocks)
    regions: Dict[int, int] = {}
    if n_blocks > 0 and L > 0:
        # 每块的环上位置 = 块内所有配对残基的中点在环上的平均位置 ∈ [0, L/2]
        positions = []
        for b in far_blocks:
            pos = 0.0
            for i, j in zip(b.residues_i, b.residues_j):
                pos += min(abs(i - j), L - abs(i - j))
            pos /= max(len(b.residues_i), 1)
            positions.append(pos)

        if max(positions) - min(positions) > 1e-6:
            # 三等分环上位置 (DRAG 的 task division 简化版)
            lo, hi = min(positions), max(positions)
            step = (hi - lo) / 3.0
            for bidx, pos in enumerate(positions):
                rid = min(int((pos - lo) / step), 2)
                regions[b.block_idx] = rid
        else:
            # 退化: 所有块位置相同, 单区域
            for bidx, b in enumerate(far_blocks):
                regions[b.block_idx] = 0

    return RLOptimizerState(
        p_coords=p_coords, sequence=sequence,
        far_blocks=far_blocks, far_pairs=far_pairs,
        block_edges=block_edges, coding_mask=coding_mask,
        regions=regions,
    )


# ---------- Reward ----------
# 正则系数 (防作弊: 拉拢远端配对时不能撞原子/扭曲骨架)
# 实测 λ2=0.1 太强 (单残基平移破坏邻居骨架键, R_distort 暴增压过 R_pair,
# MCTS 不敢动)。降到 0.01 让 R_pair 主导, 正则只在严重扭曲时介入。
LAMBDA_CLASH = 0.05   # 非键 P-P 太近惩罚
LAMBDA_DISTORT = 0.01  # 相邻 P-P 偏离 5.9Å 惩罚 (弱, 不压过 R_pair)
CLASH_THRESH = 3.0    # P-P < 此值算位障 (CG 粒度近似)
BOND_LEN_CG = 5.9     # CG 相邻 P-P 目标距离
BOND_TOL = 1.0        # 相邻 P-P 偏离 5.9±1.0 算扭曲
LAMBDA_CLOSURE = 0.10  # BSJ 闭合惩罚 (独立于骨架扭曲, 更高权重)
LAMBDA_COMPACT = 0.15  # 紧凑度奖励 (长序列核心, 提强让RL积极拉拢分散茎)

# cgRNASP 全局 CG 质量项 (FebRNA/BIophys J 2022)
# 用 long-range term (skip E_bonded, 因 P-only→3-bead 转换的 bonded 不准)
# 系数用 -1.0 (cgRNASP score 越低越好, 我们取负数让 reward 越大越好)
# scale=0.01 让量级跟 R_pair 接近 (long-range term ~50-100, scale 后 ~0.5-1.0)
LAMBDA_CGRNAS = 0.01   # cgRNASP long-range term 权重

# 结构匹配距离改进量 (DRAG-style R_improve)
# 配对比初始状态靠近时给奖励, 让 agent 能"感知距离在缩小"
# 只在 distance > improvement_threshold 时才计 (避免噪声)
DIST_IMPROVE_THRESH = 2.0  # 至少靠近 2Å 才计 reward
DIST_IMPROVE_SCALE = 20.0  # 每 20Å 改善给 1.0 reward (量级与 R_pair 相当)
LAMBDA_IMPROVE = 0.5       # R_improve 权重 (中等, 不压过 R_pair)


def _cgRNAS_score(p_coords: np.ndarray, sequence: Optional[str] = None) -> Optional[float]:
    """P-only CG → 3-bead → cgRNASP 评分 (lazy import, 失败时返回 None)。

    sequence: 实际 RNA 序列。若 None 则用 AUGC 循环占位 (非推荐, 因 cgRNASP
    对序列敏感, atom type 完全依赖序列中的碱基类型)。

    带坐标级缓存 (LRU, 16 条), 键 = (序列hash, 四舍五入到 0.01Å 的坐标元组)。
    避免 MCTS 中对同一/相近坐标重复调用 score_cgrnas。
    """
    try:
        from .p_to_3bead import p_to_3bead
        from .cgRNASP import load_all_potentials, score_cgrnas
    except ImportError:
        return None

    L = len(p_coords)
    seq = sequence or "AUGC" * (L // 4) + "A" * (L % 4)

    # 缓存键: (序列hash, 坐标) — 同一坐标但不同序列给出不同结果
    cache_key = (hash(seq), tuple(round(float(x), 2) for x in p_coords.ravel()))
    _cgRNAS_cache_lock.acquire()
    try:
        cache = _cgRNAS_cache
        if cache_key in cache:
            return cache[cache_key]
    finally:
        _cgRNAS_cache_lock.release()

    try:
        coords_3bead = p_to_3bead(p_coords)
        pots = load_all_potentials()
        total, breakdown = score_cgrnas(coords_3bead, seq, pots)
        E_long_range = (
            breakdown["E_0_1"]
            + breakdown["E_1_2"]
            + 6.0 * breakdown["E_2_4"]
            + 8.0 * breakdown["E_long"]
        )
        result = float(E_long_range)
    except Exception:
        return None

    _cgRNAS_cache_lock.acquire()
    try:
        _cgRNAS_cache[cache_key] = result
        # LRU: 超过 16 条时删最老
        if len(_cgRNAS_cache) > 16:
            first_key = next(iter(_cgRNAS_cache))
            del _cgRNAS_cache[first_key]
    finally:
        _cgRNAS_cache_lock.release()

    return result


_cgRNAS_cache: dict = {}
from threading import Lock
_cgRNAS_cache_lock = Lock()


def _cgRNAS_score_cached(p_coords: np.ndarray, sequence: str | None) -> Optional[float]:
    """带缓存的 cgRNAS 评分, 同一坐标不重复计算。

    长序列(>800nt)跳过: cgRNASP 对长序列 O(L²) 评分太慢, MCTS 每节点算会拖死 RL.
    长序列用纯物理 reward (配对/clash/闭合/紧凑), 不依赖统计势.
    """
    try:
        from .p_to_3bead import p_to_3bead
        from .cgRNASP import load_all_potentials, score_cgrnas

        L = len(p_coords)
        if L > 800:
            return None  # 长序列跳过 cgRNASP
        coords_3bead = p_to_3bead(p_coords)
        pots = load_all_potentials()
        total, breakdown = score_cgrnas(coords_3bead, sequence or "AUGC" * (L // 4), pots)

        E_long_range = (
            breakdown["E_0_1"]
            + breakdown["E_1_2"]
            + 6.0 * breakdown["E_2_4"]
            + 8.0 * breakdown["E_long"]
        )
        return float(E_long_range)
    except Exception:
        return None

# DRAG 复合 reward 参数 (Briefings in Bioinformatics, DRAG)
# 配对满意度 bonus: 远端配对距离 < WC_TARGET_DIST - SAT_MARGIN 时激活,
# 奖励 agent 把配对"真正拉拢到位"而非仅靠近。
# 量级设计: LAMBDA_SAT=1.0 让单个满足配对的 bonus 接近 R_pair 中单对的量级
# (exp(-dev/2) ∈ [0.18, 1.0] when dev ∈ [2.5, 0]), 在 RL 搜索中有效。
LAMBDA_SAT = 1.0       # 配对满意度 bonus 权重
SAT_MARGIN = 5.0       # 低于 WC_TARGET_DIST - SAT_MARGIN 算"已满足"

# ---------- DPO policy 全局注册 (RiboPO 式构象质量打分器) ----------
# DPO 训练的 policy (train_dpo.py) 学到了"保配对 + far 拉近"的状态价值。
# 通过 set_dpo_policy() 注册后, compute_reward 可加一个 DPO V 分量,
# 让 MCTS 探索时偏好 DPO 认为高质量 (far 拉近 + 保配对) 的构象。
# 默认不启用 (dpo_weight=0), 显式传 dpo_weight>0 才开, 避免每次节点评估都跑 GNN。
_DPO_POLICY = None  # 全局 DPO PolicyNetwork 实例 (已 eval)


def set_dpo_policy(policy) -> None:
    """注册 DPO policy 为全局构象质量打分器. 传 None 清除."""
    global _DPO_POLICY
    _DPO_POLICY = policy
    if policy is not None:
        policy.eval()


def get_dpo_policy():
    return _DPO_POLICY


def _dpo_value(p_coords: np.ndarray, far_pairs: List[Tuple[int, int]],
               sequence: Optional[str] = None) -> Optional[float]:
    """用全局 DPO policy 对构象打分 (V 值).

    从 far_pairs 直接构造 far_blocks (每个 far 配对一个 block), 建
    RLOptimizerState 跑 policy.forward(return_value=True) 取 V.
    无法打分 (无 policy / 无 far_pairs / forward 失败) 时返回 None.

    V 值绝对量级大 (-1000~100), 不能直接加进 reward. 调用方需归一化.
    """
    global _DPO_POLICY
    if _DPO_POLICY is None or not far_pairs:
        return None
    try:
        L = len(p_coords)
        far_blocks = []
        for bi, (i, j) in enumerate(far_pairs):
            if i >= L or j >= L:
                continue
            dev = float(np.linalg.norm(p_coords[i] - p_coords[j]))
            far_blocks.append(BlockState(
                block_idx=bi, residues_i=[i], residues_j=[j],
                centroid_i=p_coords[i], centroid_j=p_coords[j],
                current_deviation=dev,
                tier=0 if dev < 10.0 else (1 if dev < 20.0 else 2),
            ))
        if not far_blocks:
            return None
        st = RLOptimizerState(
            p_coords=p_coords,
            sequence=sequence or "AUGC" * (L // 4 + 1),
            far_blocks=far_blocks, far_pairs=[(b.residues_i[0], b.residues_j[0])
                                              for b in far_blocks],
            block_edges=[], coding_mask=None,
        )
        out = _DPO_POLICY.forward(st, return_value=True)
        if out[0] is None:
            return None
        v = float(out[-1].detach().squeeze())
        # ── 防塌缩物理惩罚 (2026-08-02): DPO V 只学"保近程+far拉近", 没学 clash.
        #    实测 dpo_simulate 把 2001nt 拉成 Rg=15.9A 穿模球 (每残基3A邻居16.8, 正常<8).
        #    加两项廉价惩罚 (O(L), 采样) 挡塌缩:
        #    1) P-P clash: P-P<3A 对数 (惩罚穿模)
        #    2) Rg 下限: 只挡"完全塌缩", 不挡合理拉拢 (2026-08-02 修).
        #       circRNA 远配拉拢后 Rg 40-50 合理 (初始化 Rg~50). 之前下限 63
        #       (紧密球) 太高, 挡住拉拢 (50→40 也触发惩罚).
        #       现在下限 0.45*sqrt(L)*3 (约 30Å, 只挡完全挤成一团).
        try:
            import math as _m
            # 1. clash (采样, 避免 O(L^2))
            L = len(p_coords)
            idx = np.linspace(0, L - 1, min(600, L)).astype(int)
            sub = p_coords[idx]
            from scipy.spatial import cKDTree
            tree = cKDTree(sub)
            clash_cnt = 0
            for pt in sub:
                clash_cnt += len(tree.query_ball_point(pt, 3.0)) - 1
            # 2. Rg 下限 (只挡完全塌缩 ~30A, 允许 40-50 合理拉拢)
            c = p_coords - p_coords.mean(0)
            rg = float(np.sqrt((c ** 2).sum(1).mean()))
            rg_min = 30.0
            penalty = 5.0 * clash_cnt + max(0.0, rg_min - rg) * 10.0
            v = v - penalty
        except Exception:
            pass
        return v
    except Exception:
        return None





def pull_far_pairs(
    p_coords: np.ndarray,
    far_pairs: List[Tuple[int, int]],
    target: float = 15.0,
    max_iter: int = 30,
    step_per_iter: float = 1.0,
) -> np.ndarray:
    """far 配对残基定向拉拢 (后处理, 不在 MCTS 动作空间内).

    2026-08-02: MCTS 块级移动与 far 拉拢目标不匹配 (MCTS 改善仅 19). 手动
    验证 (R_pair -41→+19, Rg 稳 56 不塌缩) 证明 far 配对残基定向拉拢有效.
    作为 MCTS 后的确定性物理操作: 移 far 配对残基 + 邻居保持骨架刚体.

    Args:
        p_coords: (L,3) P 坐标
        far_pairs: far 配对列表
        target: N-N 目标距离 (Å)
        max_iter: 最大迭代步数
        step_per_iter: 每步最大位移 (Å)

    Returns:
        拉拢后的 P 坐标 (L,3)
    """
    p = p_coords.copy()
    L = len(p)
    from torusfold.scheme2.cg_forcefield import p_coords_to_3bead
    far_list = list(far_pairs)
    for _ in range(max_iter):
        c3 = p_coords_to_3bead(p)
        N = c3[2::3]
        all_close = True
        for i, j in far_list:
            if i >= L or j >= L:
                continue
            d = float(np.linalg.norm(N[i] - N[j]))
            if d <= target:
                continue
            all_close = False
            dirv = (N[j] - N[i]) / (d + 1e-9)
            step = min(step_per_iter, (d - target) * 0.15)
            # 移 i 及邻居 (骨架刚体) 朝 +dirv (朝向 j)
            for r in (i - 1, i, i + 1):
                if 0 <= r < L:
                    p[r] += dirv * step
            # 移 j 及邻居 朝 -dirv (朝向 i)
            for r in (j - 1, j, j + 1):
                if 0 <= r < L:
                    p[r] -= dirv * step
        if all_close:
            break
    return p


def compute_reward(
    p_coords: np.ndarray,
    far_pairs: List[Tuple[int, int]],
    *,
    use_regularization: bool = True,
    target_dists: Optional[List[float]] = None,
    initial_p_coords: Optional[np.ndarray] = None,
    sequence: Optional[str] = None,
    dpo_weight: float = 0.0,
    p_init: Optional[np.ndarray] = None,
) -> float:
    """新架构 reward: 近远分治 + 方向梯度 + 自适应紧凑度.

    R = R_near_stay + R_far_pull + R_improve
      - λ_clash·R_clash - λ_distort·R_distort - λ_closure·R_closure
      + r_compact + r_cgrnas + r_dpo

    设计原则:
      1. Per-pair 目标距离: 从 p_init 的环上距离推导 (compute_pair_targets)
      2. Near/Far 分治: 近程 (≤100nt) 奖励保持, 远端 (>100nt) 奖励拉拢
      3. 方向梯度: R_far_pull 含 d_now>d_init 负奖励 (远离有惩罚)
      4. 紧凑度自适应: 目标 Rg 从配对几何推导, 不用 sqrt(L)
      5. R_improve 永远可用: p_init 从参数或 initial_p_coords 获取

    Args:
        p_init: 初始 P 坐标 (新参数, 优先级高于 initial_p_coords 用于 per-pair target)
    """
    L = len(p_coords)
    if not far_pairs:
        return 0.0

    # ── Per-pair 目标距离 ──
    if target_dists is None:
        _ref = p_init if p_init is not None else initial_p_coords
        if _ref is not None:
            target_dists = compute_pair_targets(_ref, far_pairs, L)
        else:
            target_dists = [WC_TARGET_DIST] * len(far_pairs)

    # ── R_near_stay + R_far_pull (近远分治) ──
    r_near = 0.0   # 近程配对保持
    r_far = 0.0    # 远端配对拉拢
    _ref = p_init if p_init is not None else initial_p_coords
    for k, (i, j) in enumerate(far_pairs):
        d_now = float(np.linalg.norm(p_coords[i] - p_coords[j]))
        seq_dist = min(abs(j - i), L - abs(j - i))
        target = target_dists[k]

        if seq_dist <= NEAR_SEQ_DIST:
            # 近程: 高斯保持在初始距离附近 (允许热运动)
            d_init = float(np.linalg.norm(_ref[i] - _ref[j])) if _ref is not None else target
            r_near += math.exp(-((d_now - d_init) / R_NEAR_STAY_SIGMA) ** 2)
        else:
            # 远端: 朝目标拉拢 + 方向梯度
            d_init = float(np.linalg.norm(_ref[i] - _ref[j])) if _ref is not None else d_now
            if d_init > target + 1.0:
                # 正在拉拢: (d_init - d_now) / (d_init - target), cap at 1.0
                progress = min(1.0, (d_init - d_now) / (d_init - target))
                r_far += progress
            elif d_now > d_init + 1.0:
                # 拉远了: 负奖励 (惩罚远离)
                r_far -= (d_now - d_init) / 20.0
            # d_init ≈ d_target → 0 (已在目标)

    # ── R_improve (DRAG 式, 永远可用) ──
    r_improve = 0.0
    _ref2 = p_init if p_init is not None else initial_p_coords
    if _ref2 is not None:
        for k, (i, j) in enumerate(far_pairs):
            d_now = float(np.linalg.norm(p_coords[i] - p_coords[j]))
            d_init = float(np.linalg.norm(_ref2[i] - _ref2[j]))
            improvement = d_init - d_now
            if improvement > DIST_IMPROVE_THRESH:
                r_improve += improvement / DIST_IMPROVE_SCALE
            elif improvement < -DIST_IMPROVE_THRESH:
                r_improve += improvement / (DIST_IMPROVE_SCALE * 2.0)  # 惩罚减半

    # 合并配对 reward
    r_pair = (R_FAR_PULL_WEIGHT * r_far
              + (1.0 / max(len(far_pairs), 1)) * r_near
              + LAMBDA_IMPROVE * r_improve)

    if not use_regularization or L < 3:
        return float(r_pair)

    # ── R_clash: 非键 P-P 太近 (向量化) ──
    r_clash = 0.0
    far_residues_set: Set[int] = set()
    far_pair_set: Set[Tuple[int, int]] = set()
    for (i, j) in far_pairs:
        far_residues_set.add(i); far_residues_set.add(j)
        far_pair_set.add((min(i, j), max(i, j)))
    far_indices = np.array(sorted(far_residues_set), dtype=np.int32)
    far_N = len(far_indices)

    if far_N > 0:
        far_p = p_coords[far_indices]
        dists = np.linalg.norm(far_p[:, np.newaxis, :] - p_coords[np.newaxis, :, :], axis=2)
        exclude = np.zeros((far_N, L), dtype=bool)
        for ri, r in enumerate(far_indices):
            exclude[ri, r] = True
            if r > 0: exclude[ri, r - 1] = True
            if r < L - 1: exclude[ri, r + 1] = True
            for (pi, pj) in far_pair_set:
                if pi == r: exclude[ri, pj] = True
                elif pj == r: exclude[ri, pi] = True
        valid_d = dists[~exclude]
        close = valid_d[valid_d < CLASH_THRESH]
        r_clash = float(np.sum(CLASH_THRESH - close))

    # ── R_distort: 相邻 P-P 偏离 5.9Å ──
    r_distort = 0.0
    adj_d = np.linalg.norm(np.diff(p_coords, axis=0), axis=1)
    dev_bond = np.abs(adj_d - BOND_LEN_CG)
    excess = dev_bond[dev_bond > BOND_TOL]
    r_distort += float(np.sum(excess - BOND_TOL))

    # ── R_closure: BSJ 闭合 ──
    d_bsj = float(np.linalg.norm(p_coords[0] - p_coords[-1]))
    r_closure = d_bsj

    # ── R_cgrnas: cgRNASP long-range ──
    r_cgrnas = 0.0
    cgrnas_val = _cgRNAS_score(p_coords, sequence)
    if cgrnas_val is not None:
        r_cgrnas = -LAMBDA_CGRNAS * cgrnas_val

    # ── R_compact: 自适应目标 Rg (从配对几何推导) ──
    r_compact = 0.0
    if L > 3:
        rg_now = float(np.sqrt(((p_coords - p_coords.mean(0)) ** 2).sum(1).mean()))
        # 目标 Rg: 从配对的平均序列距离推导 (非 sqrt(L) 经验公式)
        if far_pairs:
            avg_seq_dist = float(np.mean([min(abs(j-i), L-abs(j-i)) for i,j in far_pairs]))
            target_rg = avg_seq_dist * 0.15  # 经验系数
        else:
            target_rg = math.sqrt(L) * 1.0  # fallback
        scale = max(target_rg * 0.3, 5.0)  # 容忍 ±30%, 最小 5Å
        r_compact = LAMBDA_COMPACT * math.exp(-((rg_now - target_rg) / scale) ** 2) * (L / 100.0)

    total = (r_pair - LAMBDA_CLASH * r_clash - LAMBDA_DISTORT * r_distort
             - LAMBDA_CLOSURE * r_closure + r_cgrnas + r_compact)

    # ── R_dpo: DPO policy 构象质量 ──
    if dpo_weight > 0:
        v_dpo = _dpo_value(p_coords, far_pairs, sequence=sequence)
        if v_dpo is not None:
            total += dpo_weight * (v_dpo + 800.0) / 100.0
    return float(total)


# ---------- 策略网络 ----------
# 导入 torch 作为基类 (PolicyNetwork 继承 torch.nn.Module 支持 GPU)
try:
    import torch as _torch_import
except ImportError:
    raise ImportError("torch 未安装: pip install torch")


class PolicyNetwork(_torch_import.nn.Module):
    """块 GNN 策略网络 (torch, 手写消息传递, 不依赖 torch_geometric)。

    输入: RLOptimizerState
    输出: π_block (softmax over 块), π_dir (12: 6平移+6旋转), π_step (3)

    架构: 块节点特征 -> node_enc -> K 层消息传递 (block_edges 邻接) ->
          块嵌入 -> 3 个动作头 + 拓扑条件化

    消息传递 (GCN 式): h_i <- ReLU(W·h_i + W·Σ_{j∈N(i)} h_j / |N(i)|)
    边来自 state.block_edges (块间质心距<100 的稀疏邻接)。

    支持 GPU: 继承 torch.nn.Module, policy.to(device) 即可。
    """
    def __init__(self, hidden_dim: int = 128, n_mp_layers: int = 3):
        torch = _get_torch()
        torch.nn.Module.__init__(self)
        self.hidden_dim = hidden_dim
        self.n_mp_layers = n_mp_layers
        # 节点特征维度: [block_len, centroid_i(3), centroid_j(3), deviation,
        #                mean_pos(3)] = 11
        self.node_feat_dim = 11
        # 节点编码 (特征 -> hidden)
        self.node_enc = torch.nn.Sequential(
            torch.nn.Linear(self.node_feat_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        # 消息传递层 (每层一个 Linear, 残差连接)
        self.mp_layers = torch.nn.ModuleList([
            torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(n_mp_layers)
        ])
        # DRAG 式两层 block 选择: 先选拓扑区域, 再选区域内块
        self.head_region = torch.nn.Linear(hidden_dim, 3)  # 3 个拓扑区域
        # 动作头
        self.head_block = torch.nn.Linear(hidden_dim, 1)  # 每块打分, softmax over 块 (区域内)
        # ---- 两层 mutation 空间 (方案 #2: location + mutation) ----
        # 第一层: mutation type (平移 / 旋转)
        self.head_type = torch.nn.Linear(hidden_dim, 2)  # 2: [平移, 旋转]
        # 第二层: 条件动作参数
        self.head_trans_dir = torch.nn.Linear(hidden_dim, N_DIRECTIONS)  # 6: 平移方向
        self.head_rot_axis = torch.nn.Linear(hidden_dim, N_ROT_AXES)     # 6: 旋转轴
        self.head_step = torch.nn.Linear(hidden_dim, N_STEPS)            # 3: 步长/角度
        # value 头 (PPO GAE 用, 输出标量 V(s))。从块均值嵌入出, 代表整图状态价值。
        self.head_value = torch.nn.Linear(hidden_dim + 8, 1)  # +8: 拓扑向量拼接
        self.softmax = torch.nn.Softmax(dim=-1)
        # ---- 方案 C: 全局拓扑条件化 ----
        self.topo_dim = 8
        self.topo_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.topo_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, hidden_dim),
            torch.nn.ReLU(),
        )  # 8维拓扑向量 -> hidden_dim 嵌入

    def _message_passing(self, h, edge_index, edge_weight):
        """K 层消息传递。h: (n, hidden), edge_index: (2, E) tensor。

        GCN 式归一化聚合, 边权 = 1/(1+d) (块间质心越近影响越大)。
        无邻居的孤立节点只过自身变换 (保留信息)。
        """
        torch = _get_torch()
        n = h.shape[0]
        for layer in self.mp_layers:
            # 聚合邻居 (scatter_add, 边权加权)
            if edge_index is not None and edge_index.shape[1] > 0:
                src, dst = edge_index[0], edge_index[1]
                # 每个目标节点收到的加权消息
                agg = torch.zeros_like(h)
                msg = h[src] * edge_weight.unsqueeze(-1)
                agg = agg.index_add(0, dst, msg)
                # 按度归一化 (加 1 防零, 自身算一个邻居)
                deg = torch.zeros(n, dtype=h.dtype, device=h.device)
                deg = deg.index_add(0, dst, edge_weight)
                agg = agg / (deg + 1.0).unsqueeze(-1)
                h_new = torch.relu(layer(h + agg))
            else:
                # 无边: 只过自身变换 (退化为 MLP, 保留旧路径)
                h_new = torch.relu(layer(h))
            h = h_new + h  # 残差
        return h

    def _edges_to_tensor(self, state: RLOptimizerState):
        """把 state.block_edges 转成 (edge_index, edge_weight) tensor。

        block_edges: [(a, b, topo_dist), ...] (无向, 存一份, 消息传递时双向)
        返回 edge_index (2, 2E) 双向, edge_weight (2E,) = 1/(1+d)。
        所有 tensor 落在 self.device 上 (支持 GPU)。
        """
        torch = _get_torch()
        if not state.block_edges:
            return None, None
        src_l, dst_l, w_l = [], [], []
        for a, b, d in state.block_edges:
            w = 1.0 / (1.0 + d)
            src_l.extend([a, b])
            dst_l.extend([b, a])
            w_l.extend([w, w])
        edge_index = torch.tensor([src_l, dst_l], dtype=torch.long,
                                  device=self.device)
        edge_weight = torch.tensor(w_l, dtype=torch.float32,
                                   device=self.device)
        return edge_index, edge_weight

    @staticmethod
    def _global_topology(state: RLOptimizerState) -> np.ndarray:
        """从 block 间图结构提取全局拓扑向量 (条件输入)。

        8 维, 编码整张块图的全局形状, 策略网络据此判断
        "这个远端结构是多重茎嵌套还是简单发夹"。
        """
        blocks = state.far_blocks
        n_blocks = len(blocks)
        if n_blocks == 0:
            return np.zeros(8, dtype=np.float32)

        # 1-2: 块数 + 平均块大小
        avg_block_size = float(np.mean([len(b.residues_i) for b in blocks]))
        max_block_size = float(np.max([len(b.residues_i) for b in blocks]))

        # 3-4: 块间质心距离分布 (近/远)
        if state.block_edges:
            edge_dists = np.array([e[2] for e in state.block_edges], dtype=np.float32)
            mean_edge_dist = float(edge_dists.mean())
            max_edge_dist = float(edge_dists.max())
        else:
            mean_edge_dist = 0.0
            max_edge_dist = 0.0

        # 5-6: 所有块质心空间的跨度 (整体结构的"大小")
        all_c_list = []
        for b in blocks:
            all_c_list.append(b.centroid_i.reshape(1, 3))
            all_c_list.append(b.centroid_j.reshape(1, 3))
        all_c = np.concatenate(all_c_list, axis=0)
        span = float(np.linalg.norm(all_c.max(axis=0) - all_c.min(axis=0)))
        mean_pair_dist = float(np.mean([b.current_deviation for b in blocks]))

        # 7-8: 偏置 (所有偏差之和 / 偏差平方和) — 整体"离目标有多远"
        bias_sum = float(np.sum([abs(b.current_deviation - WC_TARGET_DIST)
                                  for b in blocks]))
        bias_var = float(np.sum([(abs(b.current_deviation - WC_TARGET_DIST)
                                   / (WC_TARGET_DIST + 1e-9)) ** 2 for b in blocks]))

        return np.array([
            n_blocks, avg_block_size,
            mean_edge_dist, max_edge_dist,
            span, mean_pair_dist,
            bias_sum, bias_var,
        ], dtype=np.float32)

    def _embed(self, state: RLOptimizerState):
        """共享嵌入: 状态 -> 块嵌入 h (n_blocks, hidden)。forward/value 共用。

        同时返回全局拓扑向量 (方案 C: 拓扑条件化策略)。
        所有 tensor 落在 self.device 上 (支持 GPU)。
        """
        torch = _get_torch()
        if not state.far_blocks:
            return None, None

        feats = []
        for b in state.far_blocks:
            f = np.concatenate([
                [len(b.residues_i)],
                b.centroid_i, b.centroid_j,
                [b.current_deviation],
                (b.centroid_i + b.centroid_j) / 2.0,
            ]).astype(np.float32)
            feats.append(f)
        x = torch.tensor(np.stack(feats), dtype=torch.float32,
                         device=self.device)
        h = self.node_enc(x)
        edge_index, edge_weight = self._edges_to_tensor(state)
        h = self._message_passing(h, edge_index, edge_weight)

        # 全局拓扑向量 (条件化 dir/step 头用)
        topo = torch.tensor(self._global_topology(state),
                            dtype=torch.float32, device=self.device)
        return h, topo

    def forward(self, state: RLOptimizerState, *, return_value: bool = False):
        """返回 (π_region, π_block, π_type, π_trans_dir, π_rot_axis, π_step[, V])。

        DRAG 式两层 block 选择 + 两层 mutation 空间:
          π_region:    3 维 softmax → 选拓扑区域
          π_block:     n_blocks 维 softmax → 选区域内块
          π_type:      2 维 softmax → [平移, 旋转] (mutation type)
          π_trans_dir: 6 维 softmax → 平移方向 (仅当 type=平移)
          π_rot_axis:  6 维 softmax → 旋转轴 (仅当 type=旋转)
          π_step:      3 维 softmax → 步长/角度

        两层 mutation 的分布论依据 (对标 SamplingDesign):
          不是对所有动作做扁平 softmax, 而是先选类型再条件选参数,
          显式建模"平移 vs 旋转"这个结构决策, 让分布更稀疏、更可解释。

        return_value=False (推理默认): 六元组。
        return_value=True (训练用): 七元组, 多一个标量 V(s)。

        方案 C: π_type/π_trans_dir/π_rot_axis/π_step 使用全局拓扑向量条件化,
        使策略知道"这是一个多重茎嵌套结构"还是"简单发夹"。
        """
        torch = _get_torch()
        h, topo = self._embed(state)
        if h is None:
            return (None, None, None, None, None, None, None) if return_value else (None, None, None, None, None, None)

        # π_region: 每个 block 对 3 区域的打分, 取 max 后 softmax
        region_scores = self.head_region(h)  # (n_blocks, 3)
        region_logit = region_scores.sum(dim=0, keepdim=True).squeeze(0)  # (3,)
        pi_region = self.softmax(region_logit)

        # π_block: 每块打分后 softmax (区域内)
        block_scores = self.head_block(h).squeeze(-1)  # (n_blocks,)
        pi_block = self.softmax(block_scores)

        # 全局嵌入: 块均值 + 拓扑嵌入 (方案 C)
        h_mean = h.mean(dim=0, keepdim=True)  # (1, hidden)
        topo_embed = self.topo_encoder(topo).unsqueeze(0)  # (1, hidden)
        h_global = h_mean + topo_embed

        # ---- 两层 mutation ----
        # 第一层: type (平移=0, 旋转=1)
        pi_type = self.softmax(self.head_type(h_global)).squeeze(0)
        # 第二层: 条件动作参数 (两者都输出, 采样时只取对应 type 的那个)
        pi_trans_dir = self.softmax(self.head_trans_dir(h_global)).squeeze(0)
        pi_rot_axis = self.softmax(self.head_rot_axis(h_global)).squeeze(0)
        pi_step = self.softmax(self.head_step(h_global)).squeeze(0)

        if return_value:
            v_input = torch.cat([h_mean, topo.unsqueeze(0)], dim=-1)
            v = self.head_value(v_input).squeeze(0).squeeze(-1)
            return pi_region, pi_block, pi_type, pi_trans_dir, pi_rot_axis, pi_step, v
        return pi_region, pi_block, pi_type, pi_trans_dir, pi_rot_axis, pi_step

    def value(self, state: RLOptimizerState):
        """单独算 V(s) (GAE bootstrap 用)。"""
        torch = _get_torch()
        h, topo = self._embed(state)
        if h is None:
            return None
        h_mean = h.mean(dim=0, keepdim=True)
        v_input = torch.cat([h_mean, topo.unsqueeze(0)], dim=-1)
        return self.head_value(v_input).squeeze(0).squeeze(-1)

    @property
    def device(self):
        """当前模型所在设备 (从参数权重自动推断, 支持 .to('cuda'))。"""
        return self.node_enc.weight.device

    def parameters(self):
        torch = _get_torch()
        # ⚠️ 修 bug: 之前引用了不存在的 head_dir (旧版头), 遗漏了 head_type/
        #   head_trans_dir/head_rot_axis (forward 实际用的头). 直接列全所有头.
        params = (list(self.node_enc.parameters()) +
                  list(self.mp_layers.parameters()) +
                  list(self.topo_encoder.parameters()) +
                  list(self.head_region.parameters()) +
                  list(self.head_block.parameters()) +
                  list(self.head_type.parameters()) +
                  list(self.head_trans_dir.parameters()) +
                  list(self.head_rot_axis.parameters()) +
                  list(self.head_step.parameters()) +
                  list(self.head_value.parameters()))
        return params

    @property
    def device(self):
        """从第一个参数推断设备, 跟随 .to(device) / .cuda()。"""
        return next(super().parameters()).device

    def save(self, path: str):
        torch = _get_torch()
        sd = {
            "node_enc": self.node_enc.state_dict(),
            "mp_layers": self.mp_layers.state_dict(),
            "topo_encoder": self.topo_encoder.state_dict(),
            "head_region": self.head_region.state_dict(),
            "head_block": self.head_block.state_dict(),
            # ⚠️ 修 bug: 之前存 head_dir (不存在的废弃头), 遗漏 forward 实际用的头.
            # 全部存上, 保证 load 后能完整推理.
            "head_type": self.head_type.state_dict(),
            "head_trans_dir": self.head_trans_dir.state_dict(),
            "head_rot_axis": self.head_rot_axis.state_dict(),
            "head_step": self.head_step.state_dict(),
            "head_value": self.head_value.state_dict(),
            "hidden_dim": self.hidden_dim,
            "n_mp_layers": self.n_mp_layers,
        }
        torch.save(sd, path)

    def load(self, path: str):
        torch = _get_torch()
        sd = torch.load(path, map_location="cpu", weights_only=False)
        self.hidden_dim = sd["hidden_dim"]
        self.n_mp_layers = sd["n_mp_layers"]
        # 兼容旧权重 (无 mp_layers): 重建空 ModuleList, 消息传递退化为自身变换
        if "mp_layers" in sd:
            self.mp_layers = torch.nn.ModuleList([
                torch.nn.Linear(self.hidden_dim, self.hidden_dim)
                for _ in range(self.n_mp_layers)
            ])
            self.mp_layers.load_state_dict(sd["mp_layers"])
        self.node_enc.load_state_dict(sd["node_enc"])
        # 兼容旧权重 (无 topo_encoder, 方案 C 前): 随机初始化, 训练前不影响推理
        if "topo_encoder" in sd:
            self.topo_encoder.load_state_dict(sd["topo_encoder"])
        # 兼容旧权重 (无 head_region, DRAG 分层前): 随机初始化
        if "head_region" in sd:
            self.head_region.load_state_dict(sd["head_region"])
        self.head_block.load_state_dict(sd["head_block"])
        self.head_step.load_state_dict(sd["head_step"])
        # ⚠️ 修 bug: 之前 load head_dir (不存在的废弃头). 改 load forward 实际用的头.
        # 兼容旧权重 (无新头): 保持随机初始化, 训练/DPO 前不影响.
        for hname in ("head_type", "head_trans_dir", "head_rot_axis"):
            if hname in sd:
                getattr(self, hname).load_state_dict(sd[hname])
        # 兼容旧权重 (无 head_value): 随机初始化, 训练前不影响推理
        if "head_value" in sd:
            self.head_value.load_state_dict(sd["head_value"])


# ---------- 动作执行 ----------
def _rotate_about_axis(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """罗德里格旋转: v 绕 axis 旋转 angle_rad。axis 不必归一化。"""
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    v = np.asarray(v, dtype=np.float64)
    cross = np.cross(axis, v)
    return c * v + s * cross + (1.0 - c) * np.dot(axis, v) * axis


def apply_action(
    state: RLOptimizerState,
    block_idx: int,
    dir_idx: int,
    step_idx: int,
) -> np.ndarray:
    """执行动作 (方案 B: 平移 + 旋转)。

    dir_idx 0..5  -> 平移 i 侧残基 (方向 = DIRECTIONS[dir_idx], 步长 Å)
    dir_idx 6..11 -> 绕 i→j 垂直轴旋转 i 侧残基 (角度 = ROT_STEP_DEG[step_idx])
                     旋转轴由 _rotation_axis 启发式选 (最对齐 i→j 的方向)。
    step_idx 0,1,2 -> 平移步长 0.5/2.0/5.0 Å 或 旋转角度 30°/60°/90°

    只动 i 侧: 配对距离 = |P[i] - P[j]|, 移动 i 会改变这个距离。
    旧版 i/j 同向平移, 相对距离不变 (bug, 已修)。
    """
    new_p = state.p_coords.copy()
    b = state.far_blocks[block_idx]
    step = STEP_SIZES[step_idx]

    if dir_idx < N_DIRECTIONS:
        # ---- 平移 ----
        direction = DIRECTIONS[dir_idx]
        delta = direction * step
        for r in b.residues_i:
            new_p[r] = new_p[r] + delta
    else:
        # ---- 旋转 ----
        rot_axis = _rotation_axis(state, block_idx, dir_idx)
        angle_rad = np.deg2rad(ROT_STEP_DEG[step_idx])
        c_i = b.centroid_i  # 旋转中心 = i 侧质心
        for r in b.residues_i:
            new_p[r] = _rotate_about_axis(new_p[r] - c_i, rot_axis, angle_rad) + c_i

    return new_p


def _rotation_axis(
    state: RLOptimizerState, block_idx: int, dir_idx: int
) -> np.ndarray:
    """选旋转轴。

    dir_idx >= N_DIRECTIONS 时, 动作类型 = 旋转。
    从 6 个可能轴中, 选能让 i→j 向量更对齐当前 i→j 期望方向的那个:
    即旋转后 v_rot 与 v 夹角最小 (最自然地把 i 侧朝 j 侧方向推)。

    简化: 直接返回 i→j 方向的垂直轴 (使 i 侧绕此轴旋转能扫过 j 侧)。
    """
    b = state.far_blocks[block_idx]
    v = b.centroid_j - b.centroid_i  # i 侧到 j 侧的期望方向
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-6:
        # 退化: i 侧质心 ≈ j 侧质心, 选任意垂直轴
        return np.array([0.0, 0.0, 1.0])

    # 选与 v 最垂直的主轴 (dot ≈ 0), 避免沿 v 方向旋转 (无效)
    main_axes = DIRECTIONS[:3]  # +x, -x, +y, +z 的前3
    dots = np.abs(main_axes @ v / v_norm)
    best_axis = int(np.argmin(dots))
    return DIRECTIONS[best_axis]


# ---------- MCTS ----------
@dataclass
class MCTSNode:
    """MCTS 搜索节点。"""
    p_coords: np.ndarray
    reward: float
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    # 传统扁平动作 (bidx, didx, sidx) — apply_action 用
    action_taken: Optional[Tuple[int, int, int]] = None
    # 两层 mutation 详细信息 (bidx, type_id, dir_or_axis_id, sidx)
    # type_id: 0=平移, 1=旋转
    action_detail: Optional[Tuple[int, int, int, int]] = None


class MCTS:
    """Monte Carlo Tree Search with policy prior.

    策略网络给先验概率, Simulation 阶段可选:
      - use_rollout=False (先验版): 叶节点估值直接用当前 reward (快, 但短视)
      - use_rollout=True  (默认): 叶节点后再走 rollout_depth 步启发式 rollout,
        用终点 reward 估值 (多看几步, 评估更准但慢 rollout_depth 倍)
    rollout 用启发式 (偏差大的块优先, 朝 j 侧方向拉), 不用 policy (policy 是
    待训练对象, 训练前不能用来评估自己, 否则 reward 信号有偏)。
    """
    def __init__(
        self,
        policy: Optional[PolicyNetwork] = None,
        c_puct: float = 1.5,
        n_simulations: int = 50,
        rollout_depth: int = 5,
        use_rollout: bool = True,
        dpo_weight: float = 0.0,
        dpo_rollout: bool = False,
        dpo_simulate: bool = False,
    ):
        self.policy = policy
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.rollout_depth = rollout_depth
        self.use_rollout = use_rollout
        self.dpo_weight = dpo_weight
        # dpo_rollout: rollout 的 5 步估值用 DPO V (3ms) 替代完整 compute_reward (1295ms).
        # DPO V 与 compute_reward 排序对齐 (Spearman 0.73, top-5 全一致), 见 2026-08-02.
        # 节约: 每模拟省 rollout_depth*(1295-3) ms, n_sim=15 省 ~97s (400x).
        # 节点最终 reward 仍用完整 compute_reward (保物理正则), 只换 rollout 内层.
        self.dpo_rollout = dpo_rollout
        # dpo_simulate: 整棵 MCTS 树的节点 reward 全用 DPO V (3ms), 不用完整 compute_reward.
        # 注意: UCB exploit=value/visits 要求树内 reward 同量级可比, 所以必须"全 DPO"
        # 或"全完整", 不能混合 (混合破坏 value 比较). dpo_simulate=True 隐含 dpo_rollout.
        # 每模拟从 1295*(1+rollout_depth) ms → 3*(1+rollout_depth) ms (430x), n_sim 可开大
        # 100 倍同算力. 最终构象仍由 optimize_far_pairs 用完整 compute_reward 评估.
        self.dpo_simulate = dpo_simulate
        if dpo_simulate:
            self.dpo_rollout = True

    def _heuristic_action(
        self,
        state: RLOptimizerState,
        far_pairs: List[Tuple[int, int]],
    ) -> Tuple[int, int, int]:
        """启发式选动作 (rollout 和无策略 fallback 共用)。

        块: 偏差大的块概率高 (softmax over deviation);
        方向: 块 i 侧朝 j 侧的向量量化到 6 方向, 70% 选它 30% 随机;
        动作类型: 70% 平移 / 30% 旋转 (方案 B: 扩动作空间);
        步长/角度: 偏差大用大步/大角度, 偏小用小步/小角度。
        """
        n_blocks = len(state.far_blocks)
        deviations = [abs(b.current_deviation - WC_TARGET_DIST) for b in state.far_blocks]
        probs = np.array(deviations) + 1e-6
        probs = probs / probs.sum()
        bidx = int(np.random.choice(n_blocks, p=probs))
        selected = state.far_blocks[bidx]
        target_dir = selected.centroid_j - selected.centroid_i
        norm = np.linalg.norm(target_dir)

        # 动作类型: 70% 平移 / 30% 旋转
        use_rotation = np.random.random() < 0.30

        if norm > 1e-6:
            target_dir = target_dir / norm
            dots = DIRECTIONS @ target_dir
            best_dir = int(np.argmax(dots))
            didx = (best_dir if np.random.random() < 0.7
                    else int(np.random.randint(N_DIRECTIONS)))
        else:
            didx = int(np.random.randint(N_DIRECTIONS))

        if use_rotation:
            didx = didx + N_DIRECTIONS  # 切到旋转区间 [6, 11]

        dev = abs(selected.current_deviation - WC_TARGET_DIST)
        if dev > 15:
            sidx = 2
        elif dev > 5:
            sidx = 1
        else:
            sidx = 0
        return bidx, didx, sidx

    def _rollout(
        self,
        state: RLOptimizerState,
        far_pairs: List[Tuple[int, int]],
        *,
        initial_p_coords: Optional[np.ndarray] = None,
        sequence: Optional[str] = None,
    ) -> float:
        """从叶节点启发式走 rollout_depth 步, 返回终点 reward。

        纯 numpy (不建树), 速度快。中间状态用 _rebuild_blocks 更新质心/偏差。
        initial_p_coords/sequence 透传给 compute_reward (R_improve / cgRNASP)。
        """
        p = state.p_coords.copy()
        # dpo_rollout=True 时用 DPO V (3ms) 替代完整 compute_reward (1295ms) 做 rollout 估值.
        # DPO V 与 compute_reward 排序对齐 (Spearman 0.73, top-5 全一致), 引导方向正确.
        # rollout 只是探索引导, 最终节点 reward 仍用完整 compute_reward (保物理).
        if self.dpo_rollout:
            for _ in range(self.rollout_depth):
                tmp = RLOptimizerState(
                    p_coords=p, sequence=state.sequence,
                    far_blocks=_rebuild_blocks(state, p),
                    far_pairs=far_pairs, block_edges=state.block_edges,
                )
                bidx, didx, sidx = self._heuristic_action(tmp, far_pairs)
                p = apply_action(tmp, bidx, didx, sidx)
            # 与 dpo_simulate 一致用 compute_reward (R_pair 驱动拉拢), 树内同量级.
            return compute_reward(p, far_pairs,
                                  initial_p_coords=initial_p_coords,
                                  sequence=sequence, dpo_weight=self.dpo_weight,
                                  p_init=initial_p_coords)
        for _ in range(self.rollout_depth):
            tmp = RLOptimizerState(
                p_coords=p, sequence=state.sequence,
                far_blocks=_rebuild_blocks(state, p),
                far_pairs=far_pairs, block_edges=state.block_edges,
            )
            bidx, didx, sidx = self._heuristic_action(tmp, far_pairs)
            p = apply_action(tmp, bidx, didx, sidx)
        return compute_reward(p, far_pairs,
                              initial_p_coords=initial_p_coords,
                              sequence=sequence, dpo_weight=self.dpo_weight,
                              p_init=initial_p_coords)


    def search(
        self,
        state: RLOptimizerState,
        far_pairs: List[Tuple[int, int]],
        *,
        initial_p_coords: Optional[np.ndarray] = None,
        sequence: Optional[str] = None,
    ) -> np.ndarray:
        """MCTS 搜索, 返回 reward 最高的 P 坐标。

        Selection 用 UCB1 (含 policy prior), Expansion 每次加一个子节点,
        Simulation 用当前 reward 直接评估 (no rollout, 先验版),
        Backprop 沿父链更新 visit/value。

        initial_p_coords: 初始 P 坐标 (R_improve 用)。None 时不计距离改进量。
        sequence: RNA 序列 (cgRNASP 评分用)。None 时退化为 AUGC 占位序列。
        """
        if self.dpo_simulate:
            # dpo_simulate 模式 (2026-08-02 修): root reward 也用 compute_reward
            # (R_pair 驱动拉拢 + clash 防塌缩), 与子节点同量级可比.
            root_reward = compute_reward(state.p_coords, far_pairs,
                                         initial_p_coords=initial_p_coords,
                                         sequence=sequence, dpo_weight=self.dpo_weight,
                                         p_init=initial_p_coords)
        else:
            root_reward = compute_reward(state.p_coords, far_pairs,
                                         initial_p_coords=initial_p_coords,
                                         sequence=sequence, dpo_weight=self.dpo_weight,
                                         p_init=initial_p_coords)
        root = MCTSNode(p_coords=state.p_coords, reward=root_reward)
        best = root

        n_blocks = len(state.far_blocks)
        if n_blocks == 0:
            return state.p_coords

        for sim in range(self.n_simulations):
            # --- Selection: 沿树下行, UCB1 选子节点 ---
            node = root
            cur_p = state.p_coords.copy()
            while node.children:
                # UCB1 = value/visits + c_puct * prior * sqrt(ln(parent_visits)/visits)
                best_child = None
                best_ucb = -np.inf
                for c in node.children:
                    if c.visits == 0:
                        ucb = np.inf
                    else:
                        exploit = c.value / c.visits
                        explore = self.c_puct * np.sqrt(
                            np.log(node.visits + 1) / c.visits
                        )
                        ucb = exploit + explore
                    if ucb > best_ucb:
                        best_ucb = ucb
                        best_child = c
                if best_child is None:
                    break
                node = best_child
                cur_p = best_child.p_coords

            # --- Expansion: 从 node 展开一个新子节点 (policy prior 选动作) ---
            tmp_state = RLOptimizerState(
                p_coords=cur_p, sequence=state.sequence,
                far_blocks=_rebuild_blocks(state, cur_p),
                far_pairs=far_pairs, block_edges=state.block_edges,
            )
            pi_region, pi_block, pi_type, pi_trans_dir, pi_rot_axis, pi_step = (
                None, None, None, None, None, None)
            if self.policy is not None:
                pi_region, pi_block, pi_type, pi_trans_dir, pi_rot_axis, pi_step = \
                    self.policy.forward(tmp_state)

            if pi_block is not None:
                # DRAG 式两层 block 选择: 先选区域, 再选区域内块
                regions = tmp_state.regions
                if regions:
                    ridx = int(np.random.choice(3, p=pi_region.detach().cpu().numpy()))
                    in_region = [bi for bi, rid in regions.items() if rid == ridx]
                    if in_region:
                        region_probs = pi_block[in_region].detach().cpu().numpy()
                        region_probs = region_probs / region_probs.sum()
                        bidx = int(np.random.choice(in_region, p=region_probs))
                    else:
                        bidx = int(np.random.choice(n_blocks, p=pi_block.detach().cpu().numpy()))
                else:
                    bidx = int(np.random.choice(n_blocks, p=pi_block.detach().cpu().numpy()))

                # ---- 两层 mutation (方案 #2) ----
                # 第一层: mutation type (0=平移, 1=旋转)
                pi_type_np = pi_type.detach().cpu().numpy()
                type_id = int(np.random.choice(2, p=pi_type_np))
                # 第二层: 条件动作参数
                if type_id == 0:
                    # 平移: 6 方向 × 3 步长
                    pi_dir_np = pi_trans_dir.detach().cpu().numpy()
                    dir_or_axis_id = int(np.random.choice(N_DIRECTIONS, p=pi_dir_np))
                    didx = dir_or_axis_id  # 0-5 = 平移方向
                else:
                    # 旋转: 6 轴 × 3 角度
                    pi_axis_np = pi_rot_axis.detach().cpu().numpy()
                    dir_or_axis_id = int(np.random.choice(N_ROT_AXES, p=pi_axis_np))
                    didx = N_DIRECTIONS + dir_or_axis_id  # 6-11 = 旋转轴
                sidx = int(np.random.choice(N_STEPS, p=pi_step.detach().cpu().numpy()))
            else:
                # 无策略: 启发式选动作 (与 rollout 共用 _heuristic_action)
                bidx, didx, sidx = self._heuristic_action(tmp_state, far_pairs)

            new_p = apply_action(tmp_state, bidx, didx, sidx)
            if self.dpo_simulate:
                # dpo_simulate 模式 (2026-08-02 修): 用 compute_reward 做 reward.
                # 之前用 DPO V (3ms) 但梯度太平滑, 无法驱动 far 拉拢 (MCTS 改善仅 19).
                # compute_reward 含 R_pair(拉拢驱动) + clash(防穿模) + R_compact,
                # 且 cgRNASP torch 提速后热调用仅 21ms, 快且能拉拢.
                # 树内量级一致 (全用 compute_reward), UCB 可比.
                r_exp = compute_reward(new_p, far_pairs,
                                       initial_p_coords=initial_p_coords,
                                       sequence=sequence, dpo_weight=self.dpo_weight,
                                       p_init=initial_p_coords)
            else:
                r_exp = compute_reward(new_p, far_pairs,
                                       initial_p_coords=initial_p_coords,
                                       sequence=sequence, dpo_weight=self.dpo_weight,
                                       p_init=initial_p_coords)

            # 保存两层 mutation 信息 (供 block_action_distribution 使用)
            # heuristic 路径: type_id/dira 未知, 用占位
            if pi_block is not None:
                detail = (bidx, type_id, dir_or_axis_id, sidx)
            else:
                detail = None

            # --- Simulation: 叶节点估值 (可选 rollout 多看几步) ---
            if self.use_rollout:
                roll_state = RLOptimizerState(
                    p_coords=new_p, sequence=state.sequence,
                    far_blocks=_rebuild_blocks(state, new_p),
                    far_pairs=far_pairs, block_edges=state.block_edges,
                )
                r = self._rollout(roll_state, far_pairs,
                                  initial_p_coords=initial_p_coords,
                                  sequence=sequence)
            else:
                r = r_exp

            child = MCTSNode(
                p_coords=new_p, reward=r, parent=node,
                action_taken=(bidx, didx, sidx),
                action_detail=detail,
            )
            node.children.append(child)

            # --- Backprop: 沿父链更新 visit/value ---
            cur = child
            while cur is not None:
                cur.visits += 1
                cur.value += r
                cur = cur.parent

            # best 用即时 reward (不卷入 rollout 估值, 避免 rollout 随机性污染最优解)
            if r_exp > best.reward:
                best = MCTSNode(p_coords=new_p, reward=r_exp,
                                parent=None, action_taken=(bidx, didx, sidx))

        # ── 构象分布输出 (SamplingDesign 式概率空间) ──
        # 收集所有评估过的节点 (不含 root), 用 visit 计数做概率权重。
        # 等价于: 从 MCTS 搜索树中提取构象概率分布 π(s) ∝ visit(s)。
        dist = _extract_conformation_distribution(root, state.p_coords)
        dist.metadata = {
            "n_simulations": self.n_simulations,
            "n_nodes": len(dist.samples),
            "best_reward": best.reward,
            "policy_loaded": self.policy is not None,
            "rollout_used": self.use_rollout,
        }
        return dist


def _extract_conformation_distribution(
    root: MCTSNode,
    root_coords: np.ndarray,
) -> ConformationDistribution:
    """从 MCTS 搜索树提取构象概率分布。

    采样原理:
      - 每个被评估过的节点 (有 reward) 是一个构象 sample
      - visit count = 该构象在搜索中被"到达"的次数 (概率权重的代理)
      - action_path = 从根到该节点的完整动作序列 (耦合变量)
    """
    samples: List[ConformationSample] = []
    # BFS: 收集所有节点, 重建 action_path + action_detail
    queue = [(root, [], [], 0)]
    while queue:
        node, path, detail_path, depth = queue.pop()
        for child in node.children:
            if node.action_taken is not None:
                new_path = path + [node.action_taken]
            else:
                new_path = path
            if node.action_detail is not None:
                new_detail = detail_path + [node.action_detail]
            else:
                new_detail = detail_path
            sample = ConformationSample(
                p_coords=child.p_coords,
                reward=child.reward,
                value=child.value / max(child.visits, 1),
                visits=child.visits,
                action_path=new_path,
                action_detail=new_detail,
                depth=depth + 1,
            )
            samples.append(sample)
            queue.append((child, new_path, new_detail, depth + 1))

    # 按 visit 降序排序 (方便 top-k 和 mode)
    samples.sort(key=lambda s: s.visits, reverse=True)

    # 如果没有任何展开, 退化: 返回根坐标的退化分布
    if not samples:
        samples.append(ConformationSample(
            p_coords=root_coords,
            reward=root.reward,
            value=root.value,
            visits=1,
        ))

    return ConformationDistribution(samples=samples)


def _rebuild_blocks(state: RLOptimizerState, new_p: np.ndarray) -> List[BlockState]:
    """用新 P 坐标重建块状态 (更新质心/偏差/tier)。"""
    new_blocks = []
    for b in state.far_blocks:
        ci = new_p[b.residues_i].mean(axis=0)
        cj = new_p[b.residues_j].mean(axis=0)
        dev = float(np.mean([
            np.linalg.norm(new_p[i] - new_p[j]) for i, j in zip(b.residues_i, b.residues_j)
        ]))
        tier = 0 if dev < 10.0 else (1 if dev < 20.0 else 2)
        new_blocks.append(BlockState(
            block_idx=b.block_idx, residues_i=b.residues_i, residues_j=b.residues_j,
            centroid_i=ci, centroid_j=cj, current_deviation=dev,
            tier=tier,
        ))
    return new_blocks


# ---------- 端到端入口 ----------
def optimize_far_pairs(
    p_coords: np.ndarray,
    sequence: str,
    far_pairs: List[Tuple[int, int]],
    stem_blocks: List[List[Tuple[int, int]]],
    *,
    policy_path: Optional[str] = None,
    n_simulations: int = 50,
    coding_mask: Optional[np.ndarray] = None,
    return_distribution: bool = False,
    dpo_weight: float = 0.0,
    dpo_policy_path: Optional[str] = None,
    dpo_rollout: bool = False,
    dpo_simulate: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict] | Tuple[np.ndarray, np.ndarray, Dict, ConformationDistribution]:
    """端到端: CG P 坐标 + 远端配对 -> RL 优化后 P 坐标 + CG 原坐标 + (可选)构象分布。

    核心变化 (vs 旧版单点输出):
      - MCTS 每 tier 返回 ConformationDistribution (采样概率空间)
      - 多 tier 分布通过 merge() 组合成最终分布
      - 旧版行为 (return single p_coords) 通过分布的 mode 获取 (向后兼容)

    Args:
        p_coords: (L, 3) CG 求解输出
        sequence: ACGU 字符串
        far_pairs: 远端配对 [(i, j), ...]
        stem_blocks: 茎块 [[(i, j), ...], ...]
        policy_path: 策略网络权重路径 (None 用随机策略)
        n_simulations: MCTS 模拟次数
        coding_mask: 可选 coding 标注 (L,) bool
        return_distribution: True 时返回 (p, cg, info, distribution)
                             否则返回 (p, cg, info) (向后兼容)

    Returns:
        旧版:  (optimized_p, cg_coords, info)
        新版:  (optimized_p, cg_coords, info, distribution)  当 return_distribution=True
        distribution: ConformationDistribution, 含构象概率权重 + 块级耦合分布
    """
    state = build_rl_state(
        p_coords, sequence, far_pairs, stem_blocks,
        coding_mask=coding_mask,
    )
    # 初始 P 坐标: 作为 R_improve 的基准 (整个 RL 回程不变)
    initial_p_coords = p_coords.copy()
    reward_before = compute_reward(p_coords, far_pairs,
                                   initial_p_coords=initial_p_coords,
                                   sequence=sequence, dpo_weight=dpo_weight,
                                   p_init=initial_p_coords)
    cg_coords = p_coords.copy()

    policy = None
    if policy_path is not None:
        try:
            policy = PolicyNetwork()
            policy.load(policy_path)
        except Exception as exc:
            print(f"[rl_optimizer] 策略权重加载失败, 用随机策略: {exc!r}")
            policy = None

    # DPO policy (RiboPO 式构象质量打分器): 注册为全局, compute_reward 用 dpo_weight 加权.
    if dpo_weight > 0 or dpo_policy_path is not None:
        if dpo_policy_path is not None:
            try:
                dpo_pol = PolicyNetwork()
                dpo_pol.load(dpo_policy_path)
                set_dpo_policy(dpo_pol)
                print(f"[rl_optimizer] DPO policy 已加载: {dpo_policy_path} (dpo_weight={dpo_weight})")
            except Exception as exc:
                print(f"[rl_optimizer] DPO policy 加载失败, 跳过 DPO 引导: {exc!r}")
                set_dpo_policy(None)
        if dpo_weight <= 0:
            # ⚠️ 默认 5.0 (不是 1.0): 1.0 时 DPO 分量被 R_pair 淹没 (量级 ~1 vs 几十),
            #    对 MCTS 决策无影响. 实测 weight=5 在近程密集样本上 pair_rate +13%
            #    (0.222→0.251), far 拉拢不损. 见 2026-08-02 可复现性验证.
            dpo_weight = 5.0

    # DRAG 分层优化 + 构象分布收集
    tier_order = [0, 1, 2]
    total_sim = 0
    tier_distributions: List[ConformationDistribution] = []

    for t in tier_order:
        tier_blocks = [b for b in state.far_blocks if b.tier == t]
        if not tier_blocks:
            continue
        n_tier_sim = max(10, n_simulations // len(tier_order))
        total_sim += n_tier_sim
        p = state.p_coords.copy()

        tier_far_pairs = []
        for b in tier_blocks:
            for i, j in zip(b.residues_i, b.residues_j):
                tier_far_pairs.append((i, j))
        state_tier = RLOptimizerState(
            p_coords=p, sequence=sequence,
            far_blocks=tier_blocks, far_pairs=tier_far_pairs,
            block_edges=[],
            coding_mask=coding_mask,
        )

        mcts = MCTS(policy=policy, n_simulations=n_tier_sim, dpo_weight=dpo_weight,
                    dpo_rollout=dpo_rollout, dpo_simulate=dpo_simulate)
        tier_dist = mcts.search(state_tier, tier_far_pairs,
                                initial_p_coords=initial_p_coords,
                                sequence=sequence)

        # 用当前 tier 分布的最可能构象作为该 tier 的输出
        # dpo_simulate: 用 reward 最高 (best_coords), 非 visit 最多 (mode 被 UCB 探索偏置污染)
        use_best = dpo_simulate
        tier_out = tier_dist.best_coords if use_best else tier_dist.mode_coords
        if tier_out is not None:
            p = tier_out.copy()

        tier_distributions.append(tier_dist)

        # 重建状态
        new_blocks = _rebuild_blocks(state, p)
        state = RLOptimizerState(
            p_coords=p, sequence=sequence,
            far_blocks=new_blocks, far_pairs=far_pairs,
            block_edges=state.block_edges, coding_mask=coding_mask,
            regions=state.regions,
        )

    optimized_p = state.p_coords

    # 合并多 tier 的构象分布 (每个 tier 贡献一组加权 sample)
    final_distribution = ConformationDistribution(
        samples=[],
        temperature=1.0,
        metadata={
            "tier_count": len(tier_distributions),
            "total_simulations": total_sim,
        },
    )
    for td in tier_distributions:
        final_distribution = final_distribution.merge(td)

    # OpenMM 精修 (2026-08-02 移除): 旧 1-bead openmm_refine (单粒子/平面圆/无二面角)
    # 会把 RL 优化出的立体 3-bead 结构拉回平面/塌缩 (实测 2001nt Rg 30.5→15.9 穿模球).
    # RL 优化结果 (optimized_p) 保持原样, 立体折叠交给 3-bead 精修/下游 amber.
    postprocess_energy = None

    # ── far 配对定向拉拢 (2026-08-02) ──
    # MCTS 块级移动与 far 拉拢目标不匹配, 作为独立后处理:
    # 移 far 配对残基 + 邻居保持骨架刚体 (手动验证 Rg 稳 不塌缩).
    if far_pairs:
        from torusfold.scheme2.cg_forcefield import p_coords_to_3bead
        c3_before = p_coords_to_3bead(optimized_p)
        N_before = c3_before[2::3]
        fd_before = np.array([np.linalg.norm(N_before[i]-N_before[j]) for i,j in far_pairs if i < len(optimized_p) and j < len(optimized_p)])
        print(f"[pull] before: far_mean={fd_before.mean():.1f} n={len(fd_before)}")
        optimized_p = pull_far_pairs(optimized_p, far_pairs, target=15.0, max_iter=30)
        c3_after = p_coords_to_3bead(optimized_p)
        N_after = c3_after[2::3]
        fd_after = np.array([np.linalg.norm(N_after[i]-N_after[j]) for i,j in far_pairs if i < len(optimized_p) and j < len(optimized_p)])
        print(f"[pull] after: far_mean={fd_after.mean():.1f} n={len(fd_after)}")

    reward_after = compute_reward(optimized_p, far_pairs,
                                  initial_p_coords=initial_p_coords,
                                  sequence=sequence, dpo_weight=dpo_weight,
                                  p_init=initial_p_coords)

    info = {
        "reward_before": float(reward_before),
        "reward_after": float(reward_after),
        "improvement": float(reward_after - reward_before),
        "n_blocks": len(state.far_blocks),
        "n_far_pairs": len(far_pairs),
        "n_simulations": total_sim,
        "policy_loaded": policy is not None,
        "coding_mask": coding_mask,
        "postprocess_energy": postprocess_energy,
        "conformation_distribution": final_distribution.summary(),
    }

    if return_distribution:
        return optimized_p, cg_coords, info, final_distribution
    return optimized_p, cg_coords, info


if __name__ == "__main__":
    # 自测: 合成远端配对, 验证 RL 能拉拢
    np.random.seed(42)
    L = 100
    # 构造 CG P 坐标 (环形), 远端配对 (10, 60) 故意拉远
    R = L * 5.9 / (2 * np.pi)
    angles = np.linspace(0, 2 * np.pi, L, endpoint=False)
    p = np.stack([R * np.cos(angles), R * np.sin(angles), np.zeros(L)], axis=1)
    # 远端配对 (10, 60): 环距 50, 真实 P-P 距离 ~2R*sin(25°) 偏离 10.5
    far_pairs = [(10, 60)]
    # 茎块: (10, 60) 单配对 (凑成 4 连续)
    stem_blocks = [[(10, 60), (11, 59), (12, 58), (13, 57)]]
    # 但 (11,59) 等不在 far_pairs, build_rl_state 会跳过 -- 直接造远端块
    # 简化: 让 far_pairs 包含整块
    far_pairs = [(10, 60), (11, 59), (12, 58), (13, 57)]

    d_before = np.linalg.norm(p[10] - p[60])
    print(f"优化前: pair(10,60) P-P = {d_before:.2f} Å (目标 ~10.5)")

    opt_p, _cg_coords, info = optimize_far_pairs(p, "A" * L, far_pairs, [far_pairs],
                                                   n_simulations=30)
    d_after = np.linalg.norm(opt_p[10] - opt_p[60])
    print(f"优化后: pair(10,60) P-P = {d_after:.2f} Å")
    print(f"info: {info}")
