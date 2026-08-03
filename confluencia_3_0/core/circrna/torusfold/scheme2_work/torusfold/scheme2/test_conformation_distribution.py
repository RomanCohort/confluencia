"""Self-test: ConformationDistribution 数据结构 + MCTS.search 返回分布。"""
import sys
sys.path.insert(0, 'src')
import numpy as np
from torusfold.scheme2 import (
    optimize_far_pairs, ConformationDistribution, ConformationSample,
    PolicyNetwork, MCTS, BlockState, RLOptimizerState, build_rl_state,
)

# ── 1. ConformationDistribution 数据结构测试 ──
print("=== Test 1: ConformationDistribution ===")

L = 100
np.random.seed(42)
R = L * 5.9 / (2 * np.pi)
angles = np.linspace(0, 2 * np.pi, L, endpoint=False)
p = np.stack([R * np.cos(angles), R * np.sin(angles), np.zeros(L)], axis=1)

# 构造 6 个人工 sample (模拟 MCTS 搜索树)
visits = [10, 8, 5, 3, 2, 1]
rewards = [-5.0, -4.0, -6.0, -8.0, -3.0, -9.0]
coords = [p + np.random.normal(0, 0.1, (L, 3)) for _ in range(6)]

samples = [ConformationSample(
    p_coords=coords[i],
    reward=float(rewards[i]),
    visits=visits[i],
    action_path=[(0, i, 0)],  # 伪动作
) for i in range(6)]

dist = ConformationDistribution(samples=samples)

print(f"  n_samples={len(dist.samples)}")
print(f"  probabilities: {np.round(dist.probabilities, 4)}")
print(f"  sum(p)={dist.probabilities.sum():.6f}")
print(f"  H={dist.entropy:.3f} nats")
print(f"  concentration={dist.concentration:.3f}")
print(f"  mean_reward={dist.mean_reward:.4f}")
print(f"  mode_reward={dist.mode.reward:.4f}")
print(f"  mode_coords shape: {dist.mode_coords.shape}")
assert abs(dist.probabilities.sum() - 1.0) < 1e-6, "Probabilities don't sum to 1"
assert dist.mode.reward == -5.0, "Mode should be max visit sample"
assert dist.mode_coords.shape == (L, 3), "Mode coords shape mismatch"
assert dist.entropy > 0, "Entropy should be positive for non-degenerate dist"
print("  PASS")

# ── 2. 采样测试 ──
print("\n=== Test 2: Sampling ===")
draws = dist.sample(k=1000, rng=np.random.default_rng(42))
drawn_visits = [s.visits for s in draws]
# 高 visit 样本应被采样更多次
high_visit_count = sum(1 for v in drawn_visits if v >= 5)
low_visit_count = sum(1 for v in drawn_visits if v <= 2)
print(f"  high-visit draws (v>=5): {high_visit_count}")
print(f"  low-visit draws (v<=2): {low_visit_count}")
assert high_visit_count > low_visit_count, "Sampling should favor high-visit samples"
print("  PASS")

# ── 3. top-k 和 merge 测试 ──
print("\n=== Test 3: top_k + merge ===")
top3 = dist.top_k(3)
assert len(top3.samples) == 3, "top_k(3) should return 3"
assert top3.samples[0].visits >= top3.samples[1].visits >= top3.samples[2].visits
print(f"  top3 visits: {[s.visits for s in top3.samples]}")

dist2 = ConformationDistribution(
    samples=[ConformationSample(
        p_coords=p + np.random.normal(0, 0.2, (L, 3)),
        reward=-2.0, visits=15,
    )]
)
merged = dist.merge(dist2)
print(f"  merged n_samples: {len(merged.samples)}")
assert len(merged.samples) == 7, "Merge should combine all samples"
assert merged.mode.visits == 15, "Merged mode should be max visit"
print("  PASS")

# ── 4. temperature 测试 ──
print("\n=== Test 4: Temperature ===")
dist_cold = ConformationDistribution(samples=samples, temperature=0.1)
dist_hot = ConformationDistribution(samples=samples, temperature=5.0)
print(f"  T=0.1: concentration={dist_cold.concentration:.3f}, H={dist_cold.entropy:.3f}")
print(f"  T=1.0: concentration={dist.concentration:.3f}, H={dist.entropy:.3f}")
print(f"  T=5.0: concentration={dist_hot.concentration:.3f}, H={dist_hot.entropy:.3f}")
assert dist_cold.concentration > dist.concentration, "Cold should be more concentrated"
assert dist_hot.entropy >= dist.entropy, "Hot should have higher entropy"
print("  PASS")

# ── 5. summary + to_dict 测试 ──
print("\n=== Test 5: summary ===")
print(f"  {dist.summary()}")
d = dist.to_dict()
assert d["n_samples"] == 6
assert d["mode_reward"] == -5.0
print("  PASS")

# ── 6. MCTS.search 返回 ConformationDistribution ──
print("\n=== Test 6: MCTS.search → ConformationDistribution ===")
stem_blocks = [[(10, 60), (11, 59), (12, 58), (13, 57)]]
far_pairs = [(10, 60), (11, 59), (12, 58), (13, 57)]

state = build_rl_state(
    p_coords=p,
    sequence="A" * L,
    far_pairs=far_pairs,
    stem_blocks=stem_blocks,
)

mcts = MCTS(policy=None, n_simulations=20, use_rollout=True)
result = mcts.search(state, far_pairs)
assert isinstance(result, ConformationDistribution), f"Expected ConformationDistribution, got {type(result)}"
print(f"  result type: {type(result).__name__}")
print(f"  {result.summary()}")
assert len(result.samples) >= 1, "Distribution should have at least root sample"
assert result.mode is not None, "Mode should not be None"
print("  PASS")

# ── 7. optimize_far_pairs return_distribution=True ──
print("\n=== Test 7: optimize_far_pairs(return_distribution=True) ===")
result_p, cg_coords, info, dist_opt = optimize_far_pairs(
    p, "A" * L, far_pairs, stem_blocks,
    n_simulations=20,
    return_distribution=True,
)
assert isinstance(dist_opt, ConformationDistribution), f"Expected ConformationDistribution, got {type(dist_opt)}"
print(f"  dist_opt type: {type(dist_opt).__name__}")
print(f"  dist_opt: {dist_opt.summary()}")
print(f"  info['conformation_distribution']: {info.get('conformation_distribution')}")
assert info.get('conformation_distribution') is not None, "info should contain dist summary"
print("  PASS")

# ── 8. optimize_far_pairs 向后兼容 (默认返回 3-tuple) ──
print("\n=== Test 8: Backward compatibility ===")
result_old = optimize_far_pairs(
    p, "A" * L, far_pairs, stem_blocks,
    n_simulations=20,
)
assert isinstance(result_old, tuple) and len(result_old) == 3, f"Expected 3-tuple, got {len(result_old)}-tuple"
print("  PASS")

print("\n=== ALL TESTS PASSED ===")
