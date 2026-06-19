# 深度学习 + 物理微调架构方案

## 核心思想

```
DL主引擎 (快速、可学习) → 物理精修模块 (准确、可验证)
        ↓                       ↓
    预测配对关系             验证几何约束
    预测初结构               优化能量最低
    多任务预测               确保圆形闭合
```

## 方案1: 两阶段级联 (推荐)

```python
class DLPhysicsCascade:
    """Stage1: DL预测 → Stage2: 物理精修"""

    def predict(self, sequence: str):
        # Stage 1: DL预测 (快速)
        dl_result = self.torusfold.predict(sequence)
        initial_coords = dl_result['coords']
        pair_probs = dl_result['pair_probs']

        # Stage 2: 物理精修 (准确)
        # 用DL预测作为物理模块的初始构象
        physics_result = self.physics_refiner.refine(
            initial_coords,
            pair_probs,  # 作为能量项的约束
        )

        return physics_result
```

**优势**：
- DL提供高质量初值 → 物理模块不需要全局采样
- 物理精修 → 修复DL预测的几何不合理处
- 总时间：DL(秒) + Physics(分钟) < 纯MD(小时)

## 方案2: 端到端可微物理层 (创新)

```python
class DifferentiablePhysicsLayer(nn.Module):
    """物理约束作为DL网络的最后一层"""

    def forward(self, dl_coords, pair_probs):
        # 将物理约束写成可微形式
        # 1. Bond length loss: Σ(d_i - bond_length)^2
        bond_loss = self._bond_length_loss(dl_coords)

        # 2. Pair distance loss: Σ(d_ij - pair_distance)^2 * pair_probs
        pair_loss = self._pair_distance_loss(dl_coords, pair_probs)

        # 3. Closure loss: ||coords[0] - coords[-1] - bond_length||^2
        closure_loss = self._closure_loss(dl_coords)

        # 4. Clash loss: Σmax(0, clash_dist - d)^2
        clash_loss = self._clash_loss(dl_coords)

        total_loss = bond_loss + pair_loss + closure_loss + clash_loss

        # 反向传播 → DL网络学习满足物理约束
        return dl_coords, total_loss
```

**优势**：
- 训练时DL就学习物理约束 → 不需要后处理
- 物理约束作为regularization → 提高泛化性
- 一旦训练完成 → 推理时天然满足物理约束

## 方案3: 迭代交替优化 (最准确)

```python
class IterativeDLPhysics:
    """DL → Physics → DL → Physics 循环"""

    def predict(self, sequence: str, n_iter: int = 3):
        coords = None

        for i in range(n_iter):
            # DL阶段：根据当前坐标更新预测
            dl_result = self.torusfold.predict(sequence, coords)
            coords = dl_result['coords']

            # Physics阶段：精修坐标
            coords = self.physics_refiner.refine(coords)

            # 评估能量
            energy = self.physics_refiner.energy(coords)
            if energy < threshold:
                break

        return coords
```

**优势**：
- DL可以利用物理精修后的信息 → 更准确的预测
- Physics可以利用DL的新预测 → 更好的精修
- 收敛后 → 最优结构

## 方案4: 熵正则化 (结合热力学)

```python
class EntropyRegularizedDL(nn.Module):
    """物理热力学熵作为DL的损失项"""

    def forward(self, sequence):
        # DL预测多个候选结构
        candidates = self.torusfold.predict_n(sequence, n=10)

        # 物理熵计算：候选结构的构象熵
        # S = -Σ p_i log p_i，其中p_i基于能量
        energies = [self.physics.energy(c) for c in candidates]
        probs = softmax(-energies / temperature)
        entropy = -sum(p * log(p) for p in probs)

        # DL学习最大化熵 → 结构多样性
        # 同时最小化能量 → 结构准确性
        loss = -entropy + energy_weight * min(energies)

        return candidates, loss
```

**优势**：
- DL学习预测多个低能构象 → 更全面
- 物理熵引导 → 结构多样性（生物现实）
- 类似IsRNAcirc的REMD思想

## 推荐实现优先级

| 方案 | 实现难度 | 创新性 | 实用性 | 推荐顺序 |
|------|----------|--------|--------|----------|
| 方案1 两阶段级联 | 低 | 中 | 高 | **推荐首先实现** |
| 方案2 可微物理层 | 中 | 高 | 高 | **中期目标** |
| 方案3 迭代优化 | 中 | 中 | 最高 | **可选扩展** |
| 方案4 熵正则化 | 高 | 最高 | 中 | **长期研究** |

## 具体代码位置

现有架构中的扩展点：

```
TorusFold架构:
    Input → TPE → CircEquivariantBackbone → CircPairformer →
    [扩展点1: 可微物理层] →
    [扩展点2: PhysicsStructureHead (已有)] →
    [扩展点3: 后处理精修 (可加)]

physics_structure_head.py:
    Plan B: GeometricConstraintSolver (已有)
    Plan A: CGMDRefiner (已有)
    [可加: IterativeRefiner] ← 方案3
    [可加: DifferentiablePhysicsLayer] ← 方案2
```

## 论文中的表述建议

"Future versions of TorusFold could integrate deep learning and physics-based refinement more tightly. We propose three pathways:

1. **Two-stage cascade**: DL predictions serve as initial conformations for physics refinement, reducing MD sampling cost while preserving accuracy.

2. **Differentiable physics layer**: Expressing geometric constraints (bond length, closure, clash) as loss functions enables the DL network to learn physics-compliant predictions during training.

3. **Iterative optimization**: Alternating DL predictions and physics refinement allows mutual information flow, potentially converging on globally optimal structures.

This hybrid approach borrows IsRNAcirc's physics rigor while retaining TorusFold's scalability, addressing the circRNA structure prediction challenge from both efficiency and accuracy perspectives."