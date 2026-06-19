# 六方案实验验证设计

## 评估指标

| 指标 | 计算方式 | 意义 |
|------|----------|------|
| **RMSD** | $\sqrt{\frac{1}{N}\sum_i ||x_i^{pred} - x_i^{ref}||^2}$ | 与参考结构偏差 |
| **Closure Error** | $|d_{01} - d_{bond}|$ | BSJ闭合精度 |
| **Energy Score** | $E_{bond} + E_{pair} + E_{clash} + ...$ | 物理合理性 |
| **Inference Time** | 秒/序列 | 实用性 |
| **Coverage** | 多样性指数 | 构象空间探索 |

## 实验设计

### 实验1：已知结构验证（Ground Truth）

**数据集**：PDB中唯一的circRNA结构（如果有）或IsRNAcirc测试集34条

**方法**：
```python
for seq, ref_coords in test_set:
    for scheme in [1, 2, 3, 4, 5, 6]:
        pred_coords = predict(seq, scheme=scheme)
        rmsd = compute_rmsd(pred_coords, ref_coords)
        closure = compute_closure_error(pred_coords)
        energy = compute_energy(pred_coords)
        results.append((scheme, rmsd, closure, energy))
```

**预期结果**：
| 方案 | RMSD预期 | Closure预期 | 推理时间 |
|------|----------|-------------|----------|
| 一 | 中 | 低 | 快 |
| 二 | 低 | 低 | 中 |
| 三 | 最低 | 最低 | 慢 |
| 四 | 高 | 高 | 快 |
| 五 | 中 | 中 | 快 |
| 六 | 低 | 低 | 慢 |

### 实验2：无参考结构验证（物理一致性）

**数据集**：100条人工设计circRNA序列

**方法**：无ground truth时，用物理能量评分
```python
for seq in test_set:
    for scheme in [1, 2, 3, 4, 5, 6]:
        pred_coords = predict(seq, scheme=scheme)
        # 物理合理性评分
        bond_energy = compute_bond_energy(pred_coords)
        clash_count = count_clashes(pred_coords)
        closure_error = compute_closure_error(pred_coords)
        # 综合物理分数
        physics_score = bond_energy + 10*clash_count + closure_error**2
```

### 实验3：构象多样性验证

**方法**：对同一序列生成N=20个候选，评估多样性
```python
for seq in test_set:
    for scheme in [1, 2, 3, 4, 6]:  # 方案5不支持多构象
        candidates = []
        for i in range(20):
            pred = predict(seq, scheme=scheme, seed=i)
            candidates.append(pred)
        # 计算成对RMSD矩阵
        rmsd_matrix = compute_pairwise_rmsd(candidates)
        # 多样性指数 = 平均成对RMSD
        diversity = rmsd_matrix.mean()
```

### 实验4：长序列可扩展性

**方法**：测试不同长度序列
```python
lengths = [100, 200, 500, 1000, 2000, 4000]
for L in lengths:
    seq = random_sequence(L)
    for scheme in [1, 2, 3, 4, 5, 6]:
        start = time.time()
        pred = predict(seq, scheme=scheme)
        elapsed = time.time() - start
        memory = get_peak_memory()
        results.append((scheme, L, elapsed, memory))
```

### 实验5：消融实验（方案三关键组件）

**方法**：测试方案三各组件贡献
```python
configs = [
    ('no_feedback', iterations=1),      # 无迭代
    ('no_physics', use_physics=False),  # 无物理筛选
    ('no_strain', feedback_mode='none'),# 无应变反馈
    ('full', iterations=3),             # 完整版
]
for config in configs:
    results = run_experiment(config)
```

## 推荐实验优先级

| 优先级 | 实验 | 工作量 | 论文价值 |
|--------|------|--------|----------|
| 🔴 高 | 实验2（物理一致性） | 低 | 核心证据 |
| 🔴 高 | 实验4（可扩展性） | 中 | 实用性证明 |
| 🟡 中 | 实验3（多样性） | 中 | 创新点 |
| 🟢 低 | 实验1（Ground Truth） | 高 | 数据依赖 |
| 🟢 低 | 实验5（消融） | 中 | 可解释性 |

## 实验代码框架

```python
# experiment_schemes.py

def run_all_schemes(sequences, output_dir):
    results = []
    for seq in sequences:
        for scheme_id, scheme_module in enumerate(SCHEMES, 1):
            try:
                result = scheme_module.predict(seq)
                metrics = compute_all_metrics(result, seq)
                results.append({
                    'scheme': scheme_id,
                    'seq_len': len(seq),
                    **metrics
                })
            except Exception as e:
                print(f"Scheme {scheme_id} failed: {e}")
    return pd.DataFrame(results)

SCHEMES = {
    1: DLPhysicsCascadeHead,
    2: GeometricConstraintSolver,
    3: DualEngineTorusFold,
    4: CircRNADiffusion,
    5: CircPairformerBlock,  # with physics_bias=True
    6: DiffusionGNNHybrid,
}
```

## 论文图表

- Figure 8a: RMSD boxplot by scheme
- Figure 8b: Closure error comparison
- Figure 8c: Energy score distribution
- Figure 8d: Inference time vs sequence length
- Table 8: Summary statistics per scheme
