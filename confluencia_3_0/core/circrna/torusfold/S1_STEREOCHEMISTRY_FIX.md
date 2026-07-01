# S1 立体化学漏洞修复计划

## 发现的问题

### 当前 S1 损失函数的局限

看 `train_all_schemes.py` 第 549 行：

```python
# 当前 S1 损失函数
loss = coord_loss + 5.0 * closure_loss + 0.5 * bond_loss

# bond_loss 只检查相邻碱基
bonds = torch.norm(pred[:, 1:] - pred[:, :-1], dim=1)  # |i-j| = 1
bsj_bond = torch.norm(pred[:, 0] - pred[:, -1])          # BSJ: |0-(L-1)|
```

**漏洞：**
- ❌ 不检查 `|i - j| > 1` 的非相邻碱基对
- ❌ 无法检测位置 10 和 30 的重叠（距离 = 1.5Å）
- ❌ 类似 AlphaFold3 的立体化学失效风险

---

## 修复方案

### 方案 A：添加 clash_loss（推荐）

```python
# 新增 clash_loss
loss = coord_loss + 5.0 * closure_loss + 0.5 * bond_loss + 10.0 * clash_loss

# clash_loss: 检查所有非相邻碱基对
def compute_clash_loss(coords, clash_distance=2.5):
    """检查非相邻碱基的原子重叠。
    
    Args:
        coords: (B, L, 3) 预测坐标
        clash_distance: Å，最小允许距离
        
    Returns:
        clash_loss: 重叠惩罚
    """
    B, L, _ = coords.shape
    
    # 计算距离矩阵
    diff = coords[:, :, np.newaxis, :] - coords[:, np.newaxis, :, :]  # (B, L, L, 3)
    dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)  # (B, L, L)
    
    # 创建 mask：|i-j| > 1，排除 BSJ（0, L-1）
    i_idx, j_idx = torch.triu_indices(L, L, offset=2)  # |i-j| >= 2
    mask = ~((i_idx == 0) & (j_idx == L - 1))
    
    # 提取有效距离
    valid_dists = dist_matrix[:, i_idx[mask], j_idx[mask]]  # (B, n_pairs)
    
    # Clash penalty: max(0, clash_dist - d)^2
    clashes = torch.clamp(clash_distance - valid_dists, min=0.0)
    clash_loss = torch.mean(clashes ** 2)
    
    return clash_loss
```

---

### 方案 B：添加键长/键角约束（更完整）

```python
# 新增键长 + 键角约束
loss = coord_loss + 5.0 * closure_loss + 0.5 * bond_loss
       + 10.0 * clash_loss
       + 2.0 * angle_loss
       + 1.0 * dihedral_loss

# 键角约束
def compute_angle_loss(coords, target_angle=109.5):
    """检查键角是否符合 RNA 骨架。
    
    Args:
        coords: (B, L, 3)
        target_angle: 度，理想键角
        
    Returns:
        angle_loss: 键角偏差惩罚
    """
    # 计算三个连续碱基的角度
    v1 = coords[:, 2:] - coords[:, 1:-1]  # (B, L-2, 3)
    v2 = coords[:, 1:-1] - coords[:, :-2]  # (B, L-2, 3)
    
    # 计算角度
    cos_angle = torch.sum(v1 * v2, dim=-1) / (torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1) + 1e-8)
    
    # 目标角度（RNA 骨架）
    target_cos = torch.cos(torch.deg2rad(target_angle))
    
    angle_loss = torch.mean((cos_angle - target_cos) ** 2)
    
    return angle_loss
```

---

### 方案 C：训练后验证（最小改动）

```python
# 训练循环结束后添加验证
for batch in val_loader:
    pred = model(seq_ids)
    
    # 新增：立体化学验证
    stereo_report = validate_stereochemistry(pred)
    
    if stereo_report['has_clashes']:
        print(f"⚠️ 样本 {batch['id']} 有原子重叠")
        print(f"  Clash pairs: {stereo_report['clash_pairs']}")
    
    if stereo_report['bond_errors'] > threshold:
        print(f"⚠️ 键长异常: {stereo_report['bond_errors']}")
```

---

## 实施步骤

### Step 1: 创建 validate_stereochemistry.py

```python
"""validate_stereochemistry.py — 立体化学验证模块。

检查项：
  1. Clash（原子重叠）
  2. 键长异常
  3. 键角异常
  4. 手性错误
"""

def validate_stereochemistry(coords, bond_length=5.9, clash_dist=2.5):
    """完整的立体化学验证。
    
    Returns:
        report: {
            'has_clashes': bool,
            'clash_pairs': List[Tuple[int, int]],
            'bond_errors': float,
            'angle_errors': float,
            'is_valid': bool,
        }
    """
    L = len(coords)
    
    # 1. Clash 检测
    clash_pairs = detect_clashes(coords, clash_dist)
    
    # 2. 键长检测
    bond_errors = compute_bond_errors(coords, bond_length)
    
    # 3. 键角检测
    angle_errors = compute_angle_errors(coords)
    
    # 4. 综合判断
    is_valid = len(clash_pairs) == 0 and bond_errors < threshold
    
    return {
        'has_clashes': len(clash_pairs) > 0,
        'clash_pairs': clash_pairs,
        'bond_errors': bond_errors,
        'angle_errors': angle_errors,
        'is_valid': is_valid,
    }
```

---

### Step 2: 修改 train_all_schemes.py

```python
# 在 train_scheme1() 中添加 clash_loss

def train_scheme1(train_loader, val_loader, args, device):
    # ...原有代码...
    
    # 新增 clash_loss 权重
    w_clash = getattr(args, 'w_clash', 10.0)
    
    for batch in train_loader:
        # ...原有代码...
        
        # 计算 clash_loss
        clash_loss = compute_clash_loss(pred_denorm)
        
        # 总损失
        loss = coord_loss + w_closure * closure_loss + 0.5 * bond_loss + w_clash * clash_loss
        
        # ...原有代码...
```

---

### Step 3: 测试修复效果

```python
# 测试脚本
def test_stereochemistry_fix():
    """测试修复后的 S1 是否避免立体化学失效。"""
    
    # 创建测试序列
    test_seq = "ACGUACGUACGUACGUACGUACGUACGUACGUACGUACGU"
    
    # 训练模型（修复版）
    model = Scheme1Model_with_clash_loss()
    train_scheme1(model, ...)
    
    # 验证立体化学
    pred = model.predict(test_seq)
    report = validate_stereochemistry(pred)
    
    # 检查结果
    assert not report['has_clashes'], "修复后不应有 clash"
    assert report['bond_errors'] < 0.5, "键长误差应小"
    
    print("✅ 立体化学修复成功")
```

---

## 修复优先级

| 任务 | 优先级 | 预期效果 |
|------|-------|---------|
| **添加 clash_loss** | 🔴 高 | 消除原子重叠 |
| **添加 angle_loss** | 🟡 中 | 提升键角精度 |
| **添加验证模块** | 🟢 低 | 便于调试 |
| **测试所有 Scheme** | 🟢 低 | 全面覆盖 |

---

## 预期改进

| 维度 | 修复前 | 修复后 |
|------|--------|--------|
| **clash 检测** | ❌ 无 | ✅ 完整 |
| **键长约束** | ✓ 相邻碱基 | ✓ 相邻 + BSJ |
| **键角约束** | ❌ 无 | ✅ 可选 |
| **立体化学质量** | 未知 | **可量化** |
| **类似 AlphaFold3 失效风险** | 高 | **低** |

---

## 下一步行动

1. **立即**：创建 `validate_stereochemistry.py`
2. **本周**：添加 `clash_loss` 到 S1
3. **下周**：扩展到 S4/S6/S7
4. **测试**：对比修复前后的立体化学报告

---

## 参考

- AlphaFold3 stereochemistry issues: Stein et al., 2024
- Amber force field: Zgarbová et al., 2011
- RNA backbone geometry: RNA structure database