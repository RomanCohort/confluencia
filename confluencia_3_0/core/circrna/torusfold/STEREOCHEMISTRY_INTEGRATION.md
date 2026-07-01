# 立体化学损失集成示例

本文档展示如何为每个 TorusFold Scheme 添加立体化学约束。

---

## 通用集成模式

```python
# 所有 Scheme 的通用模式
from stereochemistry_losses import StereochemistryLoss

# 创建立体化学损失模块
stereo_loss = StereochemistryLoss()

# 在训练循环中
for batch in train_loader:
    pred_coords = model(batch)
    
    # 计算立体化学损失
    stereo_losses = stereo_loss(pred_coords, batch['lengths'])
    
    # 总损失 = 原损失 + 立体化学损失
    total_loss = original_loss + stereo_losses['total']
    
    total_loss.backward()
```

---

## Scheme 1: EGNN + 立体化学

```python
# train_all_schemes.py 修改示例

def train_scheme1(train_loader, val_loader, args, device):
    """Scheme 1: EGNN + 立体化学约束。"""
    
    from stereochemistry_losses import StereochemistryLoss
    
    model = Scheme1Model(d_hidden=args.d_hidden).to(device)
    stereo_loss = StereochemistryLoss()
    
    # 超参数
    w_stereo = getattr(args, 'w_stereo', 1.0)  # 立体化学权重
    
    for batch in train_loader:
        pred = model(batch['seq_ids'])
        target = batch['coords']
        
        # 原损失
        coord_loss = F.mse_loss(pred['coords'], target)
        closure_loss = compute_closure_loss(pred['coords'])
        bond_loss = compute_bond_loss(pred['coords'])
        
        # 新增：立体化学损失
        stereo_losses = stereo_loss(pred['coords'], batch['lengths'])
        
        # 总损失
        total_loss = (
            coord_loss +
            5.0 * closure_loss +
            0.5 * bond_loss +
            w_stereo * stereo_losses['total']
        )
        
        total_loss.backward()
        
        # 日志
        if step % 100 == 0:
            print(f"coord={coord_loss:.4f} "
                  f"clash={stereo_losses['clash_loss']:.4f} "
                  f"angle={stereo_losses['angle_loss']:.4f}")
```

---

## Scheme 4: DDPM + 立体化学

```python
def train_scheme4(train_loader, val_loader, args, device):
    """Scheme 4: Diffusion + 立体化学引导。"""
    
    from stereochemistry_losses import StereochemistryLoss
    
    model = DiffusionModel()
    stereo_loss = StereochemistryLoss()
    
    for batch in train_loader:
        # Diffusion 训练
        t = torch.randint(0, T, (B,))
        noisy_coords = add_noise(batch['coords'], t)
        
        # 预测去噪
        pred_coords = model(noisy_coords, t)
        
        # 原损失
        diffusion_loss = F.mse_loss(pred_coords, batch['coords'])
        
        # 新增：立体化学引导
        stereo_losses = stereo_loss(pred_coords, batch['lengths'])
        
        # 总损失
        total_loss = diffusion_loss + 0.5 * stereo_losses['total']
        
        total_loss.backward()
```

---

## Scheme 6: GNN Latent Diffusion + 立体化学

```python
def train_scheme6(train_loader, val_loader, args, device):
    """Scheme 6: Latent Diffusion + 立体化学验证。"""
    
    from stereochemistry_losses import StereochemistryLoss
    from validate_stereochemistry import validate_stereochemistry
    
    model = GNNLatentDiffusionModel()
    stereo_loss = StereochemistryLoss()
    
    for batch in train_loader:
        # Latent diffusion
        latent = model.encode(batch['coords'])
        pred_latent = model.diffusion_step(latent)
        pred_coords = model.decode(pred_latent)
        
        # 原损失
        latent_loss = F.mse_loss(pred_latent, latent)
        
        # 新增：立体化学损失（在坐标空间计算）
        stereo_losses = stereo_loss(pred_coords, batch['lengths'])
        
        # 总损失
        total_loss = latent_loss + 0.3 * stereo_losses['total']
        
        total_loss.backward()
    
    # 验证阶段：完整立体化学报告
    with torch.no_grad():
        for batch in val_loader:
            pred_coords = model.inference(batch['seq_ids'])
            
            # NumPy 验证
            coords_np = pred_coords[0].cpu().numpy()
            report = validate_stereochemistry(coords_np)
            
            if not report.is_valid:
                print(f"⚠️ 样本 {batch['id']} 立体化学失效")
                print(report.summary())
```

---

## Scheme 7: Mamba Hybrid + 立体化学

```python
def train_scheme7(train_loader, val_loader, args, device):
    """Scheme 7: Mamba + Transformer + 立体化学。"""
    
    from stereochemistry_losses import StereochemistryLoss
    
    model = MambaTransformerHybrid()
    stereo_loss = StereochemistryLoss()
    
    # Mamba 适合长序列，立体化学检查更重要
    w_stereo = getattr(args, 'w_stereo', 2.0)  # 更高权重
    
    for batch in train_loader:
        # Mamba-Transformer 混合
        pred_coords = model(batch['seq_ids'])
        
        # 原损失
        coord_loss = F.mse_loss(pred_coords, batch['coords'])
        
        # 立体化学损失
        stereo_losses = stereo_loss(pred_coords, batch['lengths'])
        
        # 总损失
        total_loss = coord_loss + w_stereo * stereo_losses['total']
        
        total_loss.backward()
```

---

## Scheme 8: Sparse Pair + 立体化学

```python
def train_scheme8(train_loader, val_loader, args, device):
    """Scheme 8: Sparse Pair + 立体化学。"""
    
    from stereochemistry_losses import StereochemistryLoss
    
    model = SparsePairModel()
    stereo_loss = StereochemistryLoss()
    
    for batch in train_loader:
        # Sparse pair prediction
        pred_coords = model(batch['seq_ids'], batch['sparse_pairs'])
        
        # 原损失
        coord_loss = F.mse_loss(pred_coords, batch['coords'])
        pair_loss = compute_pair_loss(pred_coords, batch['pair_constraints'])
        
        # 立体化学损失
        stereo_losses = stereo_loss(pred_coords, batch['lengths'])
        
        # 总损失
        total_loss = (
            coord_loss +
            1.0 * pair_loss +
            1.5 * stereo_losses['total']
        )
        
        total_loss.backward()
```

---

## 验证阶段集成

```python
# 所有 Scheme 的验证阶段

from validate_stereochemistry import validate_stereochemistry

def validate_with_stereochemistry(model, val_loader, device):
    """验证 + 立体化学报告。"""
    
    stereo_stats = {
        'n_valid': 0,
        'n_invalid': 0,
        'mean_clash': 0.0,
        'mean_bond_error': 0.0,
    }
    
    for batch in val_loader:
        pred_coords = model(batch['seq_ids'])
        
        # 逐样本验证
        for i in range(len(batch['ids'])):
            coords_np = pred_coords[i].cpu().numpy()
            report = validate_stereochemistry(coords_np)
            
            if report.is_valid:
                stereo_stats['n_valid'] += 1
            else:
                stereo_stats['n_invalid'] += 1
            
            stereo_stats['mean_clash'] += report.n_clashes
            stereo_stats['mean_bond_error'] += report.bond_mean_error
    
    # 平均值
    n_total = stereo_stats['n_valid'] + stereo_stats['n_invalid']
    stereo_stats['mean_clash'] /= n_total
    stereo_stats['mean_bond_error'] /= n_total
    
    # 打印报告
    print("\n立体化学统计:")
    print(f"  有效: {stereo_stats['n_valid']}/{n_total}")
    print(f"  无效: {stereo_stats['n_invalid']}/{n_total}")
    print(f"  平均 clash: {stereo_stats['mean_clash']:.2f}")
    print(f"  平均键长误差: {stereo_stats['mean_bond_error']:.2f}Å")
    
    return stereo_stats
```

---

## 超参数建议

| Scheme | w_stereo | 原因 |
|--------|---------|------|
| **S1** | 1.0 | 平衡精度和立体化学 |
| **S4** | 0.5 | Diffusion 已有结构约束 |
| **S6** | 0.3 | Latent space 损失间接约束 |
| **S7** | 2.0 | 长序列更容易失效 |
| **S8** | 1.5 | Sparse pair 需要额外约束 |

---

## 计算成本分析

| 损失项 | 复杂度 | 额外开销 |
|--------|--------|---------|
| **clash_loss** | O(L²) | ~3% |
| **bond_loss** | O(L) | <1% |
| **angle_loss** | O(L) | <1% |
| **dihedral_loss** | O(L) | <1% |
| **总计** | O(L²) | **~5%** |

**结论：计算成本可接受（<5% 额外开销）。**

---

## 迁移价值

为什么值得迁移到所有 Scheme？

1. **代码简单**：每个 Scheme 只需添加 3-5 行代码
2. **统一接口**：所有 Scheme 使用相同的 `StereochemistryLoss`
3. **可解释性**：每个损失项有明确物理意义
4. **防止失效**：避免类似 AlphaFold3 的立体化学失效
5. **量化质量**：验证阶段输出立体化学统计报告

---

## 下一步

- [ ] 修改 `train_all_schemes.py`
- [ ] 测试每个 Scheme 的立体化学质量
- [ ] 生成对比图表（修复前后）
- [ ] 更新文档