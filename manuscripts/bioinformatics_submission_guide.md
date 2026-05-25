# Bioinformatics Application Note 投稿指南

## 期刊要求

**Bioinformatics (Oxford University Press)**

### Application Note 格式要求

| 项目 | 要求 |
|------|------|
| **字数** | 800-1000 words (不含参考文献) |
| **摘要** | ≤100 words |
| **关键词** | 3-6个 |
| **参考文献** | ≤15条 |
| **图表** | 1-2 figures + 1 table (可选) |
| **Availability声明** | 必须包含软件URL、版本、license |
| **软件要求** | 开源、可下载、有文档 |

### 文章结构

```
1. Abstract (≤100 words)
2. Introduction (背景+动机，~150 words)
3. Methods/Implementation (方法，~300 words)
4. Results/Features (功能，~200 words)
5. Comparison/Discussion (对比，~150 words)
6. Availability and Implementation (必填)
7. Acknowledgements
8. References (≤15条)
```

---

## 投稿准备清单

### 必需材料

| 材料 | 状态 | 说明 |
|------|------|------|
| GitHub仓库 | ✓ | https://github.com/RomanCohort/confluencia |
| 代码开源 | ✓ | MIT License |
| 文档 | ✓ | README_modules.md |
| 安装说明 | ✓ | pip install说明 |
| 测试数据 | 需补充 | 示例circRNA序列 |
| benchmark | 需补充 | 性能测试数据 |

### 需要补充的内容

1. **真实数据验证**
   - 使用TCGA或GEO数据验证临床预测
   - 展示IPS/TIDE评分的实际效果

2. **Benchmark数据**
   - 运行时间对比
   - 不同序列长度性能

3. **Figure准备**
   - Figure 1: 架构图 (矢量格式)
   - Figure 2: 示例结果 (雷达图/折线图)

---

## 文章亮点设计

### 强调的创新点

| 创新点 | 文献支持 | 强度 |
|------|---------|------|
| **文献权重评分** | Schlee/Nallagatla/Forsbach原始数据 | ★★★★★ |
| **circRNA专用** | 区别于mRNA工具 | ★★★★ |
| **临床预测** | IPS/TIDE整合 | ★★★★ |
| **进化优化** | Pareto + RL | ★★★ |
| **模块集成** | 10模块一站式 | ★★★★ |

### 与现有工具对比优势

| 对比维度 | Confluencia优势 |
|------|---------------|
| vs ViennaRNA | +免疫评分、+修饰预测、+临床 |
| vs mRNA设计工具 | +circRNA专用、+进化优化 |
| vs circRNA数据库 | +预测功能、+设计功能 |

---

## 可能的审稿意见预测

### 潜在问题

| 问题 | 回应策略 |
|------|---------|
| "文献权重如何验证？" | 可引用原始实验数据（Schlee图3等） |
| "ViennaRNA是现有工具" | 强调免疫评分是新贡献 |
| "临床预测需要验证" | 补充TCGA数据分析 |
| "进化算法复杂度高" | 提供运行时间benchmark |
| "缺少真实circRNA测试" | 补充circBase数据分析 |

---

## 投稿时间规划

| 时间 | 任务 |
|------|------|
| Week 1 | 补充benchmark数据 |
| Week 2 | 准备Figure (架构图、结果图) |
| Week 3 | 完成真实数据验证 |
| Week 4 | 内部审阅修改 |
| Week 5 | 投稿 |

---

## 投稿地址

**投稿系统:** https://academic.oup.com/bioinformatics

**文章类型:** Application Note

**审稿周期:** ~4-8 weeks

---

## 相关成功案例

| 文章 | 特点 | 引用 |
|------|------|------|
| ViennaRNA Package 2.0 | 经典RNA工具 | 1000+ |
| CIRCexplorer2 | circRNA注释工具 | 200+ |
| CircBase database | circRNA数据库 | 150+ |

---

## 下一步建议

### 优先级排序

| 优先级 | 任务 | 预计时间 |
|--------|------|---------|
| P0 | 补充benchmark运行时间 | 1天 |
| P1 | 准备架构图Figure | 2天 |
| P2 | 真实circRNA测试 | 3天 |
| P3 | 文档完善 | 1天 |

要我帮你补充benchmark数据或准备Figure吗？