# Confluencia main_compressed.tex 并发审稿汇总报告

**审稿日期:** 2026-06-02
**论文标题:** Confluencia: A Computational Framework for circRNA Therapeutic Candidate Screening
**审稿模式:** 四位专家并发审稿

---

## 审稿结果概览

| 审稿人 | 角色 | 推荐意见 | 核心关注点 |
|--------|------|----------|------------|
| **Reviewer 1** | 计算方法论 | **Minor Revision** | 权重缺乏独立标定数据 |
| **Reviewer 2** | circRNA/RNA Therapeutics | **Accept with Minor Revision** | endosomal escape参数(5.2%)超出文献上限 |
| **Reviewer 3** | 软件可复现性 | **Minor Revision** | GitHub URL不一致、IRB timeline模糊 |
| **Reviewer 4** | 统计学 | **Accept** | 统计报告诚实透明 |

**综合意见: Minor Revision**

---

## Reviewer 1: Methodology Assessment

### 关键问题评估

| 问题 | 评估 | 理由 |
|------|------|------|
| RIG-I通路约束 | Acceptable | 生物学依据正确：circRNA lacks 5'/3' termini |
| 权重合理性 | Partially Acceptable | 有文献依据但未校准 |
| 敏感性分析 | Acceptable | ±0.05→≤4.2%, ±0.10→≤8.5%，含决策稳定性验证 |
| 循环验证警示 | Acceptable | 明确声明"parameter verification, NOT independent validation" |
| 局限性章节 | Acceptable | 5点局限透明披露 |

### 建议
1. Limitations补充"权重校准需后续实验数据"
2. N=7的CI宽度(0.49)可在Abstract简要提及

**推荐: Minor Revision**

---

## Reviewer 2: circRNA/RNA Therapeutics

### 关键问题评估

| 问题 | 评估 |
|------|------|
| RIG-I dsRNA backbone pathway | ✓ 正确描述 (Line 29) |
| TLR delivery limitation | ✓ 已声明未建模 (Line 33) |
| Linear RNA contamination warning | ✓ 充分警告 (Line 33) |
| m6A context-dependent effect | ✓ YTHDF2/YTHDF1区分准确 |
| PK half-life extrapolation | ✓ 标注为author estimates |

### 遗留问题
- **Q7 (endosomal escape 5.2%):** 承认超出文献上限(5.0%)但未调整参数，建议修正至≤5.0%

**推荐: Accept with Minor Revision**

---

## Reviewer 3: Software/Reproducibility

### 关键问题评估

| 问题 | 评估 | 问题详情 |
|------|------|----------|
| Multi-backend architecture | Acceptable | fast/accurate mode清晰，但one-flag switch syntax未提供 |
| Confluencia Hub ethics | Acceptable | data source declaration, clinical-use warnings充分 |
| Wet-lab validation plan | Partially Acceptable | IRB timeline vague: "2-4 weeks post-approval"无具体日期 |
| Availability section | Acceptable | 但GitHub URL仍为RomanCohort，应为IGEM-FBH |
| Performance metrics | Acceptable | 85±23ms (N=100)完整报告 |

### 必须修改
1. GitHub URL: `RomanCohort/confluencia` → `IGEM-FBH/confluencia`
2. 提供one-flag switch的CLI/API syntax

**推荐: Minor Revision**

---

## Reviewer 4: Statistics

### 关键问题评估

| 问题 | 评估 | 详情 |
|------|------|------|
| IFN-β correlation (r=0.91, CI 0.50-0.99) | Acceptable | 诚实标注"CI width 0.49 exceeds acceptable threshold" |
| N=7 sample size limitation | Acceptable | 双重声明"precludes quantitative validation" |
| circBase pseudo-labels | Acceptable | 标注"model-generated, not experimental ground truth" |
| Performance report | Acceptable | 均值、标准差、样本量完整 |
| Sensitivity analysis | Acceptable | 方法论完整，含决策稳定性验证(90% stable) |

**推荐: Accept**

---

## 编辑综合意见

### 共识问题（需修改）

#### 必须修改 (Essential)

1. **GitHub URL不一致** (R3)
   - Paper中仍为`RomanCohort/confluencia`
   - 应改为`IGEM-FBH/confluencia`（与response_to_reviewers.md声明一致）

2. **Endosomal escape参数超限** (R2)
   - 5.2%超出文献上限5.0%
   - 建议调整至≤5.0%或提供文献支持

3. **One-flag switch syntax缺失** (R3)
   - Multi-backend architecture提到"one-flag switch"但未提供实际CLI/API语法

#### 建议修改 (Recommended)

1. **Limitations补充权重校准** (R1)
   - 添加"权重校准需后续实验数据"

2. **IRB timeline具体化** (R3)
   - 提供expected start date而非"2-4 weeks post-approval"

---

## 最终决定

**编辑决定: Minor Revision**

四位审稿人中：
- 1位建议 **Accept** (R4)
- 2位建议 **Minor Revision** (R1, R3)
- 1位建议 **Accept with Minor Revision** (R2)

### 优点
1. **统计报告诚实透明**：CI宽度、sample size limitation、pseudo-labels性质均明确标注
2. **生物学约束准确**：RIG-I通路、TLR递送限制、m6A context依赖性描述正确
3. **方法论声明充分**：循环验证警示、敏感性分析、权重heuristic性质均有说明
4. **Wet-lab plan清晰**：IRB approval in progress，后续验证路径明确

### 需修改项
1. GitHub URL修正
2. Endosomal escape参数调整
3. One-flag switch syntax补充

---

**审稿完成时间:** 2026-06-02
**审稿人:** Reviewer 1 (Methodology), Reviewer 2 (circRNA), Reviewer 3 (Software), Reviewer 4 (Statistics)
