# 前端Skill更新需求分析

## 当前状态

### 1. 现有前端（Streamlit）
- **位置**: `confluencia_3_0/frontend/`
- **主文件**: `app.py` (TNBC Simulacrum)
- **Tab模块**: 7个tabs（tumor_dashboard, tme_immune, treatment, biomarker, clinical, experiments, confluencia）

### 2. 现有Skill
- **位置**: `.claude/skills/confluencia/SKILL.md`
- **名称**: confluencia
- **描述**: AI+Science研究助手（文献检索、数据分析、论文写作）
- **活跃项目**: IGEM-sama, Confluencia, Civis Lucri-Faber, TorusFold

---

## 新增功能需求

### 1. Scheme 0-8管理界面
**新增Tab**: `scheme_manager.py`

功能：
- Scheme选择器（下拉菜单：Scheme 0-7）
- Scheme描述展示
- Scheme运行控制（启动/停止/监控）
- Scheme输出可视化

### 2. Circ-CASP 2026界面
**新增Tab**: `casp_dashboard.py`

功能：
- 参赛队伍列表（13支队伍）
- Scheme-Team对应关系表
- 实时排名展示
- Team 9（官方基线）特殊标注
- 预测获胜排名

### 3. CircFold Baseline可视化
**新增Tab**: `circfold_baseline.py`

功能：
- 5-stage Pipeline可视化（流程图）
- Stage进度追踪
- 实时质量指标（confidence, BSJ距离, energy）
- 输出结构预览（3D渲染）

### 4. Scheme 3双引擎蒸馏界面
**新增Tab**: `scheme3_distillation.py`

功能：
- Teacher-Student关系图
- 蒸馏进度监控
- 损失曲线（Teacher vs Student）
- 知识转移可视化

---

## Skill文件更新

### 需要更新的内容

1. **项目列表更新**
   ```markdown
   ### [[TorusFold]] - circRNA 3D 结构预测
   - Scheme 0: CircFold Baseline（线性RNA环化法）
   - Scheme 1-7: 8种训练方案
   - Scheme 3: 双引擎蒸馏（Teacher: Scheme 0）
   - Circ-CASP 2026: 13支参赛队伍
   - Team 9: 官方基线方法
   ```

2. **新增命令**
   ```markdown
   **Scheme管理：**
   /scheme-run 0    # 运行CircFold Baseline
   /scheme-train 7  # 训练Mamba+Transformer
   /casp-monitor    # 监控Circ-CASP进度
   ```

3. **新增技术栈**
   ```markdown
   ### circRNA 3D预测
   - Pipeline: ViennaRNA → trRosettaRNA2 → OpenMM → MD → Filter
   - Models: EGNN, Diffusion, Mamba+Transformer
   - CASP: Team 0-13, Scheme 0-8
   ```

---

## 实施计划

### Phase 1: Skill文件更新（1小时）
- 更新 `.claude/skills/confluencia/SKILL.md`
- 添加Scheme 0-8描述
- 添加Circ-CASP 2026信息
- 添加新命令说明

### Phase 2: Streamlit Tabs创建（3小时）
- 创建 `scheme_manager.py`
- 创建 `casp_dashboard.py`
- 创建 `circfold_baseline.py`
- 创建 `scheme3_distillation.py`

### Phase 3: 集成测试（1小时）
- 测试Tab导入
- 测试Scheme切换
- 测试Circ-CASP展示

---

## 关键文件路径

| 文件 | 路径 | 操作 |
|------|------|------|
| Skill主文件 | `.claude/skills/confluencia/SKILL.md` | 更新 |
| Streamlit主文件 | `confluencia_3_0/frontend/app.py` | 新增Tab |
| Scheme管理 | `confluencia_3_0/frontend/tabs/scheme_manager.py` | 新建 |
| CASP面板 | `confluencia_3_0/frontend/tabs/casp_dashboard.py` | 新建 |

---

## 优先级

| 优先级 | 任务 | 原因 |
|--------|------|------|
| P0 | Skill文件更新 | 基础知识库更新 |
| P1 | CASP Dashboard | 官方比赛展示 |
| P2 | Scheme Manager | 系统管理界面 |
| P3 | CircFold可视化 | 技术细节展示 |

---

**建议立即执行Phase 1（Skill文件更新）**