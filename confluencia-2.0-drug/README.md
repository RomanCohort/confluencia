# confluencia-2.0-drug
![App Icon](app.png)

> TL;DR：快速启动（演示）—— 在 Windows 下创建虚拟环境、安装依赖后运行 Streamlit 前端。

```powershell
cd "d:\IGEM集成方案\confluencia-2.0-drug"
python -m pip install -r requirements.txt
streamlit run app.py
```

Confluencia 2.0 药物模块原型：面向小样本场景的微机制疗效预测与分子优化流程（circRNA 方向）。

## 摘要

本项目提出一个面向小样本生物医药场景的可解释建模流程，目标是在有限样本条件下联合完成分子层输入、微机制指标预测与动力学轨迹刻画。系统以 Mixture of Experts（MOE）作为核心学习范式，耦合 CTM/NDP4PD 动力学后端，并通过免疫 ABM 与分子生成-进化模块形成端到端原型链路。该实现强调三类特性：

- 方法可解释性：提供多指标输出与动力学参数分解。
- 工程可复现性：提供脚本化复现与环境记录机制。
- 模块可扩展性：支持 ED2Mol 等外部生成器接入。

## 研究问题与目标

本项目对应如下研究问题：在输入分子表示、给药强度与时间上下文已知时，如何在样本量有限条件下稳定预测疗效及其机制相关指标，并输出可用于后续优化的结构化信号。

目标分解如下：

- 预测层：估计疗效与机制指标（如结合、免疫激活、炎症与毒性风险）。
- 动力学层：在时间轴上刻画药效变化过程。
- 优化层：将预测反馈用于候选分子迭代优化。

## 方法框架

已集成能力：

- MOE（Mixture of Experts）自动建模，适配小样本数据。
- 分子特征工程：优先 RDKit 指纹/描述符，缺失 RDKit 时自动降级到轻量哈希特征。
- 动力学后端双实现：
  - CTM（Compartmental Time Model）
  - NDP4PD（非线性 4 阶段药效动力学）
- 多指标输出：`efficacy`、`target_binding`、`immune_activation`、`immune_cell_activation`、`inflammation_risk`、`toxicity_risk`。
- 自适应校准系统（可选）：按样本分布动态校准疗效/风险预测，并输出剂量与给药频次的自适应系数。
- 免疫 ABM 桥接：支持将表位序列转换为触发事件，并对接 NetLogo 模型。
- ED2Mol 生成 + 反思式 RL 进化（含 Pareto 导向目标权重搜索，可选接入自适应校准进行进化评分，并纳入高风险超阈值目标约束）。

进化阶段风险门控支持两种阈值模式：

- 固定阈值（`risk_gate_threshold`）
- 分位数自适应阈值（`risk_gate_threshold_mode=quantile`，按每轮风险分布计算阈值）

反思日志额外输出惩罚引起的策略偏移诊断：

- `policy_shift_l1`：惩罚前后策略更新差异的 $L_1$ 强度
- `shift_peak_action`：偏移最大的动作
- Streamlit 可视化前端（训练、预测、解释与导出）。

### 形式化描述

设样本 $i$ 的输入为

$$
x_i = \{m_i, d_i, f_i, t_i, s_i, g_i\}
$$

其中 $m_i$ 为分子表示（SMILES/特征），$d_i$ 为剂量，$f_i$ 为给药频次，$t_i$ 为治疗时间，$s_i$ 为可选表位序列信息，$g_i$ 为可选分组标识。模型学习映射：

$$
\hat{y}_i = F_\theta(x_i), \quad
\hat{y}_i = [\hat{y}^{eff}, \hat{y}^{bind}, \hat{y}^{imm}, \hat{y}^{imm\_cell}, \hat{y}^{infl}, \hat{y}^{tox}]_i
$$

并在动力学后端中估计轨迹函数：

$$
\hat{c}_i(\tau) = G_\phi(\tau \mid x_i)
$$

其中 $G_\phi$ 可由 CTM 或 NDP4PD 实现。

### 疗效 AUC 定义（与实现一致）

在 CTM 轨迹中，综合疗效信号可写为：

$$
s_i(t) = \frac{\gamma_i \cdot E_i(t)}{1 + M_i(t)}
$$

其中 $E_i(t)$ 为效应室状态，$M_i(t)$ 为代谢负荷，$\gamma_i$ 为信号增益（`ctm_signal_gain`）。

样本 $i$ 的疗效 AUC 定义为时间积分：

$$
\mathrm{AUC}^{eff}_i = \int_{0}^{T} s_i(t)\,dt
$$

在离散时间点 $\{t_k\}_{k=0}^{K}$ 上，使用梯形法近似（与 `np.trapezoid/np.trapz` 一致）：

$$
\mathrm{AUC}^{eff}_i \approx \sum_{k=0}^{K-1} \frac{s_i(t_k) + s_i(t_{k+1})}{2}\,(t_{k+1}-t_k)
$$

该值在结果表中对应字段 `ctm_auc_efficacy`。

#### 变量与实现字段对照

| 数学符号 | 含义 | 实现字段 |
| --- | --- | --- |
| $t_k$ | 第 $k$ 个离散时间点（小时） | `time_h` |
| $E_i(t)$ | 效应室状态 | `effect_E` |
| $M_i(t)$ | 代谢负荷状态 | `metabolism_M` |
| $s_i(t)$ | 综合疗效信号 | `efficacy_signal` |
| $\gamma_i$ | 疗效信号增益 | `ctm_signal_gain` |
| $\mathrm{AUC}^{eff}_i$ | 疗效曲线下面积 | `ctm_auc_efficacy` |

#### 论文写作可引用的实现细节

- 时间网格默认采用 1 小时步长（`dt=1.0`），仿真窗口默认 72 小时（`horizon=72`）。
- 给药频次 `freq` 被转换为脉冲给药间隔 `pulse_every = round(24/freq)`，并在离散时点进行加药。
- AUC 使用梯形积分在离散网格上计算；当数值库可用时优先 `np.trapezoid`，否则回退 `np.trapz`，两者在本场景下等价为同一离散积分定义。
- 若曲线为空（异常或缺失场景），实现中将 `auc_efficacy` 置为 0 以保证训练/推理流程稳定。

### 系统流程

1. 输入标准化与质量检查（字段、单位、数值边界）。
2. 特征构建与模型选择（MOE + 候选回归器）。
3. 多任务输出与动力学曲线仿真。
4. 可选免疫 ABM 触发仿真。
5. 可选自适应校准（置信度-风险联合校正）。
6. 可选 ED2Mol + RL 反思式进化闭环（可在进化评分阶段启用自适应校准）。

## 目录说明

- `app.py`：Streamlit 主入口。
- `core/`：建模、特征、动力学、进化、可靠性与训练主逻辑。
- `tests/`：烟雾测试与回归测试。
- `tools/`：复现实验、ED2Mol 辅助脚本、NetLogo 资源。
- `build_full.ps1` / `release_full.ps1`：Windows 下打包与发布脚本。

## 环境准备

推荐 Python 3.10+（Windows）。

```powershell
cd "d:\IGEM集成方案\confluencia-2.0-drug"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> 若仅做算法与 UI 验证，不强依赖 RDKit；安装 RDKit 后将自动启用更完整的分子特征。

## 启动应用

```powershell
cd "d:\IGEM集成方案\confluencia-2.0-drug"
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

## 输入数据格式

应用中用于训练/预测的核心字段（与当前实现保持一致）：

- 必需字段：
  - `smiles`（string）
  - `dose`（float, > 0）
  - `freq`（float, > 0）
  - `treatment_time`（float, >= 0）
- 推荐字段：
  - `epitope_seq`（string，用于免疫相关特征）
  - `group_id`（string，用于分组分析）
- 可选单位字段：
  - `dose_unit`（例如 `mg`）
  - `time_unit`（例如 `h` / `day` / `min`）
- 可选监督标签（提供后进入监督训练模式）：
  - `efficacy`
  - `target_binding`
  - `immune_activation`
  - `immune_cell_activation`
  - `inflammation_risk`
  - `toxicity_risk`
  - CTM 参数标签：`ctm_ka`、`ctm_kd`、`ctm_ke`、`ctm_km`、`ctm_signal_gain`

启用自适应系统后，结果表会新增字段：
在应用结果页会进一步生成可执行建议字段（并随预测 CSV 导出）：

- `adaptive_gate_flag`（`ok`/`review`）

数据校验原则：

- `dose` 与 `freq` 必须严格大于 0。
- `treatment_time` 必须大于等于 0。

## 测试与复现
python .\tests\smoke_test.py
python .\tests\robustness_regression_test.py

一键复现日志（记录时间戳、Python 版本、`pip freeze`，并默认执行 smoke test）：

cd "d:\IGEM集成方案\confluencia-2.0-drug"
powershell -ExecutionPolicy Bypass -File .\tools\reproduce_pipeline.ps1
```

VS Code 任务（可直接运行）：
### 可复现性协议

- 输入数据摘要（样本量、缺失率、标签可用率）。
- 关键随机性设置（随机种子、数据划分策略）。

### 本地构建 dist 目录
```

说明：

- 构建输出目录：`dist/confluencia-2.0-drug`
- 若未指定 `-PythonExe`，脚本将自动查找 `python`

### 生成 release 压缩包

```powershell
cd "d:\IGEM集成方案\confluencia-2.0-drug"
powershell -ExecutionPolicy Bypass -File .\release_full.ps1 -Build -Version full
```

输出位于 `release/`，文件名形如：`confluencia-2.0-drug-full-YYYYMMDD_HHMM.zip`。

## NetLogo 免疫 ABM

1. 从输入或结果 CSV 导出触发事件：

```powershell
cd "d:\IGEM集成方案\confluencia-2.0-drug"
python .\tools\export_epitope_triggers.py --input .\your_data.csv --output .\logs\epitope_triggers.csv
```

2. 打开 `tools/netlogo/immune_response_abm.nlogo`，或将 `tools/netlogo/immune_response_abm.nls` 粘贴到 NetLogo Code 面板。
3. 在 NetLogo Command Center 运行：

```netlogo
setup
load-trigger-events "D:/IGEM集成方案/confluencia-2.0-drug/logs/epitope_triggers.csv" 0
repeat 120 [ go ]
```

4. 重点观察指标：`antigen-pool`、`activated-t-count`、`plasma-b-count`、`antibody-titer`。

## ED2Mol 集成

上游仓库：<https://github.com/pineappleK/ED2Mol>

初始化辅助：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup_ed2mol.ps1
```

运行时在应用内提供：

- `ED2Mol repo dir`（包含 `Generate.py`）
- `ED2Mol config path`（可运行的 YAML）
- `ED2Mol python cmd`（ED2Mol 环境的 Python）

更多说明见 `tools/README-ED2Mol.md`。

## 评价指标建议

根据任务类型推荐：

- 回归任务：MAE、RMSE、$R^2$。
- 稳定性评估：重复划分下的方差统计与置信区间。
- 多目标优化：Pareto front 覆盖率与超体积（若定义参考点）。

在小样本条件下，建议报告均值与不确定性区间，而非单次最优结果。

## 附录 A：符号表（Notation）

| 符号 | 含义 | 备注 |
| --- | --- | --- |
| $i$ | 样本索引 | $i=1,2,\dots,N$ |
| $x_i$ | 第 $i$ 个样本输入向量 | 含分子、剂量、频次、时间与可选上下文 |
| $m_i$ | 分子表示 | SMILES 或特征化后向量 |
| $d_i$ | 剂量 | 对应字段 `dose` |
| $f_i$ | 给药频次 | 对应字段 `freq` |
| $t_i$ | 治疗时间 | 对应字段 `treatment_time` |
| $s_i$ | 表位序列上下文 | 可选字段 `epitope_seq` |
| $g_i$ | 分组标识 | 可选字段 `group_id` |
| $F_\theta$ | 预测模型映射 | MOE 驱动的多任务预测函数 |
| $\hat{y}_i$ | 多指标预测输出 | 含疗效、结合、免疫、炎症、毒性 |
| $G_\phi$ | 动力学轨迹函数 | CTM 或 NDP4PD 后端 |
| $\hat{c}_i(\tau)$ | 在时间 $\tau$ 的预测轨迹值 | 用于动力学分析 |
| $\theta$ | 预测模型参数 | 学习得到 |
| $\phi$ | 动力学模型参数 | 学习或推断得到 |

## 附录 B：默认实验超参数（建议报告）

下表为建议在论文中显式报告的默认实验配置。若与实际实现不一致，请以运行日志与代码为准。

| 模块 | 参数项 | 建议默认值 | 说明 |
| --- | --- | --- | --- |
| 数据划分 | 训练/测试比例 | 0.8 / 0.2 | 小样本下建议多次重复划分 |
| 交叉验证 | 折数 $K$ | 5 | 可报告均值与标准差 |
| 随机性 | random seed | 42 | 需在报告中固定并公开 |
| 预处理 | 数值缩放 | StandardScaler | 与回归器组合为 pipeline |
| 候选模型 | 线性模型 | LinearRegression / Ridge | 作为低方差基线 |
| 候选模型 | 树模型 | RandomForest / HistGBR | 捕捉非线性关系 |
| 多任务输出 | 指标集合 | efficacy + 机制指标 | 支持部分标签训练 |
| 动力学后端 | backend | ctm 或 ndp4pd | 建议做消融对比 |
| 评估指标 | 回归指标 | MAE / RMSE / $R^2$ | 主结果建议三者同时报告 |
| 不确定性 | 统计方式 | 重复实验置信区间 | 建议报告 95% CI |

补充建议：

- 在数据量较小时，优先报告方差与稳定性结论。
- 在多目标情形下，同时报告单指标最优与 Pareto 集性能。

## 附录 C：实验报告模板（可直接复用）

以下模板可直接用于实验记录、附录或内部复盘。

```markdown
# Experiment Report: Confluencia 2.0 Drug Module

## 1. Run Metadata
- Date:
- Commit ID:
- Operator:
- Host OS:
- Python Version:
- Key Dependencies Snapshot: (attach `pip freeze`)

## 2. Data Summary
- Dataset Name:
- Sample Size (N):
- Required Field Completeness (`smiles`, `dose`, `freq`, `treatment_time`):
- Optional Label Availability:
  - efficacy:
  - target_binding:
  - immune_activation:
  - immune_cell_activation:
  - inflammation_risk:
  - toxicity_risk:
- Distribution Notes / Potential Shift:

## 3. Experimental Settings
- Split Strategy:
- Cross Validation K:
- Random Seed:
- Dynamics Backend (`ctm` / `ndp4pd`):
- Feature Mode (RDKit / fallback hash):
- Training Profile / Compute Profile:

## 4. Main Results
- Regression Metrics:
  - MAE:
  - RMSE:
  - R^2:
- Stability (Repeated Runs):
  - Mean:
  - Std:
  - 95% CI:

## 5. Ablation / Sensitivity
- Remove epitope features:
- Remove group information:
- CTM vs NDP4PD:
- Notes:

## 6. Optimization Loop (If ED2Mol Enabled)
- Number of Rounds:
- Reward Design:
- Pareto Front Size:
- Best Candidate Summary:

## 7. Reproducibility Artifacts
- Reproduce Script Log Path:
- Model Export Path:
- Input Snapshot Path:
- Output Snapshot Path:

## 8. Risk & Limitation Notes
- Data bias:
- Label noise:
- External tool dependency risk:
- Conclusion boundary:
```

## 论文草稿 A：方法章节（中文期刊风格）

以下文字可直接作为方法部分初稿。

### A.1 问题定义

本文关注小样本条件下的药物疗效与机制联合预测问题。对第 $i$ 个样本，输入记为：

$$
x_i = \{m_i, d_i, f_i, t_i, s_i, g_i\}
$$

其中，$m_i$ 为分子表示，$d_i$ 为给药剂量，$f_i$ 为给药频次，$t_i$ 为治疗时间，$s_i$ 为可选表位序列上下文，$g_i$ 为可选分组信息。模型学习映射 $F_\theta$，输出多任务预测向量：

$$
\hat{y}_i = [\hat{y}^{eff}, \hat{y}^{bind}, \hat{y}^{imm}, \hat{y}^{imm\_cell}, \hat{y}^{infl}, \hat{y}^{tox}]_i
$$

### A.2 模型总体框架

方法由两级模块构成。第一级为静态预测模块，采用 MOE 与多头预测器估计疗效及机制相关指标。第二级为动力学模块，将第一级输出映射为动力学参数，进而在 CTM 或 NDP4PD 后端上生成时间轨迹。该结构兼顾了预测性能与机制可解释性。

### A.3 CTM 轨迹与疗效信号构建

在 CTM 分支中，系统分别模拟吸收、分布、效应与代谢状态。综合疗效信号定义为：

$$
s_i(t) = \frac{\gamma_i \cdot E_i(t)}{1 + M_i(t)}
$$

其中，$E_i(t)$ 表示效应室状态，$M_i(t)$ 表示代谢负荷，$\gamma_i$ 为样本级信号增益参数。

### A.4 疗效 AUC 的计算

样本级疗效强度采用曲线下面积表示：

$$
\mathrm{AUC}^{eff}_i = \int_0^T s_i(t)\,dt
$$

在离散时间网格上，采用梯形积分近似：

$$
\mathrm{AUC}^{eff}_i \approx \sum_{k=0}^{K-1} \frac{s_i(t_k)+s_i(t_{k+1})}{2}(t_{k+1}-t_k)
$$

实现中默认时间步长为 1 小时、仿真窗口为 72 小时，对应输出字段为 `ctm_auc_efficacy`。

### A.5 训练与推理流程

在机制标签不完整场景中，系统支持代理标签与启发式参数先验；在标签可用场景中，采用监督参数学习。最终结果同时输出静态任务预测与动力学汇总指标（峰值、AUC、风险量化），用于候选筛选与后续优化。

## 论文草稿 B：实验章节模板（中文期刊风格）

以下结构可直接用于实验部分。

### B.1 数据集与预处理

实验数据至少包含 smiles、dose、freq、treatment_time 四项核心字段。针对具备标签的样本，纳入 target_binding、immune_activation、immune_cell_activation、inflammation_risk 与 toxicity_risk 等机制指标。所有数值变量先进行数据清洗与边界约束，确保剂量与频次严格为正。

### B.2 对比方法与消融设计

1. 主方法：MOE + 动力学后端（CTM 或 NDP4PD）。
2. 对比方法 1：仅静态回归，不构建动力学轨迹。
3. 对比方法 2：动力学后端替换（CTM 与 NDP4PD 互换）。
4. 消融实验 1：移除表位相关输入。
5. 消融实验 2：移除分组信息。

### B.3 评价指标

静态预测性能采用 MAE、RMSE 与 $R^2$。动力学层采用疗效峰值与疗效 AUC。稳定性通过重复随机划分的均值、标准差与 95% 置信区间评估。

### B.4 实验协议与复现设置

采用固定随机种子和重复实验策略。每次实验记录运行时间戳、依赖快照、关键超参数及模型后端配置，确保结果具备可追溯性与可复现性。结果报告以重复统计为主，不采用单次最优结果作为主要结论。

### B.5 结果呈现建议

正文建议按“主结果-稳定性-消融分析”三段式组织：先报告总体性能，再说明动力学模块对峰值与 AUC 的增益，最后讨论小样本场景下的方差特征与泛化边界。

## 论文草稿 C：附录模板（中文期刊风格）

### C.1 运行环境说明

1. 硬件与操作系统信息。
2. Python 版本与关键依赖版本（附依赖清单）。
3. 代码版本标识（commit id 或发布版本号）。

### C.2 超参数与实现细节

1. 数据划分比例、重复次数与随机种子。
2. 主模型与候选基学习器超参数。
3. 动力学仿真参数（时间步长、仿真窗口、后端选择）。

### C.3 补充结果

1. 各次重复实验的完整指标表。
2. 关键样本轨迹补充图（效应、毒性、AUC）。
3. 失败案例与异常样本讨论。

### C.4 可复现材料清单

1. 输入数据快照（脱敏）。
2. 训练与推理命令及参数。
3. 输出文件与日志路径。

## 论文草稿 D：结果表格模板（可直接填数值）

可使用脚本自动从复现实验日志生成 D.1-D.3 对应表格：

```powershell
cd "d:\IGEM集成方案\confluencia-2.0-drug"
.\.venv\Scripts\python.exe .\tools\export_paper_tables.py
```

默认输出：`dist/logs/reproduce/paper_tables.md`。

### D.0 字段映射（与当前实现一致）

| 论文指标名 | 代码字段/键名 | 来源 |
| --- | --- | --- |
| MAE | `efficacy_mae` | `train_report.metrics` |
| RMSE | `efficacy_rmse` | `train_report.metrics` |
| $R^2$ | `efficacy_r2` | `train_report.metrics` |
| Peak Efficacy | `ctm_peak_efficacy` | 结果 DataFrame 列 |
| AUC(Efficacy) | `ctm_auc_efficacy` | 结果 DataFrame 列 |

建议口径：`ctm_peak_efficacy` 与 `ctm_auc_efficacy` 按测试集样本取均值后填入主表。

### D.1 主结果表（建议放正文）

| 方法 | Dynamics Backend | MAE (`efficacy_mae`) ↓ | RMSE (`efficacy_rmse`) ↓ | $R^2$ (`efficacy_r2`) ↑ | Peak Efficacy (`ctm_peak_efficacy`) ↑ | AUC(Efficacy) (`ctm_auc_efficacy`) ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| 静态回归基线 | - |  |  |  | - | - |
| MOE（无动力学） | - |  |  |  | - | - |
| 本方法 | CTM |  |  |  |  |  |
| 本方法 | NDP4PD |  |  |  |  |  |

注：箭头表示指标优化方向。

### D.2 消融实验表

| 配置 | MAE (`efficacy_mae`) ↓ | RMSE (`efficacy_rmse`) ↓ | $R^2$ (`efficacy_r2`) ↑ | Peak Efficacy (`ctm_peak_efficacy`) ↑ | AUC(Efficacy) (`ctm_auc_efficacy`) ↑ |
| --- | --- | --- | --- | --- | --- |
| Full Model |  |  |  |  |  |
| w/o epitope feature |  |  |  |  |  |
| w/o group id |  |  |  |  |  |
| CTM -> NDP4PD |  |  |  |  |  |

### D.3 稳定性统计表（重复实验）

| 指标 | Mean | Std | 95% CI Lower | 95% CI Upper |
| --- | --- | --- | --- | --- |
| MAE (`efficacy_mae`) |  |  |  |  |
| RMSE (`efficacy_rmse`) |  |  |  |  |
| $R^2$ (`efficacy_r2`) |  |  |  |  |
| Peak Efficacy (`ctm_peak_efficacy`) |  |  |  |  |
| AUC(Efficacy) (`ctm_auc_efficacy`) |  |  |  |  |

### D.4 统计显著性报告模板（可选）

| 对比组 | 指标 | 差值 (Ours-Baseline) | 检验方法 | p-value |
| --- | --- | --- | --- | --- |
| Ours vs Baseline-1 | MAE |  | 配对 t 检验 / Wilcoxon |  |
| Ours vs Baseline-1 | AUC(Efficacy) |  | 配对 t 检验 / Wilcoxon |  |
| Ours vs Baseline-2 | MAE |  | 配对 t 检验 / Wilcoxon |  |
| Ours vs Baseline-2 | AUC(Efficacy) |  | 配对 t 检验 / Wilcoxon |  |

## 局限性与使用边界

- 当前实现属于研究型原型，主要用于方法验证与流程联调。
- 模型预测受样本规模、标签噪声与分布偏移影响。
- ED2Mol 与外部模拟工具的结果受其独立配置与环境状态影响。
- 免疫 ABM 为简化机制模型，不等同于真实临床免疫动力学。

## 免责声明

本项目用于研究原型与方法验证，不构成临床建议。
