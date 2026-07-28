# Circ-CASP: circRNA 3D Structure Prediction Challenge

## 竞赛概述

Circ-CASP（Critical Assessment of circRNA Structure Prediction）是首个专注于 circRNA 三维结构预测的公开竞赛，旨在推动 circRNA 结构预测方法的发展。

### 背景
circRNA（环状 RNA）是一种通过反向剪接形成的共价闭合环状 RNA 分子。由于其独特的拓扑结构（5'-3' 端连接形成环），传统的线性 RNA 结构预测方法难以直接应用。Circ-CASP 旨在建立 circRNA 3D 结构预测的评估标准和基准数据集。

---

## 竞赛规则

### 1. 数据集

| 数据类型 | 数量 | 来源构成 | 用途 |
|----------|------|----------|------|
| 训练集 | **130,000** | 见下方"训练集来源构成"明细 | 公开提供（CC-BY 4.0） |
| 测试集 | 30 | 物理高仿真结构（IsRNAcirc + Rosetta FARFAR2 交叉验证） | 预测时公开序列，结果保密 |

**训练集来源构成（130,000 条明细）：**

| 来源 | 数量 | 占比 | 置信度 | 说明 |
|------|------|------|--------|------|
| **合成数据** | 60,000 | 46.2% | 中（伪标签） | 基于理想 A-form 螺旋 + 随机基序插入生成，覆盖 50-500 nt 与 1000-5000 nt 两档 |
| **IsRNAcirc 物理求解扩增** | 50,000 | 38.5% | 高（物理仿真） | IsRNAcirc 求解器生成，经能量最小化与构象采样，覆盖全长度区间 |
| **公共数据库融合** | 20,000 | 15.3% | 高（实验/文献） | circBase + CIRCpedia 过滤后的高质量条目，附实验来源标注 |

> **来源透明度声明：** 三类数据在 metadata 中通过 `source_type` 字段（`synthetic` / `isrnacirc` / `public_database`）显式标注，参赛者可按置信度筛选训练子集。合成 circRNA 样本（混合在 13 万条中）采用三层加权融合（合成 0.3 / IsRNAcirc 0.5 / 公共库 0.8），权重写入每条数据的 `confidence_weight` 字段。

**训练集特征：**
- 序列长度范围：50-5000 nt（分档：50-500 nt 合成主档 / 500-1000 nt 补充档 / 1000-5000 nt 长序列档）
- 包含：序列、二级结构（部分）、伪标签 3D 坐标、来源标注、置信度权重
- 格式：JSON + NPY（兼容 PyTorch/TensorFlow）

> **设计说明：** 训练集覆盖 50-5000 nt 全区间，重点补充 500-1000 nt 中等长度档（真实 circRNA 最常见区间，初版在该档信号不足导致 M7/Scheme 7 在 L>500 性能断崖式下降）。三类来源交叉覆盖三档长度，确保每个长度区间都有高置信度样本。

**测试集特征：**
- 30 个真实 circRNA（来自纯物理预测的高质量结构）
- 预测阶段：仅公开序列信息
- 评估阶段：公开真实 3D 结构和评分

### 2. 预测目标

每个 circRNA 包含以下预测任务：

| 任务编号 | 任务名称 | 描述 | 权重 |
|----------|----------|------|------|
| T1 | **整体结构** | 全原子 RMSD | 40% |
| T2 | **BSJ 闭合** | 5'-3' 端距离误差 | 20% |
| T3 | **骨架构象** | 相邻核苷酸距离一致性 | 15% |
| T4 | **二级结构** | 配对碱基预测准确性 | 15% |
| T5 | **构象多样性** | 提供多个候选构象 | 10% |

### 3. 评分标准

#### T1: 整体结构 RMSD
$$\text{RMSD} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}||p_i - t_i||^2}$$

**对齐要求：** 计算 RMSD 前，先对 C3' 原子做 **Kabsch 最优刚体对齐**（最小化旋转+平移使预测与真实结构重合），再计算 RMSD。未做对齐的 RMSD 不予采纳。

评分规则（阶梯式，越短越好）：
| RMSD | 得分 |
|------|------|
| < 5 Å | 100 |
| < 10 Å | 80 |
| < 15 Å | 60 |
| < 20 Å | 40 |
| < 30 Å | 20 |
| ≥ 30 Å | 0 |

> **注意：** 此为阶梯评分，非附录脚本中的线性衰减公式 `max(0, 100 - 3.33 × RMSD)`。后者仅作参考，正式评分以本表为准。

#### T2: BSJ 闭合距离
$$d_{\text{BSJ}} = ||p_0 - p_{N-1}||$$

真实磷酸二酯键长度约 5.9 Å（首尾核苷酸 C3' 原子间距）。

**碱基对定义：** circRNA 二级结构配对采用 Watson-Crick 几何判定——两碱基的 **C3'-C3' 距离 < 15 Å** 且碱基朝向满足标准 A-form 堆叠角度；BSJ 跨越区的配对单独处理：若 i 与 j 跨越剪接位点（i < BSJ_idx ≤ j），则该配对不计入 T4 评分（避免拓扑干扰）。

评分规则（绝对合理性，以理想值 5.9 Å 为基准）：
| |d - 5.9| | 得分 |
|------|---------|------|
| < 1 Å | 100 |
| < 2 Å | 80 |
| < 5 Å | 60 |
| < 10 Å | 40 |
| ≥ 10 Å | 0 |

#### T3: 骨架距离一致性
相邻核苷酸 C3' 原子距离应接近 5.9 Å（A-form RNA）。此任务评估**预测结构是否符合 A-form 几何约束**，而非与真实结构的匹配程度。

$$\text{Bond\_Score} = \frac{1}{N-1}\sum_{i=1}^{N-1} \max(0, 100 - 20 \times |d_i - 5.9|)$$

> **注意：** 此为"绝对合理性"评分——只要预测键长偏离理想值 5.9 Å 就扣分，无论真实结构如何。不同于附录脚本中的 `|bond_pred - bond_true|`（相对准确性），本规则采用绝对合理性以避免"两边都错但错得一致"的钻空子行为。

#### T4: 二级结构配对
碱基配对预测准确性（AU/GC/GU 配对）。

**配对定义：** 两个核苷酸 i < j 构成配对，当且仅当满足以下全部条件：
1. **距离阈值：** C3'-C3' 原子间距 < 15 Å
2. **几何约束：** 碱基朝向满足 Watson-Crick A-form 堆叠角度（与相邻碱基的螺旋参数一致）
3. **BSJ 跨越排除：** 若 i < BSJ_idx ≤ j（即配对跨越剪接位点），该配对不计入评分

$$\text{Pair\_F1} = \frac{2 \times \text{TP}}{2 \times \text{TP} + \text{FP} + \text{FN}}$$

其中 TP/FP/FN 基于上述配对集合计算。

#### T5: 构象多样性（可选）
提交多个候选构象（最多 5 个），取最佳评分。

**评分细则：**
1. **基础分：** 从 5 个构象中选择 T1-T4 综合评分最高的一个，作为该目标的 T5 基础分（满分 60 分）
2. **多样性奖励：** 若提交的构象之间 RMSD 标准差 > 5 Å（即有实质性结构差异），额外奖励 40 分
3. **无多样性惩罚：** 若 RMSD 标准差 < 1 Å（即 5 个构象几乎相同），T5 总分仅得基础分 60 分 × 0.5 = 30 分

$$\text{T5\_Score} = \min(\text{best\_among\_5}, 60) + \begin{cases}
40, & \sigma_{\text{RMSD}} > 5Å \\
20, & \sigma_{\text{RMSD}} > 2Å \\
0, & \text{otherwise}
\end{cases}$$

> **注意：** 此设计防止"提交 5 个完全一样的构象"钻空子，鼓励真实的构象搜索。

### 4. 总分计算

$$\text{Total\_Score} = \sum_{i=1}^{5} w_i \times \text{Score}_i$$

总分范围：0-100 分。

---

## 学术产出与署名

Circ-CASP 旨在成为 circRNA 结构预测领域的长期学术基准。竞赛本身即是学术产出——参赛者不仅是"打比赛"，更是参与一项可被领域永久引用的学术基准建设。本章节明确参赛者从竞赛中可获得的学术资本。

### 数据集公开与引用

| 机制 | 说明 |
|------|------|
| **协议** | 训练集采用 CC-BY 4.0 协议发布，允许商用，仅需引用 |
| **永久标识** | 训练集与揭盲后测试集发布至 Zenodo，分配 DOI，版本化（v1.0, v1.1…） |
| **数据集论文** | 主办方撰写 circRNA 结构数据集论文，投稿至 *NAR Database Issue* / *Bioinformatics* |
| **测试集揭盲** | 评估结束后，30 个真实结构 + 所有参赛预测 + 评分永久公开，成为领域 benchmark |
| **数据卡** | 每条数据附 metadata：序列来源、长度、GC 含量、二级结构来源、3D 生成方法、置信度 |

> 任何使用 Circ-CASP 数据集的论文须引用数据集 DOI。数据集被引次数随时间复利增长，是竞赛给领域的长期公共资产。

### 盲测公信力

| 机制 | 说明 |
|------|------|
| **提交即锁定** | 每队最多 3 次提交，每次提交生成 SHA-256 哈希并公开存档，截止后不可修改 |
| **时间戳公证** | 截止时刻所有提交打包计算 Merkle root，公开发布至 arXiv/GitHub，任何人都可验证"预测早于揭盲" |
| **测试集来源声明** | 30 个测试集真实结构的生成方法（物理求解器版本、参数、收敛判据）公开，并经 2+ 种独立方法交叉验证 |
| **第三方评估委员会** | 邀请非参赛的 RNA 结构专家（海外/非吉大）组成评估委员会，独立审核评分过程，消除"既当裁判又当参赛者"嫌疑 |

> 这些机制确保 Circ-CASP 的盲测结果可被任何论文引用而不被质疑——"我们方法在 Circ-CASP 2026 上达到 RMSD X Å"这句话才有学术分量。

### 竞赛论文产出（三类）

竞赛结束后将产出三类论文，参赛者按贡献获得署名：

| 论文类型 | 内容 | 目标期刊 | 署名机制 |
|---------|------|----------|---------|
| **数据集论文** | Circ-CASP 数据集描述 + 基线评估 | *Bioinformatics* / *NAR Database* | 主办方 + 数据贡献者 |
| **方法学评估论文** | 所有参赛方法系统对比 + 方法学洞察 | *Nucleic Acids Res.* / *Nature Methods* (letter) | 主办方 + **所有有效方法参赛者**（集体署名 / 贡献者列表） |
| **领域综述论文** | 基于竞赛结果综述"circRNA 结构预测领域现状与挑战" | *Trends Biochem. Sci.* / *WIREs RNA* | 主办方 + 优胜者代表 |

> **关键：** 只要提交一个有效方法（达到参赛门槛），即可在方法学评估论文上署名。这是竞赛给参赛者最实在的学术资本——一篇领域顶刊评估论文的共署作者，对申博/求职/基金申请是硬通货。

### 方法学贡献署名

- 参赛方法若开源，将被收录进 Circ-CASP **官方评估框架**（GitHub 开源项目）
- 方法贡献者在评估框架中署名，可被后续工作正式引用
- 评估框架包含：评估脚本、数据加载器、baseline 方法、参赛方法包装器——成为领域基础设施

### 届次化

- Circ-CASP 2026 为第一届，计划**每两年举办一届**
- 历届数据集与结果永久可查，形成领域长期基准
- 优胜者方法自动入选下届 baseline，持续积累学术声誉

### Hub 集成：模型永久托管与流转

Circ-CASP 与 Confluencia Hub（`hub.confluencia.org`，HuggingFace 后端）深度集成——参赛模型提交后**自动入 Hub**，获得永久 DOI、下载追踪与 baseline 流转。这是参赛者学术资本的核心载体。

| 流程 | 机制 | 参赛者的"利" |
|------|------|-------------|
| **提交即托管** | 参赛者调用 `hub.push_circ_casp_submission()` 上传 `.joblib` 模型，task 强制为 `circRNA`，竞赛成绩（RMSD/T1-T5/总分/排名）自动绑定到 metadata | 模型永久可下载、可复现 |
| **ORCID 绑定** | 上传须提供 `uploader_orcid`（ISO 7064 校验），模型 ID 格式 `hub:circRNA:{orcid_short}:{hash}` | 贡献可追溯到具体学者 |
| **Zenodo DOI 申领** | 配置 `CONFLUENCIA_ZENODO_TOKEN` 后，每次上传自动申领 DOI 并写入 model card 的 BibTeX 引用块 | 论文可正式引用 `doi:10.5281/zenodo.XXX` |
| **质量分层** | 提供推理代码 repo → `reproducible` 层；盲测发布后经委员会审核 → `verified`/`benchmark_top` 层 | 优质模型浮顶，劣质下沉 |
| **下载追踪** | Hub 自动统计每个模型的下载次数，按梯度发影响力徽章：🥉≥100 / 🥈≥500 / 🥇≥1000 | "我的模型被下载 N 次"可写进简历 |
| **贡献者年报** | `hub.get_contributor_stats(orcid)` 聚合该学者所有模型的下载/引用/徽章 | CV/基金申请硬通货 |
| **Baseline 流转** | `benchmark_top` 层的 circRNA 模型自动打 `circ-casp-baseline` tag，入选下届 Circ-CASP baseline | 跨届持续被对比引用 |

**部署与使用：**
- Hub 后端部署指南见 `docs/hub_deployment.md`
- Python 接口：`from confluencia_cli.hub import ConfluenciaHub`
- R 接口：`cf_hub_push_model()` / `cf_hub_contributor_stats()` 等（见 `confluencia-rpkg`）

**降级策略：** 无 `CONFLUENCIA_HF_TOKEN` 时 Hub 自动降级为本地缓存模式，不报错但不跨机器共享；无 `CONFLUENCIA_ZENODO_TOKEN` 时跳过 DOI 申领，模型仍可上传（model card 显示"Cite as: Confluencia Hub model `hub:...`"）。

---

## 提交格式

### 文件结构
```
submission/
├── team_info.json         # 队伍信息 + 署名同意
├── predictions/
│   ├── circ_001_coords.npy  # 预测坐标 (N, 3)
│   ├── circ_001_pairs.json  # 预测配对
│   ├── circ_002_coords.npy
│   ├── ...
│   └── circ_030_coords.npy
├── method_description.md   # 方法描述
├── inference.py            # 最小可复现推理脚本（算力合规验证）
└── LICENSE                 # 方法开源协议（可选，MIT/Apache-2.0/CC-BY-4.0）
```

### team_info.json

```json
{
  "team_name": "Your Team Name",
  "contact_email": "email@example.com",
  "members": [
    {"name": "Member 1", "affiliation": "Institution", "orcid": "0000-0000-0000-0000"}
  ],
  "method_description": "Brief description",
  "method_repo_url": "https://github.com/team/circ-casp-method",
  "method_license": "MIT",
  "attribution_consent": true,
  "publish_consent": true
}
```

**字段说明：**
- `orcid`：推荐填写，用于论文署名时的身份绑定
- `method_repo_url`：方法开源仓库地址（若开源，将被收录进官方评估框架）
- `method_license`：开源协议（MIT / Apache-2.0 / CC-BY-4.0 推荐）
- `attribution_consent`：是否同意在 Circ-CASP 评估论文/数据集论文中署名（须为 true 方可参赛）
- `publish_consent`：是否同意预测结果在揭盲后公开（须为 true 方可参赛）

### predictions/circ_XXX_coords.npy

- NumPy array: shape (N, 3), dtype float32
- N = 序列长度
- 坐标单位：Ångstrom
- 原子类型：C3' 原子（可选用 P 原子）

### 多构象提交（T5 可选）

若参与 T5 构象多样性任务，每个目标可提交最多 5 个构象：
```
predictions/
├── circ_001_coords.npy          # 构象 1（主构象，必交）
├── circ_001_conf_2.npy          # 构象 2（可选）
├── circ_001_conf_3.npy          # 构象 3（可选）
├── ...
└── circ_001_conf_5.npy          # 构象 5（可选）
```

---

## 时间安排

| 时间节点 | 事项 |
|----------|------|
| 7月10日 | 公布训练集（CC-BY 4.0，Zenodo DOI） |
| 第 1-3 周 | 模型训练阶段 |
| 第 4 周 | 公布测试集序列（揭盲前） |
| 第 5 周 | 提交预测结果（SHA-256 锁定，截止时 Merkle root 公开） |
| 第 6 周 | 第三方评估委员会审核评分，公布评分结果 |
| 第 7 周 | 揭盲：30 个真实结构 + 所有预测 + 评分永久公开 |
| 第 8-12 周 | 数据集论文与方法学评估论文撰写，参赛者确认署名信息 |
| 第 6 个月 | 数据集论文投稿（*NAR Database* / *Bioinformatics*） |
| 第 9 个月 | 方法学评估论文投稿（*Nucleic Acids Res.* / *Nature Methods*） |

---

## 参赛门槛

为确保竞赛质量，所有提交必须满足以下**最低有效性要求**：

### 署名与公开同意（必填）

参赛者须在 `team_info.json` 中确认以下两项均为 `true`，否则不予评估：

| 同意项 | 含义 | 缺失后果 |
|--------|------|----------|
| `attribution_consent` | 同意在 Circ-CASP 数据集论文/方法学评估论文中署名 | 不予评估 |
| `publish_consent` | 同意预测结果在揭盲后永久公开 | 不予评估 |

> 这两项是竞赛学术产出的法律基础。署名同意确保参赛者获得应得学术资本，公开同意确保 benchmark 可被领域永久引用。

### 结构合理性门槛

| 检查项 | 最低要求 | 不合格后果 |
|--------|----------|------------|
| BSJ 闭合距离 | < 20 Å（真实值 ~5.9 Å） | 该目标 0 分 |
| 骨架键长均值 | 3.0-10.0 Å（真实值 ~5.9 Å） | 该目标 0 分 |
| 坐标非全零/全等 | std(coords) > 1.0 Å | 整个提交 disqualified |
| 坐标非随机 | 与随机坐标的 RMSD 必须优于 1σ | 整个提交 disqualified |

### 最低总分门槛

| 条件 | 阈值 | 说明 |
|------|------|------|
| 平均总分 | ≥ 10 分 | 30 个目标平均分 |
| 有效目标数 | ≥ 20 / 30 | 至少 20 个目标得分 > 0 |

未达门槛的提交不予排名，但可以修改后重新提交（限 3 次）。

**注意：** 随机数挑战赛道不受此门槛限制（但得分另行排名）。

### 违规示例

| ❌ 淘汰操作 | 原因 |
|-------------|------|
| `coords = np.random.randn(N, 3)` | 随机坐标，BSJ ~0，骨架键长 ~1.4 |
| `coords = np.zeros((N, 3))` | 全零坐标 |
| `coords = np.ones((N, 3)) * 5.9` | 所有原子重叠 |
| `coords = np.loadtxt('training_set/sample_001.npy')` | 抄训练集 |
| 手动在 PyMOL 里拖拽 | 不可复现 |

---

## 算力限制

为确保公平竞争，防止"暴力物理模拟"碾压所有方法，竞赛对算力做出以下限制：

### 单目标推理限制

| 资源 | 限制 | 说明 |
|------|------|------|
| **GPU 时间** | ≤ 10 分钟 / 目标 | 单个 circRNA 单卡推理时间 |
| **GPU 内存** | ≤ 24 GB | 单卡最大显存占用 |
| **GPU 数量** | ≤ 1 | 禁止多卡并行推理 |
| **CPU 时间** | ≤ 60 分钟 / 目标 | 纯 CPU 方法的时间限制 |
| **内存** | ≤ 64 GB | 系统内存上限 |

### 总算力预算

| 项目 | 限制 |
|------|------|
| **训练总 GPU 时间** | ≤ 100 GPU·小时 |
| **训练 GPU 型号** | 不限（需报告） |
| **推理总 GPU 时间** | ≤ 5 GPU·小时（30 目标合计） |
| **物理模拟步数** | ≤ 10,000 步 / 目标 |

### 合规方法 vs 违规方法 （示例）

| ✅ 允许 | ❌ 禁止 |
|---------|---------|
| 深度学习前向推理（秒级） | Rosetta 全原子模拟（小时级） |
| EGNN/GNN 快速预测 | IsRNAcirc 物理求解器 |
| Diffusion 快速采样 | 分子动力学 MD 模拟 |
| Mamba SSM 前向推理（O(L)，秒级） | 大规模构象搜索 |
| 轻量物理约束优化 | 蒙特卡洛大规模采样 |
| 预训练 + 微调 | |

### 合规声明

提交结果时必须附上：

1. **硬件信息**：GPU 型号、数量、推理时间
2. **方法描述**：是否使用物理模拟、模拟步数
3. **代码提交**：核心推理代码（用于验证算力合规性）
4. **最小可复现单元**：`README.md` + `inference.py`（或等价脚本），无需依赖完整训练流程即可在单卡上重跑预测
5. **Batching 规则**：若一次推理处理多个目标（batch > 1），该次推理计为所有 batch 内目标各消耗 10 min GPU 时间；禁止多卡并行推理

无法提供以上信息或信息不实的，成绩不予认可。

### 举报与申诉

- 任何参赛者可举报疑似违规的算力使用
- 组委会将要求被举报方提供推理日志（含 GPU 时间戳、显存峰值记录）
- 确认违规的，取消该队全部成绩

---

## 基准方法

主办方提供 6 种基准方法供参考，这些基准方法也会并行参赛（但不占名额），
也欢迎开发自己的算法(另：除了机器学习方法只要你数学/物理基础够强，纯数学/物理预测模型也被接受)：

| 方法编号 | 方法名称 | 类型 | 预计性能 |
|----------|----------|------|----------|
| M1 | Helical Baseline | Physics | RMSD ~25 Å |
| M2 | EGNN + Physics | DL + Physics | RMSD ~15 Å |
| M3 | Dual-Engine Iterative | Hybrid | RMSD ~12 Å |
| M4 | DDPM Guided Diffusion | DL | RMSD ~10 Å |
| M5 | Physics-Biased Transformer | DL + Physics | RMSD ~8 Å |
| M6 | GNN Latent Diffusion | DL | RMSD ~10 Å |
| M7 | Mamba+Transformer Hybrid | DL | RMSD ~8 Å (L≤500), ~12 Å (L>500) |

---

## 竞赛奖项

### 奖励说明

> *"奖金有限，但影响力无价。"*

除现金奖励外，所有获奖者将获得：
- 📌 **树洞置顶**：获奖帖子在吉大树洞置顶展示
- 👤 **个人主页**：在 Circ-CASP 主页永久展示获奖者信息与方法介绍
- 📜 **电子证书**：官方认证获奖证书

### 正式奖项（同一队可同时获得）

| 奖项 | 条件 | 数量 | 奖金 |
|------|------|------|------|
| 🏆 冠军 | 总分最高 | 1 队 | 100元 |
| 🥈 亚军 | 总分第二 | 1 队 | 30元 |
| 🥉 季军 | 总分第三 | 1 队 | 10元 |
| ⭐ 创新奖 | 方法新颖性 | 2 队 | 10元 |
| 📊 最佳单项 | T1-T5 单任务满分 | 各1队 | 各10元 |

### 无限制奖项（至少满足门槛）

| 奖项 | 条件 | 奖金 | 说明 |
|------|------|------|------|
| 💡 **最省油的灯奖** | 最小算力消耗 | 5元 | 训练+推理总 GPU 时间最少 |
| 🧠 **最小脑容量奖** | 最小显存占用 | 5元 | 推理时峰值显存最小（Scheme 7 O(L)可达 ~8GB） |
| 🏃 **长跑冠军奖** | 长序列最佳 | 5元 | L>1000 的目标平均分最高（Scheme 7 唯一方案） |
| 🌱 **自食其力奖** | 零预训练模型 | 5元 | 不使用任何预训练权重 |
| 🎨 **最佳美术奖** | 可视化最漂亮 | 5元 | 提交的结构图最美观 |
| 🐢 **稳扎稳打奖** | 方差最小 | 5元 | 30 个目标得分标准差最小 |
| 🎯 **神枪手奖** | BSJ 完美闭合 | 5元 | T2 任务平均分最高 |
| 🔄 **环形大师奖** | 环结构最准 | 5元 | 整体拓扑最接近真实 circRNA（Scheme 7 circular scan+BSJ flank） |

### 无限制奖项评选规则

1. **最省油的灯奖**：`total_gpu_seconds / mean_score` 最小
2. **最小脑容量奖**：`peak_memory_mb / mean_score` 最小
3. **长跑冠军奖**：仅统计长度 >1000 的目标，平均分最高
4. **自食其力奖**：声明"无预训练"且提供从头训练代码
5. **最佳美术奖**：第三方专业人员投票（可提交 PyMOL/Matplotlib 渲染图）
6. **稳扎稳打奖**：`std(scores)` 最小
7. **神枪手奖**：T2 平均分最高
8. **环形大师奖**：BSJ + Bond 综合评分最高

### 🎰 特别赛道：随机数挑战（欧皇奖）

> *"如果运气也算实力的话。"*

**规则：** 允许参赛者提交一个随机种子，组委会用该种子生成 3D 坐标，并纳入正式评分。

**生成方式：**
```python
import numpy as np
def oracle_predict(sequence, seed):
    rng = np.random.RandomState(seed)
    N = len(sequence)
    # 生成 A-form 螺旋 + 随机扰动
    coords = np.zeros((N, 3))
    for i in range(N):
        angle = 2 * np.pi * i / N
        radius = 5.9 * N / (2 * np.pi) * 0.5
        coords[i] = [radius * np.cos(angle), radius * np.sin(angle), 2.8 * i]
    coords += rng.normal(0, 3.0, (N, 3))  # 随机扰动尺度
    return coords
```

**提交格式：** `seeds.json`
```json
{
  "circ_001": 42,
  "circ_002": 2026,
  "circ_003": 114514,
  ...
}
```

**评分：** 与正式赛道相同评分标准，但单独排名。

**奖项：**

| 奖项 | 条件 | 奖金 |
|------|------|------|
| 🎰 **欧皇奖** | 随机赛道总分最高 | 5元 |
| 🍀 **锦鲤奖** | 单目标随机得分最高 | 5元 |

**数学背景：** 对 N=500 的序列，随机坐标与真实结构的期望 RMSD 约 30-50 Å，得 0 分。能卡到 10 分以上，说明你对随机种子的直觉超越了高斯分布。

**限制：** 每队最多提交 3 次种子（防止暴力枚举，这比赛不是算力竞赛）。

---

### ⚔️ 特别赛道：神仙打架（无限制赛道）

> *"限制算力？不存在的。资源也是一种实力。"*

**规则：** 不限算力、不限数据、不限方法。Rosetta 可以，IsRNAcirc 可以，MD 模拟 72 小时也可以，自己搭集群也行。但是Rosetta与IsRNAcirc的标准将作为基准方案参赛（不占名额）

**唯一限制：** 🧬 **只接受碳基生物参赛。** AI 代理自主参赛恕不接受（但用 AI 辅助写代码可以）。

**提交要求：**
1. 预测结果（同正式赛道格式）
2. 方法描述（详细说明用了什么算力）
3. 可选：精彩运行过程截图/日志（供大家膜拜）

**奖项：**

| 奖项 | 条件 | 奖励 |
|------|------|------|
| ⚔️ **神仙打架奖** | 无限制赛道总分最高 | 树洞置顶 3 天 + 个人主页展示 |
| 🐉 **屠龙勇士奖** | 无限制赛道亚军 | 树洞置顶 2 天 + 个人主页展示 |
| 💎 **氪金玩家奖** | 算力消耗最大且得分 > 正式赛道冠军 | 树洞置顶1天 + 个人主页展示 |

**与正式赛道的关系：**
- 神仙打架赛道**单独排名**，不影响正式赛道
- 如果神仙打架赛道的成绩被正式赛道（受限算力）超越了，正式赛道的队伍额外获得 **"凡人弑神奖"** 🗡️,现场奖励1000元，只要大佬发Nature的时候致谢能提到我就行
- 两个赛道的方法互相借鉴、交流鼓励

**特别说明：**
- 本赛道旨在探索 circRNA 结构预测的**理论上限**
- 高算力方法的结果有助于理解：物理模拟与深度学习之间的差距还有多大
- 欢迎实验室/课题组组团参加（展示你们实验室的算力底蕴！）

---

## 报名方式

1. 发送邮件至：18806370529@163.com
2. 邮件内容：
   - 队伍名称
   - 成员信息
   - 联系邮箱
3. 收到确认后在7月10日后获得训练集附件

---

## 联系方式

- 邮箱：18806370529@163.com
- 微信群：扫码加入（海报中提供）

---

## 附录：评估脚本示例

```python
import numpy as np

def kabsch_align(pred_coords, true_coords):
    """对 C3' 原子做 Kabsch 最优刚体对齐，返回对齐后的预测坐标。"""
    pred_centered = pred_coords - pred_coords.mean(axis=0)
    true_centered = true_coords - true_coords.mean(axis=0)
    H = pred_centered.T @ true_centered
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return pred_centered @ R.T + true_coords.mean(axis=0)

def evaluate_prediction(pred_coords, true_coords, pairs_pred, pairs_true, multi_conf=None):
    """计算预测评分。

    Args:
        pred_coords: (N, 3) C3' 坐标
        true_coords: (N, 3) C3' 坐标
        pairs_pred: set of (i, j) 碱基对
        pairs_true: set of (i, j) 真实碱基对
        multi_conf: list of (N, 3) arrays，多构象提交（T5），最多 5 个
    """
    N = len(pred_coords)
    IDEAL_BOND = 5.9   # A-form C3'-C3' 距离
    IDEAL_BSJ = 5.9    # 磷酸二酯键长度

    # ===== T1: RMSD（Kabsch 对齐 + 阶梯评分）=====
    pred_aligned = kabsch_align(pred_coords, true_coords)
    rmsd = float(np.sqrt(np.mean(np.sum((pred_aligned - true_coords) ** 2, axis=1))))
    if rmsd < 5:    t1_score = 100
    elif rmsd < 10: t1_score = 80
    elif rmsd < 15: t1_score = 60
    elif rmsd < 20: t1_score = 40
    elif rmsd < 30: t1_score = 20
    else:           t1_score = 0

    # ===== T2: BSJ 闭合（绝对合理性，以 5.9Å 为基准）=====
    bsj_pred = float(np.linalg.norm(pred_coords[0] - pred_coords[-1]))
    bsj_err = abs(bsj_pred - IDEAL_BSJ)
    if bsj_err < 1:  t2_score = 100
    elif bsj_err < 2: t2_score = 80
    elif bsj_err < 5: t2_score = 60
    elif bsj_err < 10: t2_score = 40
    else:             t2_score = 0

    # ===== T3: 骨架键长（绝对合理性，以 5.9Å 为基准）=====
    bond_pred = np.linalg.norm(pred_coords[1:] - pred_coords[:-1], axis=1)
    bond_dev = np.abs(bond_pred - IDEAL_BOND)
    t3_score = float(np.mean(np.maximum(0, 100 - 20 * bond_dev)))

    # ===== T4: 碱基对 F1 =====
    pairs_pred_set = set(map(tuple, pairs_pred))
    pairs_true_set = set(map(tuple, pairs_true))
    tp = len(pairs_pred_set & pairs_true_set)
    fp = len(pairs_pred_set - pairs_true_set)
    fn = len(pairs_true_set - pairs_pred_set)
    denom = 2 * tp + fp + fn
    t4_score = (2 * tp / denom * 100) if denom > 0 else 0.0

    # ===== T5: 构象多样性（可选）=====
    if multi_conf is not None and len(multi_conf) > 1:
        # 基础分：5 个构象中 T1-T4 综合最佳者
        best_base = min(60.0, max(
            _t1_to_t4_subscore(c, true_coords, pairs_pred, pairs_true)
            for c in multi_conf
        ))
        # 多样性奖励：构象间 RMSD 标准差
        rmsds = _pairwise_rmsd(multi_conf)
        sigma = float(np.std(rmsds)) if len(rmsds) > 0 else 0.0
        if sigma > 5:   diversity_bonus = 40
        elif sigma > 2: diversity_bonus = 20
        else:           diversity_bonus = 0
        # 无多样性惩罚
        if sigma < 1:
            t5_score = best_base * 0.5
        else:
            t5_score = best_base + diversity_bonus
    else:
        t5_score = 0.0  # 未提交多构象则 T5 为 0

    # ===== 总分 =====
    weights = [0.4, 0.2, 0.15, 0.15, 0.1]
    total = (weights[0]*t1_score + weights[1]*t2_score +
             weights[2]*t3_score + weights[3]*t4_score +
             weights[4]*t5_score)

    return {
        'rmsd': rmsd,
        'bsj_pred': bsj_pred,
        'bsj_error': bsj_err,
        'bond_deviation_mean': float(np.mean(bond_dev)),
        't1': t1_score,
        't2': t2_score,
        't3': t3_score,
        't4': t4_score,
        't5': t5_score,
        'total': total,
    }


def _t1_to_t4_subscore(pred_coords, true_coords, pairs_pred, pairs_true):
    """辅助：单构象的 T1-T4 加权子分（用于 T5 基础分比较）。"""
    res = evaluate_prediction(pred_coords, true_coords, pairs_pred, pairs_true, multi_conf=None)
    w = [0.4, 0.2, 0.15, 0.15]
    return w[0]*res['t1'] + w[1]*res['t2'] + w[2]*res['t3'] + w[3]*res['t4']


def _pairwise_rmsd(confs):
    """辅助：构象集合两两 RMSD（Kabsch 对齐后）。"""
    n = len(confs)
    rmsds = []
    for i in range(n):
        for j in range(i+1, n):
            a = kabsch_align(confs[i], confs[j])
            rmsds.append(np.sqrt(np.mean(np.sum((a - confs[j]) ** 2, axis=1))))
    return rmsds
```

---

**Circ-CASP 组织委员会**
2026年6月20日