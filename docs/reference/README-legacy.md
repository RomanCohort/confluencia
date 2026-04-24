多任务实验预测平台
===================

本仓库提供一个多模块实验预测平台，支持分子输入、序列输入与实验条件的联合建模，包含统一
的 Streamlit 前端与 CLI 工具，适用于训练、预测、批量筛选与敏感性分析。

**致谢**：衷心感谢 DLEPS 团队在方法论、实现与示例数据方面的贡献与帮助。

目录
----
- 功能特性
- 项目概览
- 功能矩阵
- 目录结构
- 快速开始（前端）
- 一分钟验证（CLI）
- 安装
- 关键脚本说明
- 数据与产物约定
- 药物疗效预测（SMILES + 条件 -> 疗效）
- 分子对接预测（交叉注意力：SMILES x 蛋白序列）
- 表位虚拟筛选预测器（序列 + 实验条件 -> 疗效）
- 多尺度 GNN-PINN 说明
- 运行与打包
- 常见问题

功能特性
--------
- 数据准备：可用本地 CSV 或公开来源数据
- 少样本训练与模型选择：适配小规模数据
- GBDT + Hyperopt：在剂量/频次参数空间搜索最优区域
- 敏感性分析：评估条件变量对预测影响
- 自训练：伪标签扩展未标注数据
- 可视化：筛选、热力图与分析图表
- 数据增强与去噪：表格 VAE（可选）
- 多尺度 GNN-PINN 原型（可选）

项目概览
--------
平台包含以下模块：
- 表位虚拟筛选预测器（序列 + 实验条件 -> 疗效）
- 药物疗效预测器（SMILES + 条件 -> 疗效）
- 分子对接预测（交叉注意力：SMILES x 蛋白序列 -> 对接分数）
- 数据增强与去噪（VAE 表格增强与重构去噪）
- 统一前端（Streamlit）：训练、预测、批量筛选、敏感性分析、自训练

功能矩阵
--------
| 模块 | minimal | denoise | full | 说明 |
| --- | --- | --- | --- | --- |
| 表位预测 | ✅ | ✅ | ✅ | 轻量、无 RDKit |
| 药物表格回归 | ⚠️ | ⚠️ | ✅ | 完整功能需 RDKit 或 Torch |
| 分子对接预测 | ⚠️ | ⚠️ | ✅ | 需 Torch |
| 数据增强与去噪 | ❌ | ✅ | ✅ | 需 TensorFlow 或 Keras |
| Streamlit 前端 | ✅ | ✅ | ✅ | 统一入口 |

按功能发包策略（降低打包成本）
------------------------------
- 普通演示：优先发 minimal（只保留前端 + 轻量预测）
- 实验室用户：优先发 denoise（包含去噪能力，不引入 full 全量负担）
- 算法同学：按需发 full（仅在需要 RDKit/Torch 全能力时使用）
- 默认禁止“全员 full”发包，避免时间、磁盘与失败成本放大

体积审计与定向裁剪（PostBuildTopN）
------------------------------------
每次构建后都保留 TopN 大文件清单，并基于清单做定向裁剪，而不是盲目删包。

推荐流程：
1) 日常构建使用 minimal 或 denoise。
2) 构建时显式传入 -PostBuildTopN（例如 30 或 50）。
3) 从清单中定位最大依赖，按模块价值决定保留、降级到更轻档位或改为按需发包。
4) 若某模块仅少数用户使用，迁移到 full 专用包，不进入 minimal/denoise。

示例命令（PowerShell）：
- minimal 审计：powershell -ExecutionPolicy Bypass -File .\build_windows.ps1 -BuildProfile minimal -PostBuildTopN 30
- denoise 审计：powershell -ExecutionPolicy Bypass -File .\build_windows.ps1 -BuildProfile denoise -PostBuildTopN 30
- full 审计：powershell -ExecutionPolicy Bypass -File .\build_windows.ps1 -BuildProfile full -PostBuildTopN 50

裁剪准则：
- 出现于 TopN 且不在当前档位能力范围内的依赖，优先排查是否被误打入。
- 优先删测试包、调试包、兼容层与未使用后端，避免误删运行时主路径。
- 一次只做一类裁剪并复测启动，保留可回滚记录。

目录结构
--------
- src/：核心源码
  - src/epitope/：表位预测模块
  - src/drug/：药物预测模块
  - src/docking_cli.py：对接 CLI
  - src/data_aug_denoise/：数据增强与去噪模块
  - src/frontend.py：前端入口
- scripts/：辅助脚本（训练、构建、发布）
- data/：示例与输入数据
- models/：训练输出模型
- logs/：前端反馈日志

快速开始（前端）
----------------
1) 选择安装档位（minimal / denoise / full）并安装依赖
2) 启动前端：

```powershell
python -m streamlit run src/frontend.py
```

3) 在前端选择模块，完成训练/预测/筛选

Stack 一键自动化
----------------
提供统一入口，支持“安装依赖 + 构建 + 启动”一条命令串联：

```powershell
powershell -ExecutionPolicy Bypass -File .\stack_auto.ps1 -AllInOne -Role demo
```

常用模式：
- 仅安装依赖：`powershell -ExecutionPolicy Bypass -File .\stack_auto.ps1 -InstallDeps -Role lab`
- 仅构建：`powershell -ExecutionPolicy Bypass -File .\stack_auto.ps1 -Build -Role algorithm`
- 仅启动源码前端：`powershell -ExecutionPolicy Bypass -File .\stack_auto.ps1 -Launch`
- 预演不执行（检查命令拼接）：`powershell -ExecutionPolicy Bypass -File .\stack_auto.ps1 -AllInOne -Role demo -DryRun`

Windows 双击入口：`one_click_stack.bat`（默认 `-AllInOne -Role algorithm`，即 full 档位）。

傻瓜式使用指南（给医学生）
--------------------------
只要准备一份 CSV 表格，不需要写代码，不需要懂机器学习。

**三步完成预测**
1) 进入前端的“新手指南”页，下载样例 CSV 看看格式。
2) 选择模块：
  - **药效预测**：表里有 `smiles` 列（分子简写）。
  - **表位预测**：表里有 `sequence` 列（氨基酸序列）。
3) 有标签就“训练”，没标签就“批量筛选”。导出结果 CSV。

**你只需要认识 3 个词**
- **SMILES**：分子的简写公式（如 `CCO`）。
- **序列**：氨基酸字母串（如 `SIINFEKL`）。
- **目标列**：你想预测的数值列名（如 `efficacy`、`fluorescence`）。

**最常见的 4 个错误**
1) 列名写错（大小写、空格）。
2) 目标列不是数值（混入中文或单位）。
3) 表位模型和药效模型混用。
4) CSV 里有空行/注释导致解析失败。

**最快出结果的方法**
- 进入“药效预测/表位预测”的“单条预测”，输入一条数据即可出结果。
- 有一批候选就用“批量筛选”，上传 CSV 一键导出。

一分钟验证（CLI）
-----------------
表位训练 + 预测：

```bash
python -m src.epitope_cli train --data data/example_epitope.csv --target efficacy --model-out models/epitope_model.joblib
python -m src.epitope_cli predict --model models/epitope_model.joblib --sequence SIINFEKL
```

药物训练 + 预测：

```bash
python -m src.drug_cli train --data data/example_drug.csv --target efficacy --model-out models/drug_model.joblib
python -m src.drug_cli predict --model models/drug_model.joblib --smiles CCO
```

对接训练 + 预测：

```bash
python -m src.docking_cli train --data data/docking.csv --smiles-col smiles --protein-col protein --target docking_score --model-out models/docking_crossattn.pt
python -m src.docking_cli predict --model models/docking_crossattn.pt --smiles CCO --protein "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAG"
```

安装
----
推荐使用工作区根目录的共用虚拟环境，避免每个子项目各建一套环境。

脚本安装（推荐）：

- minimal：`powershell -ExecutionPolicy Bypass -File scripts/install_shared_env.ps1 -Profile minimal`
- denoise：`powershell -ExecutionPolicy Bypass -File scripts/install_shared_env.ps1 -Profile denoise`
- full：`powershell -ExecutionPolicy Bypass -File scripts/install_shared_env.ps1 -Profile full`

安装成本控制建议：
- 开发默认安装 minimal，只有确实需要时再切到 denoise 或 full。
- 构建失败优先降档位复现（full -> denoise -> minimal），确认问题属于依赖还是业务代码。
- 统一团队的默认档位，减少环境分叉导致的重复排障。

手动安装：

- `python -m pip install -r requirements-shared.txt`
- 表位轻量：`python -m pip install -r requirements-epitope.txt`
- 全量功能：`python -m pip install -r requirements.txt`
- 去噪模块：`python -m pip install -r requirements-aug-denoise.txt`

Windows 提示：建议先 `cd` 到包含 `src/` 的目录再运行命令。

关键脚本说明
------------
- stack_auto.ps1：统一自动化入口（安装、构建、启动）
- one_click_stack.bat：Windows 一键执行入口
- scripts/install_shared_env.ps1：安装共享依赖（minimal/denoise/full）
- scripts/build_integrated.ps1：一键构建
- scripts/release_integrated.ps1：一键发布
- scripts/audit_build_size.ps1：对 dist/dist_onefile 产物做 TopN 体积审计并生成报告
- scripts/package_by_role.ps1：按角色自动选择打包档位（demo->minimal, lab->denoise, algorithm->full）
- scripts/prune_packaged_artifact.ps1：裁剪打包产物中的冗余 DLEPS 构建目录（build/dist）
- scripts/compute_embeddings_example.py：生成示例 embedding
- scripts/train_transformer_example.py：小型 Transformer 训练示例
- scripts/train_transformer_full_example.py：全量 Transformer 训练示例

体积审计脚本使用：

- `powershell -ExecutionPolicy Bypass -File scripts/audit_build_size.ps1 -TopN 30`
- 可选参数：`-DistPath dist -DistOneFilePath dist_onefile -OutputDir logs`
- 输出文件：`logs/build_size_audit_*.md` 与 `logs/build_size_audit_*.csv`

按角色发包（推荐）：

- 普通演示：`powershell -ExecutionPolicy Bypass -File scripts/package_by_role.ps1 -Role demo`
- 实验室用户：`powershell -ExecutionPolicy Bypass -File scripts/package_by_role.ps1 -Role lab`
- 算法同学：`powershell -ExecutionPolicy Bypass -File scripts/package_by_role.ps1 -Role algorithm`

说明：`scripts/package_by_role.ps1` 会在打包后自动执行 `scripts/prune_packaged_artifact.ps1`，
先裁剪冗余目录，再执行 `scripts/audit_build_size.ps1`，报告即为裁剪后体积。

常用附加参数：
- 首次环境准备：`-InstallDeps`
- 单文件打包：`-OneFile`
- 指定构建盘符：`-DriveLetter D`
- 轻量全量包（不含 TensorFlow/Keras）：`-LeanNoTensorFlow`
- 轻量全量包（不含 DLEPS 资源）：`-LeanNoDLEPS`

示例（full 进一步降体积）：
- `powershell -ExecutionPolicy Bypass -File scripts/package_by_role.ps1 -Role algorithm -LeanNoTensorFlow -LeanNoDLEPS`
- 对应能力影响：去噪模块不可用（LeanNoTensorFlow）；DLEPS 模块不可用（LeanNoDLEPS）

数据与产物约定
--------------
- 输入数据：data/
- 模型产物：models/
- 日志输出：logs/
- 缓存与中间文件：data/cache/

药物疗效预测（SMILES + 条件 -> 疗效）
-----------------------------------
该模块将 SMILES 分子指纹与可选实验条件拼接后训练回归模型。

数据格式（CSV）：
- smiles：例如 CCO
- 目标列：例如 efficacy
- 条件数值列：dose、freq、route（可选）

训练：

```bash
python -m src.drug_cli train --data data/drugs.csv --target efficacy --model-out models/drug_model.joblib
```

单条预测：

```bash
python -m src.drug_cli predict --model models/drug_model.joblib --smiles CCO --param dose=10 --param freq=2
```

批量筛选：

```bash
python -m src.drug_cli screen --model models/drug_model.joblib --candidates data/drug_candidates.csv --out drug_predictions.csv --out-col pred
```

分子对接预测（交叉注意力：SMILES x 蛋白序列）
---------------------------------------------
训练：

```bash
python -m src.docking_cli train --data data/docking.csv --smiles-col smiles --protein-col protein --target docking_score --model-out models/docking_crossattn.pt
```

预测：

```bash
python -m src.docking_cli predict --model models/docking_crossattn.pt --smiles CCO --protein "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAG"
```

扩展用法
--------
本节提供更完整的用法组合与实践建议，便于从数据准备、训练到批量筛选形成稳定流程。

1) 从示例数据到稳定模型

```bash
python -m src.epitope_cli train --data data/example_epitope.csv --target efficacy --model-out models/epitope_model.joblib
python -m src.epitope_cli screen --model models/epitope_model.joblib --candidates data/example_epitope.csv --out data/epitope_preds.csv --out-col pred
```

2) 条件列与参数输入
- 条件列建议为数值型；若列名不一致，请先在数据中统一列名。
- 单条预测可用多个 `--param` 传入条件值（示例见药物与表位单条预测）。
- 复杂列选择或参数规则请以 `--help` 输出为准：

```bash
python -m src.epitope_cli --help
python -m src.drug_cli --help
```

3) 批量筛选与结果排序
将候选表交给 `screen`，再在下游按预测列排序：

```bash
python -m src.drug_cli screen --model models/drug_model.joblib --candidates data/drug_candidates.csv --out data/drug_predictions.csv --out-col pred
```

4) 自训练（伪标签扩展）
当未标注数据量较大时，可先训练基线模型，再用自训练扩展样本：

```bash
python -m src.drug_cli self-train --help
python -m src.epitope_cli self-train --help
```

5) 训练后诊断与绘图
训练完成后可用诊断图判断偏差与残差分布：

```bash
python -m src.drug_cli plot --help
python -m src.epitope_cli plot --help
```

6) 生成与筛选联动（分子生成）
先生成候选，再交给已有模型筛选：

```bash
python -m src.drug_cli generate --data data/example_drug.csv --out data/generated_candidates.csv
python -m src.drug_cli screen --model models/drug_model.joblib --candidates data/generated_candidates.csv --out data/generated_scored.csv --out-col pred
```

7) 前端与 CLI 结合
- CLI 用于训练与批量筛选，前端用于可视化和快速验证。
- 训练得到的模型可直接在前端加载进行单条预测与筛选。

8) 参数说明（通用）
- `--data`：训练或生成所需的输入数据文件（CSV 等）。
- `--target`：目标列名（回归目标）。
- `--model-out`：训练后模型输出路径。
- `--model`：预测/筛选时加载的已训练模型。
- `--candidates`：候选列表文件（批量筛选）。
- `--out` / `--out-col`：输出文件与预测列名。
- `--smiles` / `--sequence` / `--protein`：单条预测输入。
- `--smiles-col` / `--protein-col`：指定对接数据列名。
- `--param key=value`：传入条件变量（可重复多次）。
- `--env-cols`：手动指定条件列（以 CLI 帮助为准）。

9) 真实场景流程（示例）

药物筛选一条龙：
1) 训练基线模型：

```bash
python -m src.drug_cli train --data data/example_drug.csv --target efficacy --model-out models/drug_model.joblib
```

2) 批量筛选候选库：

```bash
python -m src.drug_cli screen --model models/drug_model.joblib --candidates data/example_drug_unlabeled.csv --out data/drug_screen_out.csv --out-col pred
```

3) 结果排序与复核：
- 在表格软件或脚本中按 `pred` 由高到低排序。
- 抽取 Top-N 进入后续实验或更细粒度筛选。

表位筛选一条龙：
1) 训练模型：

```bash
python -m src.epitope_cli train --data data/example_epitope.csv --target fluorescence --model-out models/fluor.joblib
```

2) 批量筛选：

```bash
python -m src.epitope_cli screen --model models/fluor.joblib --candidates data/example_epitope_unlabeled.csv --out data/epitope_screen_out.csv --out-col pred
```

3) 可选敏感性分析与自训练：
- 若需要解释变量影响，优先做敏感性分析。
- 若未标注数据充足，可尝试自训练扩展样本。

对接筛选一条龙：
1) 训练模型：

```bash
python -m src.docking_cli train --data data/docking.csv --smiles-col smiles --protein-col protein --target docking_score --model-out models/docking_crossattn.pt
```

2) 批量筛选：

```bash
python -m src.docking_cli screen --model models/docking_crossattn.pt --candidates data/docking_candidates.csv --smiles-col smiles --protein-col protein --out data/docking_predictions.csv --out-col dock_pred
```

3) 结果复核：
- 先筛 Top-N，再检查蛋白序列长度/异常字符。
- 如需多模型对比，可用不同模型重复筛选并做一致性分析。

多尺度 GNN-PINN 研究流程
-----------------------
该流程适合做方法验证与小样本实验，不建议直接用于生产任务。

1) 准备基础分子数据：
- 以 SMILES 为主，必要时准备 3D 构象或原子坐标。

2) 训练分子嵌入：
- 先在 GNN 上做短训练或加载已有权重。
- 视需求选择 EGNN 或标准 GNN 作为 backbone。

3) 构造 PINN 任务：
- 定义 PDE 形式、边界/初始条件与采样域。
- 设定 Collocation 采样策略与残差权重。

4) 联合训练与调参：
- 先对 BC/IC 做 warm-up，再引入 PDE 残差。
- 记录收敛曲线，必要时调低学习率或加梯度裁剪。

5) 复现实验：
- 固定随机种子，记录关键超参数与数据版本。

前端使用细化（页面级步骤）
--------------------------
1) 选择模块：表位 / 药物 / 对接 / 去噪。
2) 数据导入：选择文件 -> 预览 -> 校验列名与类型。
3) 列设置：选择目标列、特征列、条件列。
4) 训练设置：选择模型与评估策略 -> 开始训练。
5) 结果查看：下载模型与预测结果，必要时查看诊断图。
6) 批量筛选：上传候选表与模型 -> 导出结果。
7) 自训练：上传未标注数据 -> 选择阈值 -> 生成伪标签。

CLI 参数详解（逐参数）
---------------------
以下为常见参数说明，实际以 `--help` 为准。

- `--data`：训练或生成所需的输入数据文件。
- `--target`：目标列名（回归任务）。
- `--model-out`：模型输出路径。
- `--model`：预测或筛选时加载的模型路径。
- `--candidates`：候选列表文件（批量筛选）。
- `--out` / `--out-col`：输出文件与预测列名。
- `--smiles` / `--sequence` / `--protein`：单条预测输入。
- `--smiles-col` / `--protein-col`：指定对接数据列名。
- `--param key=value`：条件变量（可重复多次）。
- `--env-cols`：手动指定条件列（以 CLI 帮助为准）。
- `--model-type`：选择模型类型（例如 hgb、rf、gbr、mlp 等）。
- `--cv` / `--split`：交叉验证或留出比例设置。
- `--seed`：随机种子。

常见问题排查（快速定位）
------------------------
1) 训练报错或结果异常
- 先确认输入文件编码与分隔符（CSV/TSV）。
- 检查是否存在空列、字符串列被误当数值列。
- 目标列必须为数值型，且不能全为空或常数。

2) 预测结果全相同
- 可能是模型未训练成功或输入特征恒定。
- 检查训练日志、确认模型文件大小非 0。
- 对单条预测输入做小幅扰动，观察是否变化。

3) 批量筛选输出为空
- 检查 `--candidates` 路径与列名是否匹配。
- 对接模块需同时存在 `smiles` 与 `protein` 列。

性能优化建议
------------
- 大数据先抽样调参，再全量训练。
- 将高维稀疏特征先做降维或筛选。
- 使用 `hgb` 或 `sgd` 作为速度优先的基线。
- 批量筛选优先用 CLI，前端用于可视化与复核。

数据准备与清洗建议
------------------
- 统一列名（小写、无空格），减少映射错误。
- 处理缺失值：数值列用中位数/均值填充，类别列用常量。
- 剔除异常值或采用分位数截断。
- SMILES/序列列建议去重，避免重复样本干扰评估。

评估与报告
----------
- 回归任务建议至少报告 MAE、RMSE、R2。
- 小样本优先用交叉验证并记录方差。
- 记录模型版本、训练数据版本与关键超参数，便于复现。

模型管理与复现
--------------
- 为模型文件建立命名规则：模块_日期_数据版本.joblib
- 保存训练配置（模型类型、随机种子、列名映射）。
- 重要模型建议做一次“同参数复训”以验证稳定性。

部署/打包细化步骤
-----------------
1) 确认依赖安装完成，且可正常运行前端：

```powershell
python -m streamlit run src/frontend.py
```

2) 关闭所有占用的 Python 进程，避免打包时锁文件。

3) 运行打包脚本（按实际脚本参数为准）：

```powershell
./build_windows.ps1
```

或发布版本：

```powershell
./release_windows.ps1
```

4) 检查产物目录：
- build/
- dist_onefile/
- release/

5) 若打包失败：
- 先尝试 minimal 或 denoise 档位构建，再逐步增加依赖。
- 检查是否缺少系统组件或 GPU 相关依赖。

指标可视化模板
--------------
推荐在训练完成后导出评估指标与图表：

示例指标表（CSV）：

| model | split | mae | rmse | r2 |
| --- | --- | --- | --- | --- |
| hgb | cv5 | 0.12 | 0.19 | 0.83 |

示例图表建议：
- y_true vs y_pred 散点图
- 残差直方图
- 残差 vs 预测值

CLI 可用的诊断命令：

```bash
python -m src.drug_cli plot --help
python -m src.epitope_cli plot --help
```

案例数据说明
------------
仓库内常见示例数据：
- data/example_drug.csv：药物示例数据
- data/example_drug_unlabeled.csv：药物未标注候选
- data/example_epitope.csv：表位示例数据
- data/example_epitope_unlabeled.csv：表位未标注候选

字段约定：
- 药物：smiles + 目标列 + 条件数值列
- 表位：sequence + 目标列 + 条件数值列

API/脚本接口汇总
---------------
CLI 入口：
- `python -m src.drug_cli --help`
- `python -m src.epitope_cli --help`
- `python -m src.docking_cli --help`

常用脚本：
- scripts/train_transformer_example.py
- scripts/train_transformer_full_example.py
- scripts/compute_embeddings_example.py

前端入口：
- `python -m streamlit run src/frontend.py`

差分进化建议环境（DE）与序列编码说明
---------------------------------
新增功能摘要：

- 差分进化建议环境（DE）：通过差分进化搜索数值型环境变量（env）空间，找到使模型预测值最大化或最小化的环境参数组合。前端已在“表位预测 / 药物预测”单条预测页面添加相应面板，CLI 新增 `suggest-env` 子命令。
- 序列编码（AAIndex / one-hot）：已提供 `src/epitope/encoding.py`，包含 one-hot 编码、AAIndex 从 CSV 加载、序列到 AAIndex 映射以及 one-hot + 连续属性拼接函数，便于特征构造与解释分析。

快速使用示例：

CLI（表位）—— 自动 bounds：
```bash
python -m src.epitope_cli suggest-env --model models/epitope_model.joblib --sequence SIINFEKL
```

CLI（药物）—— 对 sklearn/pt 模型均可：
```bash
python -m src.drug_cli suggest-env --model models/drug_model.joblib --smiles CCO --bounds "0.5:2,10:100"
```

前端（Streamlit）：
1. 启动：`streamlit run src/frontend.py`。
2. 打开“表位预测”或“药物预测”单条预测，展开“差分进化建议环境”，设置每个 env 的上下界，点击运行并查看建议值。

开发者说明：

- DE 实现文件：`src/common/optim/differential_evolution.py`。
- 各模型模块已暴露使用接口：
  - `src/epitope/predictor.py`: `suggest_env_by_de_epitope`
  - `src/drug/predictor.py`: `suggest_env_by_de_drug`
  - `src/drug/torch_predictor.py`: `suggest_env_by_de_torch`
  - `src/drug/transformer_predictor.py`: `suggest_env_by_de`
- 序列编码工具：`src/epitope/encoding.py`，函数包括 `one_hot_encode`, `load_aaindex_from_csv`, `sequence_to_aaindex`, `continuous_onehot_encode`。

调优建议：
- 初次运行建议使用较小 `max_iter`（例如 50）和较小 `pop_size`，确认搜索方向后再增加资源预算。前端与 CLI 默认使用合理保守参数。
- DE 为随机算法，建议多次运行或在关键实验中设置随机种子以复现结果。

性能基线对比模板
----------------
建议在固定数据集与划分下做基线对比：

| model | dataset | split | mae | rmse | r2 | notes |
| --- | --- | --- | --- | --- | --- | --- |
| hgb | drug_v1 | cv5 | 0.12 | 0.19 | 0.83 | default |
| rf | drug_v1 | cv5 | 0.15 | 0.23 | 0.78 | tuned |

对比建议：
- 保持同一划分策略（cv5 或固定留出）。
- 固定随机种子，记录完整超参。
- 仅替换模型或特征，避免多变量同时变化。

数据集版本管理规范
------------------
- 采用版本号或日期戳：dataset_v1_2026-02-10.csv
- 记录来源、清洗步骤与筛选规则。
- 训练与评估统一指向同一版本。
- 重要版本建议做只读归档。

实验记录模板
------------
建议记录：
- 实验编号、日期、数据集版本
- 模型类型与超参数
- 划分策略与随机种子
- 指标与对比结论

示例：

| exp_id | date | dataset | model | split | seed | mae | rmse | r2 | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp_001 | 2026-02-10 | drug_v1 | hgb | cv5 | 42 | 0.12 | 0.19 | 0.83 | baseline |

发布说明模板
------------
版本：v1.0.0

新增：
- 新增批量筛选与自训练流程示例
- 完善扩展用法与打包说明

修复：
- 修正部分 CLI 示例命令

已知问题：
- full 档位打包耗时较长

批量预测：

```bash
python -m src.docking_cli screen --model models/docking_crossattn.pt --candidates data/docking_candidates.csv --smiles-col smiles --protein-col protein --out docking_predictions.csv --out-col dock_pred
```

表位虚拟筛选预测器（序列 + 实验条件 -> 疗效）
--------------------------------------------
数据格式（CSV）：
- sequence：表位序列
- 目标列：fluorescence 或 killing_rate
- 条件数值列：concentration、cell_density、incubation_hours（可选）

训练：

```bash
python -m src.epitope_cli train --data data/epitopes.csv --target fluorescence --model-out models/fluor.joblib
```

单条预测：

```bash
python -m src.epitope_cli predict --model models/fluor.joblib --sequence SIINFEKL --param concentration=10 --param cell_density=1000000
```

批量筛选：

```bash
python -m src.epitope_cli screen --model models/fluor.joblib --candidates data/candidates.csv --out predictions.csv --out-col pred_fluor
```

多尺度 GNN-PINN 说明
--------------------
该原型将分子级 GNN 与 PINN 耦合，用于扩散/反应动力学的多尺度推断：

SMILES -> GNN -> GAT -> readout -> 分子嵌入 -> PINN(x, t, embedding)

PDE 形式（示例）：
  dC/dt - D * Laplacian(C) + Vmax * C / (Km + C) = 0

物理势调制（PhysicsMessageGNN）：
- Lennard-Jones（LJ）：V(r) = 4 * eps * ((sigma/r)^12 - (sigma/r)^6)
  - lj_epsilon：势深度，建议小值
  - lj_sigma：距离标度，约 3.0 到 4.0 A
- Electrostatic（静电近似）：q1*q2 / (dielectric * r)
  - dielectric：环境介电常数（溶液约 80）
- Auto：特征相似性衰减与 LJ 软项组合

运行示例：

```powershell
python -m streamlit run src/frontend.py --server.port 8505
```

```bash
python src/run_multiscale.py "CCO"
```

运行与打包
----------
- Windows 打包脚本：build_windows.ps1 / release_windows.ps1
- 产物目录：build/、dist_onefile/、release/

常见问题
--------
Q：是否必须安装 RDKit 或 Torch？
A：完整药物与对接功能需要；表位与前端在 minimal 模式可用。

Q：为什么需要 TensorFlow？
A：仅用于数据增强与去噪模块。

Q：候选表缺少条件列怎么办？
A：缺失值按空值处理，建议补齐以获得一致结果。

默认会自动把除 `smiles` 和 target 以外的“数值列”识别为条件输入；也可用 `--env-cols` 手动指定。

模型简述（药物预测）：
- hgb：大规模友好、支持早停，默认推荐
- gbr：稳健但训练较慢
- rf：抗噪、可解释性较好
- ridge：线性基线，速度快、可作为对照
- mlp：非线性强但需更多数据与调参

## Torch 训练（可选）

当需要自定义网络结构或更大规模训练时，可使用 Torch 训练脚本：

`python scripts/train_drug_torch.py --data data/drugs.csv --target-col efficacy --hidden-sizes 512,256 --epochs 200 --out models/drug_torch_model.pt`

说明：
- `--hidden-sizes` 逗号分隔，表示多层 MLP 的隐藏层宽度。
- `--env-cols` 可手动指定条件列（逗号分隔），不提供时自动使用数值列。
- 也可以在前端「药物疗效预测 → Torch训练」直接训练并下载模型。

## 单条预测

`python -m src.drug_cli predict --model models/drug_model.joblib --smiles CCO --param dose=10 --param freq=2`

## 批量虚拟筛选（候选 CSV）

候选 CSV 至少包含 `smiles` 列，且可选包含训练时用到的条件列：

`python -m src.drug_cli screen --model models/drug_model.joblib --candidates data/drug_candidates.csv --out drug_predictions.csv --out-col pred`

## 爬虫自主训练（PubChem BioAssay proxy 标签）

如果你暂时没有自己的“药效标注数据”，可以用 PubChem 的 BioAssay 汇总结果生成一个代理标签：

\- `activity_score = n_active / (n_active + n_inactive)`

注意：这不是临床疗效，只是“在公开生物测定中更可能 Active”的一个 proxy，用于快速做一个可跑通的基线模型。

1) 仅抓数据集：

`python -m src.drug_cli crawl --start-cid 1 --n 500 --min-total 5 --out data/pubchem_activity.csv`

2) 抓取并直接训练导出模型：

`python -m src.drug_cli crawl-train --start-cid 1 --n 2000 --min-total 5 --model-out models/drug_pubchem_activity.joblib --data-out data/pubchem_activity_labeled.csv`

# 分子对接预测（交叉注意力：SMILES × 蛋白序列）

该模块使用交叉注意力将配体 SMILES 与蛋白序列进行配对建模，回归预测对接效果（如打分/能量）。

## 数据格式（CSV）

训练集需要至少包含：
- `smiles`：配体 SMILES
- `protein`：蛋白/受体序列
- `docking_score`：对接效果数值（可用其他列名）

示例（列名仅示意）：

| smiles | protein | docking_score |
| --- | --- | --- |
| CCO | MTEYKLVV... | -7.8 |

## 训练

`python -m src.docking_cli train --data data/docking.csv --smiles-col smiles --protein-col protein --target docking_score --model-out models/docking_crossattn.pt`

## 单条预测

`python -m src.docking_cli predict --model models/docking_crossattn.pt --smiles CCO --protein "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAG"`

## 批量预测（候选 CSV）

`python -m src.docking_cli screen --model models/docking_crossattn.pt --candidates data/docking_candidates.csv --smiles-col smiles --protein-col protein --out docking_predictions.csv --out-col dock_pred`

# 表位虚拟筛选预测器（序列 + 实验条件 → 疗效）

本仓库额外提供一个轻量的“虚拟筛选”回归模型：
将表位序列特征（氨基酸组成、疏水性、粗略电荷、极性比例等）与实验环境参数（如给药浓度、细胞密度等数值列）拼接后训练模型。

注意：该模块不依赖 rdkit，适合在没有化学工具链时单独使用。

## 数据格式（CSV）

训练集需要至少包含：
- `sequence`：表位（肽）序列，例如 `SIINFEKL`
- 1个目标列：例如 `fluorescence` 或 `killing_rate`
- 0个或多个实验参数列（数值型）：例如 `concentration`, `cell_density`, `incubation_hours` 等

示例（列名仅示意）：

| sequence | concentration | cell_density | fluorescence |
| --- | --- | --- | --- |
| SIINFEKL | 10 | 1000000 | 12345 |

## 训练

在仓库根目录运行：
`python -m src.epitope_cli train --data data/epitopes.csv --target fluorescence --model-out models/fluor.joblib`

默认会自动把除 `sequence` 和 target 以外的“数值列”识别为实验参数输入。

训练建议：
- 数据量较大：优先使用 `hgb` 或 `sgd`
- 数据量较小：可尝试 `rf` 或 `gbr`
- 需要强非线性拟合：尝试 `mlp`（注意调参）

大规模数据推荐：可选模型 `sgd`（线性 + 弹性网正则），速度快、内存占用低：

`python -m src.epitope_cli train --data data/epitopes.csv --target fluorescence --model sgd --model-out models/fluor_sgd.joblib`

模型简述（表位预测）：
- hgb（基于直方图的梯度提升决策树）：大规模友好、支持早停，通常是稳健默认选项
- gbr（梯度提升回归树）：稳健但训练较慢
- rf（随机森林）：抗噪、可解释性较好
- mlp（多层感知机）：非线性强但需更多数据与调参
- sgd（随机梯度下降）：线性 + 弹性网，速度快、内存占用低，适合大规模

## 单条预测

`python -m src.epitope_cli predict --model models/fluor.joblib --sequence SIINFEKL --param concentration=10 --param cell_density=1000000`

## 批量虚拟筛选（候选序列列表）

候选 CSV 至少包含 `sequence` 列，且可选包含训练时用到的实验参数列：

`python -m src.epitope_cli screen --model models/fluor.joblib --candidates data/candidates.csv --out predictions.csv --out-col pred_fluor`

## 前端（Streamlit）

启动一个可视化前端，集成：
- 表位：训练 / 单条预测 / 批量虚拟筛选
- 表位：参数敏感性分析（局部梯度/数值梯度；用于解释哪些特征对预测最敏感）
- 药物：训练 / 单条预测 / 批量筛选（SMILES + 可选条件数值列；不依赖 torch）
- 药物：旧版脚本演示（若 rdkit/torch 缺失会提示不可用；支持批量 CSV 预测）
- 实验数据增强与去噪（IGEM项目2集成）：VAE 表格增强/重构去噪（若未安装 tensorflow/keras 会提示不可用）

运行：

建议在工作区根目录先安装共享环境，然后进入本项目目录运行：

1) 在 `D:/IGEM集成方案` 安装依赖（推荐用脚本）：

- minimal：`powershell -ExecutionPolicy Bypass -File scripts/install_shared_env.ps1 -Profile minimal`
- denoise：`powershell -ExecutionPolicy Bypass -File scripts/install_shared_env.ps1 -Profile denoise`
- full：`powershell -ExecutionPolicy Bypass -File scripts/install_shared_env.ps1 -Profile full`

2) 进入本项目目录（包含 src/ 的目录）：

`cd <项目目录>`

3) 启动前端：

`D:/IGEM集成方案/.venv/Scripts/python.exe -m streamlit run src/frontend.py`

如果你不在仓库根目录运行，也可以先进入根目录：

`cd <本项目目录>`

## 运行与打包（可选）

- Windows 打包脚本：`build_windows.ps1` / `release_windows.ps1`
- 已打包产物：`build/`、`dist_onefile/`、`release/`

> 提示：打包涉及本地系统环境与依赖版本，建议优先使用脚本进行构建。

---

# 用户指导（前端）

1) **选择模块**：表位 / 药物 / 数据增强与去噪。
2) **训练**：上传数据 → 选择列 → 设置参数 → 训练并下载模型。
3) **预测/筛选**：上传模型与候选表格 → 获取预测结果并下载 CSV。
4) **自训练**：上传有标注 + 无标注数据，自动伪标签扩展训练集。
5) **数据增强/去噪**：上传表格 → 选择数值/分类型列 → 生成或去噪。

**推荐流程（新手）**
1) 先用示例数据试跑：`data/example_epitope.csv` 或 `data/example_drug.csv`。
2) 在“导入预处理”里做列名清洗与缺失值处理。
3) 先训练一个默认模型（hgb/rf），验证流程通畅后再调参。
4) 训练完成后立刻做“单条预测”检查模型是否合理。
5) 再进行“批量筛选”导出 CSV，进入下游分析。

**训练页关键参数说明**
- **目标列**：必填，只能选择一个数值列。
- **特征列/条件列**：默认自动识别数值列；如需固定，使用手动选择。
- **模型类型**：
  - 表位：推荐 `hgb`（稳健默认）或 `sgd`（大规模）。
  - 药物：推荐 `hgb`，数据少可用 `rf` 作为对照。
- **评估策略**：默认 5 折交叉验证；数据很少时可改为留出法。

**预测与筛选说明**
- 单条预测用于快速 sanity check；批量筛选用于候选列表打分排序。
- 批量筛选输出会包含原始列与预测列（默认 `pred`）。
- 若候选表缺少训练时的条件列，默认按空值处理（建议补齐）。

**自训练（伪标签）建议**
- 先确保模型基线可用，再启用自训练。
- 优先选择“不确定性阈值”更严格的设置，减少噪声伪标签。
- 每轮自训练后都建议用验证集复核指标。

**性能与稳定性**
- 大数据集建议先抽样调参，再用全量训练。
- 若前端卡顿，建议先在 CLI 训练模型，再回前端做预测/筛选。

**表格导入说明**
- 支持 CSV/TSV/TXT/Excel/JSON/JSONL/Parquet/Feather 等常见格式。
- 自动识别并剔除表格**开头的注释行**（如以 `#`/`//`/`;`/`%` 开头的行）。
- 可在“导入预处理”中进行列名清洗、缺失填充、异常值截断、标准化/归一化、分组聚合与列映射。

---

# 新增功能（集成版）

## 自训练（伪标签 / 不确定性筛选）

CLI 已支持“自训练（self-train）”：
- 通过 bootstrap/集成预测估计不确定性（预测方差/标准差）
- 选取低不确定性的未标注样本作为伪标签加入训练

用法示例（以 CLI 帮助为准）：
- `python -m src.drug_cli self-train --help`
- `python -m src.epitope_cli self-train --help`

## 绘图（回归诊断）

CLI 已支持训练后绘图（y_true vs y_pred、残差直方图、残差 vs 预测）：
- `python -m src.drug_cli plot --help`
- `python -m src.epitope_cli plot --help`

## 分子生成（GAN + 进化算法）

新增“分子生成”命令：先用分子指纹 GAN 扩增种群，再用进化算法（交叉 + 变异）生成候选 SMILES，并按打分函数排序输出。

- 默认评分：QED（当不提供模型时）
- 如提供模型：用已训练的药物模型预测作为评分
- 支持评分模式：QED / 模型 / 组合加权（combined）
- 支持性质过滤（QED/MW/LogP/HBD/HBA/TPSA）与多样性约束（Tanimoto 相似度）

示例：
- 仅用进化算法（不启用 GAN）：
  - `python -m src.drug_cli generate --data data/example_drug.csv --out data/generated_ga.csv --population 200 --generations 30`
- 启用 GAN 扩增种群：
  - `python -m src.drug_cli generate --data data/example_drug.csv --use-gan --gan-epochs 200 --gan-samples 256 --out data/generated_gan_ga.csv`
- 使用已训练模型作为评分函数：
  - `python -m src.drug_cli generate --data data/example_drug.csv --model models/drug_model.joblib --out data/generated_scored.csv`

依赖提示：
- 进化算法需要 `rdkit`
- GAN 部分需要 `torch`（full 安装档位包含）

前端入口：
- Streamlit → **药物疗效预测** → **分子生成** 标签

## 用户反馈（前端）

前端侧边栏提供“用户反馈”入口：评分/是否有帮助/备注等。

- 默认仅保存到：`logs/feedback.csv`
- 提交后会自动带上当前页面上下文（module/page），并在预测/筛选后附带关键字段（如预测值摘要）

### （可选）飞书通知

如需“联网时提交反馈就通知飞书”，可配置 Webhook（不配置则不发送）。

需要设置环境变量：
- `FEISHU_WEBHOOK_URL=...`
- `FEISHU_SECRET=...`（可选）

PowerShell 示例（当前会话）：
- `$env:FEISHU_WEBHOOK_URL="https://open.feishu.cn/open-apis/bot/v2/hook/xxx"`
- `$env:FEISHU_SECRET="..."`

---

# 常见问题（FAQ）

1) **训练时报错找不到列**
  - 检查 CSV 中是否包含必需列（表位：`sequence`；药物：`smiles`；目标列）

2) **模型维度不匹配**
  - 可能是模型与当前代码版本不一致；请用当前版本重新训练并导出模型

3) **Torch 模型无法加载**
  - 确认模型文件后缀为 `.pt`，且来自本项目的 Torch 训练流程

4) **rdkit 安装失败**
  - 建议使用 `requirements-epitope.txt` 先跑通表位模块，再考虑完整安装

5) **Windows 上双击 exe 无反应**
  - 确认发布包是整包拷贝（`confluencia.exe` 与 `_internal` 同级）
  - 先安装 `VC++ 2015-2022 x64` 运行库（可将 `vc_redist.x64.exe` 放在 exe 同级并先运行）
  - 提供 `install_and_run.bat` 时可一键安装运行库并启动程序
  - 仍无反应时，用 `cmd` 在 exe 目录运行一次以查看报错

**发布包结构示例**（同级目录）：

```text
confluencia/
  confluencia.exe
  _internal/
  vc_redist.x64.exe
  install_and_run.bat
```

# 参考文献与数据来源

## 数据与接口

- PubChem PUG-REST（化合物与 BioAssay 数据接口）：https://pubchemdocs.ncbi.nlm.nih.gov/pug-rest
- PubChem BioAssay 概览：https://pubchem.ncbi.nlm.nih.gov/docs/bioassay
- ChEMBL 生物活性数据库：https://www.ebi.ac.uk/chembl/
- ZINC15 化合物库：https://zinc15.docking.org/

## 方法参考

- Kingma & Welling, Auto-Encoding Variational Bayes (VAE), 2013: https://arxiv.org/abs/1312.6114
- Snell et al., Prototypical Networks for Few-shot Learning, 2017: https://arxiv.org/abs/1703.05175
- Chen et al., A Simple Framework for Contrastive Learning of Visual Representations, 2020: https://arxiv.org/abs/2002.05709
- Chen et al., GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks, 2018: https://arxiv.org/abs/1711.02257
- Chen & Guestrin, XGBoost: A Scalable Tree Boosting System, 2016: https://arxiv.org/abs/1603.02754
- Bergstra et al., Hyperopt: A Python Library for Optimizing the Hyperparameters of Machine Learning Algorithms, 2013: https://arxiv.org/abs/1206.2944

## 第三方库

- Streamlit（前端框架）：https://streamlit.io/
- scikit-learn（传统机器学习）：https://scikit-learn.org/
- pandas（表格数据处理）：https://pandas.pydata.org/
- NumPy（数值计算）：https://numpy.org/
- Matplotlib（绘图）：https://matplotlib.org/
- RDKit（化学信息学，药物模块可选依赖）：https://www.rdkit.org/
- TensorFlow/Keras（数据增强与去噪模块可选依赖）：https://www.tensorflow.org/
- Requests（HTTP 请求）：https://requests.readthedocs.io/
- Twilio（可选短信服务）：https://www.twilio.com/docs/sms
- Feishu SDK（可选飞书通知）：https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/im-v1/message/create
## 开源项目（GitHub）

- Streamlit：https://github.com/streamlit/streamlit
- scikit-learn：https://github.com/scikit-learn/scikit-learn
- pandas：https://github.com/pandas-dev/pandas
- NumPy：https://github.com/numpy/numpy
- Matplotlib：https://github.com/matplotlib/matplotlib
- RDKit：https://github.com/rdkit/rdkit
- PyTorch：https://github.com/pytorch/pytorch
- torchvision：https://github.com/pytorch/vision
- TensorFlow：https://github.com/tensorflow/tensorflow
- Keras：https://github.com/keras-team/keras
- Altair：https://github.com/vega/altair
- joblib：https://github.com/joblib/joblib
- PyArrow：https://github.com/apache/arrow
- openpyxl：https://github.com/ericgazoni/openpyxl
- Hyperopt：https://github.com/hyperopt/hyperopt
- Py4J：https://github.com/py4j/py4j
- Requests：https://github.com/psf/requests

## 版本日志

- 
0.6.0 - 2026-01-22：
  - 轻量化安装包
  - 计划面向wetlab进行试用反馈收集
- 0.5.0 - 2026-01-20：
  - 前端新增药物旧版脚本演示模块（支持批量 CSV 预测）
  - 优化前端数据导入预处理功能
- 0.4.0 - 2026-01-17：
  - 药物模块新增“表格回归”预测器（SMILES + 条件数值列 → 疗效）
  - 前端集成药物表格回归模块
- 0.3.0 - 2026-01-15：
  - 新增前端用户反馈功能，支持飞书通知
  - 新增自训练（伪标签 / 不确定性筛选）功能
  - 新增训练后绘图（回归诊断）功能
- 0.2.0 - 2026-01-12：
  - 集成版新增药物模块（SMILES + 条件 → 疗效）
  - 集成版新增表位虚拟筛选预测器（序列 + 实验条件 → 疗效）
  - 前端集成表位、药物与数据增强/去噪模块
- 0.1.0 - 2026-01-07：
  - 前端构建优化与修复
  - 数据导入支持更多格式（Parquet/Feather）

## 0.6.1 - 2026-01-26：
  - 新增字符级 Transformer 药效预测模块（`src/drug/transformer_predictor.py`），
    支持在前端进行训练、单条预测与批量筛选（可保存为 `models/drug_transformer_model.pt`）。
  - 前端集成：在“药物疗效预测”标签下新增子标签 “Transformer训练” 与 “Transformer预测”。
  - 提供编程接口：`train_transformer_bundle(...)` 与 `predict_transformer_one(...)`，可用于脚本化训练/预测。

## 0.6.2 - 2026-01-26：
  - 新增分子生成：GAN + 进化算法（交叉 + 变异）生成候选 SMILES，支持用已训练模型评分。

### Transformer 模块 快速使用

- 前端：启动 Streamlit 前端（参见上文），进入 **药物 → Transformer训练**，上传训练 CSV，设置参数，点击“开始 Transformer 训练”。训练完成后可下载模型文件（.pt）。
- 批量筛选：在 **药物 → 批量虚拟筛选** 中选择模型类型为 `transformer(pt)`，上传/选择模型并上传候选 CSV，开始筛选并下载结果。
- 编程接口（示例）：

```python
from src.drug.transformer_predictor import train_transformer_bundle, predict_transformer_one, load_transformer_bundle
import pandas as pd

df = pd.read_csv('data/example_drug.csv')
bundle, metrics = train_transformer_bundle(df, smiles_col='smiles', target_col='efficacy')
print(metrics)

# 单条预测
pred = predict_transformer_one(bundle, smiles='CCO', env_params={'dose': 10, 'freq': 2})
print(pred)

# 加载已保存模型并预测
loaded = load_transformer_bundle('models/drug_transformer_model.pt')
pred2 = predict_transformer_one(loaded, smiles='CCO')
```

注意：Transformer 模型依赖 PyTorch（`torch`），训练与推理在没有 GPU 时仍可在 CPU 上运行，但训练速度较慢，建议在有 CUDA 的环境下启用 `use_cuda=True`。

**集成与运行示例**

- 在本地安装依赖（建议在虚拟环境中）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- 启动前端（Streamlit）：

```powershell
$env:PYTHONPATH='.'; streamlit run src/frontend.py --server.port 8503
```

- 运行完整训练示例（CSV 或合成数据）：

```powershell
$env:PYTHONPATH='.'; python .\scripts\train_transformer_full_example.py
```

生成的模型保存在 `build/transformer_full.pt`，生成的嵌入示例保存在 `build/emb_drug.npy` 与 `build/emb_epitope.npy`。