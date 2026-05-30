# Autodl 服务器实验运行指南

## 文件清单

需要上传到服务器 `scripts/` 目录的文件：
- `experiment_D_case_study.py` — 实验 D（端到端案例研究）+ 实验 A（adaptive vs fixed 权重）
- `experiment_C_bio_gated_moe.py` — 实验 C（BioGatedMOE vs MOERegressor）
- `experiment_B_cross_module_consistency.py` — 实验 B（跨模块一致性对比）
- `run_all_experiments.py` — 一键运行所有实验

## 运行步骤

### 方法 1: 一键运行（推荐）
```bash
cd /root/IGEM集成方案   # 或你的项目目录
python scripts/run_all_experiments.py
```

### 方法 2: 逐个运行（更可控）
```bash
cd /root/IGEM集成方案

# 实验 D (最重要 — 端到端案例研究)
python scripts/experiment_D_case_study.py

# 实验 C (BioGatedMOE 对比)
python scripts/experiment_C_bio_gated_moe.py

# 实验 B (跨模块一致性对比)
python scripts/experiment_B_cross_module_consistency.py
```

## 预期输出

所有结果保存到 `benchmarks/results/` 目录：
- `experiment_D_case_study.json` — 3个案例的完整推理链 + adaptive vs fixed 权重对比
- `experiment_C_bio_gated_moe.json` — BioGatedMOE vs MOERegressor MAE/R² 对比
- `experiment_B_cross_module_consistency.json` — 跨模块一致性开启/关闭对比
- `experiment_runner_log.json` — 运行日志

## 拿回结果

实验完成后，把 `benchmarks/results/` 下的 4 个 JSON 文件下载回本地，
放到 `D:\IGEM集成方案\benchmarks\results\` 目录下，然后告诉我——我会：
1. 分析结果
2. 修正论文事实错误（CTM描述、5D权重描述）
3. 用实验结果重写论文叙事（协同推理框架）

## 可能遇到的问题

1. **import 报错**：确保 confluencia 相关包在 Python path 中
   - 脚本已自动添加 sys.path，但如果项目结构不同，需要调整路径

2. **RDKit 报错**：确保服务器安装了 rdkit
   ```bash
   pip install rdkit-pypi
   ```

3. **pandas/scikit-learn 报错**：
   ```bash
   pip install pandas scikit-learn numpy
   ```

4. **单个实验超时**：run_all_experiments.py 设置了5分钟超时
   - 如果超时，可以单独运行该实验，或修改 timeout 参数

5. **数据不匹配**：如果服务器上的 epitope benchmark 数据路径不同
   - 实验 C 使用的是合成数据，不依赖 benchmark JSON
   - 实验 D/B 直接调用 pipeline，不需要额外数据文件