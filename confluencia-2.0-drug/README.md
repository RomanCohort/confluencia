# Confluencia 2.0 — 小样本 circRNA 药物发现多模块平台

> Adaptive Mixture-of-Experts with Pharmacokinetic Dynamics for Small-Sample circRNA Drug Discovery

## 定位

Confluencia 2.0 将 1.0 的单页原型重构为**多模块架构**，核心创新是样本量自适应 MOE 集成学习和 RNACTM 六室药代动力学模型。2.0 由 Drug 和 Epitope 两个独立模块组成，共享 `confluencia_shared` 库。

## 架构

```
confluencia-2.0-drug/          # 药物预测模块
├── app.py                      # Streamlit 前端 (129KB)
├── core/
│   ├── predictor.py            # DrugModelBundle, build_model()
│   ├── featurizer.py           # SMILES 分子特征化
│   ├── moe.py                  # MOE 集成回归器
│   ├── evolution.py            # 分子/序列进化优化
│   ├── pkpd.py                 # PK/PD 模拟
│   ├── ctm.py                  # RNACTM 六室 PK 模型
│   ├── innate_immune.py        # 先天免疫评估
│   ├── immune_abm.py           # 免疫 ABM 模拟
│   ├── admet.py                # ADMET 毒性预测
│   ├── gnn.py                  # 图神经网络
│   ├── generative.py           # 分子生成
│   └── ... (50+ 核心文件)
├── api/                        # FastAPI 接口层
└── tests/

confluencia-2.0-epitope/       # 表位预测模块
├── app.py                      # Streamlit 前端 (57KB)
├── core/
│   ├── predictor.py            # EpitopeModelBundle
│   ├── featurizer.py           # 序列特征化
│   ├── moe.py                  # MOE 回归器
│   ├── esm2_encoder.py         # ESM-2 蛋白质嵌入
│   ├── mamba3.py               # Mamba 序列模型
│   ├── mhc_features.py         # MHC 等位基因特征
│   └── ...
└── tests/

confluencia_shared/             # 共享库
├── moe.py                      # MOERegressor (56KB, 核心集成器)
├── models.py                   # ModelFactory, ModelConfig
├── lang.py                     # 国际化 (t(), lang_toggle())
├── ui_common.py                # 通用 UI 组件
├── metrics.py                  # rmse(), reg_metrics()
├── optim/                      # 差分进化, 超参优化
└── utils/                      # EMA, 绘图, 数据工具

confluencia_joint/              # 联合评估
├── fusion_layer.py             # 融合层
├── joint_evaluator.py          # 三维联合评估
└── scoring.py                  # 评分系统
```

## 核心创新

| 创新点 | 说明 |
|--------|------|
| **样本量自适应 MOE 集成** | 根据数据量自动选择和加权回归专家 (Ridge/HGB/RF/MLP/XGB/LGB/ET) |
| **RNACTM 药代动力学模型** | 首个针对 circRNA 的六室 PK 模型（注射→LNP→内吞→胞质释放→翻译→清除） |
| **Mamba3Lite 序列编码器** | 三时间常数自适应状态空间递归 + 四尺度池化 + 自注意力增强 |
| **反思式 RL 进化** | Pareto 导向目标权重搜索 + 风险门控 + 策略偏移诊断 |
| **Bootstrap CI 置信区间** | 小样本 t 分布 / 大样本 bootstrap percentile |
| **分层交叉验证** | 按目标变量分位数分箱，确保各 fold 效能分布均衡 |

## 核心实验结果

| 指标 | 数值 | 说明 |
|------|------|------|
| 288K IEDB AUC (allele-aware) | **0.80** | HGB，MHC 等位基因特征编码 |
| Drug Ridge R² | **0.984** | 小样本药物预测最优 |
| MOE MAE (表位) | 0.389 | 比 Ridge 降低 39.2% (p<0.001) |
| Mamba3Lite+Attn(d=16) | MAE=0.395, R²=0.802 | 注意力增强最佳单编码器 |
| Drug 无 FP R² | **0.960** | 移除过拟合的 Morgan FP |
| TCCIA circRNA 验证 | r=0.888 | N=75 |

## 快速启动

### Drug 模块
```bash
cd confluencia-2.0-drug
pip install -r requirements.txt
streamlit run app.py
```

### Epitope 模块
```bash
cd confluencia-2.0-epitope
pip install -r requirements.txt
streamlit run app.py
```

### API 服务
```bash
cd confluencia-2.0-drug
python server.py    # FastAPI 服务 (localhost:8000)
```

## 模块间关系

```
confluencia_shared (核心)
    ├── moe.py, models.py, lang.py, utils
    │
    ├── confluencia-2.0-drug (药物预测)
    │   ├── MOE 集成 + RNACTM PK + 分子进化
    │   └── api/ → FastAPI 接口
    │
    ├── confluencia-2.0-epitope (表位预测)
    │   ├── MOE 集成 + ESM-2 + MHC 特征
    │   └── cloud/ → 云端推理
    │
    └── confluencia_joint (联合评估)
        └── 三维融合评分
```

## 与其他版本的关系

- **1.0**：单页桌面原型，2.0 完全重构为多模块架构
- **3.0**：在 2.0 基础上整合 circRNA 子系统和 TNBC 模拟环境，2.0 模块通过桥接接入 3.0

## 依赖

```toml
[核心]
numpy>=1.24, pandas>=2.0, scikit-learn>=1.3, scipy>=1.10
streamlit>=1.30, joblib>=1.3, rich>=13.0

[可选]
torch>=2.0       # Mamba3Lite, ESM-2, GNN
rdkit>=2023.03   # 分子指纹/描述符
```
