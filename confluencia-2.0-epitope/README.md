# Confluencia 2.0 — 表位预测模块

> MHC 表位疗效预测：Mamba3Lite 编码 + MOE 集成 + 多尺度敏感性分析

## 定位

Confluencia 2.0 Epitope 模块面向 circRNA 免疫激活场景的微观疗效预测，预测 MHC-I 表位诱导 T 细胞应答的效能。与 Drug 模块共享 `confluencia_shared` 库，提供独立的 Streamlit 前端和训练/预测 API。

## 架构

```
confluencia-2.0-epitope/
├── app.py                      # Streamlit 前端 (57KB)
├── epitope_frontend.py         # 表位专用前端入口
├── core/
│   ├── training.py             # 训练/预测 API、模型导入导出
│   ├── pipeline.py             # 一体化流水线
│   ├── features.py             # 序列+环境特征工程
│   ├── featurizer.py           # 表位特征化
│   ├── torch_mamba.py          # PyTorch/Mamba 训练推理
│   ├── mamba3.py               # Mamba3Lite 序列模型
│   ├── moe.py                  # MOE 回归器
│   ├── esm2_encoder.py         # ESM-2 蛋白质嵌入 (实验性)
│   ├── mhc_features.py         # MHC 等位基因特征
│   ├── sensitivity.py          # 敏感性分析
│   ├── reliability.py          # 可靠性评估
│   └── ... (21 核心文件)
├── cloud/                      # 云端推理配置
└── tests/
```

## 核心创新

| 创新点 | 说明 |
|--------|------|
| **Mamba3Lite 序列编码器** | 三时间常数自适应状态空间递归 + 四尺度池化 (mean/local/meso/global) + 自注意力增强 |
| **样本量自适应 MOE 集成** | Ridge/HGB/RF/MLP/XGB/LGB/ET 专家按 OOF-RMSE 反比加权 |
| **多尺度敏感性分析** | 邻域贡献聚合 (local/meso/global) + 梯度×激活 saliency |
| **代理监督目标** | 无标签时自动构建弱监督目标 (dose+freq+circ_expr+ifn_score) |
| **MHC pseudo-sequence 编码** | AUC=0.917 的 MHC 结合预测核心特征 |
| **Windows 打包发布** | PyInstaller 一键打包 + release 脚本 |

## 核心实验结果

| 指标 | 数值 | 说明 |
|------|------|------|
| 288K IEDB AUC (allele-aware) | **0.80** | HGB + MHC 等位基因特征 |
| MOE MAE | **0.389** | 比 Ridge 降低 39.2% (p<0.001) |
| Mamba3Lite+Attn(d=16) | MAE=0.395, R²=0.802 | 注意力增强最佳单编码器 |
| MHC pseudo-sequence AUC | **0.917** | 远超 ESM-2 均值池化 (0.537) |
| ESM-2 650M 实验 | AUC=0.537 | **失败** — 均值池化不适合短肽 (8-11 AA) |

## 快速启动

### 表位专用前端
```bash
cd confluencia-2.0-epitope
pip install -r requirements.txt
streamlit run epitope_frontend.py
```

### 通用入口
```bash
streamlit run app.py
```

### 训练/预测 API
```python
from core.training import train_epitope_model, predict_epitope_model

# 训练
model_bundle, report = train_epitope_model(
    train_df,
    compute_mode="auto",
    model_backend="torch-mamba",  # 或 sklearn-moe/hgb/ridge
)

# 预测
pred_df, sens = predict_epitope_model(model_bundle, infer_df)
```

## 输入数据格式

必需列：
- `epitope_seq` — 氨基酸序列 (8-11 AA)

可选数值上下文列：
- `dose`, `freq`, `treatment_time`, `circ_expr`, `ifn_score`

可选标签列：
- `efficacy` — 缺失时使用代理监督目标

最小示例 CSV：
```csv
epitope_seq,dose,freq,treatment_time,circ_expr,ifn_score,efficacy
SLYNTVATL,2.0,1.0,24,1.2,0.7,1.80
GILGFVFTL,1.0,0.8,48,0.6,0.5,1.10
```

## 原理公式

### 特征拼接
$$\mathbf{x}=[\mathbf{x}_{\text{seq-summary}}, \mathbf{x}_{\text{local}}, \mathbf{x}_{\text{meso}}, \mathbf{x}_{\text{global}}, \mathbf{x}_{\text{kmer2}}, \mathbf{x}_{\text{kmer3}}, \mathbf{x}_{\text{bio}}, \mathbf{x}_{\text{env}}]$$

### MOE 权重
$$w_k = \frac{1/\max(\text{RMSE}_k, \epsilon)}{\sum_j 1/\max(\text{RMSE}_j, \epsilon)}$$

### 预测加权和
$$\hat{y} = \sum_k w_k \hat{y}^{(k)}$$

详见完整公式推导与符号表（原 README 第 143-308 行）。

## ESM-2 实验记录 (失败)

> 2026年4月22-23日，ESM-2 均值池化不适合短肽 MHC 结合预测

| 策略 | 结果 | 原因 |
|------|------|------|
| ESM-2 PCA 64D 替换传统特征 | AUC=0.508 | 比基线 0.537 更差 |
| 传统特征 + ESM-2 PCA 补充 (35M) | AUC=0.594 | PCA 丢失 anchor position 判别方向 |
| 传统特征 + ESM-2 PCA 补充 (650M) | AUC=0.537 | 均值池化丢失 position-specific motifs |

**结论**：MHC pseudo-sequence 编码 (AUC=0.917) 为当前最优方案，ESM-2 仅适用于长蛋白质 (>50 AA)。

## 复现与测试

### 冒烟测试
```bash
python tests/smoke_test.py
```

### 复现流水线
```powershell
powershell -ExecutionPolicy Bypass -File tools/reproduce_pipeline.ps1
```

输出日志：`logs/reproduce/`

## Windows 打包

### 构建目录包
```powershell
powershell -ExecutionPolicy Bypass -File build_full.ps1 -InstallDeps -Clean
```

输出：`dist/confluencia-2.0-epitope/`

### 构建 release zip
```powershell
powershell -ExecutionPolicy Bypass -File release_full.ps1 -Build -InstallDeps -Version full
```

输出：`release/`

## 模块间关系

```
confluencia_shared (核心)
    ├── moe.py, models.py, lang.py, utils
    │
    ├── confluencia-2.0-drug (药物预测)
    │   └── MOE 集成 + RNACTM PK + 分子进化
    │
    └── confluencia-2.0-epitope (表位预测) ← 本模块
        ├── MOE 集成 + Mamba3Lite + MHC 特征
        └── cloud/ → 云端推理配置
```

## 与其他版本的关系

- **1.0**：单页桌面原型，无 MOE/Mamba
- **2.0 Drug**：药物预测模块，共享 MOE 库
- **3.0**：TNBC simulacrum + circRNA 统一架构，通过 `EpitopeBridge` 连接本模块

## 依赖

```toml
[核心]
numpy>=1.24, pandas>=2.0, scikit-learn>=1.3, scipy>=1.10
streamlit>=1.34, joblib>=1.3

[可选]
torch>=2.2           # Mamba3Lite
mamba-ssm>=2.2       # Linux/macOS only，Windows 自动回退
```

## 常见问题

1. **Windows 没有 mamba-ssm 会失败吗？**
   不会。系统自动使用回退模块，torch-mamba 仍可运行。

2. **只想验证流程？**
   运行 `tests/smoke_test.py`。

3. **如何接入 3.0 simulacrum？**
   通过 `confluencia_3_0/core/confluencia/epitope_bridge.py` 懒加载连接。

---

详见完整开发者文档：目录结构、模块职责图、变更影响矩阵、从 0 到 1 调试清单 (原 README 第 412-522 行)。