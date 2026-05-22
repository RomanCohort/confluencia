# Confluencia 2.2 两周冲刺计划（Epitope）

## 目标
- 从"可运行原型"推进到"跨环境可比、分层可解释、可复现"的研究版本。

## 第 1 周

### 1. 可信评估增强（已落地）
- 严格 train/val/test 切分。
- 5-fold 交叉验证统计（mean/std/95%CI）。
- 固定基线（Linear/RF/HGB）门槛判定。
- 失败样本 Top-N 导出。
- 外部独立测试集评估。
- 预测区间校准表（验证集残差校准测试覆盖率）。
- 泄漏审计（train/test 精确重叠率）。

### 2. 结构化分析（已落地）
- 序列长度分层评估（short/mid/long）。
- Real-Mamba 与 Fallback 标定，显式报告差值。

## 第 2 周

### 3. 显著性验证（已落地）
- 关键模型对比引入统计显著性检验（配对符号检验）。
- 输出 effect size（Cohen's dz），避免仅看平均分。
- 前端展示 p-value、effect size、非零配对数。

### 4. 鲁棒性与偏置检查（已落地）
- 按氨基酸组成分层（疏水/带电/芳香比例）评估。
- OOD 子集表现报告（基于训练集 5%-95% 分位阈值）。
- 前端展示分层表格与 OOD 对比。

### 5. 发布级复现（已落地）
- 每次训练自动保存：配置、数据哈希、环境依赖、核心指标（`core/training.py` `_auto_save_repro`）。
- 统一实验报告模板（Markdown + CSV），含可信评估全量字段（`core/report_template.py`）。
- 前端 `_save_repro_bundle` 与核心 `_auto_save_repro` 共享报告模板。

## 清理

- 移除 `app.py` 中与 `core/reliability.py` 重复的死代码（`_credible_eval_epitope`、`_safe_metrics`、`_mean_std_ci`、`_build_sklearn_regressor`、`_predict_epitope_backend`）。
- 移除 `app.py` 中未使用的 sklearn 和 core 导入。

## 验收门槛
- Test RMSE 优于固定基线最佳模型。
- CV RMSE 95%CI 收敛（区间宽度可接受）。
- 泄漏重叠率接近 0。
- Smoke test 全绿。
