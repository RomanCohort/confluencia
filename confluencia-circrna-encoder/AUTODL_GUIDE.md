# AutoDL GPU 部署指南

## 1. 创建实例

- 镜像: PyTorch 2.0 + Python 3.10
- GPU: RTX 3090 / A100 (单卡即可)
- 数据盘: 挂载 /root/autodl-tmp

## 2. 克隆代码

```bash
cd /root/autodl-tmp
git clone https://github.com/RomanCohort/confluencia.git
cd confluencia/confluencia-circrna-encoder
```

## 3. 安装依赖

```bash
pip install esm fair-esm  # ESM2
pip install ViennaRNA     # circRNA structure
pip install scikit-learn  # metrics
```

## 4. 上传数据

将 `sequences_enhanced.csv` 上传到 `/root/autodl-tmp/`:

```bash
# 如果数据在项目中
cp data/circrna/sequences_enhanced.csv /root/autodl-tmp/
```

## 5. 运行训练

### 推荐配置 (ESM2 150M + CircPairformer)

```bash
python scripts/run_pathway_classification.py \
    --esm-model esm2_t30_150M_UR50D \
    --epochs 30 \
    --batch-size 8 \
    --lr 5e-4 \
    --c-z 64 \
    --n-pf-blocks 2 \
    --max-seq-len 200 \
    --device cuda \
    --data /root/autodl-tmp/sequences_enhanced.csv
```

### 更大模型 (ESM2 650M)

```bash
python scripts/run_pathway_classification.py \
    --esm-model esm2_t36_3B_UR50D \
    --epochs 20 \
    --batch-size 4 \
    --lr 3e-4 \
    --c-z 128 \
    --n-pf-blocks 4 \
    --max-seq-len 200 \
    --device cuda \
    --data /root/autodl-tmp/sequences_enhanced.csv
```

### 快速测试 (Mock backbone, 不需要 GPU)

```bash
python scripts/run_pathway_classification.py \
    --mock \
    --epochs 5 \
    --batch-size 16 \
    --device cpu \
    --c-z 16 \
    --n-pf-blocks 1 \
    --max-seq-len 128
```

## 6. 预期结果

| 模型 | 通路分类 | 免疫原性 AUC | 说明 |
|------|---------|-------------|------|
| Mock backbone | ~14% (随机) | ~0.50 | 无序列信息 |
| ESM2 8M + MLP | 30-40% | 0.6-0.7 | 有序列嵌入 |
| ESM2 150M + CircPairformer | 50-65% | 0.7-0.8 | +环形拓扑 |
| ESM2 650M + CircPairformer | 60-75% | 0.75-0.85 | 最佳 (需要大GPU) |

## 7. 结果文件

训练完成后:
- `models/pathway_best.pt` — 最佳模型权重
- `models/pathway_results.json` — 评估结果
- `models/pathway_history.json` — 训练曲线

## 8. 下载结果

```bash
# 从 AutoDL 下载到本地
scp -P <port> root@<address>:/root/autodl-tmp/confluencia/confluencia-circrna-encoder/models/pathway_results.json ./
scp -P <port> root@<address>:/root/autodl-tmp/confluencia/confluencia-circrna-encoder/models/pathway_best.pt ./
```
