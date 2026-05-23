"""
Confluencia circRNA Module
==========================

基于 drug 2.0 架构的 circRNA 预测模块，集成 RNA-FM 序列编码器。

## 架构

```
confluencia-circrna-encoder/
├── app.py                  # Streamlit 应用
├── core/
│   ├── encoder.py          # RNA-FM 编码器 ★
│   ├── predictor.py        # 预测接口
│   ├── pipeline.py         # 流程管道
│   ├── scoring.py          # 评分模块
│   ├── features.py         # 特征构建
│   └── training.py         # 训练模块
├── api/
│   └ routers/
│       └── circrna.py      # FastAPI 路由
├── scripts/
│   └ train.py              # 训练脚本
└ data/
│   └ models/               # 模型存储
```

## 运行

```bash
# Streamlit 应用
streamlit run confluencia-circrna-encoder/app.py

# 训练模型
python confluencia-circrna-encoder/scripts/train.py \
    --training-data confluencia_circrna/data/training/circrna_training_pairs_real.csv \
    --output-dir confluencia-circrna-encoder/data/models \
    --epochs 30
```

## 双模态联动

- circRNA 模块 (本模块)
- 药物模块 (confluencia-2.0-drug)
- 表位模块 (confluencia-2.0-epitope)
"""