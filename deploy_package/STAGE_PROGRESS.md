# circRNA 3D 结构预测流水线 - 进度记录

## 流水线概览

5 个 Stage，为 TorusFold 生成训练数据：

| Stage | 模块 | 作用 | 状态 |
|-------|------|------|------|
| 1 | `stage1_vienna.py` | ViennaRNA 二级结构（dot-bracket + mfe） | ✅ 完成 |
| 2 | `stage2_trrosetta.py` | trRosettaRNA2 3D 预测 | ⏸️ 部分（等大内存机器） |
| 3 | `stage3_cyclize.py` | BSJ 环化（线性→环状） | ⏳ 待续 |
| 4 | `stage4_md.py` | OpenMM MD 弛豫 | ⏳ 待续 |
| 5 | `stage5_quality.py` | 质量过滤 + 导出 TorusFold 格式 | ⏳ 待续 |

## 当前进度（2026-07-03）

### ✅ Stage 1 完成
- **数据量**: 129695 / 130000 条（99.8%）
- **输出位置**: `/root/autodl-tmp/confluencia/confluencia/deploy_package/stage1_merged/`
- **格式**: `<seq_id>/stage1_result.json`（含 sequence, dot_bracket, mfe, bsj_start/end）

### 📊 数据分布（129695 条）
```
  0-  500: 41.8% (54159)  ████████████████████
500- 1000: 26.0% (33726)  █████████████
1000-1500: 12.3% (15982)  ██████
1500-2000:  7.1% ( 9196)  ███
2000-3000:  7.2% ( 9354)  ███   ← Stage 2 跳过（OOM）
3000-5000:  5.6% ( 7278)  ██    ← Stage 2 跳过（OOM）

平均: 964 nt | 中位数: 618 nt | 范围: 1-4999 nt
短序列(≤2000nt): 87.2% (113078)  ← 现在可跑
长序列(>2000nt): 12.8% ( 16617)  ← 等 DGX Spark
```

### ⏸️ Stage 2 状态
- 云 GPU（32GB V100）跑长序列 OOM
- 单条长序列（~5000nt）需 16.87 GiB
- **方案**: 跳过 >2000nt 序列，等 DGX Spark (128GB 统一内存) 跑剩余

## 关键脚本

| 脚本 | 用途 |
|------|------|
| `run_stage1_only.py` | Stage 1 单进程（参数 --fasta --output） |
| `run_stage2_only.py` | Stage 2 单进程（参数 --input --output） |
| `run_stage2_batch.sh` | Stage 2 分批并行（控制 GPU 内存，默认并发 5） |
| `split_stage1_for_parallel.py` | 把 stage1_merged 切成 N 份软链接 |
| `inspect_stage1_data.py` | 检查 Stage 1 数据结构 + 长度分布 |

## 环境变量配置

```bash
# Stage 2 跳过长序列阈值（默认 2000）
export STAGE2_MAX_SEQ_LEN=2000

# PyTorch 内存碎片优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 启动命令

### Stage 1（已完成）
```bash
python run_stage1_only.py --fasta circbase_filtered_5000.fa --output stage1_merged
```

### Stage 2（待续 - DGX Spark）
```bash
# 切分 30 份
python split_stage1_for_parallel.py stage1_merged 30

# 分批并行（DGX Spark 128GB 可用 8 并发）
bash run_stage2_batch.sh 8 stage1_chunk stage2_output
```

## 硬件决策

- **AutoDL V100 32GB**: 长序列 OOM，只能跑 ≤2000nt
- **DGX Spark 128GB 4TB**（¥30000）: 一次性解决内存+存储，69天回本
