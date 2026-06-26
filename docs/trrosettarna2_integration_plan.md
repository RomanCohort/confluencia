# trRosettaRNA2 高通量 circRNA 3D 结构生成方案

## 目标

- **月产量**: 6000 条 circRNA 序列
- **硬件**: DGX Spark (8× A100/H100) 或 云GPU (AutoDL等)
- **方法**: trRosettaRNA2 (替代 RoseTTAFold2NA) + OpenMM 环化

---

## 为什么选择 trRosettaRNA2

| 对比项 | RoseTTAFold2NA | trRosettaRNA2 |
|--------|----------------|---------------|
| 速度 | 5-10 min/seq | 1-2 min/seq |
| MSA需求 | 需要 | 可选 |
| RNA特异性 | 通用核酸 | RNA优化 |
| 预测输出 | 3D坐标 | 距离/朝向约束 |
| 环化兼容性 | 需要额外处理 | 约束可直接引导 |

**吞吐量提升**: trRosettaRNA2 + fast cyclization 可达 **3-5x** 加速

---

## 管线架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  trRosettaRNA2 circRNA 3D Pipeline                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌────────────────┐    ┌─────────────────┐             │
│  │   Input      │───▶│   Stage 1      │───▶│    Stage 2      │             │
│  │   FASTA      │    │   ViennaRNA    │    │   trRosettaRNA2 │             │
│  │  circRNA seq │    │   2D Structure │    │   Distance/Orie │             │
│  └──────────────┘    └────────────────┘    └─────────────────┘             │
│                              │                      │                        │
│                              ▼                      ▼                        │
│                       ┌────────────────┐    ┌─────────────────┐             │
│                       │  dot-bracket   │    │  restraints.npy │             │
│                       │  bp_probs      │    │  (dist/ori)     │             │
│                       └────────────────┘    └─────────────────┘             │
│                                                     │                        │
│                              ┌──────────────────────┘                        │
│                              ▼                                               │
│                       ┌─────────────────┐    ┌─────────────────┐            │
│                       │    Stage 3      │───▶│    Stage 4      │            │
│                       │   OpenMM        │    │   MD Relax      │            │
│                       │   Cyclization   │    │   (fast: 2ns)   │            │
│                       └─────────────────┘    └─────────────────┘            │
│                              │                      │                        │
│                              ▼                      ▼                        │
│                       ┌─────────────────┐    ┌─────────────────┐            │
│                       │   Cyclized PDB  │    │   Relaxed PDB   │            │
│                       └─────────────────┘    └─────────────────┘            │
│                                                     │                        │
│                                                     ▼                        │
│                                              ┌─────────────────┐            │
│                                              │    Stage 5      │            │
│                                              │   Quality       │            │
│                                              │   Filter        │            │
│                                              └─────────────────┘            │
│                                                     │                        │
│                                                     ▼                        │
│                                              ┌─────────────────┐            │
│                                              │   Output        │            │
│                                              │   TorusFold     │            │
│                                              │   Training Data │            │
│                                              └─────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 吞吐量分析

### 单序列时间预算

| 阶段 | Fast模式 | Quality模式 |
|------|----------|-------------|
| Stage 1: ViennaRNA | 1-2 sec | 1-2 sec |
| Stage 2: trRosettaRNA2 | 30-60 sec | 60-120 sec |
| Stage 3: OpenMM环化 | 10-20 sec | 30-60 sec |
| Stage 4: MD弛豫 | 60-120 sec (2ns) | 300-600 sec (10ns) |
| Stage 5: 质量筛选 | 1-2 sec | 1-2 sec |
| **总计** | **2-3 min** | **7-12 min** |

### 6000序列时间估算

| 配置 | GPU时间/序列 | 总GPU时间 | 8-GPU并行时间 |
|------|-------------|----------|---------------|
| Fast模式 | 3 min | 300 GPU-h | **37.5 h** |
| Quality模式 | 10 min | 1000 GPU-h | **125 h** |

**结论**: Fast模式 + 8 GPU 并行 → 37.5小时完成6000条序列

---

## 并行策略

### DGX Spark (8× GPU) 调度

```python
# 使用 Ray 或 multiprocessing 并行
import ray

@ray.remote(num_gpus=1)
class trRosettaRNA2Worker:
    def process_batch(self, sequences, gpu_id):
        pipeline = CircRNAPipeline(mode='fast', device=f'cuda:{gpu_id}')
        results = []
        for seq in sequences:
            result = pipeline.run(seq)
            results.append(result)
        return results

def run_parallel(sequences, num_gpus=8):
    # 分片
    chunks = np.array_split(sequences, num_gpus)
    
    workers = [trRosettaRNA2Worker.remote() for _ in range(num_gpus)]
    futures = [w.process_batch.remote(chunk, i) for i, (w, chunk) in enumerate(zip(workers, chunks))]
    
    results = ray.get(futures)
    return results
```

---

## trRosettaRNA2 安装

### 方法1: 官方安装 (推荐)

```bash
# 从官方仓库安装
git clone https://github.com/pylelab/trRosettaRNA2.git
cd trRosettaRNA2

# 创建环境
conda create -n trrosetta python=3.9
conda activate trrosetta

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 下载预训练模型
wget https://yanglab.qd.sdu.edu.cn/trRosettaRNA2/models/trRosettaRNA2.pt
```

### 方法2: Docker镜像

```dockerfile
FROM nvidia/cuda:11.8-cudnn8-devel-ubuntu22.04

# 安装依赖
RUN apt-get update && apt-get install -y \
    python3 python3-pip git wget \
    && rm -rf /var/lib/apt/lists/*

# 安装Python包
RUN pip3 install torch torchvision openmm viennarna biopython numpy

# 安装trRosettaRNA2
RUN git clone https://github.com/pylelab/trRosettaRNA2.git /opt/trRosettaRNA2
WORKDIR /opt/trRosettaRNA2
RUN wget -O models/trRosettaRNA2.pt https://yanglab.qd.sdu.edu.cn/trRosettaRNA2/models/trRosettaRNA2.pt

ENV PYTHONPATH="/opt/trRosettaRNA2:${PYTHONPATH}"
```

---

## 云GPU测试方案 (AutoDL)

### 租赁配置

| 平台 | GPU | 价格 | 推荐配置 |
|------|-----|------|----------|
| AutoDL | RTX 3090 | ¥1.5/h | 测试用 |
| AutoDL | A100 40G | ¥12/h | 生产用 |
| 阿里云 | V100 | ¥15/h | 企业用 |

### 测试流程

```bash
# 1. 启动AutoDL实例 (选择PyTorch镜像)

# 2. 安装依赖
pip install openmm viennarna biopython

# 3. 安装trRosettaRNA2
cd /root
git clone https://github.com/pylelab/trRosettaRNA2.git
cd trRosettaRNA2
wget -O models/trRosettaRNA2.pt https://yanglab.qd.sdu.edu.cn/trRosettaRNA2/models/trRosettaRNA2.pt

# 4. 运行测试
python /root/IGEM集成方案/confluencia_3_0/core/circrna/torusfold/circrna_3d_pipeline/pipeline_trrosetta.py \
    --test --mode fast
```

---

## 下一步

1. **立即**: 我创建 `stage2_trrosetta.py` 集成代码
2. **准备好GPU后**: 安装测试 trRosettaRNA2
3. **验证**: 在10条序列上对比 RoseTTAFold2NA vs trRosettaRNA2 质量
4. **量产**: 部署到DGX Spark，运行6000条序列
