# TorusFold S10 — A800 训练部署清单

适用：A800 / A100 / H100 等 CUDA GPU（非本地 ROCm）。训练入口：
`train_s10_curriculum.py`（4 阶段课程 + Phase 0 PDB 混合预训练）。

---

## 0. 快速启动（数据已齐）

```bash
conda activate <env>       # torch ≥2.0 + CUDA, 见 §1.1
cd confluencia_3_0/core/circrna/torusfold
python train_s10_curriculum.py
```

Phase 0 自动加载 PDB 混合数据（11214 条），结束后自动 `restore_cg_data()` 恢复 CG 训练。

---

## 1. 环境准备（首次）

### 1.1 Python 环境
```bash
conda create -n circrna python=3.10 -y
conda activate circrna
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy biopython
# ViennaRNA (必需, pair_probs 来源)
conda install -c bioconda viennarna -y
python -c "import RNA; print('ViennaRNA OK')"
```

### 1.2 数据文件（须在 `data/` 下）
| 文件 | 用途 | 状态 |
|------|------|------|
| `data/circrna_3d_all_consolidated.npz` | CG 训练数据 82106 条 | ✅ 已有 |
| `data/circrna/circbase_seqs.fa.gz` | CG 样本序列 (FASTA) | ✅ 已有 |
| `data/pdb_cyclized/consolidated.npz` | Phase 0 PDB 混合 11214 条 (is_circular) | ✅ 已有 |
| `data/circrna_3d_all_pair_probs.npz` | ViennaRNA bpp (**Phase 1-4 必需**) | ⚠️ **本地续跑中, 需合并后上传** |
| `data/pdb_cyclized/consolidated_pair_probs.npz` | PDB 距离核 (Phase 0 不需要) | ✅ Phase 0 用几何核实时算 |

> ⚠️ **`circrna_3d_all_pair_probs.npz` 缺失不是"可选"**——它是 Phase 1-4 的
> AnchorScorer 输入。缺失时 collate 填全零矩阵：训练不崩，但 `anchor_aux_loss=0`、
> 锚点选择退化（完全失去物理配对先验）。
>
> **本地已有 38/42 分片**（`data/.precompute_tmp/bp_0..37.npz`, 共 162GB,
> ViennaRNA bpp 每样本 [L,L] float32）。分片 38-41（~4248 样本）曾在算到
> shard_38 时中断，**续跑命令**（resumable, 自动跳过已完成的 38 分片）:
> ```bash
> python precompute_pair_probs.py \
>     --consolidated ../../../../data/circrna_3d_all_consolidated.npz \
>     --fasta ../../../../data/circrna/circbase_seqs.fa.gz \
>     --output ../../../../data/circrna_3d_all_pair_probs.npz \
>     --tmp-dir ../../../../data/.precompute_tmp \
>     --chunk-size 2000 --max-len 1000 --n-threads 32
> ```
> 完成后合并出 `circrna_3d_all_pair_probs.npz`（bp_probs 是 82106 个 [L,L] 的
> object 数组），再连同 5 个数据文件一起上传 A800。
>
> `--max-len` 默认 1000: L>1000 的 4749 条 xlong 样本会被写入**全零矩阵**
> (ViennaRNA O(L²) 对超长序列太慢)。如需长序列配对先验可 `--max-len 5000`,
> 但单条可能需数分钟, 建议先默认 1000 跑通, 后续再补长序列。

### 1.3 推理/验证环境
- 批量筛选: `generate_32_workers.py`（默认 `refine=False` 高通量直出坐标）
- 单条验证: `validate_phase0.py`

---

## 2. 训练启动与监控

```bash
# 前台跑 (推荐 tmux/screen 保活)
tmux new -s train
conda activate circrna
cd confluencia_3_0/core/circrna/torusfold
python train_s10_curriculum.py 2>&1 | tee train_a800.log
```

### 关键日志行（确认启动正确）
```
[PDB Phase 0] 11214 samples loaded (5607 circular, 5607 linear), buckets=...
[stop-grad] latent→diffusion detached (P1前25%)
MC-Dropout uncertainty weighting per bucket...
```

### 常见失败
| 症状 | 原因 | 解决 |
|------|------|------|
| `[skip] PDB 3D pretrain data not found` | `data/pdb_cyclized/consolidated.npz` 缺失 | 上传 PDB npz (10MB) |
| `pair_probs not found ... run precompute_pair_probs.py` | 未跑预计算 | 跑 precompute 或接受 fallback |
| MC-Dropout UQ 崩 (pred=None) | **旧代码 bug (已修)** | 确认 v5 commit `44f96b4e` 已合入 |
| normalization 崩 (tensor size 80 vs 4) | **旧代码 bug (已修)** | 同上, 确认 `[B,1,1]` 分母版本 |

---

## 3. 训练后验证

```bash
# 探针 acc (Encoder latent 是否学会 3D): 预期 85%+ 提升
python validate_phase0.py

# 高通量筛选 (refine=False 直出)
python generate_32_workers.py --input seqs.fa --output results/

# 终选结构 (refine=True, 100 步 physics_refine)
python generate_32_workers.py --input candidates.fa --refine True
```

---

## 4. 后续待办（数据可行后）⚠️

### 4.1 MSA 共进化特征接入（最大缺口, 接口已建数据待接）
`msa_features.py` 提供接口，但**未接入训练**。接入步骤：
1. 下载 Rfam.seed（实验验证的家族共识二级结构, ~50MB）：
   ```bash
   wget https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.seed.gz
   gunzip Rfam.seed.gz   # → data/rfam/Rfam.seed
   ```
2. 测试解析 + 家族匹配：
   ```bash
   python msa_features.py   # 自测 (consensus 配对/融合)
   python -c "from msa_features import load_rfam_consensus; e=load_rfam_consensus('data/rfam/Rfam.seed'); print(len(e), 'entries')"
   ```
3. 在 `train_s10_curriculum.py` 的 `load_pdb_phase0_data` / CG 加载处接入：
   `pair_probs_fused = fuse_pair_probs(vienna, consensus, alpha=0.7)`
4. 验证融合后训练收敛（对比 α=0.7 vs α=1.0 的 anchor_aux / diff_loss）

> 依赖：外网下载 Rfam.seed；ViennaRNA（已有）。零依赖 JackHMMER（k-mer 代理匹配）。

### 4.2 其他可选
- `consolidate_pdb_cyclized.py` 重建 PDB npz（如需换数据/加族）
- `physics_refine` 推理步数 100 已在 v5 默认；若高通量需求大可临时 `refine=False`

---

## 5. 版本基线（A800 必须 >= 这些 commit）
| commit | 内容 | 必需 |
|--------|------|------|
| `771379bd` | v4.1 CRBPSA 氢键 + Phase 0 数据切换 | ✅ |
| `44f96b4e` | v5 混合线性 RNA + anchor 扩展 + circular 感知 + MC-UQ/normalization 修复 | ✅ **必须** |
| `252d1d47` | 6 点评审回应 + msa_features 接口 | ✅ |
| `a7a20f24` | 架构图 v5 | 可选 |

```bash
git log --oneline feat/scheme8-iterative-refinement   # 确认上述 commit 在
git checkout feat/scheme8-iterative-refinement
```
