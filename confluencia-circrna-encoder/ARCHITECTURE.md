# TorusFold v2 — 最终架构

```
输入: circRNA 序列 (A,C,G,U)
 │
 ▼
┌─────────────────────────────────────────────────────┐
│  1. FASTA 生成                                       │
│  circRNA 序列 → FASTA 文件 (U 不 T)                  │
│  同时用于 RNA-FM / RhoFold+ / ViennaRNA             │
└────────────────────┬────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
  ┌──────────┐ ┌──────────┐ ┌──────────────┐
  │ RNA-FM   │ │ RiNALMo  │ │ RhoFold+     │
  │ (5M)     │ │ (650M)   │ │ (~200M)      │
  │ 冻结     │ │ 冻结     │ │ 可选, 独立   │
  │          │ │          │ │              │
  │ 23M ncRNA│ │ 36M ncRNA│ │ PDB RNA结构  │
  │ d=640    │ │ d=1280   │ │ 输出: 3D坐标 │
  └────┬─────┘ └────┬─────┘ └──────┬───────┘
       │            │              │
       ▼            ▼              ▼
  ┌─────────────────────────────────────────────┐
  │  2. Torus Positional Encoding (TPE)         │
  │  sin(2π·n·i/L), cos(2π·n·i/L)  n=1..8      │
  │  关键: TPE(x, L) = TPE(x+L, L) 环形周期性 │
  └────────────────────┬────────────────────────┘
                       │  序列嵌入 (B, L, d_model)
                       ▼
  ┌─────────────────────────────────────────────┐
  │  3. Pair 初始化 (AF3-style)                 │
  │  z[i,j] = Linear_left(x[i])                │
  │        + Linear_right(x[j])                 │
  │        + Embed(d_circ(i,j))   ← 环形距离!  │
  │                                             │
  │  d_circ(i,j) = min(|i-j|, L-|i-j|)         │
  │  输出: (B, L, L, c_z=128)                  │
  └────────────────────┬────────────────────────┘
                       │  pair_repr (B, L, L, c_z)
                       ▼
  ┌─────────────────────────────────────────────┐
  │  4. CircPairformer Stack ×4 (核心创新)      │
  │                                             │
  │  每个 block:                                │
  │  ┌───────────────────────────────────────┐  │
  │  │ 4a. TriangleMulUpdate (outgoing)      │  │
  │  │     z[i,j] += Σ_k a[i,k]·b[j,k]      │  │
  │  │     环形距离 bias 在 attention 中      │  │
  │  │                                       │  │
  │  │ 4b. TriangleMulUpdate (incoming)      │  │
  │  │     z[i,j] += Σ_k a[j,k]·b[i,k]      │  │
  │  │                                       │  │
  │  │ 4c. TriangleAttention (starting node) │  │
  │  │     对每个 i, attend over k for j     │  │
  │  │     + CircularRelativeBias           │  │
  │  │                                       │  │
  │  │ 4d. TriangleAttention (ending node)   │  │
  │  │     对每个 j, attend over k for i     │  │
  │  │     + CircularRelativeBias           │  │
  │  │                                       │  │
  │  │ 4e. PairTransition                   │  │
  │  │     FFN(c_z → 4·c_z → c_z)          │  │
  │  └───────────────────────────────────────┘  │
  │                                             │
  │  循环拓扑约束: d_circ 贯穿所有三角形操作    │
  │  BSJ 配对: d_circ > L/2 的 pair 自然涌现   │
  └────────────────────┬────────────────────────┘
                       │  refined pair_repr
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
  ┌──────────┐  ┌───────────┐  ┌───────────────────┐
  │ 5a. Pair │  │ 5b. 结构  │  │ 5c. Multi-task     │
  │ 预测头   │  │ 预测头    │  │ 预测头             │
  │          │  │           │  │                    │
  │ P[i,j]=  │  │ Simple:   │  │ 通路分类 (7类)     │
  │ σ(w·z)  │  │ MDS初始化 │  │ RIG-I/TLR/PKR/... │
  │ 对称+BSJ │  │ + refine  │  │                    │
  │ 对称     │  │           │  │ 免疫原性 (binary) │
  │          │  │ Diffusion:│  │ σ(imm_head)       │
  │ 输出:    │  │ 去噪网络   │  │                    │
  │ pair     │  │ t→x_0     │  │ 免疫等级 (3类)    │
  │ 概率矩阵 │  │ +闭合约束  │  │ low/med/high      │
  │          │  │           │  │                    │
  │          │  │ 输出:     │  │ 输入:             │
  │          │  │ (B,L,3)   │  │ [seq_emb,         │
  │          │  │ 坐标+置信  │  │  pair_feat,       │
  │          │  │           │  │  bsj_strength]     │
  └──────────┘  └───────────┘  └───────────────────┘
        │              │              │
        ▼              ▼              ▼
  ┌──────────────────────────────────────────────┐
  │  6. 可视化 & 验证                              │
  │                                               │
  │  • BSJ 跨接配对热力图 (pair_probs > L/2 区域) │
  │  • ViennaRNA circ vs linear MFE 差异对比      │
  │  • RhoFold+ 3D 结构 (独立预测, FASTA 输入)     │
  │  • 通路分类混淆矩阵                            │
  │  • 免疫原性 ROC 曲线                          │
  └──────────────────────────────────────────────┘
```

## 文件清单

| 文件 | 作用 | 状态 |
|------|------|------|
| `core/tpe.py` | TPE + CircularRelativeBias | ✅ |
| `core/equivariant_backbone.py` | ESM2/RNA-FM 冻结 backbone + TPE | ✅ |
| `core/triangle_update.py` | CircPairformerStack (AF3-style) | ✅ |
| `core/diffusion_structure.py` | CircDiffusionStructure + SimpleHead | ✅ |
| `core/torusfold.py` | 主模型整合 (v2) | ✅ |
| `core/__init__.py` | 导出 | ✅ |
| `scripts/run_circrna_analysis.py` | **AutoDL 跑的全流程脚本** | ✅ |
| `tests/test_torusfold_v2.py` | 10 个测试 | ✅ 全通过 |
| `AUTODL_GUIDE.md` | AutoDL 部署指南 | ✅ |

## AutoDL 上跑什么

```bash
# 主命令: RNA-FM backbone + CircPairformer → 通路分类 + 免疫原性
python scripts/run_circrna_analysis.py \
    --backbone rna-fm --device cuda \
    --epochs 30 --batch-size 8 --c-z 64

# 可选: 同时跑 RhoFold+ 出 3D 结构
python scripts/run_circrna_analysis.py \
    --backbone rna-fm --device cuda --rhofold
```

## 核心创新 vs 现有方法

| | ESM2 | RNA-FM | AlphaFold3 | RhoFold+ | **TorusFold** |
|---|---|---|---|---|---|
| 拓扑 | 线性 | 线性 | 线性 | 线性 | **环形 (S¹)** |
| 位置编码 | Sinusoidal | Sinusoidal | Sinusoidal | Sinusoidal | **TPE (周期性)** |
| Pair 更新 | 无 | 无 | Evoformer | 无 | **CircPairformer** |
| BSJ 配对 | 不可能 | 不可能 | 不可能 | 不可能 | **自然涌现** |
| 闭合约束 | 无 | 无 | 无 | 无 | **x[0]≈x[-1]** |
| 目标 | 蛋白质 | RNA 嵌入 | 蛋白质 3D | RNA 3D | **circRNA 全栈** |
