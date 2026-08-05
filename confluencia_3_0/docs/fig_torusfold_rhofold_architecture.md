# TorusFold + RhoFold+ 完整架构图

## 1. 整体数据流

```mermaid
graph TB
    subgraph Input["输入"]
        SEQ["circRNA 序列<br/>(B, L) token IDs<br/>A=0 U=1 G=2 C=3 N=4"]
    end

    subgraph Backbone["RhoFold+ RNA FM Encoder (冻结)"]
        direction TB
        RNAFM["RNA FM Encoder<br/>12-layer Transformer<br/>d=640, 20 heads<br/>99.5M params"]
        NODE_R["node_repr<br/>(B, L, 640)<br/>碱基配对+保守性"]
        PAIR_R["pair_repr<br/>(B, L, L, 128)<br/>残基对关系"]
    end

    subgraph Projection["Projection Layer (可训练)"]
        NODE_P["node_proj<br/>Linear(640→256)<br/>+ LayerNorm + GELU"]
        PAIR_P["pair_proj<br/>Linear(128→64)<br/>+ LayerNorm + GELU"]
    end

    subgraph PairTrack["PairTrack (可训练, 2层)"]
        INIT_RNA["init_from_rna_fm_pair()<br/>从 RNA FM pair_repr 初始化<br/>稀疏 K=30 近邻"]
        TRIMUL["TriMulUpdate ×2<br/>三角一致性<br/>d(i,j) ≤ d(i,k)+d(k,j)"]
        Z["z: (B, L, 30, 64)<br/>稀疏 pair repr"]
    end

    subgraph Fusion["Pair → Node 融合"]
        P2N["pair_to_node<br/>Linear(64→256)×2<br/>+ LayerNorm"]
        FUSED["node_feat + pair_enhance<br/>(B, L, 256)"]
    end

    subgraph CGDecoder["CG Decoder (可训练)"]
        DECODER["MLP: 256→256→3<br/>+ LayerNorm + GELU<br/>+ Dropout(0.1)"]
    end

    subgraph Output["输出"]
        COORDS["CG 3-bead 坐标<br/>(B, L, 3)<br/>P atom positions"]
    end

    subgraph Loss["损失函数 (15项)"]
        L_COORD["coord_L2 ×10.0<br/>坐标 MSE"]
        L_BSJ["BSJ_FAPE ×2.0<br/>局部坐标系闭合"]
        L_BOND["bond ×0.5<br/>键长 5.9Å"]
        L_CLOSURE["closure ×5.0<br/>标量闭合距离"]
        L_STEREO["stereo ×1.0<br/>手性约束"]
        L_PHYS["physics ×1.0<br/>力场正则"]
        L_PAIR["pairing ×1.0<br/>ViennaRNA bpp"]
        L_CONTACT["contact ×1.0<br/>接触图"]
        L_TORUS["torus ×0.5<br/>环坐标 MSE"]
        L_CHIRAL["chirality ×0.5<br/>手性嵌入"]
        L_CONTRAST["contrastive ×0.1<br/>几何对比"]
        L_BRIDGE["bridge ×0.1<br/>键+对约束"]
        L_DISTILL["distillation ×0.1<br/>接触图蒸馏"]
        L_ANCHOR["anchor ×0.5<br/>动态锚点"]
        L_BSJ_GEO["bsj_geometry ×dynamic<br/>BSJ 角度+二面角"]
    end

    SEQ --> RNAFM
    RNAFM --> NODE_R & PAIR_R
    NODE_R --> NODE_P
    PAIR_R --> PAIR_P
    PAIR_P --> INIT_RNA
    INIT_RNA --> TRIMUL
    TRIMUL --> Z
    Z --> P2N
    NODE_P --> FUSED
    P2N --> FUSED
    FUSED --> DECODER
    DECODER --> COORDS
    COORDS --> L_COORD & L_BSJ & L_BOND & L_CLOSURE
```

## 2. 三种训练模式切换

```mermaid
graph LR
    subgraph ENV["环境变量切换"]
        E1["默认<br/>原版 S10"]
        E2["USE_PAIR_TRACK=1<br/>S10 + PairTrack"]
        E3["USE_RHOFOLD=1<br/>RhoFold+ backbone"]
        E4["TORUSFOLD_RESUME=...<br/>断点续跑"]
    end

    subgraph MODEL["模型"]
        M1["StrictlyEquivariantS10<br/>(SO(2)×SO(2)×R⁺)"]
        M2["S10 + PairTrack<br/>wrapper"]
        M3["TorusFoldRhoFold<br/>(RhoFold+ FM + CG Decoder)"]
    end

    E1 --> M1
    E2 --> M2
    E3 --> M3
    E4 -.->|恢复状态| M1 & M2 & M3
```

## 3. 训练课程 (5-Phase Curriculum)

```mermaid
graph TB
    subgraph Phase0["Phase 0: PDB 3D 预训练"]
        P0_DATA["PDB cyclized RNA<br/>5,607 files<br/>真实实验结构"]
        P0_GOAL["学 3D 空间感<br/>(螺旋/假结/茎环)"]
        P0_DETACH["detach_frac=0.0<br/>联合训练"]
    end

    subgraph Phase1["Phase 1: CG short-heavy"]
        P1_DATA["CG circRNA 151-200nt<br/>60% 短序列"]
        P1_GOAL["学环化拓扑<br/>BSJ 闭合"]
        P1_DETACH["detach_frac=1.0<br/>stop-grad"]
    end

    subgraph Phase2["Phase 2: medium"]
        P2_DATA["CG circRNA 201-500nt<br/>20-40% 中等"]
        P2_GOAL["逐步拉长<br/>远端配对"]
    end

    subgraph Phase3["Phase 3: long-dominant"]
        P3_DATA["CG circRNA 501-1000nt<br/>40-50% 长序列"]
        P3_GOAL["主力训练<br/>复杂折叠"]
    end

    subgraph Phase4["Phase 4: xlong + BSJ stress"]
        P4_DATA["CG circRNA 1001-5000nt<br/>20-30% 超长"]
        P4_GOAL["BSJ 压力测试<br/>大规模闭合"]
    end

    P0_DATA --> P0_GOAL
    P1_DATA --> P1_GOAL
    Phase0 -.->|val_rmsd < target| Phase1
    Phase1 -.->|收敛| Phase2
    Phase2 -.->|收敛| Phase3
    Phase3 -.->|收敛| Phase4
```

## 4. PairTrack 内部结构

```mermaid
graph TB
    subgraph Input["输入"]
        Z_IN["z: (B, L, K, 64)<br/>稀疏 pair repr"]
    end

    subgraph Layer1["PairTrackLayer ×2"]
        TMU["TriMulUpdate<br/>Starting Edge"]
        TMU2["TriMulUpdate<br/>Ending Edge"]
        FFN["FFN<br/>d→d_ffn→d"]
        NORM["LayerNorm"]
    end

    subgraph Output["输出"]
        Z_OUT["z': (B, L, K, 64)<br/>三角一致性增强"]
    end

    Z_IN --> TMU --> TMU2 --> FFN --> NORM --> Z_OUT
```

## 5. BSJ FAPE 损失

```mermaid
graph TB
    subgraph Input["输入"]
        PRED["pred_coords<br/>(B, L, 3)"]
        TARGET["target_coords<br/>(B, L, 3)"]
    end

    subgraph LocalFrame["局部坐标系构建"]
        ORIGIN["原点: P(L-1)<br/>BSJ junction 位置"]
        ZAXIS["z-axis: P(L-1) → P(0)<br/>闭合方向"]
        XAXIS["x-axis: Gram-Schmidt<br/>正交化"]
    end

    subgraph Transform["坐标变换"]
        TO_LOCAL["to_local_frame()<br/>R^T @ (coords - origin)"]
        REGION["BSJ 区域<br/>[L-10, ..., L-1, 0, ..., 10]<br/>20 residues"]
    end

    subgraph Loss["损失计算"]
        L2["L2(pred_local, target_local)<br/>+ clamp(max=50)"]
    end

    PRED --> TO_LOCAL
    TARGET --> TO_LOCAL
    ORIGIN --> TO_LOCAL
    ZAXIS --> TO_LOCAL
    REGION --> L2
    TO_LOCAL --> L2
    L2 --> BSJ_FAPE["BSJ FAPE Loss"]
    L2 --> BSJ_CONF["BSJ Confidence<br/>exp(-fape/5.0)"]
```

## 6. 关键参数一览

| 组件 | 参数量 | 状态 | 说明 |
|------|--------|------|------|
| RNA FM Encoder | 99,521,546 | 冻结 9/12 层 | 碱基配对+保守性 |
| PairTrack | 132,864 | 可训练 | 三角一致性×2层 |
| CG Decoder | 67,075 | 可训练 | MLP 256→256→3 |
| Projections | ~200,000 | 可训练 | 640→256, 128→64 |
| PairToNode | ~130,000 | 可训练 | 64→256→256 |
| **总计** | **99,994,253** | **25.5% 可训练** | |

## 7. 训练数据

| 数据集 | 数量 | 大小 | 来源 |
|--------|------|------|------|
| 主训练 | 82,106 | 523MB | circrna_3d_all_consolidated.npz |
| Pair probs | — | 20.3GB | ViennaRNA bpp |
| PDB cyclized | 5,607 | — | Phase 0 预训练 |

### 长度分布
```
151-200nt:    6,962  (8.5%)   ← Phase 1
201-500nt:   38,104  (46.4%)  ← Phase 2
501-1000nt:  32,291  (39.3%)  ← Phase 3
1001-5000nt:  4,749  (5.8%)   ← Phase 4
```

## 8. 文件索引

| 文件 | 行数 | 功能 |
|------|------|------|
| `rhofold_backbone.py` | ~200 | RhoFold+ RNA FM 集成 |
| `pair_track.py` | ~350 | 稀疏三角一致性 |
| `cg_decoder.py` | ~100 | P atom → 3-bead |
| `bsj_fape.py` | ~250 | 局部坐标系损失 |
| `torusfold_rhofold.py` | ~250 | 完整模型组装 |
| `train_s10_curriculum.py` | ~1300 | 主训练脚本 (15 loss, 5 phase) |
| `train_rhofold_cg.py` | ~300 | 独立训练 (简化版) |
| `test_2oiu_rhofold.py` | ~150 | 2OIU 验证 |

## 9. 使用方式

```bash
# 原版 S10
python train_s10_curriculum.py --labels ./data/circrna_3d_all_consolidated --device cuda

# S10 + PairTrack
USE_PAIR_TRACK=1 python train_s10_curriculum.py --labels ./data/circrna_3d_all_consolidated --device cuda

# RhoFold+ backbone
USE_RHOFOLD=1 python train_s10_curriculum.py --labels ./data/circrna_3d_all_consolidated --device cuda

# 断点续跑
TORUSFOLD_RESUME=models/s10_curriculum/phase0_end_full.pt \
  python train_s10_curriculum.py --labels ./data/circrna_3d_all_consolidated --device cuda
```
