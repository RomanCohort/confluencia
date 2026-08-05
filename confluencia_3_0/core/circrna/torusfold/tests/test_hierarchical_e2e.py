"""test_hierarchical_e2e.py — 端到端测试: 真实 circRNA 序列 -> 层次化预测.

测试内容:
  1. 加载真实 circRNA 序列 (hsa_circ_0000002, 251nt)
  2. 模拟 MSA (用单序列 + 变异)
  3. 运行 MSA 聚类
  4. 运行 chunk 切分
  5. 运行 overlap 一致性损失
  6. 运行特征提取
  7. 运行全链融合
"""
import sys
sys.path.insert(0, r'C:\Users\颜子壹\deploy\IGEM集成方案\confluencia_3_0\core\circrna\torusfold')

import torch
import numpy as np

# ====== 配置 ======
DEVICE = 'cpu'
CHUNK_SIZE = 200  # 小一点, 确保能跑
OVERLAP = 30

print("=" * 60)
print("  Hierarchical Predictor — End-to-End Test")
print("=" * 60)

# ====== Step 1: 加载真实序列 ======
print("\n[Step 1] Loading real circRNA sequence...")

data = np.load(
    r'C:\Users\颜子壹\deploy\IGEM集成方案\data\circrna_3d_all_consolidated.npz',
    allow_pickle=True,
)
# 选 hsa_circ_0000002 (251nt)
idx = 0
seq_id = str(data['ids'][idx])
length = int(data['lengths'][idx])
coords_gt = data['coords'][idx, :length]  # (L, 3)

print(f"  Sample: {seq_id}")
print(f"  Length: {length}nt")
print(f"  GT coords range: [{coords_gt.min():.1f}, {coords_gt.max():.1f}]A")

# 生成随机 token 序列 (模拟输入)
seq_tokens = torch.randint(0, 4, (length,), device=DEVICE)
print(f"  seq_tokens: {seq_tokens.shape}")

# ====== Step 2: 模拟 MSA ======
print("\n[Step 2] Simulating MSA...")

N_MSA = 20  # 模拟 20 条 MSA 序列
msa_tokens = seq_tokens.unsqueeze(0).expand(N_MSA, -1).clone()

# 加一些随机变异 (5% mutation rate)
for i in range(1, N_MSA):
    mutation_mask = torch.rand(length) < 0.05
    random_bases = torch.randint(0, 4, (length,))
    msa_tokens[i][mutation_mask] = random_bases[mutation_mask]

print(f"  MSA shape: {msa_tokens.shape} ({N_MSA} seqs, {length}nt)")
print(f"  Mutation rate: ~5%")

# ====== Step 3: MSA 聚类 ======
print("\n[Step 3] MSA Clustering...")

from msa_cluster import MSACluster
cluster = MSACluster(n_representatives=8, method='embedding')

rep_ids, rep_seqs, weights = cluster(msa_tokens)
print(f"  Representatives: {rep_ids.shape[0]}")
print(f"  Rep IDs: {rep_ids.tolist()}")
print(f"  Weights: {weights.tolist()}")
print(f"  Weights sum: {weights.sum():.1f}")

# ====== Step 4: Chunk 切分 ======
print("\n[Step 4] Chunk Splitting...")

from chunk_splitter import ChunkSplitter
splitter = ChunkSplitter(chunk_size=CHUNK_SIZE, overlap=OVERLAP)

chunks = splitter.split(seq_tokens, is_circular=True, bsj_pos=0)
print(f"  Chunks: {len(chunks)}")
for c in chunks:
    bsj = f"BSJ@{c.bsj_local_pos}" if c.is_circular_chunk else "no BSJ"
    print(f"    Chunk {c.chunk_id}: [{c.start}:{c.end}] len={c.end-c.start} {bsj}")

# ====== Step 5: Overlap 一致性损失 ======
print("\n[Step 5] Overlap Consistency Loss...")

from overlap_loss import OverlapConsistencyLoss
from chunk_splitter import ChunkInfo

criterion = OverlapConsistencyLoss()

# 模拟 chunk 坐标 (随机)
chunk_coords = []
for c in chunks:
    L_chunk = c.end - c.start
    coord = torch.randn(L_chunk, 3, requires_grad=True)
    chunk_coords.append(coord)

losses = criterion(chunk_coords, chunks, total_length=length)
print(f"  Total loss: {losses['total'].item():.4f}")
for k, v in losses.items():
    if k != 'total':
        print(f"    {k}: {v.item():.4f}")

# 验证梯度
losses['total'].backward()
print(f"  Gradient flows: {chunk_coords[0].grad is not None}")
print(f"  Grad norm: {chunk_coords[0].grad.norm().item():.4f}")

# ====== Step 6: 特征提取 ======
print("\n[Step 6] Feature Extraction...")

from chunk_predictor import ChunkFeatureExtractor, ChunkPrediction

extractor = ChunkFeatureExtractor(d_feature=256)

features = []
for i, c in enumerate(chunks):
    L_chunk = c.end - c.start
    pred = ChunkPrediction(
        coords=chunk_coords[i].detach(),
        node_repr=torch.randn(L_chunk, 256),
        contact_map=torch.randn(L_chunk, L_chunk).abs(),
        bsj_confidence=0.85,
        chunk_id=c.chunk_id,
        start=c.start,
        end=c.end,
    )
    feat = extractor(pred)
    features.append(feat)
    print(f"  Chunk {c.chunk_id}: feature shape = {feat.shape}")

features_tensor = torch.stack(features)
print(f"  All features: {features_tensor.shape}")

# ====== Step 7: 全链融合 ======
print("\n[Step 7] Chunk Fusion GNN...")

from chunk_fusion import ChunkFusionGNN

fusion = ChunkFusionGNN(d_chunk=256, n_heads=8, n_layers=2)

global_coords = fusion(features_tensor, chunk_coords, chunks, total_length=length)
print(f"  Global coords: {global_coords.shape}")
print(f"  Coord range: [{global_coords.min():.1f}, {global_coords.max():.1f}]")

# ====== Step 8: Stitch 拼接 ======
print("\n[Step 8] Stitch Chunks...")

stitch_coords = ChunkSplitter.stitch_chunks(
    [c.detach() for c in chunk_coords],
    chunks,
    total_length=length,
)
print(f"  Stitched coords: {stitch_coords.shape}")

# ====== Step 9: 评估 ======
print("\n[Step 9] Evaluation...")

# BSJ 闭合距离
closure_dist = torch.norm(global_coords[0] - global_coords[-1]).item()
print(f"  BSJ closure distance: {closure_dist:.2f}A")
print(f"  Target bond length: 5.9A")
print(f"  Closure error: {abs(closure_dist - 5.9):.2f}A")

# 坐标范围
print(f"  X range: [{global_coords[:,0].min():.1f}, {global_coords[:,0].max():.1f}]")
print(f"  Y range: [{global_coords[:,1].min():.1f}, {global_coords[:,1].max():.1f}]")
print(f"  Z range: [{global_coords[:,2].min():.1f}, {global_coords[:,2].max():.1f}]")

# ====== 完成 ======
print("\n" + "=" * 60)
print("  All steps completed! OK")
print("=" * 60)

print("\nSummary:")
print(f"  Input: {seq_id} ({length}nt)")
print(f"  MSA: {N_MSA} seqs -> {rep_ids.shape[0]} representatives")
print(f"  Chunks: {len(chunks)} (size={CHUNK_SIZE}, overlap={OVERLAP})")
print(f"  Output: {global_coords.shape}")
print(f"  BSJ closure: {closure_dist:.2f}A")
print(f"  Overlap loss: {losses['total'].item():.4f}")
