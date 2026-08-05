"""smoke_hierarchical.py — 层次化预测 smoke test."""
import sys
sys.path.insert(0, r'C:\Users\颜子壹\deploy\IGEM集成方案\confluencia_3_0\core\circrna\torusfold')

import torch

def test_msa_cluster():
    from msa_cluster import MSACluster
    cluster = MSACluster(n_representatives=8, method='embedding')
    msa = torch.randint(0, 4, (20, 100))
    rep_ids, rep_seqs, weights = cluster(msa)
    assert rep_ids.shape[0] <= 8
    assert rep_seqs.shape == (rep_ids.shape[0], 100)
    print(f"  MSA Cluster: {msa.shape[0]} seqs -> {rep_ids.shape[0]} reps, weights sum={weights.sum():.1f}")
    print("  OK")

def test_chunk_splitter():
    from chunk_splitter import ChunkSplitter
    splitter = ChunkSplitter(chunk_size=200, overlap=20)
    seq = torch.randint(0, 4, (600,))
    chunks = splitter.split(seq, is_circular=False)
    assert len(chunks) >= 3
    print(f"  Linear 600nt -> {len(chunks)} chunks")
    for c in chunks:
        print(f"    Chunk {c.chunk_id}: [{c.start}:{c.end}] len={c.end-c.start}")
    seq_c = torch.randint(0, 4, (600,))
    chunks_c = splitter.split(seq_c, is_circular=True, bsj_pos=300)
    print(f"  Circular 600nt (BSJ@300) -> {len(chunks_c)} chunks")
    print("  OK")

def test_overlap_loss():
    from overlap_loss import OverlapConsistencyLoss
    from chunk_splitter import ChunkInfo
    criterion = OverlapConsistencyLoss()
    chunk1 = torch.randn(200, 3, requires_grad=True)
    chunk2 = torch.randn(200, 3, requires_grad=True)
    chunk2.data[:20] = chunk1.data[180:200] + 0.01 * torch.randn(20, 3)
    info1 = ChunkInfo(chunk_id=0, start=0, end=200, seq_tokens=torch.randint(0,4,(200,)))
    info2 = ChunkInfo(chunk_id=1, start=180, end=380, seq_tokens=torch.randint(0,4,(200,)))
    losses = criterion([chunk1, chunk2], [info1, info2], total_length=380)
    assert 'total' in losses
    assert losses['total'].requires_grad
    print(f"  Overlap Loss: total={losses['total'].item():.4f}")
    for k, v in losses.items():
        if k != 'total':
            print(f"    {k}: {v.item():.4f}")
    print("  OK")

def test_feature_extractor():
    from chunk_predictor import ChunkFeatureExtractor, ChunkPrediction
    extractor = ChunkFeatureExtractor(d_feature=256)
    pred = ChunkPrediction(
        coords=torch.randn(200, 3),
        node_repr=torch.randn(200, 256),
        contact_map=torch.randn(200, 200).abs(),
        bsj_confidence=0.85,
        chunk_id=0, start=0, end=200,
    )
    feature = extractor(pred)
    assert feature.shape == (256,), f"Expected (256,), got {feature.shape}"
    print(f"  Feature Extractor: -> {feature.shape}")
    print("  OK")

def test_fusion_gnn():
    from chunk_fusion import ChunkFusionGNN
    from chunk_splitter import ChunkInfo
    fusion = ChunkFusionGNN(d_chunk=256, n_heads=8, n_layers=2)
    chunk_features = torch.randn(3, 256)
    chunk_coords = [torch.randn(200, 3), torch.randn(200, 3), torch.randn(140, 3)]
    chunk_infos = [
        ChunkInfo(chunk_id=0, start=0, end=200, seq_tokens=torch.randint(0,4,(200,))),
        ChunkInfo(chunk_id=1, start=180, end=380, seq_tokens=torch.randint(0,4,(200,))),
        ChunkInfo(chunk_id=2, start=360, end=500, seq_tokens=torch.randint(0,4,(140,))),
    ]
    global_coords = fusion(chunk_features, chunk_coords, chunk_infos, total_length=500)
    assert global_coords.shape == (500, 3)
    print(f"  Chunk Fusion: 3 chunks -> {global_coords.shape}")
    print("  OK")

def test_stitch():
    from chunk_splitter import ChunkSplitter, ChunkInfo
    chunks_coords = [torch.randn(200, 3), torch.randn(200, 3), torch.randn(140, 3)]
    chunk_infos = [
        ChunkInfo(chunk_id=0, start=0, end=200),
        ChunkInfo(chunk_id=1, start=180, end=380),
        ChunkInfo(chunk_id=2, start=360, end=500),
    ]
    full_coords = ChunkSplitter.stitch_chunks(chunks_coords, chunk_infos, total_length=500)
    assert full_coords.shape == (500, 3)
    print(f"  Stitch: 3 chunks -> {full_coords.shape}")
    print("  OK")

if __name__ == '__main__':
    print("=== Hierarchical Predictor Smoke Tests ===\n")
    test_msa_cluster()
    test_chunk_splitter()
    test_overlap_loss()
    test_feature_extractor()
    test_fusion_gnn()
    test_stitch()
    print("\n=== All tests passed! ===")
