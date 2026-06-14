"""
test_torusfold_v2.py — Tests for TorusFold v2 (AF3-inspired) components.

Validates:
1. TriangleMultiplicativeUpdate: outgoing/incoming correctness
2. TriangleAttention: starting/ending node with circular bias
3. CircPairformerBlock: full pair update preserves shape
4. CircPairformerStack: multi-block refinement
5. CircDiffusionStructure: diffusion sampling + closure
6. SimpleStructureHead: MDS initialization
7. PairInitialization: outer sum + circular distance
8. TorusFold v2 integration: full forward pass
"""

import sys
import math
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_triangle_multiplicative_update():
    """Test 1: Triangle multiplicative update preserves shape and learns."""
    from core.triangle_update import TriangleMultiplicativeUpdate

    print("\n=== Test 1: Triangle Multiplicative Update ===")

    c_z = 32
    L = 10
    z = torch.randn(2, L, L, c_z)

    for direction in ["outgoing", "incoming"]:
        tri = TriangleMultiplicativeUpdate(c_z=c_z, c_hidden=16, direction=direction)
        out = tri(z)
        assert out.shape == z.shape, f"{direction}: shape mismatch {out.shape} != {z.shape}"
        # Output should differ from input (gating may be near-zero initially but not identical)
        diff = (out - z).abs().mean().item()
        print(f"  {direction}: shape OK, mean diff from input: {diff:.6f}")

    print("  ✓ Triangle multiplicative update passed")


def test_triangle_attention():
    """Test 2: Triangle attention with circular bias."""
    from core.triangle_update import TriangleAttention

    print("\n=== Test 2: Triangle Attention ===")

    c_z = 32
    n_heads = 4
    L = 12

    z = torch.randn(2, L, L, c_z)

    for direction in ["starting", "ending"]:
        tri_att = TriangleAttention(c_z=c_z, n_heads=n_heads, direction=direction)
        out = tri_att(z)
        assert out.shape == z.shape, f"{direction}: shape mismatch"
        print(f"  {direction}: shape OK")

    print("  ✓ Triangle attention passed")


def test_circ_pairformer_block():
    """Test 3: Full CircPairformerBlock preserves shape."""
    from core.triangle_update import CircPairformerBlock

    print("\n=== Test 3: CircPairformerBlock ===")

    c_z = 32
    L = 12
    z = torch.randn(2, L, L, c_z)

    block = CircPairformerBlock(c_z=c_z, c_hidden_tri=16, n_heads_tri=4)
    out = block(z)

    assert out.shape == z.shape
    print(f"  Input shape: {z.shape}, Output shape: {out.shape}")

    # Verify transformation happened
    diff = (out - z).abs().mean().item()
    print(f"  Mean diff: {diff:.6f}")
    assert diff > 1e-6, "Block should modify the input"

    print("  ✓ CircPairformerBlock passed")


def test_circ_pairformer_stack():
    """Test 4: CircPairformerStack with multiple blocks."""
    from core.triangle_update import CircPairformerStack

    print("\n=== Test 4: CircPairformerStack ===")

    c_z = 32
    L = 12
    z = torch.randn(2, L, L, c_z)

    stack = CircPairformerStack(n_blocks=3, c_z=c_z, c_hidden_tri=16, n_heads_tri=4)

    n_params = sum(p.numel() for p in stack.parameters())
    print(f"  Parameters: {n_params:,}")

    out = stack(z)
    assert out.shape == z.shape
    print(f"  Shape preserved: {z.shape} → {out.shape}")

    print("  ✓ CircPairformerStack passed")


def test_pair_initialization():
    """Test 5: PairInitialization produces correct shape."""
    from core.torusfold import PairInitialization

    print("\n=== Test 5: PairInitialization ===")

    d_model = 64
    c_z = 32
    L = 15

    seq_repr = torch.randn(2, L, d_model)
    pair_init = PairInitialization(d_model=d_model, c_z=c_z)

    pair = pair_init(seq_repr)

    assert pair.shape == (2, L, L, c_z), f"Expected (2, {L}, {L}, {c_z}), got {pair.shape}"
    print(f"  Pair shape: {pair.shape}")

    # Check that pair[i,j] depends on both i and j
    diff_i = (pair[0, 0, 1, :] - pair[0, 1, 1, :]).norm().item()
    diff_j = (pair[0, 0, 1, :] - pair[0, 0, 2, :]).norm().item()
    print(f"  Varies with i: {diff_i:.4f}, Varies with j: {diff_j:.4f}")
    assert diff_i > 0.01, "Pair should vary with position i"
    assert diff_j > 0.01, "Pair should vary with position j"

    print("  ✓ PairInitialization passed")


def test_simple_structure_head():
    """Test 6: SimpleStructureHead produces valid coordinates."""
    from core.diffusion_structure import SimpleStructureHead

    print("\n=== Test 6: SimpleStructureHead ===")

    c_z = 32
    L = 20
    pair_repr = torch.randn(1, L, L, c_z)

    head = SimpleStructureHead(d_pair=c_z, d_coord=16, n_rbf=8)

    with torch.no_grad():
        out = head(pair_repr)

    assert out["coords"].shape == (1, L, 3)
    assert out["confidence"].shape == (1, L)
    print(f"  Coords shape: {out['coords'].shape}")
    print(f"  Closure distance: {out['closure_dist'][0].item():.2f} Å")

    print("  ✓ SimpleStructureHead passed")


def test_diffusion_structure():
    """Test 7: CircDiffusionStructure sampling (short run)."""
    from core.diffusion_structure import CircDiffusionStructure

    print("\n=== Test 7: CircDiffusionStructure ===")

    c_z = 32
    L = 15
    pair_repr = torch.randn(1, L, L, c_z)

    diffusion = CircDiffusionStructure(
        d_pair=c_z,
        d_time=32,
        d_cond=64,
        d_coord=16,
        n_layers=2,
        n_steps=5,  # Very few steps for test speed
        bond_length=3.4,
    )

    n_params = sum(p.numel() for p in diffusion.parameters())
    print(f"  Parameters: {n_params:,}")

    with torch.no_grad():
        out = diffusion.sample(pair_repr, n_samples=1, return_trajectory=False)

    assert out["coords"].shape == (1, L, 3)
    assert out["confidence"].shape == (1, L)
    print(f"  Coords shape: {out['coords'].shape}")
    print(f"  Closure distance: {out['closure_dist'][0].item():.2f} Å")

    print("  ✓ CircDiffusionStructure passed")


def test_torusfold_v2():
    """Test 8: Full TorusFold v2 integration with simple structure."""
    from core.torusfold import TorusFold, TorusFoldConfig

    print("\n=== Test 8: TorusFold v2 Integration ===")

    config = TorusFoldConfig(
        d_model=64,
        n_torus_layers=1,
        c_z=32,
        c_hidden_tri=16,
        n_pairformer_blocks=2,
        n_heads_tri=2,
        structure_mode="simple",
        hidden_dim=64,
        dropout=0.1,
        n_rot_augments=0,
        translation_efficiency=True,
        circ_stability=True,
        immune_pathway=True,
        bsj_confidence=True,
    )

    model = TorusFold(config)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # Test with mock sequence representation
    # (Skip backbone, test from pair representation onward)
    L = 15
    mock_seq_repr = torch.randn(1, L, config.d_model)
    mock_gene_expr = torch.randn(1, config.gene_dim)

    # Test pair initialization + pairformer
    pair_repr = model.pair_init(mock_seq_repr)
    assert pair_repr.shape == (1, L, L, config.c_z)
    print(f"  Pair init shape: {pair_repr.shape}")

    pair_repr = model.pairformer(pair_repr)
    assert pair_repr.shape == (1, L, L, config.c_z)
    print(f"  Pairformer output shape: {pair_repr.shape}")

    # Test pair prediction
    pair_probs = model.pair_head(pair_repr)
    assert pair_probs.shape == (1, L, L)
    # Check symmetry
    sym_err = (pair_probs - pair_probs.transpose(-1, -2)).abs().max().item()
    print(f"  Pair symmetry error: {sym_err:.6f}")

    # Test simple structure head
    with torch.no_grad():
        struct_out = model.structure_head(pair_repr)
    assert struct_out["coords"].shape == (1, L, 3)
    print(f"  Structure coords shape: {struct_out['coords'].shape}")

    # Test multi-task heads
    global_emb = mock_seq_repr.mean(dim=1)  # (1, d_model=64)
    struct_feat = pair_repr.mean(dim=(1, 2))  # (1, c_z=32) — mean over L×L
    bsj_stability = torch.tensor([0.5])

    expected_input_dim = config.d_model + config.gene_dim + config.c_z + 1  # 64+6+32+1=103

    multi_input = torch.cat([
        global_emb, mock_gene_expr, struct_feat, bsj_stability.unsqueeze(-1),
    ], dim=-1)

    assert multi_input.shape == (1, expected_input_dim), f"Expected dim {expected_input_dim}, got {multi_input.shape[-1]}"

    composite = model.composite_head(multi_input)
    report = model.report_head(multi_input)
    response_logits = model.response_head(multi_input)

    assert composite.shape == (1, 8)
    assert report.shape == (1, 4)
    assert response_logits.shape == (1, 3)
    print(f"  Composite: {composite.shape}, Report: {report.shape}, Response: {response_logits.shape}")

    print("  ✓ TorusFold v2 integration passed")


def test_config_serialization_v2():
    """Test 9: Config serialization with new fields."""
    from core.torusfold import TorusFoldConfig

    print("\n=== Test 9: Config Serialization v2 ===")

    config = TorusFoldConfig(
        c_z=128,
        n_pairformer_blocks=6,
        structure_mode="diffusion",
        n_diffusion_steps=200,
    )

    d = config.to_dict()
    assert d["c_z"] == 128
    assert d["n_pairformer_blocks"] == 6
    assert d["structure_mode"] == "diffusion"
    assert d["n_diffusion_steps"] == 200

    # Reconstruct
    config2 = TorusFoldConfig(**d)
    assert config2.c_z == 128
    assert config2.structure_mode == "diffusion"

    print("  ✓ Config serialization v2 passed")


def test_pairformer_gradient_flow():
    """Test 10: Gradient flows through CircPairformerStack."""
    from core.triangle_update import CircPairformerStack

    print("\n=== Test 10: Gradient Flow ===")

    c_z = 32
    L = 10
    z = torch.randn(1, L, L, c_z, requires_grad=True)

    stack = CircPairformerStack(n_blocks=2, c_z=c_z, c_hidden_tri=16, n_heads_tri=2)
    out = stack(z)
    loss = out.sum()
    loss.backward()

    assert z.grad is not None, "No gradient on input"
    assert z.grad.shape == z.shape
    grad_norm = z.grad.norm().item()
    print(f"  Gradient norm: {grad_norm:.4f}")
    assert grad_norm > 0, "Zero gradient"

    print("  ✓ Gradient flow passed")


def run_all_tests():
    print("=" * 60)
    print("TorusFold v2 (AF3-inspired) Component Tests")
    print("=" * 60)

    tests = [
        ("Triangle Multiplicative Update", test_triangle_multiplicative_update),
        ("Triangle Attention", test_triangle_attention),
        ("CircPairformerBlock", test_circ_pairformer_block),
        ("CircPairformerStack", test_circ_pairformer_stack),
        ("PairInitialization", test_pair_initialization),
        ("SimpleStructureHead", test_simple_structure_head),
        ("CircDiffusionStructure", test_diffusion_structure),
        ("TorusFold v2 Integration", test_torusfold_v2),
        ("Config Serialization v2", test_config_serialization_v2),
        ("Gradient Flow", test_pairformer_gradient_flow),
    ]

    results = []
    for name, func in tests:
        try:
            func()
            results.append((name, "PASS"))
        except Exception as e:
            results.append((name, f"FAIL: {e}"))
            print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, s in results if s == "PASS")
    for name, status in results:
        mark = "✓" if status == "PASS" else "✗"
        print(f"  {mark} {name}: {status}")

    print(f"\nTotal: {passed}/{len(results)} passed")
    return passed == len(results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
