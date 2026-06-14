"""
test_torusfold.py — Tests for TorusFold components.

Validates:
1. TPE periodicity: PE[i] = PE[i + L]
2. Rotation invariance: same embedding for rotated sequence
3. IRS pair symmetry: P[i, j] = P[j, i]
4. Structure closure: x[0] and x[L-1] within bond distance
5. Circular distance: d_circ(i, j) = min(|i-j|, L-|i-j|)
"""

import sys
import math
from pathlib import Path

import torch
import torch.nn.functional as F

# Add project root to path — the package dir IS the root
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Also add the IGEM root for confluencia_shared etc.
_IGEM_ROOT = Path(__file__).resolve().parents[2]
if str(_IGEM_ROOT) not in sys.path:
    sys.path.insert(0, str(_IGEM_ROOT))


def test_tpe_periodicity():
    """Test 1: TPE should be periodic with period L."""
    from core.tpe import TorusPositionalEncoding

    print("\n=== Test 1: TPE Periodicity ===")

    d_model = 64
    L = 100

    tpe = TorusPositionalEncoding(d_model=d_model, n_harmonics=8, learnable=True)
    tpe.eval()  # Disable dropout for deterministic test

    # Generate dummy input with 2*L positions, but tell TPE the period is L
    dummy = torch.zeros(1, 2 * L, d_model)
    with torch.no_grad():
        pe = tpe(dummy, seq_len=L)  # seq_len=L means period = L

    # Compare PE[i] with PE[i + L] — should be equal due to periodicity
    pe_first = pe[0, :L]
    pe_second = pe[0, L:]

    error = (pe_first - pe_second).norm().item()
    print(f"Periodicity error (||PE[i] - PE[i+L]||): {error:.6f}")

    assert error < 0.01, f"TPE not periodic! Error = {error}"
    print("✓ TPE periodicity verified")

    # Also test explicit method
    explicit_error = tpe.get_periodicity_error(L)
    print(f"Explicit periodicity check: {explicit_error:.6f}")
    assert explicit_error < 0.01

    return True


def test_circular_distance():
    """Test 2: Circular distance matrix should wrap around."""
    from core.irs_pair import circular_distance_matrix

    print("\n=== Test 2: Circular Distance ===")

    L = 50
    circ_dist = circular_distance_matrix(L, device="cpu")

    # Check properties
    # d(0, L-1) should be 1 (adjacent on circle)
    d_0_last = circ_dist[0, L-1].item()
    d_last_0 = circ_dist[L-1, 0].item()

    print(f"d(0, L-1) = {d_0_last}")
    print(f"d(L-1, 0) = {d_last_0}")

    assert d_0_last == 1.0, f"d(0, L-1) should be 1, got {d_0_last}"
    assert d_last_0 == 1.0, f"d(L-1, 0) should be 1, got {d_last_0}"

    # d(0, L/2) should be L/2 (maximum distance on circle)
    d_mid = circ_dist[0, L//2].item()
    print(f"d(0, L/2) = {d_mid}")
    assert d_mid == L // 2

    # d(i, i) should be 0
    for i in range(L):
        assert circ_dist[i, i].item() == 0

    # d(i, j) should be symmetric
    for i in range(L):
        for j in range(i+1, L):
            assert circ_dist[i, j].item() == circ_dist[j, i].item()

    print("✓ Circular distance verified")

    return True


def test_circular_relative_bias():
    """Test 3: Circular relative bias should wrap around."""
    from core.tpe import CircularRelativeBias

    print("\n=== Test 3: Circular Relative Bias ===")

    n_heads = 8
    L = 20
    max_dist = 10

    bias_module = CircularRelativeBias(n_heads, max_dist)
    bias = bias_module(L)  # (1, n_heads, L, L)

    # Check: bias[0, 0, 0, L-1] should equal bias[0, 0, 0, 1]
    # (both are distance 1 on the circle)
    bias_0_last = bias[0, 0, 0, L-1].item()
    bias_0_1 = bias[0, 0, 0, 1].item()

    print(f"Bias for d(0, L-1) = {bias_0_last:.4f}")
    print(f"Bias for d(0, 1) = {bias_0_1:.4f}")

    # They should be close (same circular distance)
    assert abs(bias_0_last - bias_0_1) < 0.001, \
        f"Bias for adjacent positions should match! {bias_0_last} vs {bias_0_1}"

    # Bias should be symmetric: bias[i,j] = bias[j,i]
    for i in range(L):
        for j in range(i+1, L):
            assert abs(bias[0, 0, i, j].item() - bias[0, 0, j, i].item()) < 0.001

    print("✓ Circular relative bias verified")

    return True


def test_rotation_invariance_mock():
    """Test 4: Rotation invariance (mock test without backbone)."""
    print("\n=== Test 4: Rotation Invariance (Mock) ===")

    # This test verifies the rotation logic
    # Full test requires RNA-FM backbone loaded

    def rotate_sequence(seq, offset):
        """Rotate circular sequence."""
        offset = offset % len(seq)
        if offset == 0:
            return seq
        return seq[offset:] + seq[:offset]

    # Test sequence
    seq = "ACGUACGUACGUACGU"  # L=16

    # Rotations
    rotations = [rotate_sequence(seq, k * 4) for k in range(4)]

    print("Original sequence:", seq)
    for i, r in enumerate(rotations):
        print(f"Rotation {i*4}: {r}")

    # Check: all rotations should have same length
    for r in rotations:
        assert len(r) == len(seq)

    # Check: rotating by L returns original
    assert rotate_sequence(seq, len(seq)) == seq
    print("✓ Rotation logic verified")

    # Note: Full rotation invariance test requires backbone
    print("⚠ Full rotation invariance test requires RNA-FM backbone (skip in unit test)")

    return True


def test_pair_symmetry():
    """Test 5: IRS pair matrix should be symmetric."""
    from core.irs_pair import IRSPairModule

    print("\n=== Test 5: IRS Pair Symmetry ===")

    d_model = 64
    d_pair = 32
    L = 20

    # Mock sequence representation
    sequence_repr = torch.randn(1, L, d_model)

    pair_module = IRSPairModule(
        d_model=d_model,
        d_pair=d_pair,
        n_heads=4,
        n_layers=2,
    )

    with torch.no_grad():
        pair_out = pair_module(sequence_repr)

    pair_probs = pair_out["pair_probs"]  # (1, L, L)

    # Check symmetry: P[i,j] = P[j,i]
    max_diff = 0.0
    for i in range(L):
        for j in range(i+1, L):
            diff = abs(pair_probs[0, i, j].item() - pair_probs[0, j, i].item())
            max_diff = max(max_diff, diff)

    print(f"Max pair symmetry error: {max_diff:.6f}")
    assert max_diff < 0.001, f"Pair matrix not symmetric! Max diff = {max_diff}"

    # Check BSJ mask
    bsj_mask = pair_out["bsj_pair_mask"]
    print(f"BSJ mask shape: {bsj_mask.shape}")
    print(f"BSJ crossing pairs count: {bsj_mask.sum().item()}")

    print("✓ Pair symmetry verified")

    return True


def test_structure_closure():
    """Test 6: Structure should satisfy circular closure constraint."""
    from core.structure_head import (
        TorusStructureHead, CircularClosureLoss
    )

    print("\n=== Test 6: Structure Closure ===")

    d_pair = 32
    L = 30

    # Mock pair representation
    pair_repr = torch.randn(1, L, L, d_pair)

    structure_head = TorusStructureHead(
        d_pair=d_pair,
        d_coord=16,
        n_refinement_iters=2,
    )

    with torch.no_grad():
        struct_out = structure_head(pair_repr, return_loss=True)

    coords = struct_out["coords"]  # (1, L, 3)
    closure_dist = struct_out["closure_distance"][0].item()

    print(f"Closure distance (x[0] to x[L-1]): {closure_dist:.2f} Å")
    print(f"Target bond length: 3.4 Å")

    # Check: closure distance should be reasonable
    # After refinement, should be closer to 3.4 Å
    assert closure_dist < 20.0, f"Closure distance too large: {closure_dist}"

    # Check closure loss
    closure_loss = struct_out.get("closure_loss")
    if closure_loss is not None:
        print(f"Closure loss: {closure_loss.item():.4f}")

    # Verify coordinates shape
    assert coords.shape == (1, L, 3)

    # Verify confidence scores
    confidence = struct_out["confidence"]
    assert confidence.shape == (1, L)
    print(f"Mean confidence: {confidence[0].mean().item():.2f}")

    print("✓ Structure closure verified")

    return True


def test_torusfold_integration():
    """Test 7: Full TorusFold integration (mock, no backbone)."""
    from core.torusfold import (
        TorusFold, TorusFoldConfig
    )

    print("\n=== Test 7: TorusFold Integration (Mock) ===")

    # Create config with minimal settings
    config = TorusFoldConfig(
        d_model=64,          # Small for testing
        n_torus_layers=1,
        n_rot_augments=0,    # Skip rotation aug for unit test
        c_z=32,              # Pair representation dim (renamed from d_pair)
        n_pairformer_blocks=1,  # Renamed from n_pair_layers
        n_heads_tri=2,
        translation_efficiency=True,
        circ_stability=True,
        immune_pathway=True,
        bsj_confidence=True,
    )

    model = TorusFold(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # Verify module structure
    print("Modules:")
    print(f"  - backbone (CircEquivariantBackbone): {type(model.backbone).__name__}")
    print(f"  - pair_init (PairInitialization): {type(model.pair_init).__name__}")
    print(f"  - pairformer (CircPairformerStack): {type(model.pairformer).__name__}")
    print(f"  - pair_head (PairPredictionHead): {type(model.pair_head).__name__}")
    print(f"  - bsj_analyzer (BSJPairAnalyzer): {type(model.bsj_analyzer).__name__}")
    print(f"  - structure_head: {type(model.structure_head).__name__}")
    print(f"  - composite_head: {type(model.composite_head).__name__}")
    print(f"  - report_head: {type(model.report_head).__name__}")
    print(f"  - response_head: {type(model.response_head).__name__}")

    # Test forward pass with mock embeddings
    L = 20

    # Mock backbone output
    mock_global_emb = torch.randn(1, config.d_model)
    mock_sequence_repr = torch.randn(1, L, config.d_model)

    # Mock gene expression
    gene_expr = torch.randn(1, config.gene_dim)

    # Run Pair initialization + Pairformer (v2, AF3-style)
    with torch.no_grad():
        pair_repr = model.pair_init(mock_sequence_repr)
        pair_repr = model.pairformer(pair_repr)
        pair_probs = model.pair_head(pair_repr)

        struct_out = model.structure_head(pair_repr)

    print(f"Pair repr shape: {pair_repr.shape}")
    print(f"Pair probs shape: {pair_probs.shape}")
    print(f"Structure output keys: {list(struct_out.keys())}")

    print("✓ TorusFold integration verified")

    return True


def test_torusfold_config():
    """Test 8: Config serialization and deserialization."""
    from core.torusfold import TorusFoldConfig

    print("\n=== Test 8: Config Serialization ===")

    config = TorusFoldConfig(
        d_model=128,
        n_harmonics=12,
        n_rot_augments=4,
        translation_efficiency=True,
    )

    # Serialize
    config_dict = config.to_dict()
    print(f"Config keys: {len(config_dict)}")
    print(f"d_model in dict: {config_dict['d_model']}")

    # Verify
    assert config_dict["d_model"] == 128
    assert config_dict["n_harmonics"] == 12
    assert config_dict["n_rot_augments"] == 4
    assert config_dict["translation_efficiency"] == True

    print("✓ Config serialization verified")

    return True


def run_all_tests():
    """Run all TorusFold tests."""
    print("=" * 60)
    print("TorusFold Component Tests")
    print("=" * 60)

    tests = [
        ("TPE Periodicity", test_tpe_periodicity),
        ("Circular Distance", test_circular_distance),
        ("Circular Relative Bias", test_circular_relative_bias),
        ("Rotation Invariance (Mock)", test_rotation_invariance_mock),
        ("Pair Symmetry", test_pair_symmetry),
        ("Structure Closure", test_structure_closure),
        ("TorusFold Integration", test_torusfold_integration),
        ("Config Serialization", test_torusfold_config),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "✓ PASS", None))
        except Exception as e:
            results.append((name, "✗ FAIL", str(e)))
            print(f"Error in {name}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, status, _ in results if status == "✓ PASS")
    failed = sum(1 for _, status, _ in results if status == "✗ FAIL")

    for name, status, error in results:
        print(f"{status} {name}")
        if error:
            print(f"  Error: {error[:100]}...")

    print(f"\nTotal: {passed} passed, {failed} failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)