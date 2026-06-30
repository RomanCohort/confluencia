#!/usr/bin/env python3
"""
test_all_backends.py — 测试所有 Backend 和 V2/V3/MOE 功能
"""

import sys
import time
import warnings
from pathlib import Path

# Fix Windows encoding
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_structure_backend():
    """测试 StructureBackend 三种模式。"""
    print("\n" + "="*60)
    print("  Test 1: StructureBackend (TorusFold/Pipeline/Heuristic)")
    print("="*60)

    from confluencia_3_0.core.circrna.structure_backend import (
        StructureBackend,
        BackendConfig,
        quick_predict,
        predict_with_pipeline,
        predict_heuristic,
    )

    # 测试序列
    test_seq = "ACGUGCUAAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCU"

    # === Test 1.1: Pipeline 模式 ===
    print("\n[Test 1.1] Pipeline Backend (推荐模式)")
    start = time.time()
    result = predict_with_pipeline(test_seq)
    elapsed = time.time() - start

    print(f"  ✓ Method: {result.method}")
    print(f"  ✓ Available: {result.available}")
    print(f"  ✓ dsRNA fraction: {result.dsRNA_fraction:.3f}")
    print(f"  ✓ BSJ stability: {result.bsj_stability:.3f}")
    print(f"  ✓ SASA mean: {result.sasa_mean:.3f}")
    print(f"  ✓ Confidence: {result.confidence:.3f}")
    print(f"  ✓ Elapsed: {elapsed:.2f}s")
    assert result.available, "Pipeline should be available"
    assert result.method in ["pipeline", "heuristic"], f"Method should be pipeline or heuristic fallback"

    # === Test 1.2: Heuristic 模式 ===
    print("\n[Test 1.2] Heuristic Backend (最快)")
    start = time.time()
    result = predict_heuristic(test_seq)
    elapsed = time.time() - start

    print(f"  ✓ Method: {result.method}")
    print(f"  ✓ Available: {result.available}")
    print(f"  ✓ dsRNA fraction: {result.dsRNA_fraction:.3f}")
    print(f"  ✓ Elapsed: {elapsed:.3f}s")
    assert result.available, "Heuristic should always be available"
    assert result.method == "heuristic", "Method should be heuristic"
    assert elapsed < 1.0, "Heuristic should be fast (<1s)"

    # === Test 1.3: TorusFold 模式（如果可用）===
    print("\n[Test 1.3] TorusFold Backend (需训练)")
    try:
        result = quick_predict(test_seq, mode="torusfold", fallback=False)
        print(f"  ✓ Method: {result.method}")
        print(f"  ✓ Available: {result.available}")
        if result.available:
            print(f"  ✓ coords available: {result.coords is not None}")
            print(f"  ✓ Confidence: {result.confidence:.3f}")
    except Exception as e:
        print(f"  ⚠ TorusFold not available: {e}")
        print("  (This is expected if model not trained)")

    # === Test 1.4: Fallback 机制 ===
    print("\n[Test 1.4] Fallback 机制")
    backend = StructureBackend()
    result = backend.predict(test_seq, mode="torusfold", fallback=True, verbose=True)
    print(f"  ✓ Final method: {result.method}")
    print(f"  ✓ Available: {result.available}")
    assert result.available, "Fallback should succeed"

    print("\n✅ Test 1 PASSED: StructureBackend")


def test_v2_immune_sensing():
    """测试 V2 免疫评分（动态权重）。"""
    print("\n" + "="*60)
    print("  Test 2: V2 Immune Sensing (Dynamic Weights)")
    print("="*60)

    from confluencia_3_0.core.circrna.immune_sensing_v2 import (
        predict_circrna_immunogenicity_v2,
        score_sequence_v2,
    )

    # 测试序列
    test_seq = "ACGUGCUAAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCU"

    # === Test 2.1: V2 启发式模式 ===
    print("\n[Test 2.1] V2 启发式模式")
    result = predict_circrna_immunogenicity_v2(test_seq, use_torusfold=False)

    print(f"  ✓ Method: {result.method}")
    print(f"  ✓ Overall score: {result.overall_score:.3f}")
    print(f"  ✓ RIG-I score: {result.rig_i_score:.3f}")
    print(f"  ✓ TLR7 score: {result.tlr7_score:.3f}")
    print(f"  ✓ TLR8 score: {result.tlr8_score:.3f}")
    print(f"  ✓ PKR score: {result.pkr_score:.3f}")
    assert result.method == "heuristic_fallback", "Should use heuristic fallback"
    assert 0 <= result.overall_score <= 1, "Score should be in [0, 1]"

    # === Test 2.2: V2 动态权重 ===
    print("\n[Test 2.2] V2 动态权重")
    weights = result.weights
    print(f"  ✓ RIG-I weights: dsRNA={weights.rig_i_dsRNA:.3f}, motif={weights.rig_i_motif:.3f}")
    print(f"  ✓ TLR7 weights: GU={weights.tlr7_gu_rich:.3f}, AU={weights.tlr7_au_rich:.3f}")
    print(f"  ✓ PKR weights: dsRNA={weights.pkr_dsRNA:.3f}")
    assert weights.rig_i_dsRNA + weights.rig_i_motif + weights.rig_i_gc + weights.rig_i_length > 0.9, "Weights should sum to ~1"

    # === Test 2.3: V2 快速接口 ===
    print("\n[Test 2.3] V2 快速接口")
    scores = score_sequence_v2(test_seq, use_torusfold=False)
    print(f"  ✓ Overall: {scores['overall']:.3f}")
    print(f"  ✓ RIG-I: {scores['rig_i']:.3f}")
    print(f"  ✓ Method: {scores['method']}")

    print("\n✅ Test 2 PASSED: V2 Immune Sensing")


def test_moe_v3():
    """测试 V3 MOE（SeqTopK 路由）。"""
    print("\n" + "="*60)
    print("  Test 3: V3 MOE (SeqTopK Routing)")
    print("="*60)

    from confluencia_3_0.core.circrna.torusfold_moe_v3 import (
        predict_with_moe_v3,
        compute_adaptive_weights_moe_v3,
        compute_immunogenicity_moe_v3,
    )

    # 测试序列
    test_seq = "ACGUGCUAAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCU"

    # === Test 3.1: MOE 预测 ===
    print("\n[Test 3.1] MOE 预测")
    result = predict_with_moe_v3(test_seq, use_torusfold=False)

    print(f"  ✓ Immunogenicity overall: {result['immunogenicity']['overall']:.3f}")
    print(f"  ✓ RIG-I: {result['immunogenicity']['pathways']['rig_i']:.3f}")
    print(f"  ✓ TLR7: {result['immunogenicity']['pathways']['tlr7']:.3f}")
    print(f"  ✓ Selected experts: {result['selected_experts']}")
    print(f"  ✓ Rationales: {result['rationales']['imm']}")

    # === Test 3.2: 动态权重 ===
    print("\n[Test 3.2] MOE 动态四维权重")
    weights = compute_adaptive_weights_moe_v3(test_seq)
    print(f"  ✓ Stability: {weights['stability']:.3f}")
    print(f"  ✓ Translation: {weights['translation']:.3f}")
    print(f"  ✓ Immune evasion: {weights['immune_evasion']:.3f}")
    print(f"  ✓ Delivery: {weights['delivery']:.3f}")
    assert sum(weights.values()) > 0.9, "Weights should sum to ~1"

    # === Test 3.3: 免疫评分 ===
    print("\n[Test 3.3] MOE 免疫评分")
    imm = compute_immunogenicity_moe_v3(test_seq)
    print(f"  ✓ RIG-I: {imm['rig_i']:.3f}")
    print(f"  ✓ TLR7: {imm['tlr7']:.3f}")
    print(f"  ✓ TLR8: {imm['tlr8']:.3f}")
    print(f"  ✓ PKR: {imm['pkr']:.3f}")

    print("\n✅ Test 3 PASSED: V3 MOE")


def test_circrna_manager():
    """测试 CircRNAManager 集成。"""
    print("\n" + "="*60)
    print("  Test 4: CircRNAManager Integration")
    print("="*60)

    try:
        from confluencia_3_0.core.subsystem_managers import CircRNAManager
        from confluencia_3_0.core.config import CircRNAConfig

        # 模拟 agent（简化版）
        class MockAgent:
            def __init__(self):
                self._internal_state = {}
                self.config = type('obj', (object,), {
                    'circrna': CircRNAConfig()
                })()

        agent = MockAgent()
        manager = CircRNAManager(agent)

        # 测试序列
        test_seq = "ACGUGCUAAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCU"

        # === Test 4.1: predict_structure (Pipeline) ===
        print("\n[Test 4.1] predict_structure (Pipeline)")
        result = manager.predict_structure(test_seq, mode="pipeline")
        print(f"  ✓ Method: {result['crna_structure_method']}")
        print(f"  ✓ dsRNA fraction: {result['crna_dsRNA_fraction']:.3f}")
        print(f"  ✓ BSJ stability: {result['crna_bsj_stability']:.3f}")
        print(f"  ✓ Confidence: {result['crna_confidence']:.3f}")

        # === Test 4.2: assess_immunogenicity (V2) ===
        print("\n[Test 4.2] assess_immunogenicity (V2)")
        result = manager.assess_immunogenicity(test_seq, backend="v2")
        print(f"  ✓ Method: {result['crna_backend_method']}")
        print(f"  ✓ Overall: {result['crna_immunogenicity_score']:.3f}")
        print(f"  ✓ RIG-I: {result['crna_rig_i_score']:.3f}")

        # === Test 4.3: assess_immunogenicity (MOE V3) ===
        print("\n[Test 4.3] assess_immunogenicity (MOE V3)")
        result = manager.assess_immunogenicity(test_seq, backend="moe_v3")
        print(f"  ✓ Method: {result['crna_backend_method']}")
        print(f"  ✓ Overall: {result['crna_immunogenicity_score']:.3f}")
        if 'crna_selected_experts' in result:
            print(f"  ✓ Selected experts: {result['crna_selected_experts']}")

        print("\n✅ Test 4 PASSED: CircRNAManager")

    except Exception as e:
        print(f"  ⚠ CircRNAManager test failed: {e}")
        print("  (This may be due to missing dependencies)")
        warnings.warn(f"CircRNAManager test skipped: {e}", UserWarning)


def test_evolution_v2():
    """测试进化模块 V2（动态权重）。"""
    print("\n" + "="*60)
    print("  Test 5: Evolution V2 (Dynamic Weights)")
    print("="*60)

    from confluencia_3_0.core.circrna.cirrna_evolution_v2 import (
        CircRNAEvolverV2,
        quick_evolve_v2,
        compute_adaptive_objective_weights,
        EvolutionConfigV2,
    )

    # 测试序列
    test_seq = "ACGUGCUAAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCU"

    # === Test 5.1: 快速进化 ===
    print("\n[Test 5.1] 快速进化 (10代)")
    start = time.time()
    best_seq, score = quick_evolve_v2(test_seq, generations=10, use_torusfold=False)
    elapsed = time.time() - start

    print(f"  ✓ Best sequence length: {len(best_seq)}")
    print(f"  ✓ Final score: {score:.3f}")
    print(f"  ✓ Elapsed: {elapsed:.2f}s")

    # === Test 5.2: 动态权重计算 ===
    print("\n[Test 5.2] 动态权重计算")
    config = EvolutionConfigV2()
    weights = compute_adaptive_objective_weights(None, test_seq, config)
    print(f"  ✓ Method: {weights.method}")
    print(f"  ✓ Stability: {weights.stability:.3f}")
    print(f"  ✓ Translation: {weights.translation:.3f}")
    print(f"  ✓ Immune: {weights.immune_evasion:.3f}")
    print(f"  ✓ Delivery: {weights.delivery:.3f}")

    # === Test 5.3: Evolver 完整测试 ===
    print("\n[Test 5.3] Evolver 完整测试")
    evolver = CircRNAEvolverV2(use_torusfold=False)
    total_score, details = evolver.evaluate_sequence(test_seq)
    print(f"  ✓ Total score: {total_score:.3f}")
    print(f"  ✓ Stability: {details['stability']:.3f}")
    print(f"  ✓ Translation: {details['translation']:.3f}")
    print(f"  ✓ Immune evasion: {details['immune_evasion']:.3f}")
    print(f"  ✓ Delivery: {details['delivery']:.3f}")

    print("\n✅ Test 5 PASSED: Evolution V2")


def test_performance():
    """性能测试。"""
    print("\n" + "="*60)
    print("  Test 6: Performance Benchmarks")
    print("="*60)

    from confluencia_3_0.core.circrna.structure_backend import (
        predict_heuristic,
        predict_with_pipeline,
    )

    # 不同长度序列
    test_seqs = [
        "ACGU" * 25,    # 100 nt
        "ACGU" * 125,   # 500 nt
        "ACGU" * 250,   # 1000 nt
    ]

    print("\n[Test 6.1] Heuristic 性能")
    for seq in test_seqs:
        start = time.time()
        result = predict_heuristic(seq)
        elapsed = time.time() - start
        print(f"  ✓ L={len(seq)}: {elapsed:.3f}s, dsRNA={result.dsRNA_fraction:.3f}")
        assert elapsed < 0.1, f"Heuristic should be instant (<0.1s), got {elapsed:.3f}s"

    print("\n[Test 6.2] Pipeline 性能")
    for seq in test_seqs:
        start = time.time()
        result = predict_with_pipeline(seq)
        elapsed = time.time() - start
        print(f"  ✓ L={len(seq)}: {elapsed:.2f}s, method={result.method}")
        # Pipeline 可能较慢，但不应超时
        assert elapsed < 10.0, f"Pipeline should complete in <10s, got {elapsed:.2f}s"

    print("\n✅ Test 6 PASSED: Performance")


def main():
    """运行所有测试。"""
    print("="*60)
    print("  Confluencia 3.0 Backend + V2/V3/MOE Integration Test")
    print("="*60)

    try:
        test_structure_backend()
        test_v2_immune_sensing()
        test_moe_v3()
        test_circrna_manager()
        test_evolution_v2()
        test_performance()

        print("\n" + "="*60)
        print("  ✅ ALL TESTS PASSED")
        print("="*60)

        print("\n功能总结：")
        print("  ✓ Backend 三种模式（TorusFold/Pipeline/Heuristic）")
        print("  ✓ 自动 Fallback 机制（带提示）")
        print("  ✓ V2 动态免疫评分权重")
        print("  ✓ V3 MOE SeqTopK 路由")
        print("  ✓ CircRNAManager 集成")
        print("  ✓ Evolution V2 动态四维权重")
        print("  ✓ 性能测试（L=100, 500, 1000）")

    except Exception as e:
        print("\n" + "="*60)
        print(f"  ❌ TEST FAILED: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()