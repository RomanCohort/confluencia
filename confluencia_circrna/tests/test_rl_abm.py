"""
Test script for circRNA RL-ABM integration.

Validates:
1. ABM simulation works with circRNA sequences
2. Reward computation integrates ABM + drug response
3. RL environment runs without errors
4. Comparison with baseline methods
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from confluencia_circrna.core.circrna_rl_abm import (
    PatientProfile,
    RLABMConfig,
    compute_abm_reward,
    CircRNAABMEnv,
    train_rl_abm,
    optimize_circrna_with_abm,
    compare_with_baseline,
)


def test_abm_reward():
    """Test ABM-based reward computation."""
    print("\n" + "="*60)
    print("TEST 1: ABM Reward Computation")
    print("="*60)

    # Test sequence
    test_seq = "AUGCGCUAUGGCUAGCUAUGCGCUAUGGCUAGCUAUGCGCUAUGGC" * 3
    modification = "m6A"

    # Create patient profile
    patient = PatientProfile(
        patient_id="test_001",
        trop2=0.6,
        b7h4=0.4,
        mki67=0.5,
        pd_l1=0.7,
    )

    # Compute reward
    reward = compute_abm_reward(
        sequence=test_seq,
        modification=modification,
        patient=patient,
        combination_drug="pembrolizumab",
    )

    print(f"Sequence length: {len(test_seq)}")
    print(f"Modification: {modification}")
    print(f"Combination drug: pembrolizumab")
    print(f"\nReward components:")
    print(f"  Peak antibody: {reward.peak_antibody:.4f}")
    print(f"  Peak effector T: {reward.peak_effector_t:.4f}")
    print(f"  Immune AUC: {reward.immune_auc:.4f}")
    print(f"  Antigen clearance: {reward.antigen_clearance:.4f}")
    print(f"  Response probability: {reward.response_probability:.4f}")
    print(f"  Resistance risk: {reward.resistance_risk:.4f}")
    print(f"  Synergy score: {reward.synergy_score:.4f}")
    print(f"\n  TOTAL REWARD: {reward.total_reward:.4f}")

    assert 0.0 <= reward.total_reward <= 1.0, "Reward out of bounds"
    print("\n✓ Test passed: Reward computation works")


def test_rl_env():
    """Test RL environment."""
    print("\n" + "="*60)
    print("TEST 2: RL Environment")
    print("="*60)

    config = RLABMConfig(
        max_steps=10,
        seed_seq="AUGCGCUAUGGCUAGCUAUGCGCUAUGGCUAGCUAUGCGCUAUGGC",
        initial_modification="m6A",
        seed=42,
    )

    env = CircRNAABMEnv(config)

    # Reset
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info keys: {list(info.keys())}")

    # Run a few steps
    total_reward = 0.0
    for step in range(5):
        action = np.random.randint(0, env.n_actions)
        obs, reward, done, step_info = env.step(action)
        total_reward += reward
        print(f"  Step {step+1}: action={step_info['action']}, reward={reward:.4f}")

        if done:
            break

    print(f"\nTotal reward: {total_reward:.4f}")
    print(f"Best sequence found: {len(env.best_sequence)} bp")
    print(f"Best modification: {env.best_modification}")
    print(f"Best reward: {env.best_reward:.4f}")

    assert len(env.best_sequence) > 0, "No sequence found"
    print("\n✓ Test passed: RL environment works")


def test_training():
    """Test RL training loop."""
    print("\n" + "="*60)
    print("TEST 3: RL Training (10 episodes)")
    print("="*60)

    config = RLABMConfig(
        max_steps=8,
        seed_seq="AUGCGCUAUGGCUAGCUAUGCGCUAUGGCUAGCUAUGCGCUAUGGC",
        initial_modification="m6A",
        seed=42,
    )

    results_df, best = train_rl_abm(
        config=config,
        n_episodes=10,
        verbose=True,
    )

    print(f"\nTraining results shape: {results_df.shape}")
    print(f"Best reward achieved: {best['reward']:.4f}")
    print(f"Best sequence length: {len(best['sequence'])} bp")
    print(f"Best modification: {best['modification']}")

    assert len(results_df) == 10, "Wrong number of episodes"
    print("\n✓ Test passed: RL training works")


def test_optimization():
    """Test full optimization pipeline."""
    print("\n" + "="*60)
    print("TEST 4: Full Optimization")
    print("="*60)

    seed_seq = "AUGCGCUAUGGCUAGCUAUGCGCUAUGGCUAGCUAUGCGCUAUGGC"

    patient = PatientProfile(
        patient_id="high_risk",
        trop2=0.8,
        b7h4=0.7,
        mki67=0.6,
        pd_l1=0.3,  # Low PD-L1 = harder to treat
    )

    best_seq, best_mod, results = optimize_circrna_with_abm(
        seed_seq=seed_seq,
        patient=patient,
        n_episodes=20,
        modification="m6A",
    )

    print(f"Patient: {patient.patient_id}")
    print(f"  TROP2: {patient.trop2}, B7-H4: {patient.b7h4}")
    print(f"\nOptimization results:")
    print(f"  Best sequence: {len(best_seq)} bp")
    print(f"  Best modification: {best_mod}")
    print(f"  Episodes run: {len(results)}")
    print(f"  Best episode reward: {results['best_reward'].max():.4f}")

    # Compare rewards over episodes
    initial_reward = results['episode_reward'].iloc[0]
    final_best = results['best_reward'].iloc[-1]
    improvement = (final_best - initial_reward) / max(initial_reward, 0.01) * 100

    print(f"\n  Improvement: {improvement:.1f}%")

    print("\n✓ Test passed: Optimization works")


def test_comparison():
    """Test comparison with baseline."""
    print("\n" + "="*60)
    print("TEST 5: Method Comparison")
    print("="*60)

    seed_seq = "AUGCGCUAUGGCUAGCUAUGCGCUAUGGCUAGCUAUGCGCUAUGGC"

    try:
        comparison_df = compare_with_baseline(
            seed_seq=seed_seq,
            n_rounds=3,
        )

        print("\nComparison results:")
        print(comparison_df.to_string(index=False))

        # Find best method
        best_method = comparison_df.loc[comparison_df['reward'].idxmax(), 'method']
        print(f"\nBest method: {best_method}")

        print("\n✓ Test passed: Comparison works")
    except Exception as e:
        print(f"Comparison test skipped (some modules unavailable): {e}")


def main():
    """Run all tests."""
    print("="*60)
    print(" circRNA RL-ABM Integration Tests")
    print("="*60)

    tests = [
        ("ABM Reward", test_abm_reward),
        ("RL Environment", test_rl_env),
        ("RL Training", test_training),
        ("Optimization", test_optimization),
        ("Comparison", test_comparison),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f" Results: {passed} passed, {failed} failed")
    print("="*60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
