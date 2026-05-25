#!/usr/bin/env python
"""
test_pipeline_full.py — Full pipeline integration test.

Usage:
    python test_pipeline_full.py
"""

import sys
sys.path.insert(0, 'confluencia_circrna')
from confluencia_circrna import run_pipeline


def main():
    print("=" * 60)
    print("circRNA Pipeline Full Integration Test")
    print("=" * 60)

    # Test 1: High immunogenicity sequence
    print("\n=== Test 1: High Immunogenicity ===")
    result1 = run_pipeline(
        'GCGCGUGUGUACGU' * 30,
        {'TROP2': 8.0, 'B7-H4': 7.0},
        enable_structure=True
    )

    print(f'IPS: {result1.composite_scores["ips"]:.2f}')
    print(f'Response: {result1.composite_scores["predicted_response"]}')
    print(f'RIG-I: {result1.immune_scores["rig_i_score"]:.3f}')
    print(f'TLR: {result1.immune_scores["tlr_score"]:.3f}')
    print(f'PKR: {result1.immune_scores["pkr_score"]:.3f}')
    print(f'Overall: {result1.composite_scores["overall_immunogenicity"]:.3f}')

    if result1.structure_features:
        print(f'Stability: {result1.structure_features.structure_stability:.2f}')
        print(f'MFE: {result1.structure_features.mfe:.1f} kcal/mol')
        print(f'Method: {result1.structure_features.prediction_method}')
        print(f'dsRNA fraction: {result1.structure_features.dsrna_fraction:.2%}')

    print('Recommendations:')
    for rec in result1.recommendations[:4]:
        print(f'  - {rec}')

    # Test 2: Low immunogenicity sequence
    print("\n=== Test 2: Low Immunogenicity ===")
    result2 = run_pipeline(
        'AUAUAUAUAUAUAU' * 30,
        {'TROP2': 3.0, 'B7-H4': 2.0},
        enable_structure=True
    )

    print(f'IPS: {result2.composite_scores["ips"]:.2f}')
    print(f'Response: {result2.composite_scores["predicted_response"]}')
    print(f'RIG-I: {result2.immune_scores["rig_i_score"]:.3f}')
    print(f'TLR: {result2.immune_scores["tlr_score"]:.3f}')
    print(f'PKR: {result2.immune_scores["pkr_score"]:.3f}')

    if result2.structure_features:
        print(f'Stability: {result2.structure_features.structure_stability:.2f}')
        print(f'MFE: {result2.structure_features.mfe:.1f} kcal/mol')
        print(f'Method: {result2.structure_features.prediction_method}')

    print('Recommendations:')
    for rec in result2.recommendations[:4]:
        print(f'  - {rec}')

    # Test 3: Mixed sequence (no structure prediction)
    print("\n=== Test 3: Mixed Sequence (no structure) ===")
    result3 = run_pipeline(
        'ACGUACGUACGUACGU' * 20,
        {'TROP2': 5.0, 'MKI67': 6.0},
        enable_structure=False
    )

    print(f'IPS: {result3.composite_scores["ips"]:.2f}')
    print(f'Response: {result3.composite_scores["predicted_response"]}')
    print(f'Structure: {result3.structure_features}')

    print("\n" + "=" * 60)
    print("Test Complete!")


if __name__ == "__main__":
    main()