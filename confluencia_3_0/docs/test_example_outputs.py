"""Quick test to generate example outputs for wiki."""
import sys
sys.path.insert(0, '..')

from confluencia_3_0.core.circrna.immune_sensing import score_sequence
from confluencia_3_0.core.pk.rnactm import (
    simulate_rna_ctm, infer_rna_ctm_params, summarize_rna_ctm_curve
)

# Test 1: Immunogenicity scoring
print("=" * 60)
print("Immunogenicity Scoring Example")
print("=" * 60)

test_sequences = [
    ("Low risk sequence", "AUGCGCUAUAGCAUGCGCUAUAUGC"),  # Low AU/GU content
    ("High RIG-I risk", "GUGUGUGUCCUCCGUGUGUGUCCUCCGU"),  # GU-rich, CCUCC motifs
    ("High TLR7 risk", "GUUGGUUGGUUGGUUGGUUG"),  # GUUG motifs
    ("High TLR8 risk", "AUUAUUAUUAUAUUUAUUUAU"),  # AUUA motifs
]

for name, seq in test_sequences:
    scores = score_sequence(seq)
    print(f"\n{name}:")
    print(f"  Sequence: {seq[:30]}...")
    print(f"  RIG-I:  {scores.get('rig_i', 0):.3f}")
    print(f"  TLR7:   {scores.get('tlr7', 0):.3f}")
    print(f"  TLR8:   {scores.get('tlr8', 0):.3f}")
    print(f"  PKR:    {scores.get('pkr', 0):.3f}")
    print(f"  Overall: {scores.get('overall', 0):.3f}")

# Test 2: PK simulation
print("\n" + "=" * 60)
print("CirculaPK Pharmacokinetic Simulation")
print("=" * 60)

params = infer_rna_ctm_params(modification='m6a')
curve = simulate_rna_ctm(dose=1.0, freq=1.0, params=params, horizon=72)
summary = summarize_rna_ctm_curve(curve)

print("\nm6A-modified circRNA PK parameters:")
for key, value in summary.items():
    print(f"  {key}: {value:.4f}")

# Test 3: Modification comparison
print("\n" + "=" * 60)
print("Effect of Modifications")
print("=" * 60)

print("\nDegradation rates by modification:")
mods = ['none', 'm6a', 'psi', '5mc']
for mod in mods:
    p = infer_rna_ctm_params(modification=mod)
    print(f"  {mod}: k_degrade = {p.k_degrade:.4f}")

print("\nTest completed successfully.")