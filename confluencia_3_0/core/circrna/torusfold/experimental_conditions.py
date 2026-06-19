"""
experimental_conditions.py — Experimental condition encoding for RNA structure.

RNA folding is sensitive to:
- pH: affects protonation of bases, changes pairing preferences
- Mg²⁺ concentration: stabilizes tertiary structure, essential for ribozymes
- Na⁺/K⁺ concentration: affects duplex stability
- Temperature: affects folding kinetics and equilibrium

This module provides:
1. ConditionEncoder: encode experimental conditions as neural network conditioning
2. ViennaRNAConditionWrapper: pass conditions to ViennaRNA folding
3. PhysicsConditionAdjuster: adjust physics energy terms based on conditions
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class ExperimentalConditions:
    """Experimental conditions affecting RNA folding."""
    temperature: float = 310.0      # K (37°C default)
    pH: float = 7.4                 # physiological pH
    Mg_concentration: float = 1.0   # mM, Mg²⁺
    Na_concentration: float = 150.0 # mM, Na⁺/K⁺
    K_concentration: float = 5.0    # mM, intracellular K⁺

    def to_dict(self) -> Dict:
        return {
            'T': self.temperature,
            'pH': self.pH,
            'Mg': self.Mg_concentration,
            'Na': self.Na_concentration,
            'K': self.K_concentration,
        }

    def validate(self) -> bool:
        """Check if conditions are physically reasonable."""
        if self.temperature < 250 or self.temperature > 400:
            return False
        if self.pH < 3 or self.pH > 11:
            return False
        if self.Mg_concentration < 0 or self.Mg_concentration > 100:
            return False
        if self.Na_concentration < 0 or self.Na_concentration > 1000:
            return False
        return True


class ConditionEncoder(nn.Module):
    """Encode experimental conditions as neural network conditioning.

    Maps physical parameters to embedding vectors that can be
    injected into the structure prediction model.
    """

    def __init__(
        self,
        d_cond: int = 64,
        use_temperature: bool = True,
        use_ph: bool = True,
        use_ions: bool = True,
    ):
        super().__init__()
        self.d_cond = d_cond

        # Temperature encoder (K → normalized → embedding)
        self.temp_encoder = nn.Sequential(
            nn.Linear(1, d_cond // 4),
            nn.GELU(),
            nn.Linear(d_cond // 4, d_cond // 3),
        ) if use_temperature else None

        # pH encoder
        self.ph_encoder = nn.Sequential(
            nn.Linear(1, d_cond // 4),
            nn.GELU(),
            nn.Linear(d_cond // 4, d_cond // 3),
        ) if use_ph else None

        # Ion encoder (Mg²⁺, Na⁺, K⁺)
        self.ion_encoder = nn.Sequential(
            nn.Linear(3, d_cond // 2),
            nn.GELU(),
            nn.Linear(d_cond // 2, d_cond // 3),
        ) if use_ions else None

        # Combine all conditions
        total_dim = d_cond // 3 * sum([use_temperature, use_ph, use_ions])
        self.combiner = nn.Sequential(
            nn.Linear(total_dim, d_cond),
            nn.LayerNorm(d_cond),
            nn.GELU(),
        )

    def forward(
        self,
        conditions: ExperimentalConditions,
    ) -> torch.Tensor:
        """Encode conditions to conditioning vector.

        Returns:
            (d_cond,) conditioning embedding
        """
        embeddings = []

        # Normalize temperature (250-400K → 0-1)
        if self.temp_encoder is not None:
            T_norm = (conditions.temperature - 250) / 150
            T_norm = max(0, min(1, T_norm))
            T_tensor = torch.tensor([[T_norm]], dtype=torch.float32)
            embeddings.append(self.temp_encoder(T_tensor).squeeze(0))

        # Normalize pH (3-11 → 0-1)
        if self.ph_encoder is not None:
            pH_norm = (conditions.pH - 3) / 8
            pH_norm = max(0, min(1, pH_norm))
            pH_tensor = torch.tensor([[pH_norm]], dtype=torch.float32)
            embeddings.append(self.ph_encoder(pH_tensor).squeeze(0))

        # Normalize ions (log scale, concentration range varies)
        if self.ion_encoder is not None:
            Mg_norm = math.log10(conditions.Mg_concentration + 1) / 2  # 0-100 mM → 0-1
            Na_norm = math.log10(conditions.Na_concentration + 1) / 3  # 0-1000 mM → 0-1
            K_norm = math.log10(conditions.K_concentration + 1) / 2   # 0-100 mM → 0-1
            ion_tensor = torch.tensor([[Mg_norm, Na_norm, K_norm]], dtype=torch.float32)
            embeddings.append(self.ion_encoder(ion_tensor).squeeze(0))

        # Combine
        combined = torch.cat(embeddings, dim=-1)
        cond_embedding = self.combiner(combined)

        return cond_embedding


class ViennaRNAConditionWrapper:
    """Pass experimental conditions to ViennaRNA folding.

    ViennaRNA supports:
    - Temperature: -T parameter
    - Salt concentration: --saltConc parameter (since 2.4)
    - Mg²⁺: via parameters file or --saltConc with Mg²⁺ model
    """

    def __init__(self, vienna_path: str = "RNAfold"):
        self.vienna_path = vienna_path

    def fold_with_conditions(
        self,
        sequence: str,
        conditions: ExperimentalConditions,
    ) -> Dict:
        """Fold RNA with experimental conditions.

        Args:
            sequence: RNA sequence
            conditions: experimental conditions

        Returns:
            Dict with structure, mfe, conditions used
        """
        import subprocess
        import tempfile

        # Build command with conditions
        cmd = [self.vienna_path]

        # Temperature (ViennaRNA uses °C)
        T_celsius = conditions.temperature - 273.15
        cmd.extend(["-T", str(T_celsius)])

        # Salt concentration (Na⁺ + K⁺)
        total_salt = conditions.Na_concentration + conditions.K_concentration
        if total_salt > 0:
            cmd.extend(["--saltConc", str(total_salt / 1000)])  # ViennaRNA expects M

        # For Mg²⁺, ViennaRNA 2.5+ has --saltConc with ion type
        # Older versions: use parameter adjustment
        if conditions.Mg_concentration > 0:
            # Mg²⁺ stabilizes structure, adjust energy parameters
            # This is a simplified model; real Mg²⁺ effects are complex
            cmd.extend(["--saltConcMg", str(conditions.Mg_concentration / 1000)])

        # Run ViennaRNA
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
            f.write(f">seq\n{sequence}\n")
            fa_path = f.name

        try:
            result = subprocess.run(
                cmd + [fa_path],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Parse output
            lines = result.stdout.strip().split('\n')
            structure = ""
            mfe = 0.0

            for line in lines:
                if line.startswith('.'):
                    structure = line.split()[0]
                elif '(' in line and ')' in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        structure = parts[0]
                        mfe_str = parts[-1]
                        if mfe_str.endswith(')') and mfe_str.startswith('('):
                            mfe = float(mfe_str[1:-1])

            return {
                'sequence': sequence,
                'structure': structure,
                'mfe': mfe,
                'conditions': conditions.to_dict(),
                'cmd': ' '.join(cmd),
            }

        finally:
            import os
            os.unlink(fa_path)


class PhysicsConditionAdjuster:
    """Adjust physics energy terms based on experimental conditions.

    Key effects:
    - Temperature: affects Boltzmann distribution, thermal fluctuations
    - pH: affects base protonation (A, C can be protonated)
    - Mg²⁺: stabilizes tertiary structure, reduces electrostatic repulsion
    - Na⁺/K⁺: shielding of phosphate repulsion
    """

    def __init__(
        self,
        conditions: ExperimentalConditions,
    ):
        self.conditions = conditions

    def adjust_bond_energy(self, base_energy: float) -> float:
        """Temperature affects bond fluctuations."""
        # Higher T → larger thermal fluctuations → higher bond energy tolerance
        T_ref = 310.0  # 37°C reference
        scale = (self.conditions.temperature / T_ref) ** 0.5
        return base_energy / scale  # Higher T, lower penalty

    def adjust_electrostatic_energy(self, base_energy: float) -> float:
        """Ion concentration shields phosphate repulsion."""
        # Debye-Hückel screening length: λ_D = 0.304 / sqrt(I) nm
        # where I is ionic strength
        I = 0.5 * (
            self.conditions.Na_concentration +
            self.conditions.K_concentration +
            4 * self.conditions.Mg_concentration  # Mg²⁺ contributes 4x
        ) / 1000  # Convert mM to M

        if I > 0:
            screening_factor = 1.0 / (1.0 + math.sqrt(I))
        else:
            screening_factor = 1.0

        return base_energy * screening_factor

    def adjust_pair_stability(self, base_energy: float, pair_type: str) -> float:
        """Mg²⁺ stabilizes certain pair types."""
        # Mg²⁺ preferentially stabilizes:
        # - G-C pairs (3 H-bonds)
        # - Non-canonical pairs
        # - Tertiary interactions

        Mg = self.conditions.Mg_concentration

        if pair_type == 'GC':
            # Mg²⁺ enhances G-C stability
            stabilization = 0.1 * math.log(Mg + 1) * base_energy
            return base_energy - stabilization
        elif pair_type == 'AU':
            # AU less affected
            return base_energy
        elif pair_type == 'GU':
            # GU wobble pairs benefit from Mg²⁺
            stabilization = 0.05 * math.log(Mg + 1) * base_energy
            return base_energy - stabilization
        else:
            return base_energy

    def adjust_closure_tolerance(self, base_tolerance: float) -> float:
        """Higher temperature → larger closure tolerance."""
        T_ref = 310.0
        scale = (self.conditions.temperature / T_ref) ** 0.5
        return base_tolerance * scale

    def get_thermal_fluctuation(self) -> float:
        """Get expected thermal fluctuation amplitude (Å)."""
        # RMS fluctuation: sqrt(kT / k_bond)
        k_bond = 1.0  # arbitrary units
        kB = 1.38e-23  # J/K
        T = self.conditions.temperature
        # Convert to Å scale (rough approximation)
        fluctuation = math.sqrt(kB * T / k_bond) * 1e10 * 0.1
        return min(fluctuation, 2.0)  # Cap at 2Å


class ConditionAwareStructureBackend:
    """Structure prediction backend with experimental condition support.

    Combines:
    - TorusFold for 3D prediction
    - ViennaRNA for secondary structure
    - Physics refinement with condition-adjusted parameters
    """

    def __init__(
        self,
        default_conditions: Optional[ExperimentalConditions] = None,
    ):
        self.default_conditions = default_conditions or ExperimentalConditions()
        self.condition_encoder = ConditionEncoder()

    def predict(
        self,
        sequence: str,
        conditions: Optional[ExperimentalConditions] = None,
        backend: str = "torusfold",
    ) -> Dict:
        """Predict structure with experimental conditions.

        Args:
            sequence: circRNA sequence
            conditions: experimental conditions (uses default if None)
            backend: "torusfold", "vienna", "physics"

        Returns:
            Dict with coords/structure, conditions, energy
        """
        cond = conditions or self.default_conditions

        if not cond.validate():
            raise ValueError(f"Invalid conditions: {cond}")

        # Encode conditions
        cond_embedding = self.condition_encoder(cond)

        result = {
            'sequence': sequence,
            'conditions': cond.to_dict(),
        }

        if backend == "vienna":
            vienna = ViennaRNAConditionWrapper()
            vienna_result = vienna.fold_with_conditions(sequence, cond)
            result['structure'] = vienna_result['structure']
            result['mfe'] = vienna_result['mfe']

        elif backend == "physics":
            # Use physics solver with condition-adjusted parameters
            from .constraint_solver import GeometricConstraintSolver, SolverConfig

            adjuster = PhysicsConditionAdjuster(cond)

            config = SolverConfig(
                closure_tolerance=adjuster.adjust_closure_tolerance(0.5),
            )
            solver = GeometricConstraintSolver(config)

            class MinimalConstraintSet:
                def __init__(self, seq_len):
                    self.seq_len = seq_len
                    self.pair_constraints = []

            constraint_set = MinimalConstraintSet(len(sequence))
            conformations = solver.solve(constraint_set)

            if conformations:
                coords = conformations[0]
                # Adjust energy based on conditions
                energy = solver._compute_cg_energy(coords, constraint_set)
                energy_adj = adjuster.adjust_electrostatic_energy(energy)

                result['coords'] = coords
                result['energy'] = energy_adj
                result['thermal_fluctuation'] = adjuster.get_thermal_fluctuation()

        elif backend == "torusfold":
            # Inject condition embedding into TorusFold
            # (requires model modification to accept conditioning)
            result['condition_embedding'] = cond_embedding.detach().numpy()
            result['note'] = "Condition embedding available for TorusFold conditioning"

        return result