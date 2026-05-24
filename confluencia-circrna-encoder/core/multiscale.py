"""
multiscale.py — Multi-scale modeling for circRNA therapeutics.

Adapted from drug 2.0's multiscale.py for circRNA context.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@dataclass
class MultiscaleConfig:
    """Configuration for multi-scale modeling."""

    # Time scales (hours)
    molecular_timescale: float = 0.1  # Molecular interactions
    cellular_timescale: float = 4.0  # Cellular response
    tissue_timescale: float = 24.0  # Tissue level
    organism_timescale: float = 168.0  # Whole organism (7 days)

    # Spatial scales
    molecular_scale: float = 1e-9  # nm
    cellular_scale: float = 1e-6  # um
    tissue_scale: float = 1e-3  # mm


class MolecularModel:
    """Molecular level model for circRNA."""

    def simulate(self, sequence: str) -> Dict:
        """Simulate molecular interactions."""
        from .innate_immune import quick_predict

        immune = quick_predict(sequence)

        # Molecular binding kinetics
        seq = sequence.upper()
        length = len(seq)

        # RIG-I binding affinity (GC-rich)
        gc = sum(1 for c in seq if c in 'GC') / max(length, 1)
        rig_i_binding = gc * 0.7 + 0.3

        # TLR binding affinity (GU-rich)
        gu = sum(1 for i in range(len(seq)-1) if seq[i:i+2] in ['GU', 'UG']) / max(length-1, 1)
        tlr_binding = gu * 0.8 + 0.2

        # PKR binding (structure)
        entropy = immune['pkr']['entropy']
        pkr_binding = min(entropy / 2, 1)

        return {
            'level': 'molecular',
            'rig_i_binding_affinity': rig_i_binding,
            'tlr_binding_affinity': tlr_binding,
            'pkr_binding_affinity': pkr_binding,
            'overall_binding': immune['overall_score'],
            'timescale': self.config.molecular_timescale if hasattr(self, 'config') else 0.1,
        }


class CellularModel:
    """Cellular level model for circRNA."""

    def simulate(self, sequence: str, molecular_result: Dict) -> Dict:
        """Simulate cellular response."""
        # Cell types involved
        cell_types = ['DC', 'Macrophage', 'NK', 'T_cell', 'B_cell']

        # Activation levels based on molecular binding
        dc_activation = molecular_result['rig_i_binding_affinity'] * 0.8
        macrophage_activation = molecular_result['tlr_binding_affinity'] * 0.7
        nk_activation = molecular_result['pkr_binding_affinity'] * 0.6
        t_cell_activation = dc_activation * 0.9  # Secondary activation
        b_cell_activation = t_cell_activation * 0.5

        return {
            'level': 'cellular',
            'cell_activations': {
                'DC': dc_activation,
                'Macrophage': macrophage_activation,
                'NK': nk_activation,
                'T_cell': t_cell_activation,
                'B_cell': b_cell_activation,
            },
            'max_activation': max(dc_activation, macrophage_activation, t_cell_activation),
            'timescale': self.config.cellular_timescale if hasattr(self, 'config') else 4.0,
        }


class TissueModel:
    """Tissue level model for circRNA."""

    def simulate(self, cellular_result: Dict) -> Dict:
        """Simulate tissue-level response."""
        activations = cellular_result['cell_activations']

        # Tissue response
        inflammation = activations['Macrophage'] * 0.6 + activations['DC'] * 0.4
        immune_infiltration = activations['T_cell'] * 0.7 + activations['NK'] * 0.3
        cytokine_release = sum(activations.values()) / len(activations) * 0.5

        return {
            'level': 'tissue',
            'inflammation_level': inflammation,
            'immune_infiltration': immune_infiltration,
            'cytokine_release': cytokine_release,
            'tissue_effect': inflammation * 0.4 + immune_infiltration * 0.6,
            'timescale': self.config.tissue_timescale if hasattr(self, 'config') else 24.0,
        }


class OrganismModel:
    """Whole organism level model."""

    def simulate(self, tissue_result: Dict) -> Dict:
        """Simulate organism-level outcome."""
        tissue_effect = tissue_result['tissue_effect']

        # Clinical outcomes
        tumor_response = tissue_effect * 0.7
        systemic_toxicity = tissue_result['inflammation_level'] * 0.4

        # Therapeutic window
        therapeutic_window = tumor_response - systemic_toxicity

        return {
            'level': 'organism',
            'tumor_response': tumor_response,
            'systemic_toxicity': systemic_toxicity,
            'therapeutic_window': therapeutic_window,
            'clinical_benefit': therapeutic_window > 0.2,
            'timescale': self.config.organism_timescale if hasattr(self, 'config') else 168.0,
        }


class MultiscaleModel:
    """
    Multi-scale modeling for circRNA therapeutics.

    Levels:
    - Molecular: circRNA-protein binding
    - Cellular: immune cell activation
    - Tissue: inflammation, infiltration
    - Organism: tumor response, toxicity
    """

    def __init__(self, config: Optional[MultiscaleConfig] = None):
        self.config = config or MultiscaleConfig()

        self.molecular = MolecularModel()
        self.cellular = CellularModel()
        self.tissue = TissueModel()
        self.organism = OrganismModel()

    def simulate_full(self, sequence: str) -> Dict:
        """
        Simulate all scales.

        Args:
            sequence: circRNA sequence

        Returns:
            Multi-scale simulation results
        """
        # Molecular level
        mol_result = self.molecular.simulate(sequence)

        # Cellular level
        cell_result = self.cellular.simulate(sequence, mol_result)

        # Tissue level
        tissue_result = self.tissue.simulate(cell_result)

        # Organism level
        organism_result = self.organism.simulate(tissue_result)

        return {
            'molecular': mol_result,
            'cellular': cell_result,
            'tissue': tissue_result,
            'organism': organism_result,
            'cascade': {
                'molecular→cellular': mol_result['overall_binding'],
                'cellular→tissue': cell_result['max_activation'],
                'tissue→organism': tissue_result['tissue_effect'],
            },
            'final_outcome': {
                'tumor_response': organism_result['tumor_response'],
                'toxicity': organism_result['systemic_toxicity'],
                'therapeutic_window': organism_result['therapeutic_window'],
            },
        }

    def simulate_timeseries(self, sequence: str, n_steps: int = 100) -> Dict:
        """Simulate across time at all scales."""
        results = []

        for step in range(n_steps):
            sim = self.simulate_full(sequence)

            # Add time progression
            time_hours = step * 2  # 2 hour steps

            sim['time'] = time_hours

            # Scale activation based on time
            if time_hours < 10:  # Early: molecular dominant
                factor = 0.3
            elif time_hours < 50:  # Mid: cellular/tissue dominant
                factor = 0.7
            else:  # Late: organism effects
                factor = 1.0

            sim['molecular']['overall_binding'] *= factor
            sim['cellular']['max_activation'] *= factor
            sim['tissue']['tissue_effect'] *= factor
            sim['organism']['tumor_response'] *= factor

            results.append(sim)

        return {
            'timeseries': results,
            'peak_response_time': max(range(n_steps), key=lambda i: results[i]['organism']['tumor_response']) * 2,
            'peak_response_value': max(r['organism']['tumor_response'] for r in results),
        }


def multiscale_simulation(sequence: str) -> Dict:
    """Quick multi-scale simulation."""
    model = MultiscaleModel()
    return model.simulate_full(sequence)