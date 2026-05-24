"""
immune_abm.py — Agent-Based Model for circRNA immune response simulation.

Adapted from drug 2.0's immune_abm.py for circRNA context.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class CellType(Enum):
    """Cell types in immune simulation."""
    DC = "dendritic_cell"
    MACROPHAGE = "macrophage"
    T_CELL = "t_cell"
    B_CELL = "b_cell"
    NK_CELL = "natural_killer"
    TUMOR = "tumor_cell"
    CIRCRNA = "circrna"


class CellState(Enum):
    """Cell states."""
    RESTING = "resting"
    ACTIVATED = "activated"
    DYING = "dying"
    DEAD = "dead"


@dataclass
class AgentConfig:
    """Configuration for ABM simulation."""

    # Grid size
    grid_size: int = 100

    # Initial populations
    n_dc: int = 50
    n_macrophage: int = 30
    n_t_cell: int = 100
    n_b_cell: int = 20
    n_nk: int = 50
    n_tumor: int = 200
    n_circrna: int = 1000

    # Time parameters
    max_steps: int = 500
    dt: float = 1.0

    # Interaction probabilities
    dc_circrna_capture: float = 0.8
    t_cell_activation: float = 0.6
    tumor_killing: float = 0.3

    # Cytokine parameters
    cytokine_decay: float = 0.95
    cytokine_threshold: float = 100


class ImmuneCell:
    """Base class for immune cells."""

    def __init__(self, cell_type: CellType, position: tuple, state: CellState = CellState.RESTING):
        self.cell_type = cell_type
        self.position = position
        self.state = state
        self.activation_level = 0.0
        self.lifetime = 0

    def move(self, grid_size: int):
        """Random movement."""
        dx = np.random.randint(-1, 2)
        dy = np.random.randint(-1, 2)

        new_x = max(0, min(grid_size - 1, self.position[0] + dx))
        new_y = max(0, min(grid_size - 1, self.position[1] + dy))

        self.position = (new_x, new_y)


class CircRNAAgent:
    """circRNA agent in the simulation."""

    def __init__(self, sequence: str, position: tuple, config: AgentConfig):
        self.sequence = sequence
        self.position = position
        self.config = config

        # Calculate properties
        self.immunogenicity = self._calc_immunogenicity()
        self.persistence = self._calc_persistence()
        self.stability = self._calc_stability()

        self.captured = False
        self.transcribed = False
        self.active = True

    def _calc_immunogenicity(self) -> float:
        """Calculate immunogenicity score."""
        seq = self.sequence.upper().replace('T', 'U')
        length = len(seq)

        gc = sum(1 for c in seq if c in 'GC') / max(length, 1)
        u = sum(1 for c in seq if c == 'U') / max(length, 1)

        return gc * 0.4 + u * 0.3 + np.random.uniform(0.1, 0.3)

    def _calc_persistence(self) -> float:
        """Calculate circRNA persistence (half-life estimate)."""
        # Longer circRNAs tend to persist longer
        length_factor = min(len(self.sequence) / 500, 1.0)

        # GC-rich sequences are more stable
        gc = sum(1 for c in self.sequence.upper() if c in 'GC') / max(len(self.sequence), 1)

        return length_factor * 0.5 + gc * 0.5

    def _calc_stability(self) -> float:
        """Calculate circRNA stability."""
        return self.persistence * 0.8 + 0.2


class CytokineField:
    """Cytokine concentration field."""

    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.il6 = np.zeros((grid_size, grid_size))
        self.tnf = np.zeros((grid_size, grid_size))
        self.ifn = np.zeros((grid_size, grid_size))

    def diffuse(self, decay: float):
        """Diffuse and decay cytokines."""
        # Simple diffusion
        self.il6 *= decay
        self.tnf *= decay
        self.ifn *= decay

    def add_cytokine(self, position: tuple, cytokine: str, amount: float):
        """Add cytokine at position."""
        if cytokine == 'IL6':
            self.il6[position[0], position[1]] += amount
        elif cytokine == 'TNF':
            self.tnf[position[0], position[1]] += amount
        elif cytokine == 'IFN':
            self.ifn[position[0], position[1]] += amount

    def get_concentration(self, position: tuple) -> Dict:
        """Get cytokine concentrations at position."""
        return {
            'IL6': self.il6[position[0], position[1]],
            'TNF': self.tnf[position[0], position[1]],
            'IFN': self.ifn[position[0], position[1]],
        }


class ImmuneABM:
    """
    Agent-Based Model for circRNA immune response.

    Simulates:
    - circRNA distribution and persistence
    - Dendritic cell capture and presentation
    - T cell activation
    - Tumor cell killing
    - Cytokine dynamics
    """

    def __init__(self, config: Optional[AgentConfig] = None, circrna_sequence: str = None):
        self.config = config or AgentConfig()
        self.sequence = circrna_sequence or "AUCCAAAAGCGGGGUAUUUG"  # Default

        self.grid_size = self.config.grid_size
        self.step = 0

        # Initialize agents
        self.cells: List[ImmuneCell] = []
        self.circrna_agents: List[CircRNAAgent] = []
        self.cytokines = CytokineField(self.grid_size)

        self._init_cells()

        # Metrics
        self.metrics = {
            'tumor_count': [],
            't_cell_activation': [],
            'cytokine_levels': [],
            'circrna_active': [],
        }

    def _init_cells(self):
        """Initialize cell populations."""
        # Dendritic cells
        for _ in range(self.config.n_dc):
            pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            self.cells.append(ImmuneCell(CellType.DC, pos))

        # T cells
        for _ in range(self.config.n_t_cell):
            pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            self.cells.append(ImmuneCell(CellType.T_CELL, pos))

        # Tumor cells
        for _ in range(self.config.n_tumor):
            pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            self.cells.append(ImmuneCell(CellType.TUMOR, pos))

        # circRNA agents
        for _ in range(self.config.n_circrna):
            pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            self.circrna_agents.append(CircRNAAgent(self.sequence, pos, self.config))

    def step_simulation(self) -> Dict:
        """Run one simulation step."""
        self.step += 1

        # Move cells
        for cell in self.cells:
            if cell.state != CellState.DEAD:
                cell.move(self.grid_size)

        # circRNA capture by DCs
        dc_cells = [c for c in self.cells if c.cell_type == CellType.DC and c.state != CellState.DEAD]

        for circrna in self.circrna_agents:
            if circrna.active and not circrna.captured:
                for dc in dc_cells:
                    dist = abs(circrna.position[0] - dc.position[0]) + abs(circrna.position[1] - dc.position[1])

                    if dist < 5 and np.random.random() < self.config.dc_circrna_capture:
                        circrna.captured = True
                        dc.state = CellState.ACTIVATED
                        dc.activation_level += circrna.immunogenicity

                        # Release cytokines
                        self.cytokines.add_cytokine(dc.position, 'IL6', circrna.immunogenicity * 50)
                        self.cytokines.add_cytokine(dc.position, 'IFN', circrna.immunogenicity * 30)
                        break

        # T cell activation
        t_cells = [c for c in self.cells if c.cell_type == CellType.T_CELL and c.state != CellState.DEAD]
        activated_dc = [c for c in dc_cells if c.state == CellState.ACTIVATED]

        for t_cell in t_cells:
            if t_cell.state == CellState.RESTING:
                for dc in activated_dc:
                    dist = abs(t_cell.position[0] - dc.position[0]) + abs(t_cell.position[1] - dc.position[1])

                    if dist < 10 and np.random.random() < self.config.t_cell_activation:
                        t_cell.state = CellState.ACTIVATED
                        t_cell.activation_level = dc.activation_level
                        break

        # Tumor killing
        tumor_cells = [c for c in self.cells if c.cell_type == CellType.TUMOR and c.state != CellState.DEAD]
        activated_t = [c for c in t_cells if c.state == CellType.ACTIVATED]

        for tumor in tumor_cells:
            for t in activated_t:
                dist = abs(tumor.position[0] - t.position[0]) + abs(tumor.position[1] - t.position[1])

                if dist < 3 and np.random.random() < self.config.tumor_killing * t.activation_level:
                    tumor.state = CellState.DYING

                    # Cytokine release
                    self.cytokines.add_cytokine(tumor.position, 'TNF', 20)
                    break

        # Cytokine diffusion
        self.cytokines.diffuse(self.config.cytokine_decay)

        # circRNA persistence decay
        for circrna in self.circrna_agents:
            if circrna.active:
                circrna.persistence -= 0.01
                if circrna.persistence < 0.1:
                    circrna.active = False

        # Collect metrics
        metrics = self._collect_metrics()

        return metrics

    def _collect_metrics(self) -> Dict:
        """Collect simulation metrics."""
        tumor_alive = len([c for c in self.cells if c.cell_type == CellType.TUMOR and c.state != CellState.DEAD])
        t_activated = len([c for c in self.cells if c.cell_type == CellType.T_CELL and c.state == CellType.ACTIVATED])
        circrna_active = len([c for c in self.circrna_agents if c.active])

        cytokine_avg = {
            'IL6': np.mean(self.cytokines.il6),
            'TNF': np.mean(self.cytokines.tnf),
            'IFN': np.mean(self.cytokines.ifn),
        }

        self.metrics['tumor_count'].append(tumor_alive)
        self.metrics['t_cell_activation'].append(t_activated)
        self.metrics['cytokine_levels'].append(cytokine_avg)
        self.metrics['circrna_active'].append(circrna_active)

        return {
            'step': self.step,
            'tumor_alive': tumor_alive,
            't_cell_activated': t_activated,
            'circrna_active': circrna_active,
            'cytokines': cytokine_avg,
            'tumor_kill_rate': 1 - tumor_alive / max(self.config.n_tumor, 1),
        }

    def run_simulation(self, max_steps: int = None) -> Dict:
        """Run full simulation."""
        max_steps = max_steps or self.config.max_steps

        print(f"Running ABM simulation for {max_steps} steps...")

        for step in range(max_steps):
            if step % 50 == 0:
                print(f"  Step {step}/{max_steps}")

            metrics = self.step_simulation()

            # Early termination if tumor eliminated
            if metrics['tumor_alive'] < 10:
                print(f"  Tumor eliminated at step {step}")
                break

        final_metrics = {
            'total_steps': self.step,
            'final_tumor_count': self.metrics['tumor_count'][-1],
            'max_t_cell_activation': max(self.metrics['t_cell_activation']),
            'peak_cytokines': {
                'IL6': max(m['IL6'] for m in self.metrics['cytokine_levels']),
                'TNF': max(m['TNF'] for m in self.metrics['cytokine_levels']),
                'IFN': max(m['IFN'] for m in self.metrics['cytokine_levels']),
            },
            'tumor_kill_rate': 1 - self.metrics['tumor_count'][-1] / self.config.n_tumor,
        }

        return final_metrics


def simulate_circrna_response(sequence: str, n_steps: int = 100) -> Dict:
    """Quick simulation of circRNA immune response."""
    config = AgentConfig(max_steps=n_steps)
    abm = ImmuneABM(config, circrna_sequence=sequence)
    return abm.run_simulation()