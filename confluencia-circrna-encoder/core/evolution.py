"""
evolution.py — Evolutionary optimization for circRNA sequences.

Adapted from drug 2.0's evolution.py for circRNA context.
"""

from __future__ import annotations

import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary optimization."""

    # Population
    population_size: int = 50
    elite_size: int = 10

    # Evolution
    n_generations: int = 100
    mutation_rate: float = 0.05
    crossover_rate: float = 0.7

    # Selection
    selection_pressure: float = 2.0
    diversity_penalty: float = 0.1

    # Targets
    target_immunogenicity: float = 0.6
    target_stability: float = 0.7

    # Constraints
    min_length: int = 100
    max_length: int = 500
    gc_min: float = 0.35
    gc_max: float = 0.65


class CircRNAIndividual:
    """Individual circRNA sequence in evolution."""

    def __init__(self, sequence: str):
        self.sequence = sequence
        self.fitness = 0.0
        self.age = 0

    def evaluate(self) -> float:
        """Evaluate fitness."""
        from .innate_immune import quick_predict
        from .dose_tox import quick_dose_predict

        # Immunogenicity score
        immune = quick_predict(self.sequence)
        imm_score = immune['overall_score']

        # Therapeutic window
        dose = quick_dose_predict(self.sequence, dose=100)
        window = dose['therapeutic_window']

        # Combined fitness
        self.fitness = imm_score * 0.5 + window * 0.5

        return self.fitness


class CircRNAEvolution:
    """
    Evolutionary optimization for circRNA sequences.

    Methods:
    - Population initialization
    - Fitness evaluation
    - Selection (tournament, roulette)
    - Crossover (sequence blending)
    - Mutation (point, insert, delete)
    """

    NUCS = ['A', 'U', 'G', 'C']

    def __init__(self, config: Optional[EvolutionConfig] = None):
        self.config = config or EvolutionConfig()
        self.population: List[CircRNAIndividual] = []
        self.history: List[Dict] = []

    def initialize_population(self, seed_sequences: List[str] = None):
        """Initialize population."""
        from .generative import CircRNAGenerator

        generator = CircRNAGenerator()

        self.population = []

        if seed_sequences:
            # Add seed sequences
            for seq in seed_sequences[:self.config.elite_size]:
                self.population.append(CircRNAIndividual(seq))

        # Fill rest with random sequences
        while len(self.population) < self.config.population_size:
            seq = generator.generate_random(
                np.random.randint(self.config.min_length, self.config.max_length)
            )
            self.population.append(CircRNAIndividual(seq))

    def evaluate_population(self) -> Dict:
        """Evaluate all individuals."""
        fitnesses = []

        for individual in self.population:
            fitness = individual.evaluate()
            fitnesses.append(fitness)

        return {
            'mean_fitness': np.mean(fitnesses),
            'max_fitness': np.max(fitnesses),
            'min_fitness': np.min(fitnesses),
            'std_fitness': np.std(fitnesses),
        }

    def select_parents(self) -> List[CircRNAIndividual]:
        """Select parents via tournament selection."""
        parents = []

        # Elite selection
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        parents.extend(sorted_pop[:self.config.elite_size])

        # Tournament selection for rest
        n_parents = int(self.config.population_size * 0.5) - self.config.elite_size

        for _ in range(n_parents):
            candidates = random.sample(self.population, min(5, len(self.population)))
            winner = max(candidates, key=lambda x: x.fitness)
            parents.append(winner)

        return parents

    def crossover(self, parent1: CircRNAIndividual, parent2: CircRNAIndividual) -> Tuple[str, str]:
        """Crossover two sequences."""
        seq1 = parent1.sequence
        seq2 = parent2.sequence

        if random.random() > self.config.crossover_rate:
            return seq1, seq2

        # Single-point crossover
        point = random.randint(1, min(len(seq1), len(seq2)) - 1)

        child1 = seq1[:point] + seq2[point:]
        child2 = seq2[:point] + seq1[point:]

        return child1, child2

    def mutate(self, sequence: str) -> str:
        """Mutate sequence."""
        seq = list(sequence)

        for i in range(len(seq)):
            if random.random() < self.config.mutation_rate:
                # Point mutation
                choices = [n for n in self.NUCS if n != seq[i]]
                seq[i] = random.choice(choices)

        # Occasionally insert/delete
        if random.random() < 0.02:
            pos = random.randint(0, len(seq) - 1)
            if random.random() < 0.5 and len(seq) > self.config.min_length:
                # Delete
                seq.pop(pos)
            else:
                # Insert
                seq.insert(pos, random.choice(self.NUCS))

        return ''.join(seq)

    def evolve_generation(self) -> Dict:
        """Evolve one generation."""
        # Evaluate
        stats = self.evaluate_population()

        # Select parents
        parents = self.select_parents()

        # Create offspring
        offspring = []

        # Elite passes through
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        for elite in sorted_pop[:self.config.elite_size]:
            offspring.append(CircRNAIndividual(elite.sequence))

        # Breed new individuals
        while len(offspring) < self.config.population_size:
            p1, p2 = random.sample(parents, 2)
            child1, child2 = self.crossover(p1, p2)

            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            offspring.append(CircRNAIndividual(child1))
            offspring.append(CircRNAIndividual(child2))

        # Trim to population size
        offspring = offspring[:self.config.population_size]

        # Age population
        for ind in offspring:
            ind.age += 1

        self.population = offspring

        # Record history
        self.history.append(stats)

        return stats

    def run_evolution(self, n_generations: int = None) -> Dict:
        """Run full evolution."""
        n_generations = n_generations or self.config.n_generations

        print(f"Running evolution for {n_generations} generations...")

        for gen in range(n_generations):
            if gen % 10 == 0:
                print(f"  Generation {gen}: mean_fitness={np.mean([i.fitness for i in self.population]):.4f}")

            stats = self.evolve_generation()

            # Early termination if converged
            if stats['max_fitness'] > 0.9 and stats['std_fitness'] < 0.05:
                print(f"  Converged at generation {gen}")
                break

        # Get best individual
        best = max(self.population, key=lambda x: x.fitness)

        return {
            'best_sequence': best.sequence,
            'best_fitness': best.fitness,
            'generations': len(self.history),
            'final_stats': self.history[-1],
            'convergence': stats['std_fitness'] < 0.05,
        }

    def get_top_sequences(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top n sequences."""
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)

        return [(ind.sequence, ind.fitness) for ind in sorted_pop[:n]]


def evolve_sequence(
    seed_sequence: str = None,
    n_generations: int = 50,
    target_fitness: float = 0.7,
) -> Tuple[str, float]:
    """
    Quick evolutionary optimization.

    Args:
        seed_sequence: Starting sequence (optional)
        n_generations: Number of generations
        target_fitness: Target fitness

    Returns:
        Optimized sequence, fitness
    """
    config = EvolutionConfig(n_generations=n_generations, target_immunogenicity=target_fitness)
    evolution = CircRNAEvolution(config)

    seed_sequences = [seed_sequence] if seed_sequence else None
    evolution.initialize_population(seed_sequences)

    result = evolution.run_evolution()

    return result['best_sequence'], result['best_fitness']


def optimize_population(
    n_sequences: int = 20,
    n_generations: int = 100,
) -> List[Tuple[str, float]]:
    """Optimize population of sequences."""
    config = EvolutionConfig(
        population_size=50,
        n_generations=n_generations,
    )
    evolution = CircRNAEvolution(config)
    evolution.initialize_population()

    evolution.run_evolution()

    return evolution.get_top_sequences(n_sequences)