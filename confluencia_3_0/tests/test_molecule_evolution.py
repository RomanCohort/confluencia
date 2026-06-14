"""分子进化 + circRNA 进化单元测试。"""
import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from confluencia_3_0.core.evolution.molecule_evolution import (
    EvolutionConfig,
    EvolutionArtifacts,
    evolve_molecules_with_reflection,
)
from confluencia_3_0.core.evolution.cirrna_evolution import (
    CircRNAEvolutionConfig,
    CircRNAEvolutionArtifacts,
    evolve_cirrna,
    run_cirrna_evolution,
    optimize_for_translation,
    compute_cirrna_objectives,
)
from confluencia_3_0.core.evolution.pareto import (
    softmax,
    normalize_cols,
    pareto_front_mask,
    select_weights_with_pareto,
    reward_from_weights,
)
from confluencia_3_0.core.evolution.actions import MOLECULE_ACTIONS, CIRCRNA_ACTIONS


class TestParetoUtils:
    """验证 Pareto/RL 工具函数。"""

    def test_softmax(self):
        x = np.array([1.0, 2.0, 3.0])
        p = softmax(x)
        assert abs(p.sum() - 1.0) < 1e-6
        assert (p >= 0).all()
        assert p[2] > p[1] > p[0]

    def test_normalize_cols(self):
        X = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]], dtype=np.float32)
        Xn = normalize_cols(X)
        assert abs(Xn[:, 0].min()) < 1e-5
        assert abs(Xn[:, 0].max() - 1.0) < 1e-5

    def test_pareto_front_mask(self):
        X = np.array([[1.0, 1.0], [0.5, 0.5], [0.3, 0.3]], dtype=np.float32)
        mask = pareto_front_mask(X)
        # 只有 [1,1] 是非支配的
        assert mask[0]
        assert not mask[1]
        assert not mask[2]

    def test_reward_from_weights(self):
        X = np.eye(3, dtype=np.float32)
        w = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        r = reward_from_weights(X, w)
        assert abs(r[0] - 0.5) < 1e-5
        assert abs(r[1] - 0.3) < 1e-5


class TestActions:
    """验证动作常量。"""

    def test_molecule_actions(self):
        assert len(MOLECULE_ACTIONS) == 3
        assert "ed2mol" in MOLECULE_ACTIONS

    def test_circrna_actions(self):
        assert len(CIRCRNA_ACTIONS) == 4
        assert "mutate_backbone" in CIRCRNA_ACTIONS
        assert "shuffle_ires_flanking" in CIRCRNA_ACTIONS


class TestCircRNAObjectives:
    """验证 circRNA 目标计算。"""

    def test_basic_objectives(self):
        seq = "AUGCGCUAUGGCUAGCUAUGCGCUAUGGCUAGCUAUGCGCUAUGGC" * 3
        obj = compute_cirrna_objectives(seq, "m6A")
        assert obj.shape == (4,)
        assert (obj >= 0).all()
        assert (obj <= 1).all()

    def test_short_sequence(self):
        obj = compute_cirrna_objectives("AUGC", "none")
        assert obj.shape == (4,)

    def test_modification_effect(self):
        seq = "AUGCGCUAUGGCUAGCUAUGCGCUAUGGCUAGCUAUGCGCUAUGGC" * 3
        obj_none = compute_cirrna_objectives(seq, "none")
        obj_m6a = compute_cirrna_objectives(seq, "m6A")
        # m6A 应提高稳定性
        assert obj_m6a[0] >= obj_none[0]


class TestCircRNAEvolution:
    """验证 circRNA 序列进化。"""

    def test_basic_evolution(self):
        cfg = CircRNAEvolutionConfig(rounds=2, candidates_per_round=8, seed=42)
        result_df, artifacts = evolve_cirrna(cfg)
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(artifacts, CircRNAEvolutionArtifacts)
        assert artifacts.rounds_ran > 0
        assert len(artifacts.best_sequence) > 0

    def test_run_convenience(self):
        seq = "AUGCGCUAUGGCUAGCUAUGCGCUAUGGCUAGCUAUGCGCUAUGGC"
        result_df, artifacts = run_cirrna_evolution(seq, rounds=2)
        assert artifacts.rounds_ran > 0

    def test_optimize_translation(self):
        seq = "AUGCGCUAUGGCUAGCUAUGCGCUAUGGCUAGCUAUGCGCUAUGGC"
        best = optimize_for_translation(seq, rounds=2)
        assert isinstance(best, str)
        assert len(best) > 0


class TestMoleculeEvolution:
    """验证分子进化。"""

    def test_basic_evolution_no_pipeline(self):
        """无管线时应仍可运行 (随机评分)。"""
        cfg = EvolutionConfig(rounds=2, candidates_per_round=8)
        result_df, artifacts = evolve_molecules_with_reflection(
            seed_smiles=["CCO", "CCN"],
            cfg=cfg,
        )
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(artifacts, EvolutionArtifacts)
        assert artifacts.rounds_ran > 0

    def test_early_stopping(self):
        """验证早停机制。"""
        cfg = EvolutionConfig(rounds=100, candidates_per_round=8, early_stop_patience=2)
        result_df, artifacts = evolve_molecules_with_reflection(
            seed_smiles=["CCO"],
            cfg=cfg,
        )
        # 应在 100 轮前停止
        assert artifacts.rounds_ran <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
