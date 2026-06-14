"""CircRNAManager 单元测试。"""
import sys
import os
import pytest

# 确保可导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from confluencia_3_0.core.events import (
    CIRCRNA_IMMUNE_EVAL, CIRCRNA_STRUCTURE_PREDICT,
    CIRCRNA_SEQUENCE_EVOLVE, CIRCRNA_VACCINE_ASSESS,
    CIRCRNA_FOLDING_KINETICS, CIRCRNA_DRUG_RESPONSE,
)
from confluencia_3_0.core.config import Confluencia3Config, CircRNAConfig
from confluencia_3_0.core.state_schema import StateSchema
from confluencia_3_0.core.event_bus import EventBus


class TestCircRNAEvents:
    """验证 circRNA 事件常量。"""

    def test_event_values(self):
        assert CIRCRNA_IMMUNE_EVAL == "circrna_immune_eval"
        assert CIRCRNA_STRUCTURE_PREDICT == "circrna_structure_predict"
        assert CIRCRNA_SEQUENCE_EVOLVE == "circrna_sequence_evolve"
        assert CIRCRNA_VACCINE_ASSESS == "circrna_vaccine_assess"
        assert CIRCRNA_FOLDING_KINETICS == "circrna_folding_kinetics"
        assert CIRCRNA_DRUG_RESPONSE == "circrna_drug_response"


class TestCircRNAConfig:
    """验证 CircRNAConfig 配置。"""

    def test_defaults(self):
        cfg = CircRNAConfig()
        assert cfg.enabled is True
        assert cfg.immunogenicity_backend == "heuristic"
        assert cfg.mhc_backend == "local"
        assert cfg.drug_backend == "local"
        assert cfg.pk_backend == "rnactm"
        assert cfg.enable_structure_prediction is True
        assert cfg.enable_folding_kinetics is False

    def test_in_confluencia3_config(self):
        cfg = Confluencia3Config()
        assert hasattr(cfg, 'circrna')
        assert isinstance(cfg.circrna, CircRNAConfig)
        assert cfg.circrna.enabled is True


class TestCircRNAStateKeys:
    """验证 crna_* 状态键。"""

    def test_keys_registered(self):
        schema = StateSchema()
        crna_keys = schema.get_subsystem_keys("circrna")
        assert len(crna_keys) == 15
        assert "crna_immunogenicity_score" in crna_keys
        assert "crna_ips_score" in crna_keys
        assert "crna_backend_tier" in crna_keys
        assert "crna_structure_method" in crna_keys


class TestCircRNAManager:
    """验证 CircRNAManager 基本功能。"""

    def _make_mock_agent(self):
        """创建最小化的 mock agent。"""
        class MockAgent:
            def __init__(self):
                self.config = Confluencia3Config()
                self._schema = StateSchema()
                self._internal_state = self._schema.init_defaults()
                self._event_bus = EventBus()

        return MockAgent()

    def test_manager_creation(self):
        from confluencia_3_0.core.subsystem_managers import CircRNAManager
        agent = self._make_mock_agent()
        manager = CircRNAManager(agent)
        assert manager.subsystem_name == "circrna"

    def test_set_sequence(self):
        from confluencia_3_0.core.subsystem_managers import CircRNAManager
        agent = self._make_mock_agent()
        manager = CircRNAManager(agent)
        manager.set_sequence("AUGCAUGCAUGC")
        assert manager._current_sequence == "AUGCAUGCAUGC"
        # T -> U 转换
        manager.set_sequence("ATGCATGC")
        assert manager._current_sequence == "AUGCAUGC"

    def test_step_disabled(self):
        from confluencia_3_0.core.subsystem_managers import CircRNAManager
        agent = self._make_mock_agent()
        agent.config.circrna.enabled = False
        manager = CircRNAManager(agent)
        result = manager.step()
        assert result == {}

    def test_event_subscription(self):
        from confluencia_3_0.core.subsystem_managers import CircRNAManager
        agent = self._make_mock_agent()
        manager = CircRNAManager(agent)
        # EventBus 应该有 circRNA 事件订阅者
        assert CIRCRNA_IMMUNE_EVAL in agent._event_bus._subscribers
        assert CIRCRNA_STRUCTURE_PREDICT in agent._event_bus._subscribers

    def test_simulate_pk(self):
        """验证内化 RNACTM PK 模拟。"""
        from confluencia_3_0.core.subsystem_managers import CircRNAManager
        agent = self._make_mock_agent()
        manager = CircRNAManager(agent)
        result = manager.simulate_pk("AUGCGCUAUGGCUAGCUAUGCGCUAUGGCUAGCUAUGCGCUAUGGC", dose=1.0, freq=1.0)
        assert result.get("source") == "internal"
        assert result.get("available") is True
        assert "rna_ctm_auc_efficacy" in result or "auc" in result

    def test_evolve_sequence_fixed_api(self):
        """验证修复后的 evolve_sequence API（不再传不存在的参数）。"""
        from confluencia_3_0.core.subsystem_managers import CircRNAManager
        agent = self._make_mock_agent()
        manager = CircRNAManager(agent)
        result = manager.evolve_sequence(
            "AUGCGCUAUGGCUAGCUAUGCGCUAUGGCUAGCUAUGCGCUAUGGC",
            objective="stability",
            generations=2,
        )
        assert "crna_evolution_generation" in result
        assert result["crna_evolution_generation"] > 0

    def test_evolve_molecules(self):
        """验证分子进化方法。"""
        from confluencia_3_0.core.subsystem_managers import CircRNAManager
        from confluencia_3_0.core.evolution.molecule_evolution import EvolutionConfig
        agent = self._make_mock_agent()
        manager = CircRNAManager(agent)
        result = manager.evolve_molecules(
            seed_smiles=["CCO", "CCN"],
            cfg=EvolutionConfig(rounds=2, candidates_per_round=8),
        )
        assert "best_reward" in result
        assert result.get("source") == "internal"

    def test_new_event_subscription(self):
        """验证新增的 PK 和分子进化事件订阅。"""
        from confluencia_3_0.core.subsystem_managers import CircRNAManager
        agent = self._make_mock_agent()
        manager = CircRNAManager(agent)
        from confluencia_3_0.core.events import CIRCRNA_PK_SIMULATE, MOLECULE_EVOLUTION_REQUEST
        assert CIRCRNA_PK_SIMULATE in agent._event_bus._subscribers
        assert MOLECULE_EVOLUTION_REQUEST in agent._event_bus._subscribers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
