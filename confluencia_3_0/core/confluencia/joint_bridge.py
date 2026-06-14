"""Confluencia 联合评估桥接 (Joint Evaluation Bridge)

三维评分: ClinicalScore + BindingScore + KineticsScore → JointScore
"""
from typing import Dict, Any, Optional
from ..event_bus import EventBus
from ..events import CONFLUENCIA_JOINT_EVAL
from ..config import ConfluenciaConfig


class JointEvaluationBridge:
    """Confluencia 联合评估桥接"""

    def __init__(self, config: Optional[ConfluenciaConfig] = None, event_bus: Optional[EventBus] = None):
        self.config = config or ConfluenciaConfig()
        self.bus = event_bus
        self._evaluator = None
        self._loaded = False

        if self.bus:
            self.bus.subscribe(CONFLUENCIA_JOINT_EVAL, self._on_joint_eval, priority=0, name="joint_bridge")

    def _lazy_load(self):
        if self._loaded:
            return self._evaluator is not None
        if not self.config.enabled or not self.config.confluencia_path:
            self._loaded = True
            return False
        try:
            import importlib.util
            import os
            path = os.path.join(self.config.confluencia_path, "confluencia_joint", "joint_evaluator.py")
            if os.path.exists(path):
                spec = importlib.util.spec_from_file_location("joint_evaluator", path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self._evaluator = module
                    self._loaded = True
                    return True
        except Exception:
            pass
        self._loaded = True
        return False

    def evaluate(self, clinical_score: float, binding_score: float, kinetics_score: float) -> Dict[str, Any]:
        """联合评估

        Args:
            clinical_score: 临床评分 (0-1)
            binding_score: 结合评分 (0-1)
            kinetics_score: 动力学评分 (0-1)

        Returns:
            {"joint_score": float, "clinical": float, "binding": float, "kinetics": float}
        """
        weights = self.config.joint_eval_weights

        if self._lazy_load():
            try:
                result = self._evaluator.evaluate(clinical_score, binding_score, kinetics_score)
                return {
                    "joint_score": result.get("joint_score", 0.0),
                    "clinical": clinical_score,
                    "binding": binding_score,
                    "kinetics": kinetics_score,
                    "source": "confluencia",
                }
            except Exception:
                pass

        # Fallback: 加权平均
        joint = (
            clinical_score * weights.get("clinical", 0.4) +
            binding_score * weights.get("binding", 0.35) +
            kinetics_score * weights.get("kinetics", 0.25)
        )

        return {
            "joint_score": joint,
            "clinical": clinical_score,
            "binding": binding_score,
            "kinetics": kinetics_score,
            "source": "fallback",
        }

    def _on_joint_eval(self, event) -> Dict[str, Any]:
        data = event.data
        return self.evaluate(
            clinical_score=data.get("clinical_score", 0.0),
            binding_score=data.get("binding_score", 0.0),
            kinetics_score=data.get("kinetics_score", 0.0),
        )