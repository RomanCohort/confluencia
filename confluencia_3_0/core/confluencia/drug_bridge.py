"""Confluencia 药物预测桥接 (Drug Prediction Bridge)

懒加载 Confluencia MOE 集成预测器，fail-silent 模式。
"""
from typing import Dict, Any, Optional
from ..event_bus import EventBus
from ..events import CONFLUENCIA_DRUG_PREDICTION
from ..config import ConfluenciaConfig


class DrugPredictionBridge:
    """Confluencia 药物预测桥接

    调用 Confluencia 的 MOE 集成预测器（Ridge R2=0.984），
    获取药物对 TNBC 的预测疗效评分。
    """

    def __init__(self, config: Optional[ConfluenciaConfig] = None, event_bus: Optional[EventBus] = None):
        self.config = config or ConfluenciaConfig()
        self.bus = event_bus
        self._predictor = None
        self._loaded = False

        if self.bus:
            self.bus.subscribe(CONFLUENCIA_DRUG_PREDICTION, self._on_drug_prediction, priority=0, name="drug_bridge")

    def _lazy_load(self):
        """懒加载 Confluencia 药物预测模块"""
        if self._loaded:
            return self._predictor is not None

        if not self.config.enabled or not self.config.confluencia_path:
            self._loaded = True
            return False

        try:
            import importlib.util
            import os
            path = os.path.join(self.config.confluencia_path, "confluencia-2.0-drug")
            spec = importlib.util.spec_from_file_location(
                "confluencia_drug",
                os.path.join(path, "__init__.py")
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self._predictor = module
                self._loaded = True
                return True
        except Exception:
            pass

        self._loaded = True
        return False

    def predict(self, drug_smiles: str, target_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """预测药物疗效

        Args:
            drug_smiles: 药物SMILES结构式
            target_info: 靶点信息

        Returns:
            {"prediction_score": float, "confidence": float}
        """
        if not self._lazy_load():
            # Confluencia不可用时返回默认值
            return {"prediction_score": 0.5, "confidence": 0.0, "source": "fallback"}

        try:
            # 调用Confluencia预测器
            result = self._predictor.predict(drug_smiles, **(target_info or {}))
            return {
                "prediction_score": result.get("score", 0.5),
                "confidence": result.get("confidence", 0.0),
                "source": "confluencia",
            }
        except Exception:
            return {"prediction_score": 0.5, "confidence": 0.0, "source": "fallback"}

    def _on_drug_prediction(self, event) -> Dict[str, Any]:
        data = event.data
        smiles = data.get("smiles", "")
        result = self.predict(smiles, data.get("target_info"))
        return result