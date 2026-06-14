"""Confluencia 表位预测桥接 (Epitope Prediction Bridge)

懒加载 Confluencia MHC结合预测（ESM2+Mamba MOE）。
"""
from typing import Dict, Any, Optional
from ..event_bus import EventBus
from ..events import CONFLUENCIA_EPITOPE_PREDICTION
from ..config import ConfluenciaConfig


class EpitopePredictionBridge:
    """Confluencia 表位预测桥接"""

    def __init__(self, config: Optional[ConfluenciaConfig] = None, event_bus: Optional[EventBus] = None):
        self.config = config or ConfluenciaConfig()
        self.bus = event_bus
        self._predictor = None
        self._loaded = False

        if self.bus:
            self.bus.subscribe(CONFLUENCIA_EPITOPE_PREDICTION, self._on_epitope, priority=0, name="epitope_bridge")

    def _lazy_load(self):
        if self._loaded:
            return self._predictor is not None
        if not self.config.enabled or not self.config.confluencia_path:
            self._loaded = True
            return False
        try:
            import importlib.util
            import os
            path = os.path.join(self.config.confluencia_path, "confluencia_joint")
            spec = importlib.util.spec_from_file_location(
                "confluencia_joint",
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

    def predict_epitope(self, peptide_sequence: str, mhc_allele: str = "HLA-A*02:01") -> Dict[str, Any]:
        """预测MHC结合"""
        if not self._lazy_load():
            return {"binding_score": 0.5, "confidence": 0.0, "source": "fallback"}
        try:
            result = self._predictor.predict_epitope(peptide_sequence, mhc_allele)
            return {
                "binding_score": result.get("score", 0.5),
                "confidence": result.get("confidence", 0.0),
                "source": "confluencia",
            }
        except Exception:
            return {"binding_score": 0.5, "confidence": 0.0, "source": "fallback"}

    def _on_epitope(self, event) -> Dict[str, Any]:
        data = event.data
        return self.predict_epitope(
            peptide_sequence=data.get("peptide", ""),
            mhc_allele=data.get("mhc_allele", "HLA-A*02:01"),
        )