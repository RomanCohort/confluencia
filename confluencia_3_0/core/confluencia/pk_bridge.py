"""Confluencia PK模型桥接 (PK Model Bridge)

优先使用内化 RNACTM，降级到 2.0-drug 懒加载桥接。
"""
from typing import Dict, Any, Optional
from ..event_bus import EventBus
from ..events import CONFLUENCIA_PK_SIMULATION
from ..config import ConfluenciaConfig


class PKModelBridge:
    """Confluencia PK模型桥接

    优先使用 3.0 内化的 RNACTM 六室PK模型，
    降级到 2.0-drug 的 ctm.py 懒加载。
    """

    def __init__(self, config: Optional[ConfluenciaConfig] = None, event_bus: Optional[EventBus] = None):
        self.config = config or ConfluenciaConfig()
        self.bus = event_bus
        self._pk_model = None
        self._loaded = False

        if self.bus:
            self.bus.subscribe(CONFLUENCIA_PK_SIMULATION, self._on_pk_simulation, priority=0, name="pk_bridge")

    def simulate_pk(self, dose: float, circrna_sequence: str = "", **kwargs) -> Dict[str, Any]:
        """模拟circRNA PK

        Args:
            dose: 给药剂量
            circrna_sequence: circRNA序列
            **kwargs: 传递给 infer_rna_ctm_params 的参数

        Returns:
            {"concentration_time": [...], "auc": float, "half_life": float, "source": str, "available": bool}
        """
        # 优先使用内化 RNACTM
        try:
            from ..pk.rnactm import infer_rna_ctm_params, simulate_rna_ctm, summarize_rna_ctm_curve
            params = infer_rna_ctm_params(
                modification=kwargs.get("modification", "none"),
                delivery_vector=kwargs.get("delivery_vector", "LNP_standard"),
                route=kwargs.get("route", "IV"),
                ires_score=kwargs.get("ires_score", 0.5),
                gc_content=kwargs.get("gc_content", 0.5),
                struct_stability=kwargs.get("struct_stability", 0.5),
                innate_immune_score=kwargs.get("innate_immune_score", 0.0),
            )
            curve = simulate_rna_ctm(
                dose=dose,
                freq=kwargs.get("freq", 1.0),
                params=params,
                horizon=kwargs.get("horizon", 168),
                dt=kwargs.get("dt", 1.0),
            )
            summary = summarize_rna_ctm_curve(curve)
            return {
                "concentration_time": curve.to_dict("records") if hasattr(curve, 'to_dict') else [],
                "auc": summary.get("rna_ctm_auc_efficacy", 0.0),
                "half_life": summary.get("rna_ctm_rna_half_life_h", 12.0),
                "peak_protein": summary.get("rna_ctm_peak_protein", 0.0),
                "bioavailability": summary.get("rna_ctm_bioavailability_frac", 0.0),
                "source": "internal",
                "available": True,
            }
        except Exception:
            pass

        # 降级到 2.0 懒加载
        if not self._lazy_load():
            return {
                "concentration_time": [],
                "auc": 0.0,
                "half_life": 12.0,
                "source": "fallback",
                "available": False,
            }

        try:
            # 使用正确的 ctm.py API
            params = self._pk_model.infer_rna_ctm_params(
                modification=kwargs.get("modification", "none"),
                delivery_vector=kwargs.get("delivery_vector", "LNP_standard"),
                route=kwargs.get("route", "IV"),
            )
            curve = self._pk_model.simulate_rna_ctm(
                dose=dose,
                freq=kwargs.get("freq", 1.0),
                params=params,
                horizon=kwargs.get("horizon", 168),
            )
            summary = self._pk_model.summarize_rna_ctm_curve(curve)
            return {
                "concentration_time": curve.to_dict("records") if hasattr(curve, 'to_dict') else [],
                "auc": summary.get("rna_ctm_auc_efficacy", 0.0),
                "half_life": summary.get("rna_ctm_rna_half_life_h", 12.0),
                "peak_protein": summary.get("rna_ctm_peak_protein", 0.0),
                "source": "confluencia",
                "available": True,
            }
        except Exception:
            return {
                "concentration_time": [],
                "auc": 0.0,
                "half_life": 12.0,
                "source": "fallback",
                "available": False,
            }

    def _lazy_load(self):
        """懒加载 2.0-drug ctm.py"""
        if self._loaded:
            return self._pk_model is not None

        if not self.config.enabled or not self.config.confluencia_path:
            self._loaded = True
            return False

        try:
            import importlib.util
            import os
            ctm_path = os.path.join(
                self.config.confluencia_path,
                "confluencia-2.0-drug", "core", "ctm.py"
            )
            if os.path.exists(ctm_path):
                spec = importlib.util.spec_from_file_location("ctm", ctm_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self._pk_model = module
                    self._loaded = True
                    return True
        except Exception:
            pass

        self._loaded = True
        return False

    def _on_pk_simulation(self, event) -> Dict[str, Any]:
        data = event.data
        result = self.simulate_pk(
            dose=data.get("dose", 0.0),
            circrna_sequence=data.get("sequence", ""),
        )
        return result
