"""Confluencia Plugin System — extensible algorithm platform.

Users can register custom models, encoders, PK solvers, and scoring
dimensions to extend or replace any stage of the pipeline.

Quick start:
    import confluencia_cli.plugins as cf

    # Register a custom model
    @cf.register_model("my_xgboost")
    def create_xgboost(**kwargs):
        from xgboost import XGBRegressor
        return XGBRegressor(**kwargs)

    # Register a custom encoder
    @cf.register_encoder("my_esm2")
    def encode_esm2(sequence: str) -> list[float]:
        from esm import pretrained
        model, alphabet = pretrained.esm2_t6_8M_UR50D()
        # ... return embedding vector
        return embedding.tolist()

    # Register a custom PK solver
    @cf.register_pk_solver("my_two_compartment")
    def solve_two_compartment(dose, freq, params, horizon):
        from scipy.integrate import odeint
        # ... return dict of lists
        return result

    # Add a new evaluation dimension
    cf.register_dimension("manufacturability", weight=0.15,
                          scorer=my_manufacturability_scorer)

    # Modify scoring weights
    cf.set_weights(clinical=0.25, binding=0.20, kinetics=0.20,
                   gene_signature=0.15, circrna=0.10, manufacturability=0.10)

    # List all registered components
    cf.list_registry()
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Protocols — interfaces that registered components must satisfy
# ---------------------------------------------------------------------------

@runtime_checkable
class ModelCreator(Protocol):
    """Protocol for model factory functions."""
    def __call__(self, **kwargs: Any) -> Any:
        ...


@runtime_checkable
class SequenceEncoder(Protocol):
    """Protocol for sequence encoding functions."""
    def __call__(self, sequence: str, **kwargs: Any) -> List[float]:
        ...


@runtime_checkable
class PKSolver(Protocol):
    """Protocol for PK simulation functions."""
    def __call__(self, dose: float, freq: float, params: Dict[str, float],
                 horizon: float) -> Dict[str, List[float]]:
        ...


@runtime_checkable
class DimensionScorer(Protocol):
    """Protocol for evaluation dimension scoring functions."""
    def __call__(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        ...


# ---------------------------------------------------------------------------
# Registry — singleton that holds all registered components
# ---------------------------------------------------------------------------

class _Registry:
    """Global registry for pluggable components."""

    def __init__(self):
        self._models: Dict[str, Callable] = {}
        self._encoders: Dict[str, Callable] = {}
        self._pk_solvers: Dict[str, Callable] = {}
        self._dimensions: Dict[str, Dict[str, Any]] = {}
        self._weights: Dict[str, float] = {
            "clinical": 0.25,
            "binding": 0.25,
            "kinetics": 0.20,
            "gene_signature": 0.15,
            "circrna": 0.15,
        }

    # ---- Model Registration ----

    def register_model(self, name: str, creator: Optional[Callable] = None):
        """Register a custom model type.

        Usage:
            @cf.register_model("my_xgb")
            def create_xgb(**kwargs):
                return XGBRegressor(**kwargs)

            # Or:
            cf.register_model("my_xgb", lambda **kw: XGBRegressor(**kw))
        """
        def decorator(fn: Callable) -> Callable:
            self._models[name] = fn
            return fn
        if creator is not None:
            self._models[name] = creator
            return creator
        return decorator

    def get_model(self, name: str) -> Optional[Callable]:
        """Get a registered model creator by name."""
        return self._models.get(name)

    def list_models(self) -> List[str]:
        """List all registered model types."""
        return sorted(self._models.keys())

    # ---- Encoder Registration ----

    def register_encoder(self, name: str, encoder: Optional[Callable] = None):
        """Register a custom sequence encoder.

        Usage:
            @cf.register_encoder("esm2_small")
            def encode_esm2(sequence: str, **kwargs) -> list[float]:
                ...

            # Then use in pipeline:
            bridge.mamba3_encode("SLYNTVATL", encoder="esm2_small")
        """
        def decorator(fn: Callable) -> Callable:
            self._encoders[name] = fn
            return fn
        if encoder is not None:
            self._encoders[name] = encoder
            return encoder
        return decorator

    def get_encoder(self, name: str) -> Optional[Callable]:
        """Get a registered encoder by name."""
        return self._encoders.get(name)

    def list_encoders(self) -> List[str]:
        """List all registered encoder types."""
        return sorted(self._encoders.keys())

    # ---- PK Solver Registration ----

    def register_pk_solver(self, name: str, solver: Optional[Callable] = None):
        """Register a custom PK solver.

        Usage:
            @cf.register_pk_solver("two_compartment")
            def solve_two_compartment(dose, freq, params, horizon):
                from scipy.integrate import odeint
                ...
                return {"time_h": [...], "concentration": [...]}
        """
        def decorator(fn: Callable) -> Callable:
            self._pk_solvers[name] = fn
            return fn
        if solver is not None:
            self._pk_solvers[name] = solver
            return solver
        return decorator

    def get_pk_solver(self, name: str) -> Optional[Callable]:
        """Get a registered PK solver by name."""
        return self._pk_solvers.get(name)

    def list_pk_solvers(self) -> List[str]:
        """List all registered PK solvers."""
        return sorted(self._pk_solvers.keys())

    # ---- Evaluation Dimension Registration ----

    def register_dimension(self, name: str, weight: float = 0.0,
                           scorer: Optional[Callable] = None,
                           description: str = ""):
        """Register a new evaluation dimension.

        Usage:
            cf.register_dimension(
                "manufacturability",
                weight=0.15,
                scorer=my_manufacturability_scorer,
                description="Ease of large-scale circRNA production"
            )
        """
        self._dimensions[name] = {
            "weight": weight,
            "scorer": scorer,
            "description": description,
        }
        if weight > 0:
            self._weights[name] = weight

    def get_dimension(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a registered dimension by name."""
        return self._dimensions.get(name)

    def list_dimensions(self) -> List[str]:
        """List all registered evaluation dimensions."""
        return sorted(self._dimensions.keys())

    # ---- Scoring Weights ----

    def set_weights(self, **weights: float):
        """Set scoring weights for evaluation dimensions.

        Usage:
            cf.set_weights(clinical=0.30, binding=0.25, kinetics=0.15,
                           gene_signature=0.15, circrna=0.15)
        """
        for dim, w in weights.items():
            self._weights[dim] = w

    def get_weights(self) -> Dict[str, float]:
        """Get current scoring weights."""
        return dict(self._weights)

    def normalize_weights(self) -> Dict[str, float]:
        """Get weights normalized to sum to 1.0."""
        total = sum(self._weights.values())
        if total == 0:
            return {k: 0.0 for k in self._weights}
        return {k: v / total for k, v in self._weights.items()}

    # ---- Introspection ----

    def summary(self) -> Dict[str, Any]:
        """Get a summary of all registered components."""
        return {
            "models": self.list_models(),
            "encoders": self.list_encoders(),
            "pk_solvers": self.list_pk_solvers(),
            "dimensions": self.list_dimensions(),
            "weights": self.normalize_weights(),
        }


# Global singleton
_registry = _Registry()


# ---------------------------------------------------------------------------
# Public API — decorator-style registration
# ---------------------------------------------------------------------------

def register_model(name: str, creator: Optional[Callable] = None):
    """Register a custom model type. Use as decorator or function call."""
    return _registry.register_model(name, creator)


def register_encoder(name: str, encoder: Optional[Callable] = None):
    """Register a custom sequence encoder. Use as decorator or function call."""
    return _registry.register_encoder(name, encoder)


def register_pk_solver(name: str, solver: Optional[Callable] = None):
    """Register a custom PK solver. Use as decorator or function call."""
    return _registry.register_pk_solver(name, solver)


def register_dimension(name: str, weight: float = 0.0,
                       scorer: Optional[Callable] = None,
                       description: str = ""):
    """Register a new evaluation dimension."""
    return _registry.register_dimension(name, weight, scorer, description)


def set_weights(**weights: float):
    """Set scoring weights for evaluation dimensions."""
    return _registry.set_weights(**weights)


def get_weights() -> Dict[str, float]:
    """Get current scoring weights (normalized)."""
    return _registry.normalize_weights()


def list_registry() -> Dict[str, Any]:
    """List all registered components."""
    return _registry.summary()


# Module-level access to registry for internal use
def _get_registry() -> _Registry:
    """Get the global registry instance (internal use)."""
    return _registry
