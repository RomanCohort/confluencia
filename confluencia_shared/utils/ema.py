"""Simple parameter-level EMA (Mean Teacher) utilities for PyTorch models.

Provides:
- clone_model_for_ema(model): creates a detached copy of model with same architecture and parameters
- update_ema(ema_model, model, decay): in-place update of ema_model params: ema = decay*ema + (1-decay)*model
- set_requires_grad(model, requires_grad): helper

This module keeps minimal dependencies (torch) and is safe to import only when torch available.
"""
from typing import Iterator

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - if torch not installed
    torch = None
    nn = None


def clone_model_for_ema(model: "nn.Module") -> "nn.Module":
    """Create a copy of `model` suitable as EMA teacher.

    The returned model shares the same architecture and has parameters copied from
    the source. Gradients are disabled for the EMA model.
    """
    if torch is None:
        raise RuntimeError("torch is required for EMA utilities")
    # Create a new instance of the model's class if possible
    try:
        ema = type(model)()
    except Exception:
        # Fallback: deep copy
        import copy

        ema = copy.deepcopy(model)
        set_requires_grad(ema, False)
        return ema

    # Try to load state_dict
    try:
        ema.load_state_dict(model.state_dict())
    except Exception:
        # fallback to deepcopy
        import copy

        ema = copy.deepcopy(model)
    set_requires_grad(ema, False)
    # Ensure in eval mode
    ema.eval()
    return ema


def update_ema(ema_model: "nn.Module", model: "nn.Module", decay: float) -> None:
    """In-place update EMA model parameters with student model parameters.

    For each parameter/ buffer:
        ema_param = decay * ema_param + (1 - decay) * param

    Works for both parameters and buffers (e.g., batchnorm stats).
    """
    if torch is None:
        raise RuntimeError("torch is required for EMA utilities")
    if decay < 0.0 or decay > 1.0:
        raise ValueError("decay must be in [0, 1]")

    # Update parameters
    with torch.no_grad():
        for ema_v, model_v in _named_params_buffers(ema_model, model):
            if ema_v.data.dtype != model_v.data.dtype:
                model_data = model_v.data.type(ema_v.data.dtype)
            else:
                model_data = model_v.data
            ema_v.data.mul_(decay).add_(model_data, alpha=(1.0 - decay))


def _named_params_buffers(ema_model: "nn.Module", model: "nn.Module") -> Iterator[tuple]:
    # zip parameters by names to be robust
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()
    for k in ema_state.keys():
        if k in model_state:
            yield ema_state[k], model_state[k]


def set_requires_grad(model: "nn.Module", requires_grad: bool) -> None:
    if nn is None:
        return
    for p in model.parameters():
        p.requires_grad = requires_grad
