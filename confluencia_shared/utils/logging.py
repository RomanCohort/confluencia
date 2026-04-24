"""
Centralized logging configuration for Confluencia.

Provides a consistent logging setup across all modules with support for:
- Console and file output
- Configurable log levels
- JSON structured logging for production
- Simple function call tracing for debugging

Usage:
    from confluencia_shared.utils.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Starting training...")
    logger.debug(f"Training samples: {n_samples}")
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Module-level logger cache
_loggers: dict[str, logging.Logger] = {}

# Default log format
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str | Path] = None,
    console: bool = True,
) -> logging.Logger:
    """Get or create a logger with consistent formatting.

    Args:
        name: Logger name (typically __name__).
        level: Logging level (default: INFO).
        log_file: Optional file path for log output.
        console: Whether to output to console (default: True).

    Returns:
        Configured logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # Remove any existing handlers

    formatter = logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    _loggers[name] = logger
    return logger


def configure_root_logger(
    level: int = logging.INFO,
    log_dir: Optional[str | Path] = None,
    log_file_name: str = "confluencia.log",
) -> None:
    """Configure the root logger for the entire application.

    Call this once at application startup.

    Args:
        level: Logging level (default: INFO).
        log_dir: Directory for log files. If None, logs only to console.
        log_file_name: Name of the log file (default: "confluencia.log").
    """
    root_logger = logging.getLogger("confluencia")
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    formatter = logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)

    # File handler (if log_dir specified)
    if log_dir:
        log_path = Path(log_dir) / log_file_name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)


def log_function_call(logger: logging.Logger, func_name: str, **kwargs) -> None:
    """Log a function call with its parameters.

    Args:
        logger: Logger instance.
        func_name: Name of the function being called.
        **kwargs: Function parameters to log.
    """
    params = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    logger.debug(f"{func_name}({params})")


def log_training_start(
    logger: logging.Logger,
    module: str,
    n_samples: int,
    n_features: int,
    model_type: str,
) -> None:
    """Log training start with key parameters.

    Args:
        logger: Logger instance.
        module: Module name (e.g., "epitope", "drug").
        n_samples: Number of training samples.
        n_features: Number of features.
        model_type: Type of model being trained.
    """
    logger.info(
        f"[{module}] Training started: n_samples={n_samples}, "
        f"n_features={n_features}, model={model_type}"
    )


def log_training_complete(
    logger: logging.Logger,
    module: str,
    metrics: dict,
    duration_seconds: Optional[float] = None,
) -> None:
    """Log training completion with metrics.

    Args:
        logger: Logger instance.
        module: Module name.
        metrics: Dict of metric names to values.
        duration_seconds: Optional training duration.
    """
    metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
    msg = f"[{module}] Training complete: {metrics_str}"
    if duration_seconds is not None:
        msg += f", duration={duration_seconds:.2f}s"
    logger.info(msg)


def log_prediction(logger: logging.Logger, module: str, n_samples: int, model_type: str) -> None:
    """Log prediction operation.

    Args:
        logger: Logger instance.
        module: Module name.
        n_samples: Number of samples to predict.
        model_type: Type of model used.
    """
    logger.info(f"[{module}] Predicting: n_samples={n_samples}, model={model_type}")


__all__ = [
    "get_logger",
    "configure_root_logger",
    "log_function_call",
    "log_training_start",
    "log_training_complete",
    "log_prediction",
]
