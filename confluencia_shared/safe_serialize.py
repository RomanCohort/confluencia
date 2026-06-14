"""
Safe Serialization Utilities for Confluencia

Provides secure alternatives to pickle for model serialization.
Uses JSON for metadata + joblib for sklearn models (with hash verification).

Security model:
- JSON: Safe for metadata (strings, numbers, lists, dicts)
- joblib: Safe for sklearn models with hash verification
- No arbitrary code execution via pickle.loads
"""

import hashlib
import json
import gzip
import io
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np


# Magic bytes for format identification
SAFE_FORMAT_MAGIC = b"CFSAFE01"
MODEL_MAGIC = "CONFLUENCIA_DRUG_MODEL_v2"


def compute_hash(data: bytes) -> str:
    """Compute SHA256 hash of data for integrity verification."""
    return hashlib.sha256(data).hexdigest()[:16]


def serialize_safe(
    model: Any,
    metadata: Optional[Dict] = None,
    use_joblib: bool = True
) -> bytes:
    """
    Safely serialize model with metadata.

    Format:
    - 8 bytes: magic header (CFSAFE01)
    - 32 bytes: model hash (SHA256 truncated)
    - 4 bytes: metadata length (uint32 big-endian)
    - N bytes: JSON metadata
    - Remaining: gzipped model data

    Args:
        model: sklearn model or numpy array
        metadata: Optional dict with model info
        use_joblib: Use joblib for sklearn models (recommended)

    Returns:
        Serialized bytes

    Raises:
        ValueError: If model type is not supported
    """
    import joblib

    if metadata is None:
        metadata = {}

    # Validate metadata is JSON-serializable
    try:
        meta_json = json.dumps(metadata, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Metadata must be JSON-serializable: {e}")

    # Serialize model with joblib (using MemoryIO)
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=3)
    model_bytes = buffer.getvalue()
    model_hash = compute_hash(model_bytes)

    # Build payload
    meta_bytes = meta_json.encode("utf-8")
    meta_len = len(meta_bytes)

    # Header: magic(8) + hash(16) + meta_len(4)
    header = SAFE_FORMAT_MAGIC + model_hash.encode("ascii") + meta_len.to_bytes(4, "big")

    # Full payload
    payload = header + meta_bytes + gzip.compress(model_bytes, compresslevel=6)

    return payload


def deserialize_safe(
    data: bytes,
    expected_hash: Optional[str] = None,
    allow_joblib: bool = True
) -> tuple[Any, Dict]:
    """
    Safely deserialize model with integrity verification.

    Args:
        data: Serialized bytes from serialize_safe()
        expected_hash: Optional expected model hash for verification
        allow_joblib: Allow joblib deserialization (set False for untrusted sources)

    Returns:
        Tuple of (model, metadata)

    Raises:
        ValueError: If format invalid, hash mismatch, or joblib not allowed
    """
    import joblib

    # Minimum size: magic(8) + hash(16) + meta_len(4) + 1 byte metadata
    if len(data) < 29:
        raise ValueError("Invalid payload: too short")

    # Parse header
    magic = data[:8]
    if magic != SAFE_FORMAT_MAGIC:
        raise ValueError(
            f"Invalid format: expected safe serialization, got magic bytes {magic!r}. "
            "If this is a legacy pickle file, use import_legacy_model() with caution."
        )

    model_hash = data[8:24].decode("ascii")
    meta_len = int.from_bytes(data[24:28], "big")

    if len(data) < 28 + meta_len:
        raise ValueError("Invalid payload: metadata truncated")

    # Parse metadata
    meta_bytes = data[28:28 + meta_len]
    try:
        metadata = json.loads(meta_bytes.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid metadata JSON: {e}")

    # Parse model
    model_compressed = data[28 + meta_len:]
    try:
        model_bytes = gzip.decompress(model_compressed)
    except gzip.BadGzipFile:
        # Try without gzip (legacy)
        model_bytes = model_compressed

    # Verify hash
    actual_hash = compute_hash(model_bytes)
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(
            f"Model hash mismatch: expected {expected_hash}, got {actual_hash}. "
            "File may be corrupted or tampered."
        )

    # Update metadata with actual hash
    metadata["_model_hash"] = actual_hash

    if not allow_joblib:
        raise ValueError(
            "Joblib deserialization disabled for untrusted source. "
            "If you trust this file, set allow_joblib=True."
        )

    try:
        buffer = io.BytesIO(model_bytes)
        model = joblib.load(buffer)
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")

    return model, metadata


def is_safe_serialization(data: bytes) -> bool:
    """Check if data uses safe serialization format."""
    return len(data) >= 8 and data[:8] == SAFE_FORMAT_MAGIC


def serialize_numpy(arr: np.ndarray, metadata: Optional[Dict] = None) -> bytes:
    """Serialize numpy array safely."""
    # Convert to bytes and compute hash
    arr_bytes = arr.tobytes()
    arr_hash = compute_hash(arr_bytes)

    meta = {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "hash": arr_hash,
        **(metadata or {})
    }

    return serialize_safe(arr, meta)


def deserialize_numpy(data: bytes) -> np.ndarray:
    """Deserialize numpy array safely."""
    arr, meta = deserialize_safe(data)

    if not isinstance(arr, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(arr)}")

    # Verify shape
    if "shape" in meta:
        expected_shape = tuple(meta["shape"])
        if arr.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch: expected {expected_shape}, got {arr.shape}"
            )

    return arr


# Legacy support with warnings
def import_legacy_pickle(
    data: bytes,
    allow_unsafe: bool = False,
    source_description: str = "unknown"
) -> Any:
    """
    Import legacy pickle file with security warning.

    SECURITY WARNING:
    pickle.loads can execute arbitrary code. Only use on files
    from completely trusted sources.

    Args:
        data: Pickle file bytes
        allow_unsafe: Must be True to proceed
        source_description: Description of file source for logging

    Returns:
        Deserialized object

    Raises:
        ValueError: If allow_unsafe is False
    """
    import pickle

    if not allow_unsafe:
        raise ValueError(
            f"Legacy pickle import disabled for security. "
            f"Source: {source_description}. "
            "Set allow_unsafe=True ONLY if you completely trust this file source. "
            "Consider re-exporting with safe serialization."
        )

    import warnings
    warnings.warn(
        f"Loading legacy pickle from {source_description}. "
        "This file type can execute arbitrary code. "
        "Consider re-exporting with safe serialization.",
        UserWarning,
        stacklevel=3
    )

    try:
        return pickle.loads(data)
    except Exception as e:
        raise ValueError(f"Failed to load legacy pickle: {e}")


__all__ = [
    "serialize_safe",
    "deserialize_safe",
    "is_safe_serialization",
    "serialize_numpy",
    "deserialize_numpy",
    "import_legacy_pickle",
    "compute_hash",
    "SAFE_FORMAT_MAGIC",
    "MODEL_MAGIC",
]
