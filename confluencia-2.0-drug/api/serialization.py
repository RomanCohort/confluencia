"""Serialization utilities for Confluencia 2.0 Drug API.

Handles conversion between:
- pandas DataFrame <-> CSV string
- core dataclasses <-> Pydantic schemas
"""

from __future__ import annotations

import io
from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Type, TypeVar, Union

import pandas as pd


def df_to_csv_str(df: pd.DataFrame) -> str:
    """Encode a DataFrame as a CSV string."""
    if df is None or df.empty:
        return ""
    return df.to_csv(index=False)


def csv_str_to_df(csv: str) -> pd.DataFrame:
    """Decode a CSV string back into a DataFrame."""
    if not csv or not csv.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(csv))


T = TypeVar("T")


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """Convert a dataclass instance to a dictionary.

    Handles nested dataclasses and common types (list, dict, tuple).
    DataFrame values are converted to CSV strings.
    """
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        return df_to_csv_str(obj)
    if isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    if is_dataclass(obj):
        result = {}
        for f in fields(obj):
            val = getattr(obj, f.name, None)
            result[f.name] = dataclass_to_dict(val)
        return result
    # Primitive types (str, int, float, bool, None)
    return obj


def dict_to_dataclass(data: Dict[str, Any], cls: Type[T]) -> T:
    """Convert a dictionary to a dataclass instance.

    Handles nested dataclasses by recursively converting.
    Does NOT handle DataFrame fields (those should be pre-converted).
    """
    if data is None:
        return None
    if not is_dataclass(cls):
        return data

    kwargs = {}
    for f in fields(cls):
        field_name = f.name
        if field_name not in data:
            # Use default or default_factory if available
            if f.default is not dataclass.MISSING:
                continue
            if f.default_factory is not dataclass.MISSING:
                continue
            kwargs[field_name] = None
            continue

        val = data[field_name]
        field_type = f.type

        # Handle Optional types
        origin = getattr(field_type, "__origin__", None)
        args = getattr(field_type, "__args__", ())

        if origin is Union and type(None) in args:
            # Optional[X] -> extract X
            non_none_args = [a for a in args if a is not type(None)]
            if len(non_none_args) == 1:
                field_type = non_none_args[0]
                origin = getattr(field_type, "__origin__", None)

        # Handle List types
        if origin is list and args and is_dataclass(args[0]):
            val = [dict_to_dataclass(item, args[0]) for item in val] if isinstance(val, list) else val
        elif is_dataclass(field_type) and isinstance(val, dict):
            val = dict_to_dataclass(val, field_type)

        kwargs[field_name] = val

    return cls(**kwargs)
