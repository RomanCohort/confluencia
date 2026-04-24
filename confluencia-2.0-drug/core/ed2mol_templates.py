from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional


def _as_posix(p: str) -> str:
    return str(p).replace("\\", "/")


def build_ed2mol_config_text(
    mode: Literal["denovo", "hitopt"],
    output_dir: str,
    receptor_pdb: str,
    center_x: float,
    center_y: float,
    center_z: float,
    reference_core_sdf: Optional[str] = None,
) -> str:
    lines = [
        f"output_dir: {_as_posix(output_dir)}",
        f"receptor: {_as_posix(receptor_pdb)}",
        f"x: {float(center_x):.4f}",
        f"y: {float(center_y):.4f}",
        f"z: {float(center_z):.4f}",
    ]
    if mode == "hitopt":
        if not reference_core_sdf:
            raise ValueError("reference_core_sdf is required for hitopt mode")
        lines.append(f"reference_core: {_as_posix(reference_core_sdf)}")
    return "\n".join(lines) + "\n"


def write_ed2mol_config(
    save_path: str,
    mode: Literal["denovo", "hitopt"],
    output_dir: str,
    receptor_pdb: str,
    center_x: float,
    center_y: float,
    center_z: float,
    reference_core_sdf: Optional[str] = None,
) -> str:
    text = build_ed2mol_config_text(
        mode=mode,
        output_dir=output_dir,
        receptor_pdb=receptor_pdb,
        center_x=center_x,
        center_y=center_y,
        center_z=center_z,
        reference_core_sdf=reference_core_sdf,
    )
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return str(p)
