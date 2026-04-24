from __future__ import annotations

import csv
import importlib
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List


@dataclass
class ED2MolRunResult:
    smiles: List[str]
    used_fallback: bool
    message: str


class ED2MolAdapter:
    """Adapter for GitHub ED2Mol (pineappleK/ED2Mol).

    Expected ED2Mol layout:
    - <repo>/Generate.py
    - <repo>/configs/*.yml

    Runtime contract in this adapter:
    - User provides a config file path (via UI) that ED2Mol can run.
    - Adapter patches only `output_dir` in a temporary copy of config and executes:
      python Generate.py --config <temp_config>
    - Generated molecules are parsed from output_dir recursively (.csv/.smi/.txt/.sdf).
    """

    def __init__(self, repo_dir: str, python_cmd: str = "python") -> None:
        self.repo_dir = Path(repo_dir)
        self.python_cmd = str(python_cmd)

    def _patch_config_output_dir(self, config_path: Path, output_dir: Path) -> Path:
        text = config_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        patched: List[str] = []
        replaced = False
        out_str = str(output_dir).replace("\\", "/")
        for ln in lines:
            if ln.strip().startswith("output_dir:"):
                patched.append(f"output_dir: {out_str}")
                replaced = True
            else:
                patched.append(ln)
        if not replaced:
            patched.append(f"output_dir: {out_str}")
        fd, tmp_path = tempfile.mkstemp(prefix="ed2mol_cfg_", suffix=".yml")
        os.close(fd)
        Path(tmp_path).write_text("\n".join(patched) + "\n", encoding="utf-8")
        return Path(tmp_path)

    def _extract_smiles_from_text(self, p: Path) -> List[str]:
        out: List[str] = []
        for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = ln.strip()
            if not s:
                continue
            tok = s.split()[0]
            # Skip probable headers
            if tok.lower() in {"smiles", "molecule", "id"}:
                continue
            out.append(tok)
        return out

    def _extract_smiles_from_csv(self, p: Path) -> List[str]:
        out: List[str] = []
        with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            r = csv.DictReader(f)
            fields = [str(x) for x in (r.fieldnames or [])]
            cand_cols = [c for c in fields if c.lower() in {"smiles", "smile", "mol", "molecule"}]
            if not cand_cols:
                return out
            c0 = cand_cols[0]
            for row in r:
                v = str(row.get(c0, "")).strip()
                if v:
                    out.append(v)
        return out

    def _extract_smiles_from_sdf(self, p: Path) -> List[str]:
        out: List[str] = []
        try:
            Chem: Any = importlib.import_module("rdkit.Chem")
        except Exception:
            return out
        try:
            sdf_supplier = getattr(Chem, "SDMolSupplier", None)
            mol_to_smiles = getattr(Chem, "MolToSmiles", None)
            if sdf_supplier is None or mol_to_smiles is None:
                return out
            sup = sdf_supplier(str(p), removeHs=False)
            for mol in sup:
                if mol is None:
                    continue
                s = mol_to_smiles(mol)
                if s:
                    out.append(str(s))
        except Exception:
            return out
        return out

    def _collect_smiles(self, output_dir: Path, max_count: int) -> List[str]:
        if not output_dir.exists():
            return []
        pool: List[str] = []
        for p in output_dir.rglob("*"):
            if not p.is_file():
                continue
            suffix = p.suffix.lower()
            if suffix in {".smi", ".txt"}:
                pool.extend(self._extract_smiles_from_text(p))
            elif suffix == ".csv":
                pool.extend(self._extract_smiles_from_csv(p))
            elif suffix == ".sdf":
                pool.extend(self._extract_smiles_from_sdf(p))

        # Deduplicate while preserving order
        seen = set()
        out: List[str] = []
        for s in pool:
            key = s.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(key)
            if len(out) >= max_count:
                break
        return out

    def generate(self, config_path: str, max_count: int = 64, timeout_sec: int = 240) -> ED2MolRunResult:
        repo = self.repo_dir
        gen_py = repo / "Generate.py"
        cfg = Path(config_path)

        if not repo.exists() or not gen_py.exists():
            return ED2MolRunResult(smiles=[], used_fallback=True, message="ED2Mol repo or Generate.py not found")
        if not cfg.exists():
            return ED2MolRunResult(smiles=[], used_fallback=True, message="ED2Mol config file not found")

        out_dir = Path(tempfile.mkdtemp(prefix="ed2mol_out_"))
        tmp_cfg = self._patch_config_output_dir(cfg, out_dir)

        cmd = [self.python_cmd, str(gen_py), "--config", str(tmp_cfg)]
        try:
            subprocess.run(cmd, cwd=str(repo), check=True, timeout=int(timeout_sec), capture_output=True, text=True)
        except Exception as e:
            return ED2MolRunResult(smiles=[], used_fallback=True, message=f"ED2Mol run failed: {e}")
        finally:
            try:
                tmp_cfg.unlink(missing_ok=True)
            except Exception:
                pass

        smiles = self._collect_smiles(out_dir, max_count=max_count)
        if not smiles:
            return ED2MolRunResult(smiles=[], used_fallback=True, message="ED2Mol finished but no parseable SMILES found")
        return ED2MolRunResult(smiles=smiles, used_fallback=False, message="ED2Mol generation succeeded")
