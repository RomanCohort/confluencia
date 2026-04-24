from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from src.common.dataset_fetch import concat_tables


try:
    from rdkit import Chem as _Chem  # type: ignore
    Chem: Any = _Chem
except Exception:
    Chem = None  # type: ignore


_SESSION_LOCAL = threading.local()


def _get_session() -> requests.Session:
    sess = getattr(_SESSION_LOCAL, "session", None)
    if sess is None:
        sess = requests.Session()
        _SESSION_LOCAL.session = sess
    return sess


def _normalize_smiles(s: str) -> Optional[str]:
    """Normalize and validate SMILES using RDKit if available.

    Returns canonical SMILES or None if invalid or RDKit unavailable.
    """
    if not s:
        return None
    if Chem is None:
        return s
    try:
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        # canonical / isomeric SMILES
        can = Chem.MolToSmiles(m, isomericSmiles=True)
        return can
    except Exception:
        return None


_PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# PubChem property fields (PUG-REST)
PUBCHEM_PROPERTY_FIELDS: List[str] = [
    "MolecularFormula",
    "MolecularWeight",
    "XLogP",
    "TopologicalPolarSurfaceArea",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
    "Charge",
    "IUPACName",
    "InChIKey",
    "InChI",
    "CanonicalSMILES",
    "IsomericSMILES",
]


DOCKING_COLUMN_ALIASES: Dict[str, List[str]] = {
    "ligand_smiles": ["ligand_smiles", "smiles", "ligand", "ligand_smile", "lig_smiles"],
    "protein_sequence": ["protein_sequence", "sequence", "prot_seq", "protein_seq", "target_sequence"],
    "protein_pdb": ["protein_pdb", "pdb", "pdb_id", "pdbid", "protein_id", "structure_id"],
    "binding_score": ["binding_score", "affinity", "score", "pKd", "pKi", "deltaG", "dg"],
    "pocket_path": ["pocket_path", "pocket_file", "pocket_csv", "pocket"],
}

MULTISCALE_COLUMN_ALIASES: Dict[str, List[str]] = {
    "smiles": ["smiles", "ligand_smiles", "compound_smiles"],
    "target": ["target", "y", "label", "value", "response"],
    "D": ["D", "diffusion", "diffusivity"],
    "Vmax": ["Vmax", "vmax", "v_max"],
    "Km": ["Km", "km", "k_m"],
}

_PUBCHEM_PROPERTY_KEY_TO_COL: Dict[str, str] = {
    "MolecularFormula": "molecular_formula",
    "MolecularWeight": "molecular_weight",
    "XLogP": "xlogp",
    "TopologicalPolarSurfaceArea": "tpsa",
    "HBondDonorCount": "hbd",
    "HBondAcceptorCount": "hba",
    "RotatableBondCount": "rotatable_bonds",
    "Charge": "charge",
    "IUPACName": "iupac_name",
    "InChIKey": "inchikey",
    "InChI": "inchi",
    "CanonicalSMILES": "canonical_smiles",
    "IsomericSMILES": "isomeric_smiles",
}


def _hash_fields(fields: List[str]) -> str:
    text = "|".join([str(f).strip() for f in fields if str(f).strip()])
    if not text:
        return "empty"
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:12]


@dataclass(frozen=True)
class PubChemCrawlResult:
    cid: int
    smiles: Optional[str]
    activity_score: Optional[float]
    n_active: int
    n_inactive: int
    n_total: int


def _safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        return {}


def _request_text(
    url: str,
    *,
    timeout: float = 30.0,
    headers: Optional[Dict[str, str]] = None,
    retries: int = 3,
    backoff_factor: float = 0.5,
    session: Optional[requests.Session] = None,
) -> Optional[str]:
    sess = session or _get_session()
    for attempt in range(int(retries) if retries and retries > 0 else 1):
        try:
            resp = sess.get(url, timeout=timeout, headers=headers)
            code = getattr(resp, "status_code", None)
            if code == 200:
                return resp.text
            # simple transient status handling
            if code in (429, 500, 502, 503, 504) and attempt + 1 < int(retries):
                time.sleep(float(backoff_factor) * (2 ** attempt))
                continue
            return None
        except requests.RequestException:
            if attempt + 1 < int(retries):
                time.sleep(float(backoff_factor) * (2 ** attempt))
                continue
            return None


def fetch_smiles_by_cid(
    cid: int,
    *,
    cache_dir: Optional[Path] = None,
    timeout: float = 30.0,
    headers: Optional[Dict[str, str]] = None,
    retries: int = 3,
    backoff_factor: float = 0.5,
    session: Optional[requests.Session] = None,
) -> Optional[str]:
    """Fetch Canonical SMILES for a PubChem CID.

    Uses caching if cache_dir is provided.
    """

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        p = cache_dir / f"cid_{cid}_smiles.txt"
        if p.exists():
            txt = p.read_text(encoding="utf-8", errors="ignore").strip()
            return txt or None

    url = f"{_PUBCHEM_BASE}/compound/cid/{cid}/property/CanonicalSMILES/TXT"
    txt = _request_text(url, timeout=timeout, headers=headers, retries=retries, backoff_factor=backoff_factor, session=session)
    if txt is None:
        return None

    smiles = txt.strip()
    if cache_dir is not None:
        (cache_dir / f"cid_{cid}_smiles.txt").write_text(smiles, encoding="utf-8")
    return smiles or None


def fetch_assaysummary_by_cid(
    cid: int,
    *,
    cache_dir: Optional[Path] = None,
    timeout: float = 30.0,
    headers: Optional[Dict[str, str]] = None,
    retries: int = 3,
    backoff_factor: float = 0.5,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """Fetch PubChem assay summary JSON for a CID.

    Endpoint: /compound/cid/{cid}/assaysummary/JSON
    """

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        p = cache_dir / f"cid_{cid}_assaysummary.json"
        if p.exists():
            return _safe_json_loads(p.read_text(encoding="utf-8", errors="ignore"))

    url = f"{_PUBCHEM_BASE}/compound/cid/{cid}/assaysummary/JSON"
    txt = _request_text(url, timeout=timeout, headers=headers, retries=retries, backoff_factor=backoff_factor, session=session)
    if txt is None:
        return {}

    if cache_dir is not None:
        (cache_dir / f"cid_{cid}_assaysummary.json").write_text(txt, encoding="utf-8")
    return _safe_json_loads(txt)


def fetch_pubchem_properties_by_cid(
    cid: int,
    *,
    property_fields: Optional[List[str]] = None,
    cache_dir: Optional[Path] = None,
    timeout: float = 30.0,
    headers: Optional[Dict[str, str]] = None,
    retries: int = 3,
    backoff_factor: float = 0.5,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """Fetch PubChem compound properties for a CID.

    Endpoint: /compound/cid/{cid}/property/{props}/JSON
    """

    fields = [f for f in (property_fields or list(PUBCHEM_PROPERTY_FIELDS)) if str(f).strip()]
    if not fields:
        return {}

    props_key = _hash_fields(fields)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        p = cache_dir / f"cid_{cid}_properties_{props_key}.json"
        if p.exists():
            return _safe_json_loads(p.read_text(encoding="utf-8", errors="ignore"))

    props = ",".join(fields)
    url = f"{_PUBCHEM_BASE}/compound/cid/{cid}/property/{props}/JSON"
    txt = _request_text(url, timeout=timeout, headers=headers, retries=retries, backoff_factor=backoff_factor, session=session)
    if txt is None:
        return {}

    if cache_dir is not None:
        (cache_dir / f"cid_{cid}_properties_{props_key}.json").write_text(txt, encoding="utf-8")
    return _safe_json_loads(txt)


def fetch_pubchem_synonyms_by_cid(
    cid: int,
    *,
    cache_dir: Optional[Path] = None,
    timeout: float = 30.0,
    headers: Optional[Dict[str, str]] = None,
    retries: int = 3,
    backoff_factor: float = 0.5,
    session: Optional[requests.Session] = None,
) -> List[str]:
    """Fetch PubChem synonyms for a CID."""

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        p = cache_dir / f"cid_{cid}_synonyms.json"
        if p.exists():
            obj = _safe_json_loads(p.read_text(encoding="utf-8", errors="ignore"))
        else:
            obj = {}
    else:
        obj = {}

    if not obj:
        url = f"{_PUBCHEM_BASE}/compound/cid/{cid}/synonyms/JSON"
        txt = _request_text(url, timeout=timeout, headers=headers, retries=retries, backoff_factor=backoff_factor, session=session)
        if txt is None:
            return []
        obj = _safe_json_loads(txt)
        if cache_dir is not None:
            (cache_dir / f"cid_{cid}_synonyms.json").write_text(txt, encoding="utf-8")

    try:
        info_list = obj.get("InformationList", {})
        info = info_list.get("Information")
        if isinstance(info, list) and info:
            first = info[0]
            syns = first.get("Synonym") if isinstance(first, dict) else None
            if isinstance(syns, list):
                return [str(s).strip() for s in syns if str(s).strip()]
    except Exception:
        return []

    return []


def _iter_assay_outcomes(obj: Dict[str, Any], *, indication_filter: Optional[str] = None) -> Iterable[str]:
    """Yield ActivityOutcome strings from a PubChem assaysummary JSON (best-effort)."""

    # PubChem formats may vary; we defensively search a few common shapes.
    def _walk(x: Any) -> Iterable[Any]:
        if isinstance(x, dict):
            for v in x.values():
                yield from _walk(v)
        elif isinstance(x, list):
            for v in x:
                yield from _walk(v)
        else:
            yield x

    # Fast-path: common schema
    # {"AssaySummaries": {"AssaySummary": [{"ActivityOutcome": "Active"}, ...]}}
    try:
        summaries = obj.get("AssaySummaries", {})
        items = summaries.get("AssaySummary")
        if isinstance(items, list):
            for it in items:
                if not isinstance(it, dict):
                    continue
                if indication_filter:
                    matched = False
                    for k in ("Indication", "Indications", "Disease", "Diseases", "Condition", "Conditions", "Purpose", "Description"):
                        if k in it and isinstance(it.get(k), str):
                            if indication_filter.lower() in it.get(k, "").lower():
                                matched = True
                                break
                    if not matched:
                        continue
                if "ActivityOutcome" in it:
                    v = it.get("ActivityOutcome")
                    if isinstance(v, str) and v.strip():
                        yield v.strip()
            return
    except Exception:
        pass

    # Table schema (seen in cached assaysummary):
    # {"Table": {"Columns": {"Column": ["AID", ..., "Activity Outcome", ...]}, "Row": [{"Cell": [...]}, ...]}}
    try:
        table = obj.get("Table", {}) if isinstance(obj, dict) else {}
        columns = table.get("Columns", {}).get("Column") if isinstance(table, dict) else None
        rows = table.get("Row") if isinstance(table, dict) else None
        if isinstance(columns, list) and isinstance(rows, list):
            col_idx = None
            indication_col_idx = None
            for i, name in enumerate(columns):
                if not isinstance(name, str):
                    continue
                n = name.strip().lower()
                if n in {"activity outcome", "activityoutcome"}:
                    col_idx = i
                if n in {"indication", "indications", "disease", "diseases", "condition", "conditions"}:
                    indication_col_idx = i
            if col_idx is not None:
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    cells = row.get("Cell")
                    if not isinstance(cells, list) or col_idx >= len(cells):
                        continue
                    if indication_filter and indication_col_idx is not None and indication_col_idx < len(cells):
                        ind_val = cells[indication_col_idx]
                        if not (isinstance(ind_val, str) and indication_filter.lower() in ind_val.lower()):
                            continue
                    v = cells[col_idx]
                    if isinstance(v, str) and v.strip():
                        yield v.strip()
                return
    except Exception:
        pass

    # Fallback: deep search for keys named ActivityOutcome
    def _walk_dict(d: Any) -> Iterable[str]:
        if isinstance(d, dict):
            for k, v in d.items():
                if k == "ActivityOutcome" and isinstance(v, str) and v.strip():
                    yield v.strip()
                else:
                    yield from _walk_dict(v)
        elif isinstance(d, list):
            for v in d:
                yield from _walk_dict(v)
        return

    yield from _walk_dict(obj)


def _iter_assay_outcomes_with_weights(
    obj: Dict[str, Any], *, indication_filter: Optional[str] = None
) -> Iterable[Tuple[str, float]]:
    """Yield (ActivityOutcome, weight) tuples.

    Weight is inferred from common numeric fields in assay entries or table columns
    (e.g. 'Count', 'Total', 'NumberTested'); defaults to 1 when unavailable.
    """
    try:
        summaries = obj.get("AssaySummaries", {})
        items = summaries.get("AssaySummary")
        if isinstance(items, list):
            for it in items:
                if not isinstance(it, dict):
                    continue
                if indication_filter:
                    matched = False
                    for k in ("Indication", "Indications", "Disease", "Diseases", "Condition", "Conditions", "Purpose", "Description"):
                        if k in it and isinstance(it.get(k), str):
                            if indication_filter.lower() in it.get(k, "").lower():
                                matched = True
                                break
                    if not matched:
                        continue

                weight = 1.0
                for k in ("Count", "Total", "NumberTested", "N", "NumSamples", "Instances"):
                    if k in it:
                        try:
                            weight = float(it.get(k) or 0) or 1.0
                            break
                        except Exception:
                            continue

                if "ActivityOutcome" in it:
                    v = it.get("ActivityOutcome")
                    if isinstance(v, str) and v.strip():
                        yield v.strip(), float(weight)
            return
    except Exception:
        pass

    try:
        table = obj.get("Table", {}) if isinstance(obj, dict) else {}
        columns = table.get("Columns", {}).get("Column") if isinstance(table, dict) else None
        rows = table.get("Row") if isinstance(table, dict) else None
        if isinstance(columns, list) and isinstance(rows, list):
            col_idx = None
            indication_col_idx = None
            count_col_idx = None
            for i, name in enumerate(columns):
                if not isinstance(name, str):
                    continue
                n = name.strip().lower()
                if n in {"activity outcome", "activityoutcome"}:
                    col_idx = i
                if n in {"indication", "indications", "disease", "diseases", "condition", "conditions"}:
                    indication_col_idx = i
                if n in {"count", "total", "n", "num", "number tested", "number"}:
                    count_col_idx = i
            if col_idx is not None:
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    cells = row.get("Cell")
                    if not isinstance(cells, list) or col_idx >= len(cells):
                        continue
                    if indication_filter and indication_col_idx is not None and indication_col_idx < len(cells):
                        ind_val = cells[indication_col_idx]
                        if not (isinstance(ind_val, str) and indication_filter.lower() in ind_val.lower()):
                            continue
                    weight = 1.0
                    if count_col_idx is not None and count_col_idx < len(cells):
                        try:
                            weight = float(cells[count_col_idx])
                        except Exception:
                            weight = 1.0
                    v = cells[col_idx]
                    if isinstance(v, str) and v.strip():
                        yield v.strip(), float(weight)
                return
    except Exception:
        pass

    # Fallback: deep search for ActivityOutcome strings, weight=1
    def _walk_dict_weight(d: Any) -> Iterable[Tuple[str, float]]:
        if isinstance(d, dict):
            for k, v in d.items():
                if k == "ActivityOutcome" and isinstance(v, str) and v.strip():
                    yield v.strip(), 1.0
                else:
                    yield from _walk_dict_weight(v)
        elif isinstance(d, list):
            for v in d:
                yield from _walk_dict_weight(v)
        return

    yield from _walk_dict_weight(obj)


def assaysummary_to_activity_score_detailed(
    obj: Dict[str, Any], *, indication_filter: Optional[str] = None, weighted: bool = False
) -> Tuple[Optional[float], float, float, float, Dict[str, Dict[str, float]]]:
    """Return detailed outcome counts and optionally weighted activity score.

    Returns (score, n_active, n_inactive, n_total, outcomes_dict)
    where outcomes_dict maps outcome_str -> {"count": count, "weight": weight_sum}.
    """
    outcomes: Dict[str, Dict[str, float]] = {}
    if weighted:
        iterator = _iter_assay_outcomes_with_weights(obj, indication_filter=indication_filter)
        for outcome, w in iterator:
            k = outcome.strip()
            ent = outcomes.setdefault(k, {"count": 0.0, "weight": 0.0})
            ent["count"] = ent["count"] + 1.0
            ent["weight"] = ent["weight"] + float(w)
    else:
        iterator = _iter_assay_outcomes(obj, indication_filter=indication_filter)
        for outcome in iterator:
            k = outcome.strip()
            ent = outcomes.setdefault(k, {"count": 0.0, "weight": 0.0})
            ent["count"] = ent["count"] + 1.0
            ent["weight"] = ent["weight"] + 1.0

    n_active = outcomes.get("Active", {}).get("count", 0.0) + outcomes.get("active", {}).get("count", 0.0)
    n_inactive = outcomes.get("Inactive", {}).get("count", 0.0) + outcomes.get("inactive", {}).get("count", 0.0)
    w_active = outcomes.get("Active", {}).get("weight", 0.0) + outcomes.get("active", {}).get("weight", 0.0)
    w_inactive = outcomes.get("Inactive", {}).get("weight", 0.0) + outcomes.get("inactive", {}).get("weight", 0.0)

    if weighted:
        total = w_active + w_inactive
        if total <= 0:
            return None, 0.0, 0.0, 0.0, outcomes
        score = float(w_active) / float(total)
        return score, w_active, w_inactive, total, outcomes
    else:
        total = n_active + n_inactive
        if total <= 0:
            return None, 0.0, 0.0, 0.0, outcomes
        score = float(n_active) / float(total)
        return score, n_active, n_inactive, total, outcomes


def assaysummary_to_activity_score(obj: Dict[str, Any], *, indication_filter: Optional[str] = None) -> Tuple[Optional[float], int, int, int]:
    """Convert assay summary JSON into a simple activity score.

    Score definition (heuristic):
      activity_score = n_active / (n_active + n_inactive)

    Where ActivityOutcome is counted as Active/Inactive (case-insensitive).
    If no such outcomes exist, returns (None, 0, 0, 0).

    This is NOT clinical efficacy; it's a proxy label based on bioassay outcomes.
    """

    n_active = 0
    n_inactive = 0

    for outcome in _iter_assay_outcomes(obj, indication_filter=indication_filter):
        o = outcome.strip().lower()
        if o == "active":
            n_active += 1
        elif o == "inactive":
            n_inactive += 1

    n_total = n_active + n_inactive
    if n_total <= 0:
        return None, 0, 0, 0

    return float(n_active) / float(n_total), n_active, n_inactive, n_total


def crawl_pubchem_activity_dataset(
    *,
    start_cid: int = 1,
    n: int = 200,
    cids: Optional[Iterable[int]] = None,
    sleep_seconds: float = 0.2,
    min_total_outcomes: int = 5,
    min_active: int = 1,
    treat_zero_unlabeled: bool = True,
    drop_invalid: bool = True,
    cache_dir: str | Path = "data/cache/pubchem",
    timeout: float = 30.0,
    user_agent: str = "drug-efficacy-crawler/1.0 (research; contact: local)",
    # concurrency / network
    max_workers: int = 4,
    rate_limit: float = 5.0,  # requests per second (global)
    retries: int = 3,
    backoff_factor: float = 0.5,
    indication: Optional[str] = None,
    # advanced options
    weighted: bool = False,
    include_outcome_breakdown: bool = False,
    normalize_smiles: bool = False,
    include_properties: bool = False,
    property_fields: Optional[List[str]] = None,
    include_synonyms: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """Crawl PubChem and build a training dataset.

        Output columns:
            - cid, smiles
            - activity_score (float in [0,1], may be NaN)
            - n_active, n_inactive, n_total
            - (optional) PubChem properties: molecular_weight, xlogp, tpsa, hbd, hba, rotatable_bonds, etc.
            - (optional) synonyms, synonyms_count

    Notes
    -----
    - This is best-effort; many CIDs won't have assay summaries.
    - If cids is provided, start_cid/n will be ignored.
    - Uses a small sleep between requests to be polite.
    - Caches responses to disk.
    - By default, drops invalid rows (no SMILES or no usable label).
    """

    cache_path = Path(cache_dir)
    headers = {"User-Agent": user_agent}

    rows: List[Dict[str, Any]] = []

    # Simple global rate limiter (token spacing)
    class RateLimiter:
        def __init__(self, r: float):
            self.interval = 1.0 / r if r and r > 0 else 0.0
            self.lock = threading.Lock()
            self._last = 0.0

        def wait(self) -> None:
            if self.interval <= 0:
                return
            with self.lock:
                now = time.time()
                wait_for = self.interval - (now - self._last)
                if wait_for > 0:
                    time.sleep(wait_for)
                    now = time.time()
                self._last = now

    rate_limiter = RateLimiter(rate_limit)

    def _process_cid(cid: int) -> Optional[Dict[str, Any]]:
        try:
            session = _get_session()
            # smiles
            rate_limiter.wait()
            smiles = fetch_smiles_by_cid(
                cid,
                cache_dir=cache_path,
                timeout=timeout,
                headers=headers,
                retries=retries,
                backoff_factor=backoff_factor,
                session=session,
            )

            # assay
            rate_limiter.wait()
            assay = fetch_assaysummary_by_cid(
                cid,
                cache_dir=cache_path,
                timeout=timeout,
                headers=headers,
                retries=retries,
                backoff_factor=backoff_factor,
                session=session,
            )

            detailed_score, d_n_active, d_n_inactive, d_n_total, outcomes_dict = (
                assaysummary_to_activity_score_detailed(assay, indication_filter=indication, weighted=weighted)
            )
            score, n_active, n_inactive, n_total = detailed_score, d_n_active, d_n_inactive, d_n_total

            props: Dict[str, Any] = {}
            if include_properties:
                user_fields = [f for f in (property_fields or list(PUBCHEM_PROPERTY_FIELDS)) if str(f).strip()]
                request_fields = list(user_fields)
                if smiles is None:
                    for extra in ("CanonicalSMILES", "IsomericSMILES"):
                        if extra not in request_fields:
                            request_fields.append(extra)
                rate_limiter.wait()
                props = fetch_pubchem_properties_by_cid(
                    cid,
                    property_fields=request_fields,
                    cache_dir=cache_path,
                    timeout=timeout,
                    headers=headers,
                    retries=retries,
                    backoff_factor=backoff_factor,
                    session=session,
                )

                # flatten property table if needed
                if "PropertyTable" in props and isinstance(props.get("PropertyTable"), dict):
                    table = props.get("PropertyTable", {})
                    entries = table.get("Properties") if isinstance(table, dict) else None
                    if isinstance(entries, list) and entries:
                        props = entries[0] if isinstance(entries[0], dict) else props

                if smiles is None:
                    maybe = props.get("CanonicalSMILES") or props.get("IsomericSMILES")
                    if isinstance(maybe, str) and maybe.strip():
                        smiles = maybe.strip()

            if normalize_smiles and smiles:
                norm = _normalize_smiles(smiles)
                if norm is None:
                    smiles = None
                else:
                    smiles = norm

            syns: List[str] = []
            if include_synonyms:
                rate_limiter.wait()
                syns = fetch_pubchem_synonyms_by_cid(
                    cid,
                    cache_dir=cache_path,
                    timeout=timeout,
                    headers=headers,
                    retries=retries,
                    backoff_factor=backoff_factor,
                    session=session,
                )

            if score is not None and int(n_active) < int(min_active):
                score = None
            if score is not None and bool(treat_zero_unlabeled) and float(score) == 0.0:
                score = None

            invalid = score is None or n_total < int(min_total_outcomes) or smiles is None
            if not invalid or not bool(drop_invalid):
                out: Dict[str, Any] = {
                    "cid": int(cid),
                    "smiles": smiles,
                    "activity_score": (float(score) if score is not None else float("nan")),
                    "n_active": int(n_active),
                    "n_inactive": int(n_inactive),
                    "n_total": int(n_total),
                }
                if include_properties and props:
                    user_fields = [f for f in (property_fields or list(PUBCHEM_PROPERTY_FIELDS)) if str(f).strip()]
                    for k in user_fields:
                        if k in props and k in _PUBCHEM_PROPERTY_KEY_TO_COL:
                            out[_PUBCHEM_PROPERTY_KEY_TO_COL[k]] = props.get(k)
                if include_synonyms:
                    out["synonyms"] = "|".join(syns) if syns else ""
                    out["synonyms_count"] = int(len(syns))
                if include_outcome_breakdown:
                    out["outcome_counts"] = outcomes_dict
                return out
            return None
        except Exception:
            return None

    # Submit tasks in a thread pool to fetch multiple CIDs concurrently while respecting rate limit
    if cids is None:
        cid_list = list(range(int(start_cid), int(start_cid) + int(n)))
    else:
        cid_list = [int(c) for c in cids if c is not None]
        # 去重但保持原顺序
        cid_list = list(dict.fromkeys(cid_list))
    total = len(cid_list)
    completed = 0
    with ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
        futures = {ex.submit(_process_cid, cid): cid for cid in cid_list}
        for fut in as_completed(futures):
            res = None
            try:
                res = fut.result()
            except Exception:
                res = None
            if res:
                rows.append(res)
            completed += 1
            if progress_callback:
                try:
                    progress_callback(completed, total)
                except Exception:
                    pass

    df = pd.DataFrame(rows)
    return df


def _resolve_col(df: pd.DataFrame, name: Optional[str], aliases: List[str]) -> Optional[str]:
    if name and name in df.columns:
        return name
    lower_map = {str(c).lower(): c for c in df.columns}
    for a in aliases:
        key = str(a).lower()
        if key in lower_map:
            return lower_map[key]
    return None


def crawl_docking_training_dataset(
    *,
    sources: Iterable[str],
    cache_dir: str | Path = "data/cache/http",
    timeout: float = 30.0,
    sleep_seconds: float = 0.2,
    user_agent: str = "docking-crawler/1.0 (research; contact: local)",
    ligand_smiles_col: Optional[str] = None,
    protein_seq_col: Optional[str] = None,
    protein_pdb_col: Optional[str] = None,
    binding_score_col: Optional[str] = None,
    pocket_path_col: Optional[str] = None,
    normalize_smiles: bool = False,
    drop_invalid: bool = True,
) -> pd.DataFrame:
    """Merge docking datasets from user-provided tables.

    Standard output columns (best-effort):
      - ligand_smiles
      - protein_sequence (optional)
      - protein_pdb (optional)
      - binding_score (optional)
      - pocket_path (optional)
    """

    sources_list = [str(s).strip() for s in sources if str(s).strip()]
    if not sources_list:
        raise ValueError("sources 不能为空")

    df = concat_tables(
        sources_list,
        cache_dir=str(cache_dir),
        timeout=float(timeout),
        sleep_seconds=float(sleep_seconds),
        headers={"User-Agent": str(user_agent)},
    )

    lig_col = _resolve_col(df, ligand_smiles_col, DOCKING_COLUMN_ALIASES["ligand_smiles"])
    seq_col = _resolve_col(df, protein_seq_col, DOCKING_COLUMN_ALIASES["protein_sequence"])
    pdb_col = _resolve_col(df, protein_pdb_col, DOCKING_COLUMN_ALIASES["protein_pdb"])
    score_col = _resolve_col(df, binding_score_col, DOCKING_COLUMN_ALIASES["binding_score"])
    pocket_col = _resolve_col(df, pocket_path_col, DOCKING_COLUMN_ALIASES["pocket_path"])

    out = pd.DataFrame()
    if lig_col:
        out["ligand_smiles"] = df[lig_col].astype(str)
    if seq_col:
        out["protein_sequence"] = df[seq_col].astype(str)
    if pdb_col:
        out["protein_pdb"] = df[pdb_col].astype(str)
    if score_col:
        out["binding_score"] = pd.to_numeric(df[score_col], errors="coerce")
    if pocket_col:
        out["pocket_path"] = df[pocket_col].astype(str)

    if normalize_smiles and "ligand_smiles" in out.columns:
        out["ligand_smiles"] = [(_normalize_smiles(str(s)) or "") for s in out["ligand_smiles"].tolist()]

    if drop_invalid:
        keep = out["ligand_smiles"].astype(str).str.len() > 0 if "ligand_smiles" in out.columns else pd.Series([True] * len(out))
        if "protein_sequence" in out.columns:
            keep = keep & (out["protein_sequence"].astype(str).str.len() > 0)
        elif "protein_pdb" in out.columns:
            keep = keep & (out["protein_pdb"].astype(str).str.len() > 0)
        out = out[keep].copy()

    return out


def crawl_multiscale_training_dataset(
    *,
    sources: Iterable[str],
    cache_dir: str | Path = "data/cache/http",
    timeout: float = 30.0,
    sleep_seconds: float = 0.2,
    user_agent: str = "multiscale-crawler/1.0 (research; contact: local)",
    smiles_col: Optional[str] = None,
    target_col: Optional[str] = None,
    d_col: Optional[str] = None,
    vmax_col: Optional[str] = None,
    km_col: Optional[str] = None,
    default_D: float = 0.1,
    default_Vmax: float = 0.5,
    default_Km: float = 0.1,
    normalize_smiles: bool = False,
    drop_invalid: bool = True,
) -> pd.DataFrame:
    """Merge multiscale training datasets from user-provided tables.

    Standard output columns (best-effort):
      - smiles
      - target (optional)
      - D, Vmax, Km (optional; filled by defaults if missing)
    """

    sources_list = [str(s).strip() for s in sources if str(s).strip()]
    if not sources_list:
        raise ValueError("sources 不能为空")

    df = concat_tables(
        sources_list,
        cache_dir=str(cache_dir),
        timeout=float(timeout),
        sleep_seconds=float(sleep_seconds),
        headers={"User-Agent": str(user_agent)},
    )

    s_col = _resolve_col(df, smiles_col, MULTISCALE_COLUMN_ALIASES["smiles"])
    t_col = _resolve_col(df, target_col, MULTISCALE_COLUMN_ALIASES["target"])
    d_res = _resolve_col(df, d_col, MULTISCALE_COLUMN_ALIASES["D"])
    vmax_res = _resolve_col(df, vmax_col, MULTISCALE_COLUMN_ALIASES["Vmax"])
    km_res = _resolve_col(df, km_col, MULTISCALE_COLUMN_ALIASES["Km"])

    out = pd.DataFrame()
    if s_col:
        out["smiles"] = df[s_col].astype(str)
    if t_col:
        out["target"] = pd.to_numeric(df[t_col], errors="coerce")

    if d_res:
        out["D"] = pd.to_numeric(df[d_res], errors="coerce")
    else:
        out["D"] = float(default_D)
    if vmax_res:
        out["Vmax"] = pd.to_numeric(df[vmax_res], errors="coerce")
    else:
        out["Vmax"] = float(default_Vmax)
    if km_res:
        out["Km"] = pd.to_numeric(df[km_res], errors="coerce")
    else:
        out["Km"] = float(default_Km)

    if normalize_smiles and "smiles" in out.columns:
        out["smiles"] = [(_normalize_smiles(str(s)) or "") for s in out["smiles"].tolist()]

    if drop_invalid:
        keep = out["smiles"].astype(str).str.len() > 0 if "smiles" in out.columns else pd.Series([True] * len(out))
        out = out[keep].copy()

    return out
