"""
src.drug.crawler — unified drug data crawler for Confluencia.

Combines:
  - PubChem bio-activity crawling  (from confluencia-2.0-drug/tools/pubchem_crawler.py)
  - ChEMBL REST API fetching       (from scripts/legacy/fetch_chembl_drug.py)
  - Docking dataset crawling       (from scripts/legacy/fetch_docking_data.py)

Public API
----------
crawl_pubchem(...)
    Crawl PubChem PUG-REST for SMILES + activity scores.

crawl_chembl(...)
    Fetch bioactivity data from ChEMBL REST API.

crawl_docking(...)
    Merge docking datasets from user-provided tables / ChEMBL.

crawl_all_drug(...)
    One-shot: run PubChem + ChEMBL + optional docking, return concatenated DataFrame.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

# ── Re-export the full PubChem crawler from 2.0 ──────────────────────────
from confluencia_2_0_drug.tools.pubchem_crawler import (  # type: ignore[import-not-found]  # noqa: F401
    PubChemCrawlResult,
    assaysummary_to_activity_score,
    assaysummary_to_activity_score_detailed,
    crawl_docking_training_dataset,
    crawl_multiscale_training_dataset,
    crawl_pubchem_activity_dataset,
    fetch_assaysummary_by_cid,
    fetch_pubchem_properties_by_cid,
    fetch_pubchem_synonyms_by_cid,
    fetch_smiles_by_cid,
)

# ---------------------------------------------------------------------------
# ChEMBL REST crawler (legacy integration)
# ---------------------------------------------------------------------------

_CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"

_DEFAULT_TARGET_IDS: List[str] = [
    "CHEMBL203",   # EGFR
    "CHEMBL240",   # HER2
    "CHEMBL279",   # VEGFR2
    "CHEMBL4016",  # CDK2
    "CHEMBL301",   # Estrogen receptor alpha
]

_DEFAULT_STANDARD_TYPES: List[str] = ["IC50", "Ki"]


def _chembl_get_json(
    url: str,
    *,
    timeout: float = 30.0,
    retries: int = 3,
    backoff: float = 0.5,
) -> Dict[str, Any]:
    last_exc: Optional[Exception] = None
    for attempt in range(int(retries)):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:
            last_exc = exc
            if attempt + 1 < int(retries):
                time.sleep(float(backoff) * (2 ** attempt))
                continue
            break
    raise RuntimeError(f"ChEMBL request failed: {url}") from last_exc


def crawl_chembl(
    *,
    target_ids: Optional[Sequence[str]] = None,
    standard_types: Optional[Sequence[str]] = None,
    limit_per_query: int = 5000,
    page_size: int = 1000,
    default_dose: float = 10.0,
    default_freq: float = 1.0,
    sleep_seconds: float = 0.5,
    timeout: float = 30.0,
    retries: int = 3,
) -> pd.DataFrame:
    """Fetch drug bioactivity data from ChEMBL REST API.

    Parameters
    ----------
    target_ids : list[str], optional
        ChEMBL target IDs.  Defaults to EGFR / HER2 / VEGFR2 / CDK2 / ERα.
    standard_types : list[str], optional
        Activity standard types (default ``["IC50", "Ki"]``).
    limit_per_query : int
        Max records per target+type query.
    default_dose, default_freq : float
        Placeholder values (ChEMBL doesn't carry dosing info).
    sleep_seconds : float
        Polite delay between requests.

    Returns
    -------
    pd.DataFrame  columns: ``[smiles, dose, freq, efficacy]``
    """
    targets = list(target_ids or _DEFAULT_TARGET_IDS)
    stypes = list(standard_types or _DEFAULT_STANDARD_TYPES)

    records: List[Dict[str, Any]] = []
    for tid in targets:
        for stype in stypes:
            offset = 0
            while offset < limit_per_query:
                params = {
                    "target_chembl_id": tid,
                    "standard_type": stype,
                    "has_smiles": "true",
                    "format": "json",
                    "limit": page_size,
                    "offset": offset,
                }
                url = f"{_CHEMBL_API}/activity.json?{urllib.parse.urlencode(params)}"
                data = _chembl_get_json(url, timeout=timeout, retries=retries)
                activities = data.get("activities", [])
                if not activities:
                    break
                for act in activities:
                    smi = act.get("canonical_smiles")
                    pval = act.get("pchembl_value")
                    if smi and pval:
                        try:
                            records.append({
                                "smiles": smi,
                                "pchembl_value": float(pval),
                                "standard_type": act.get("standard_type", ""),
                                "target_chembl_id": tid,
                            })
                        except (ValueError, TypeError):
                            continue
                offset += page_size
                if len(activities) < page_size:
                    break
                time.sleep(float(sleep_seconds))

    if not records:
        return pd.DataFrame(columns=["smiles", "dose", "freq", "efficacy"])

    df = pd.DataFrame(records)
    # Deduplicate: keep highest pChEMBL per SMILES
    df = df.sort_values("pchembl_value", ascending=False)
    df = df.drop_duplicates(subset=["smiles"], keep="first")

    # Normalise pChEMBL → efficacy ∈ [0, 1]
    pmin, pmax = df["pchembl_value"].min(), df["pchembl_value"].max()
    df["efficacy"] = ((df["pchembl_value"] - pmin) / max(pmax - pmin, 1e-6)).round(6)

    out = pd.DataFrame({
        "smiles": df["smiles"],
        "dose": default_dose,
        "freq": default_freq,
        "efficacy": df["efficacy"],
    })
    return out.dropna(subset=["smiles", "efficacy"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Docking data fetcher (legacy integration)
# ---------------------------------------------------------------------------

_PROTEIN_TARGETS: Dict[str, Dict[str, str]] = {
    "HER2": {
        "uniprot_id": "P04626",
        "sequence": (
            "MELAALCRWGLLLALLPPGAASQVNTGVVLHRKREKISRALKELRNGNEKITSLHDCFVKFQNGNKALRGTNKHD"
            "NPNRQLVFENKTITLSEALRKLKEMEIVQRRVDDVFLRNLRENEKQQLTDLQKDVPYLKLSFNSHDPVTMPEKVT"
        ),
    },
    "EGFR": {
        "uniprot_id": "P00533",
        "sequence": (
            "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQNYKSD"
            "GLYTDLIPQKLRFPSGLTIYHAENGSLDTEKQLELQKLEQRQAELEKLKDSDSLEEKLKELPEEELKNTEKEKQEA"
        ),
    },
    "VEGFR2": {
        "uniprot_id": "P35968",
        "sequence": (
            "MVLLYMTVLSAGLLAPGSLRAQSLLPSCGPLPLPLLLLPLLPLLGAAPGQKDSASAVVLPQFVQVTVNQDSFLPSL"
            "PQPRVPPQTQLQPLQLNQVTFTLTLPSQTQTQPVNLSALTSLLSLPQLPQLPQLSAFSLPLLPVLQAPRPLPQLP"
        ),
    },
}


def _fetch_uniprot_fasta(uniprot_id: str, *, timeout: float = 15.0) -> Optional[str]:
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            lines = resp.read().decode().strip().split("\n")
            return "".join(lines[1:])
    except Exception:
        return None


def crawl_docking_sources(
    *,
    targets: Optional[Sequence[str]] = None,
    standard_types: Sequence[str] = ("Ki", "Kd", "IC50"),
    chembl_limit: int = 500,
    timeout: float = 20.0,
    sleep_seconds: float = 0.3,
    use_full_sequences: bool = True,
) -> pd.DataFrame:
    """Fetch protein-ligand docking data from ChEMBL.

    Parameters
    ----------
    targets : list[str], optional
        Protein target names (keys of built-in PROTEIN_TARGETS).
    standard_types : list[str]
        ChEMBL activity types to query.
    use_full_sequences : bool
        Fetch full UniProt sequences; False uses embedded truncations.

    Returns
    -------
    pd.DataFrame  columns: ``[smiles, protein, docking_score]``
    """
    target_names = list(targets or _PROTEIN_TARGETS.keys())
    protein_seqs: Dict[str, str] = {}
    for name in target_names:
        info = _PROTEIN_TARGETS.get(name)
        if info is None:
            continue
        if use_full_sequences:
            seq = _fetch_uniprot_fasta(info["uniprot_id"], timeout=timeout)
            protein_seqs[name] = seq or info["sequence"]
        else:
            protein_seqs[name] = info["sequence"]

    chembl_target_map = {
        "HER2": "CHEMBL240",
        "EGFR": "CHEMBL203",
        "VEGFR2": "CHEMBL279",
    }

    records: List[Dict[str, Any]] = []
    for tname in target_names:
        if tname not in protein_seqs:
            continue
        chembl_tid = chembl_target_map.get(tname)
        if not chembl_tid:
            continue
        for stype in standard_types:
            params = {
                "target_chembl_id": chembl_tid,
                "standard_type": stype,
                "has_smiles": "true",
                "format": "json",
                "limit": chembl_limit,
            }
            url = f"{_CHEMBL_API}/activity.json?{urllib.parse.urlencode(params)}"
            try:
                data = _chembl_get_json(url, timeout=timeout)
            except Exception:
                continue
            for act in data.get("activities", []):
                smi = act.get("canonical_smiles")
                pval = act.get("pchembl_value")
                if smi and pval:
                    try:
                        records.append({
                            "smiles": smi,
                            "protein": protein_seqs[tname],
                            "docking_score": float(pval),
                        })
                    except (ValueError, TypeError):
                        continue
            time.sleep(float(sleep_seconds))

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.drop_duplicates(subset=["smiles", "protein"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Unified "crawl all" convenience
# ---------------------------------------------------------------------------

def crawl_all_drug(
    *,
    # PubChem params
    pubchem_start_cid: int = 1,
    pubchem_n: int = 200,
    pubchem_cids: Optional[Iterable[int]] = None,
    pubchem_cache_dir: str | Path = "data/cache/pubchem",
    pubchem_max_workers: int = 4,
    pubchem_rate_limit: float = 5.0,
    # ChEMBL params
    chembl_target_ids: Optional[Sequence[str]] = None,
    chembl_standard_types: Optional[Sequence[str]] = None,
    chembl_limit: int = 5000,
    # Docking params
    include_docking: bool = False,
    docking_targets: Optional[Sequence[str]] = None,
    # General params
    sleep_seconds: float = 0.3,
    timeout: float = 30.0,
    progress_callback=None,
) -> Dict[str, pd.DataFrame]:
    """Run PubChem + ChEMBL (+ optional docking) crawlers and return named results.

    Returns
    -------
    dict  ``{"pubchem": DataFrame, "chembl": DataFrame, "docking": DataFrame | None}``
    """
    result: Dict[str, pd.DataFrame] = {}

    # 1. PubChem
    pubchem_df = crawl_pubchem_activity_dataset(
        start_cid=pubchem_start_cid,
        n=pubchem_n,
        cids=pubchem_cids,
        cache_dir=pubchem_cache_dir,
        max_workers=pubchem_max_workers,
        rate_limit=pubchem_rate_limit,
        timeout=timeout,
        sleep_seconds=sleep_seconds,
        progress_callback=progress_callback,
    )
    result["pubchem"] = pubchem_df

    # 2. ChEMBL
    chembl_df = crawl_chembl(
        target_ids=chembl_target_ids,
        standard_types=chembl_standard_types,
        limit_per_query=chembl_limit,
        timeout=timeout,
        sleep_seconds=sleep_seconds,
    )
    result["chembl"] = chembl_df

    # 3. Docking (optional)
    if include_docking:
        docking_df = crawl_docking_sources(
            targets=docking_targets,
            timeout=timeout,
            sleep_seconds=sleep_seconds,
        )
        result["docking"] = docking_df
    else:
        result["docking"] = pd.DataFrame()

    return result
