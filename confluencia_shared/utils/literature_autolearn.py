from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from urllib.parse import quote

import requests


_EUROPE_PMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_SESSION = requests.Session()


_DOMAIN_DEFAULT_TERMS: Dict[str, List[str]] = {
    "drug": ["drug", "compound", "bioassay", "activity", "smiles", "binding"],
    "epitope": ["epitope", "peptide", "mhc", "immunogenic", "iedb"],
    "docking": ["docking", "binding affinity", "pdbbind", "bindingdb"],
    "multiscale": ["multiscale", "kinetics", "diffusion", "vmax", "km"],
    "custom": [],
}

_DOMAIN_DEFAULT_SOURCES: Dict[str, List[str]] = {
    "drug": ["PubChem", "ChEMBL", "BindingDB", "DrugBank", "ZINC"],
    "epitope": ["IEDB", "UniProt", "PDB", "SAbDab"],
    "docking": ["PDBbind", "BindingDB", "DUD-E", "PDB"],
    "multiscale": ["BRENDA", "SABIO-RK", "BioModels", "KEGG"],
    "custom": ["Zenodo", "Figshare", "Dryad", "Dataverse"],
}

_DOMAIN_FIELD_HINTS: Dict[str, List[str]] = {
    "drug": ["smiles", "activity_score", "assay", "target", "cid"],
    "epitope": ["sequence", "allele", "label", "source", "length"],
    "docking": ["ligand_smiles", "protein_sequence", "binding_score", "protein_pdb"],
    "multiscale": ["smiles", "target", "D", "Vmax", "Km"],
    "custom": ["id", "label", "feature", "metadata"],
}

_DATASET_KEYWORDS: List[Tuple[str, str]] = [
    ("pubchem", "PubChem"),
    ("chembl", "ChEMBL"),
    ("bindingdb", "BindingDB"),
    ("pdbbind", "PDBbind"),
    ("pdb", "PDB"),
    ("drugbank", "DrugBank"),
    ("zinc", "ZINC"),
    ("iedb", "IEDB"),
    ("uniprot", "UniProt"),
    ("sabdab", "SAbDab"),
    ("dud-e", "DUD-E"),
    ("brenda", "BRENDA"),
    ("sabio", "SABIO-RK"),
    ("biomodels", "BioModels"),
    ("kegg", "KEGG"),
    ("zenodo", "Zenodo"),
    ("figshare", "Figshare"),
    ("dryad", "Dryad"),
    ("dataverse", "Dataverse"),
    ("github", "GitHub"),
]

_LINK_HINT_RE = re.compile(r"https?://[^\s)\]}>\"']+", re.IGNORECASE)


@dataclass(frozen=True)
class LiteratureItem:
    title: str
    year: Optional[int]
    journal: str
    authors: str
    doi: str
    pmid: str
    pmcid: str
    source: str
    abstract: str
    paper_url: str
    fulltext_urls: List[str]
    dataset_hints: List[str]
    link_hints: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "year": self.year,
            "journal": self.journal,
            "authors": self.authors,
            "doi": self.doi,
            "pmid": self.pmid,
            "pmcid": self.pmcid,
            "source": self.source,
            "abstract": self.abstract,
            "paper_url": self.paper_url,
            "fulltext_urls": " | ".join(self.fulltext_urls),
            "dataset_hints": " | ".join(self.dataset_hints),
            "link_hints": " | ".join(self.link_hints),
        }


def _hash_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _request_json(
    url: str,
    *,
    timeout: float,
    headers: Optional[Dict[str, str]],
    retries: int,
    backoff_factor: float,
) -> Dict[str, Any]:
    last_exc: Optional[Exception] = None
    for attempt in range(int(retries) if retries and retries > 0 else 1):
        try:
            resp = _SESSION.get(url, timeout=float(timeout), headers=headers)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc
            if attempt + 1 < int(retries):
                time.sleep(float(backoff_factor) * (2 ** attempt))
                continue
            break
    raise RuntimeError("request failed") from last_exc


def _normalize_terms(text: str) -> List[str]:
    tokens = [t.strip() for t in re.split(r"[;,\n]+", str(text)) if t.strip()]
    return [t for t in tokens if t]


def _build_query(
    *,
    domain: str,
    keywords: Sequence[str],
    sources: Optional[Sequence[str]],
    year_from: Optional[int],
    year_to: Optional[int],
    include_preprints: bool,
) -> str:
    terms: List[str] = []
    domain_key = str(domain or "custom").strip().lower()
    terms.extend(_DOMAIN_DEFAULT_TERMS.get(domain_key, []))
    for kw in keywords:
        kw = str(kw).strip()
        if kw:
            terms.append(kw)

    if not terms:
        terms = ["dataset", "training data"]

    query = " AND ".join([f"({t})" if " " in t else t for t in terms])

    if sources:
        cleaned = [str(s).strip() for s in sources if str(s).strip()]
        if cleaned:
            src_q = " OR ".join([f'"{s}"' if (" " in s or "/" in s) else s for s in cleaned])
            query = f"{query} AND ({src_q})"

    if year_from or year_to:
        y_from = int(year_from) if year_from else 1900
        y_to = int(year_to) if year_to else int(time.strftime("%Y"))
        query = f"{query} AND PUB_YEAR:[{y_from} TO {y_to}]"

    if not include_preprints:
        query = f"{query} NOT SRC:PPR"

    return query


def _extract_dataset_hints(text: str) -> List[str]:
    text_l = str(text or "").lower()
    hits: List[str] = []
    for key, label in _DATASET_KEYWORDS:
        if key in text_l:
            hits.append(label)
    return sorted(set(hits))


def _extract_link_hints(text: str) -> List[str]:
    links = _LINK_HINT_RE.findall(str(text or ""))
    return sorted(set(links))


def _search_europe_pmc(
    query: str,
    *,
    max_results: int,
    timeout: float,
    retries: int,
    backoff_factor: float,
    user_agent: str,
) -> List[Dict[str, Any]]:
    page_size = min(max(int(max_results), 1), 100)
    cursor = "*"
    out: List[Dict[str, Any]] = []
    headers = {"User-Agent": str(user_agent)}

    while len(out) < max_results:
        url = (
            f"{_EUROPE_PMC_SEARCH}?query={quote(query)}"
            f"&format=json&pageSize={page_size}&cursorMark={quote(cursor)}"
        )
        data = _request_json(url, timeout=float(timeout), headers=headers, retries=int(retries), backoff_factor=float(backoff_factor))
        results = data.get("resultList", {}).get("result", [])
        if not results:
            break
        out.extend(results)
        next_cursor = data.get("nextCursorMark")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor

    return out[:max_results]


def _build_paper_url(doi: str, pmid: str, pmcid: str) -> str:
    if doi:
        return f"https://doi.org/{doi}"
    if pmcid:
        return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
    if pmid:
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    return ""


def _extract_fulltext_urls(raw: Any) -> List[str]:
    urls: List[str] = []
    if isinstance(raw, dict):
        raw = raw.get("fullTextUrl", [])
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                u = str(item.get("url", "")).strip()
                if u:
                    urls.append(u)
    return urls


def _aggregate_suggestions(items: Sequence[LiteratureItem], domain: str) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for item in items:
        for hint in item.dataset_hints:
            counts[hint] = counts.get(hint, 0) + 1

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top_sources = [name for name, _n in ranked][:8]
    if not top_sources:
        top_sources = _DOMAIN_DEFAULT_SOURCES.get(domain, _DOMAIN_DEFAULT_SOURCES["custom"])

    field_hints = _DOMAIN_FIELD_HINTS.get(domain, _DOMAIN_FIELD_HINTS["custom"])
    actions = [
        "优先从开放数据库导出标准化表格（CSV/TSV）",
        "检查数据许可与使用条款，保留引用信息",
        "统一列名（smiles/sequence/label），记录来源字段",
        "对文本/序列字段做去重与长度过滤",
    ]

    return {
        "suggested_sources": top_sources,
        "field_hints": field_hints,
        "actions": actions,
    }


def _write_items_csv(items: Sequence[LiteratureItem]) -> bytes:
    with_csv = io.StringIO()
    writer = csv.DictWriter(with_csv, fieldnames=list(items[0].to_dict().keys()) if items else [])
    if items:
        writer.writeheader()
        for item in items:
            writer.writerow(item.to_dict())
    return with_csv.getvalue().encode("utf-8")


def literature_autolearn(
    *,
    query: str,
    domain: str = "custom",
    keywords: Optional[Sequence[str]] = None,
    sources: Optional[Sequence[str]] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    include_preprints: bool = False,
    max_results: int = 20,
    timeout: float = 30.0,
    retries: int = 3,
    backoff_factor: float = 0.5,
    user_agent: str = "literature-autolearn/1.0 (research; contact: local)",
    cache_dir: Optional[Union[str, Path]] = None,
    include_csv: bool = False,
) -> Dict[str, Any]:
    domain_key = str(domain or "custom").strip().lower()
    if domain_key not in _DOMAIN_DEFAULT_TERMS:
        domain_key = "custom"

    keywords_list = list(keywords) if keywords else []
    if query:
        keywords_list.insert(0, str(query))

    built_query = _build_query(
        domain=domain_key,
        keywords=keywords_list,
        sources=sources,
        year_from=year_from,
        year_to=year_to,
        include_preprints=bool(include_preprints),
    )

    cache_path: Optional[Path] = None
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_key = _hash_key(built_query)
        cache_file = cache_path / f"eupmc_{cache_key}.json"
        if cache_file.exists() and cache_file.stat().st_size > 0:
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8", errors="ignore"))
                raw_results = data.get("results", [])
            except Exception:
                raw_results = []
        else:
            raw_results = _search_europe_pmc(
                built_query,
                max_results=int(max_results),
                timeout=float(timeout),
                retries=int(retries),
                backoff_factor=float(backoff_factor),
                user_agent=str(user_agent),
            )
            cache_file.write_text(json.dumps({"results": raw_results}, ensure_ascii=True), encoding="utf-8")
    else:
        raw_results = _search_europe_pmc(
            built_query,
            max_results=int(max_results),
            timeout=float(timeout),
            retries=int(retries),
            backoff_factor=float(backoff_factor),
            user_agent=str(user_agent),
        )

    items: List[LiteratureItem] = []
    for res in raw_results:
        title = str(res.get("title", "")).strip()
        abstract = str(res.get("abstractText", "")).strip()
        year: Optional[int] = None
        pub_year = res.get("pubYear")
        if pub_year is not None and str(pub_year).strip():
            try:
                year = int(str(pub_year).strip())
            except Exception:
                year = None
        journal = str(res.get("journalTitle", "")).strip()
        authors = str(res.get("authorString", "")).strip()
        doi = str(res.get("doi", "")).strip()
        pmid = str(res.get("pmid", "")).strip()
        pmcid = str(res.get("pmcid", "")).strip()
        source = str(res.get("source", "")).strip()
        fulltext_urls = _extract_fulltext_urls(res.get("fullTextUrlList"))
        paper_url = _build_paper_url(doi, pmid, pmcid)
        combined_text = f"{title}\n{abstract}"
        dataset_hints = _extract_dataset_hints(combined_text)
        link_hints = _extract_link_hints(abstract)

        items.append(
            LiteratureItem(
                title=title,
                year=year,
                journal=journal,
                authors=authors,
                doi=doi,
                pmid=pmid,
                pmcid=pmcid,
                source=source,
                abstract=abstract,
                paper_url=paper_url,
                fulltext_urls=fulltext_urls,
                dataset_hints=dataset_hints,
                link_hints=link_hints,
            )
        )

    suggestions = _aggregate_suggestions(items, domain_key)

    result: Dict[str, Any] = {
        "query": built_query,
        "domain": domain_key,
        "n_items": len(items),
        "items": [item.to_dict() for item in items],
        "suggestions": suggestions,
    }

    if include_csv and items:
        csv_bytes = _write_items_csv(items)
        result["result_csv_b64"] = base64.b64encode(csv_bytes).decode("ascii")

    return result
