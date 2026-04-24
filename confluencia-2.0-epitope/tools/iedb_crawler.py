from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from src.common.dataset_fetch import concat_tables, download_to_cache, is_http_url


AA_STANDARD = set("ACDEFGHIKLMNPQRSTVWY")
_UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"
_PDB_FASTA_URL = "https://www.rcsb.org/fasta/entry/{pdb}"


def _clean_sequence(seq: str, *, allow_x: bool = False) -> Tuple[str, bool]:
    if seq is None:
        return "", False
    s = str(seq).strip().upper().replace(" ", "")
    s = s.replace("-", "").replace(".", "")
    if not s:
        return "", False

    allowed = set(AA_STANDARD)
    if allow_x:
        allowed.add("X")

    for ch in s:
        if ch not in allowed:
            return s, False
    return s, True


def _filter_sequences(
    series: pd.Series,
    *,
    min_len: int = 8,
    max_len: int = 25,
    allow_x: bool = False,
) -> Tuple[pd.Series, pd.Series]:
    cleaned = []
    valids = []
    for s in series.astype(str).tolist():
        cs, ok = _clean_sequence(s, allow_x=allow_x)
        cleaned.append(cs)
        valids.append(bool(ok))

    cleaned_series = pd.Series(cleaned, index=series.index)
    len_ok = cleaned_series.str.len().between(int(min_len), int(max_len))
    valid_mask = pd.Series(valids, index=series.index) & len_ok
    return cleaned_series, valid_mask


def parse_fasta_text(text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not text:
        return rows

    seq_id = None
    desc = None
    buf: List[str] = []

    def _flush() -> None:
        nonlocal seq_id, desc, buf
        if seq_id is None:
            return
        seq = "".join(buf).strip()
        rows.append({"sequence": seq, "seq_id": str(seq_id), "description": str(desc or "")})
        seq_id = None
        desc = None
        buf = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            _flush()
            header = line[1:].strip()
            if header:
                parts = header.split(maxsplit=1)
                seq_id = parts[0]
                desc = parts[1] if len(parts) > 1 else ""
            else:
                seq_id = ""
                desc = ""
        else:
            buf.append(line)
    _flush()
    return rows


def _is_uniprot_source(src: str) -> bool:
    s = str(src).strip().lower()
    return s.startswith("uniprot:") or s.startswith("uniprotkb:")


def _is_pdb_source(src: str) -> bool:
    s = str(src).strip().lower()
    return s.startswith("pdb:") or s.startswith("rcsb:")


def _parse_uniprot_accessions(src: str) -> List[str]:
    s = str(src).strip()
    if not s:
        return []
    if ":" in s:
        s = s.split(":", 1)[1]
    parts = re.split(r"[\s,;]+", s.strip())
    return [p for p in parts if p]


def _parse_pdb_ids(src: str) -> List[str]:
    s = str(src).strip()
    if not s:
        return []
    if ":" in s:
        s = s.split(":", 1)[1]
    parts = re.split(r"[\s,;]+", s.strip())
    return [p.strip() for p in parts if p.strip()]


def crawl_epitope_fasta_sources(
    urls_or_paths: Sequence[str],
    *,
    cache_dir: Union[str, Path] = "data/cache/epitope",
    timeout: float = 30.0,
    sleep_seconds: float = 0.2,
    user_agent: str = "epitope-crawler/1.0 (research; contact: local)",
    min_len: int = 8,
    max_len: int = 25,
    allow_x: bool = False,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    headers: Dict[str, str] = {"User-Agent": str(user_agent)}
    rows: List[Dict[str, str]] = []

    for src in urls_or_paths:
        src = str(src).strip()
        if not src:
            continue

        if _is_uniprot_source(src):
            accs = _parse_uniprot_accessions(src)
            if not accs:
                continue
            for acc in accs:
                url = _UNIPROT_FASTA_URL.format(acc=acc)
                art = download_to_cache(
                    url,
                    cache_dir=cache_dir,
                    timeout=timeout,
                    sleep_seconds=sleep_seconds,
                    headers=headers,
                )
                text = art.path.read_text(encoding="utf-8", errors="ignore")
                parsed = parse_fasta_text(text)
                for r in parsed:
                    r["_source"] = f"uniprot:{acc}"
                rows.extend(parsed)
            continue

        if _is_pdb_source(src):
            pdb_ids = _parse_pdb_ids(src)
            if not pdb_ids:
                continue
            for pdb in pdb_ids:
                url = _PDB_FASTA_URL.format(pdb=pdb)
                art = download_to_cache(
                    url,
                    cache_dir=cache_dir,
                    timeout=timeout,
                    sleep_seconds=sleep_seconds,
                    headers=headers,
                )
                text = art.path.read_text(encoding="utf-8", errors="ignore")
                parsed = parse_fasta_text(text)
                for r in parsed:
                    r["_source"] = f"pdb:{pdb}"
                rows.extend(parsed)
            continue

        if is_http_url(src):
            art = download_to_cache(
                src,
                cache_dir=cache_dir,
                timeout=timeout,
                sleep_seconds=sleep_seconds,
                headers=headers,
            )
            text = art.path.read_text(encoding="utf-8", errors="ignore")
        else:
            p = Path(src)
            if not p.exists():
                raise FileNotFoundError(str(p))
            text = p.read_text(encoding="utf-8", errors="ignore")

        parsed = parse_fasta_text(text)
        for r in parsed:
            r["_source"] = src
        rows.extend(parsed)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    cleaned, mask = _filter_sequences(df["sequence"], min_len=min_len, max_len=max_len, allow_x=allow_x)
    df["sequence"] = cleaned
    df = df.loc[mask].copy()

    if drop_duplicates and not df.empty:
        df = df.drop_duplicates(subset=["sequence"]).copy()

    return df


def clean_epitope_table(
    df: pd.DataFrame,
    *,
    sequence_col: str,
    min_len: int = 8,
    max_len: int = 25,
    allow_x: bool = False,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    if sequence_col not in df.columns:
        raise ValueError(f"sequence 列不存在: {sequence_col}")

    cleaned, mask = _filter_sequences(df[sequence_col], min_len=min_len, max_len=max_len, allow_x=allow_x)
    out = df.copy()
    out[sequence_col] = cleaned
    out = out.loc[mask].copy()

    if drop_duplicates and not out.empty:
        out = out.drop_duplicates(subset=[sequence_col]).copy()
    return out


def crawl_epitope_csv_datasets(
    urls_or_paths: Sequence[str],
    *,
    cache_dir: Union[str, Path] = "data/cache/epitope",
    timeout: float = 30.0,
    sleep_seconds: float = 0.2,
    user_agent: str = "epitope-crawler/1.0 (research; contact: local)",
    sequence_col: Optional[str] = None,
    min_len: int = 8,
    max_len: int = 25,
    allow_x: bool = False,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """Fetch user-provided CSV/TSV/Excel datasets and concatenate.

    Notes
    -----
    - 这是一个通用的“轻量爬虫”：只抓取用户提供的 URL/文件路径。
    - 默认会把下载内容缓存到 cache_dir，避免重复请求。
    """

    headers: Dict[str, str] = {"User-Agent": str(user_agent)}
    df = concat_tables(
        list(urls_or_paths),
        cache_dir=str(cache_dir),
        timeout=float(timeout),
        sleep_seconds=float(sleep_seconds),
        headers=headers,
    )
    if df.empty:
        return df

    if sequence_col:
        return clean_epitope_table(
            df,
            sequence_col=str(sequence_col),
            min_len=int(min_len),
            max_len=int(max_len),
            allow_x=bool(allow_x),
            drop_duplicates=bool(drop_duplicates),
        )

    return df
