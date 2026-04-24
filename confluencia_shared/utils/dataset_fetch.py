from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import pandas as pd
import requests
_SESSION = requests.Session()


def _request_bytes(
    url: str,
    *,
    timeout: float,
    headers: Optional[Dict[str, str]],
    retries: int,
    backoff_factor: float,
) -> bytes:
    last_exc: Optional[Exception] = None
    for attempt in range(int(retries) if retries and retries > 0 else 1):
        try:
            resp = _SESSION.get(url, timeout=float(timeout), headers=headers)
            resp.raise_for_status()
            return resp.content
        except Exception as exc:
            last_exc = exc
            if attempt + 1 < int(retries):
                time.sleep(float(backoff_factor) * (2 ** attempt))
                continue
            break
    raise RuntimeError("request failed") from last_exc


def is_http_url(s: str) -> bool:
    s = str(s).strip().lower()
    return s.startswith("http://") or s.startswith("https://")


def _hash_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


@dataclass(frozen=True)
class DownloadedArtifact:
    url: str
    path: Path
    n_bytes: int


def download_to_cache(
    url: str,
    *,
    cache_dir: Union[str, Path] = "data/cache/http",
    timeout: float = 30.0,
    sleep_seconds: float = 0.0,
    headers: Optional[Dict[str, str]] = None,
    retries: int = 3,
    backoff_factor: float = 0.5,
) -> DownloadedArtifact:
    """Download a URL to a cache directory.

    This is a best-effort helper meant for *user-provided* URLs.
    """

    if not is_http_url(url):
        raise ValueError(f"Not an http(s) URL: {url}")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = _hash_key(url)
    # keep extension if any (best effort)
    suffix = Path(url.split("?", 1)[0]).suffix
    if not suffix:
        suffix = ".bin"

    out_path = cache_dir / f"{key}{suffix}"
    if out_path.exists() and out_path.stat().st_size > 0:
        return DownloadedArtifact(url=url, path=out_path, n_bytes=int(out_path.stat().st_size))

    data = _request_bytes(url, timeout=float(timeout), headers=headers, retries=int(retries), backoff_factor=float(backoff_factor))
    out_path.write_bytes(data)

    if float(sleep_seconds) > 0:
        time.sleep(float(sleep_seconds))

    return DownloadedArtifact(url=url, path=out_path, n_bytes=len(data))


def read_table_any(
    source: str,
    *,
    cache_dir: Union[str, Path] = "data/cache/http",
    timeout: float = 30.0,
    sleep_seconds: float = 0.0,
    headers: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Read CSV/TSV/Excel from a local path or http(s) URL."""

    if is_http_url(source):
        art = download_to_cache(source, cache_dir=cache_dir, timeout=timeout, sleep_seconds=sleep_seconds, headers=headers)
        p = art.path
    else:
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(str(p))

    ext = p.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    if ext in {".tsv", ".txt"}:
        return pd.read_csv(p, sep="\t")
    # default: CSV
    return pd.read_csv(p)


def concat_tables(sources: Sequence[str], **kwargs) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for s in sources:
        df = read_table_any(str(s), **kwargs)
        df = df.copy()
        df["_source"] = str(s)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)


# ----------------------------
# Crawler registry (site selector)
# ----------------------------


@dataclass(frozen=True)
class SiteInfo:
    """Metadata for a crawl site.

    kind is used for UI filtering, e.g.:
      - table: returns a generic table suitable for CSV/Excel workflow
      - bio/chem: domain-specific sources
    """

    name: str
    description: str
    kind: str = "table"


_SITE_REGISTRY: Dict[str, tuple[SiteInfo, Callable[..., pd.DataFrame]]] = {}


def register_site(site: SiteInfo):
    """Decorator to register a crawler function under site.name."""

    def _wrap(fn: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
        key = str(site.name).strip().lower()
        if not key:
            raise ValueError("site.name 不能为空")
        _SITE_REGISTRY[key] = (site, fn)
        return fn

    return _wrap


def list_sites(*, kind: Optional[str] = None) -> List[SiteInfo]:
    items = [info for (info, _fn) in _SITE_REGISTRY.values()]
    if kind:
        items = [s for s in items if str(s.kind).lower() == str(kind).lower()]
    items.sort(key=lambda s: (s.kind, s.name))
    return items


def get_site(site: str) -> tuple[SiteInfo, Callable[..., pd.DataFrame]]:
    key = str(site).strip().lower()
    if key not in _SITE_REGISTRY:
        known = ", ".join(sorted(_SITE_REGISTRY.keys()))
        raise ValueError(f"未知站点: {site}（可用: {known}）")
    return _SITE_REGISTRY[key]


def crawl_site(site: str, **kwargs) -> pd.DataFrame:
    """Run a registered crawler by name."""

    _info, fn = get_site(site)
    return fn(**kwargs)


@register_site(
    SiteInfo(
        name="urlcsv",
        kind="table",
        description="用户提供 CSV/TSV/Excel 的 URL 或本地路径；下载后合并为一个表并添加 _source 列",
    )
)
def crawl_urlcsv(
    *,
    sources: Sequence[str],
    cache_dir: Union[str, Path] = "data/cache/http",
    timeout: float = 30.0,
    sleep_seconds: float = 0.2,
    user_agent: str = "urlcsv-crawler/1.0 (research; contact: local)",
) -> pd.DataFrame:
    if not sources:
        raise ValueError("sources 不能为空")
    headers: Dict[str, str] = {"User-Agent": str(user_agent)}
    return concat_tables(
        list(sources),
        cache_dir=str(cache_dir),
        timeout=float(timeout),
        sleep_seconds=float(sleep_seconds),
        headers=headers,
    )
