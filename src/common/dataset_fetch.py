"""
src.common.dataset_fetch — thin re-export layer.

Re-exports everything from ``confluencia_shared.utils.dataset_fetch`` so
that crawlers written against the early-version path (``src.common.dataset_fetch``)
keep working without import changes.
"""

from confluencia_shared.utils.dataset_fetch import (  # noqa: F401
    DownloadedArtifact,
    SiteInfo,
    concat_tables,
    crawl_site,
    download_to_cache,
    get_site,
    is_http_url,
    list_sites,
    read_table_any,
    register_site,
)

__all__ = [
    "DownloadedArtifact",
    "SiteInfo",
    "concat_tables",
    "crawl_site",
    "download_to_cache",
    "get_site",
    "is_http_url",
    "list_sites",
    "read_table_any",
    "register_site",
]
