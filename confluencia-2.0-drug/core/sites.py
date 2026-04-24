from __future__ import annotations

"""Register drug-related crawl sites.

Importing this module has side effects: it registers site crawlers into
`src.common.dataset_fetch` registry.

We keep this separate to avoid circular imports between common<->drug.
"""

from pathlib import Path
from typing import Union, Optional, List

import pandas as pd

from confluencia_shared.utils.dataset_fetch import SiteInfo, register_site
from tools.crawler import crawl_pubchem_activity_dataset


@register_site(
    SiteInfo(
        name="pubchem",
        kind="chem",
        description="PubChem PUG-REST API：按CID范围抓取并计算 activity_score 作为代理标签",
    )
)
def crawl_pubchem(
    *,
    start_cid: int = 1,
    n: int = 200,
    sleep_seconds: float = 0.2,
    min_total_outcomes: int = 5,
    cache_dir: Union[str, Path] = "data/cache/pubchem",
    timeout: float = 30.0,
    indication: Optional[str] = None,
    # advanced
    max_workers: int = 4,
    rate_limit: float = 5.0,
    weighted: bool = False,
    include_outcome_breakdown: bool = False,
    normalize_smiles: bool = False,
    include_properties: bool = False,
    property_fields: Optional[List[str]] = None,
    include_synonyms: bool = False,
) -> pd.DataFrame:
    return crawl_pubchem_activity_dataset(
        start_cid=int(start_cid),
        n=int(n),
        sleep_seconds=float(sleep_seconds),
        min_total_outcomes=int(min_total_outcomes),
        cache_dir=str(cache_dir),
        timeout=float(timeout),
        indication=indication,
        max_workers=int(max_workers),
        rate_limit=float(rate_limit),
        weighted=bool(weighted),
        include_outcome_breakdown=bool(include_outcome_breakdown),
        normalize_smiles=bool(normalize_smiles),
        include_properties=bool(include_properties),
        property_fields=list(property_fields) if property_fields else None,
        include_synonyms=bool(include_synonyms),
    )
