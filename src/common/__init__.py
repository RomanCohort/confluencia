"""
src.common — shared data-fetch utilities for Confluencia crawlers.

All heavy lifting is delegated to ``confluencia_shared.utils.*`` so that
a single implementation is maintained.  Submodules re-export the public
API so that crawlers can simply::

    from src.common.dataset_fetch import concat_tables, download_to_cache
"""
