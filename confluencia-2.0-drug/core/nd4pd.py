from __future__ import annotations

# Backward-compatibility shim: ND4PD old naming points to corrected NDP4PD implementation.
from .ndp4pd import NDP4PDParams as ND4PDParams
from .ndp4pd import ndp4pd_from_ctm_like as nd4pd_from_ctm_like
from .ndp4pd import simulate_ndp4pd as simulate_nd4pd
