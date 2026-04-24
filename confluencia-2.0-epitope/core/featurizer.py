from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


AA_ORDER: Tuple[str, ...] = (
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
)

# Kyte-Doolittle hydropathy index
HYDROPATHY: Dict[str, float] = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}

# Rough net charge contribution around pH~7
CHARGE_EST: Dict[str, float] = {
    "D": -1.0,
    "E": -1.0,
    "K": 1.0,
    "R": 1.0,
    "H": 0.1,
}

AROMATIC = {"F", "W", "Y"}
ACIDIC = {"D", "E"}
BASIC = {"K", "R", "H"}
POLAR_UNCHARGED = {"S", "T", "N", "Q", "C", "Y"}
NONPOLAR = {"A", "V", "L", "I", "P", "F", "M", "W", "G"}
SMALL = {"A", "G", "S", "T", "P"}


def _clean_sequence(seq: str) -> str:
    if seq is None:
        return ""
    return str(seq).strip().upper().replace(" ", "")


def _safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d else 0.0


@dataclass(frozen=True)
class SequenceFeatures:
    """Feature extractor for epitope (peptide) sequences.

    version=1:
      Output features = [AAC(20), global_stats(12)].

    version=2:
      Output features = [AAC(20), global_stats(12), region_stats(6)].

      region_stats splits the sequence into three regions (N-term / middle / C-term)
      by thirds (rounded, but each region has at least 1 residue when length>0).

      region_stats:
        - hydropathy_mean_n, hydropathy_mean_mid, hydropathy_mean_c
        - frac_nonpolar_n, frac_nonpolar_mid, frac_nonpolar_c
    """

    version: int = 1

    def feature_names(self) -> List[str]:
        names = [f"aac_{aa}" for aa in AA_ORDER]
        names += [
            "len",
            "hydropathy_mean",
            "hydropathy_std",
            "net_charge_est",
            "frac_nonpolar",
            "frac_polar_uncharged",
            "frac_acidic",
            "frac_basic",
            "frac_aromatic",
            "frac_small",
            "frac_proline",
            "frac_glycine",
        ]

        if int(self.version) >= 2:
            names += [
                "hydropathy_mean_n",
                "hydropathy_mean_mid",
                "hydropathy_mean_c",
                "frac_nonpolar_n",
                "frac_nonpolar_mid",
                "frac_nonpolar_c",
            ]

        return names

    def _region_slices(self, length: int) -> Tuple[slice, slice, slice]:
        if length <= 0:
            return (slice(0, 0), slice(0, 0), slice(0, 0))

        # Split by thirds; ensure each region gets at least 1 aa when possible.
        a = max(1, length // 3)
        b = max(1, (length - a) // 2)
        c = max(1, length - a - b)

        # If rounding caused overflow, trim from middle then C.
        while a + b + c > length and b > 1:
            b -= 1
        while a + b + c > length and c > 1:
            c -= 1

        n_end = a
        mid_end = a + b
        return (slice(0, n_end), slice(n_end, mid_end), slice(mid_end, length))

    def transform_one(self, sequence: str) -> np.ndarray:
        seq = _clean_sequence(sequence)
        length = len(seq)

        counts = {aa: 0 for aa in AA_ORDER}
        hydros: List[float] = []
        net_charge = 0.0

        n_nonpolar = 0
        n_polar = 0
        n_acidic = 0
        n_basic = 0
        n_aromatic = 0
        n_small = 0
        n_pro = 0
        n_gly = 0

        for ch in seq:
            if ch in counts:
                counts[ch] += 1
                hydros.append(HYDROPATHY.get(ch, 0.0))
                net_charge += CHARGE_EST.get(ch, 0.0)

                if ch in NONPOLAR:
                    n_nonpolar += 1
                if ch in POLAR_UNCHARGED:
                    n_polar += 1
                if ch in ACIDIC:
                    n_acidic += 1
                if ch in BASIC:
                    n_basic += 1
                if ch in AROMATIC:
                    n_aromatic += 1
                if ch in SMALL:
                    n_small += 1
                if ch == "P":
                    n_pro += 1
                if ch == "G":
                    n_gly += 1
            else:
                # Unknown residue (e.g., X, B, Z): ignore for composition categories
                # but still count towards length.
                hydros.append(0.0)

        aac = np.array([_safe_div(counts[aa], length) for aa in AA_ORDER], dtype=np.float32)

        if hydros:
            hydro_arr = np.array(hydros, dtype=np.float32)
            hydro_mean = float(hydro_arr.mean())
            hydro_std = float(hydro_arr.std())
        else:
            hydro_mean = 0.0
            hydro_std = 0.0

        stats = np.array(
            [
                float(length),
                hydro_mean,
                hydro_std,
                float(net_charge),
                _safe_div(n_nonpolar, length),
                _safe_div(n_polar, length),
                _safe_div(n_acidic, length),
                _safe_div(n_basic, length),
                _safe_div(n_aromatic, length),
                _safe_div(n_small, length),
                _safe_div(n_pro, length),
                _safe_div(n_gly, length),
            ],
            dtype=np.float32,
        )

        if int(self.version) < 2:
            return np.concatenate([aac, stats], axis=0)

        # Region features: N / middle / C thirds
        n_slice, mid_slice, c_slice = self._region_slices(length)
        hydro_arr = np.array(hydros, dtype=np.float32) if hydros else np.zeros((length,), dtype=np.float32)

        def _mean(arr: np.ndarray) -> float:
            return float(arr.mean()) if arr.size else 0.0

        def _frac_nonpolar(subseq: str) -> float:
            if not subseq:
                return 0.0
            cnt = 0
            for ch in subseq:
                if ch in NONPOLAR:
                    cnt += 1
            return _safe_div(cnt, len(subseq))

        seq_n = seq[n_slice]
        seq_m = seq[mid_slice]
        seq_c = seq[c_slice]

        region = np.array(
            [
                _mean(hydro_arr[n_slice]),
                _mean(hydro_arr[mid_slice]),
                _mean(hydro_arr[c_slice]),
                _frac_nonpolar(seq_n),
                _frac_nonpolar(seq_m),
                _frac_nonpolar(seq_c),
            ],
            dtype=np.float32,
        )

        return np.concatenate([aac, stats, region], axis=0)

    def transform_many(self, sequences: Sequence[str]) -> np.ndarray:
        if not sequences:
            return np.zeros((0, len(self.feature_names())), dtype=np.float32)
        cache: Dict[str, np.ndarray] = {}
        feats = []
        for s in sequences:
            key = "" if s is None else str(s)
            if key in cache:
                feat = cache[key]
            else:
                feat = self.transform_one(key)
                cache[key] = feat
            feats.append(feat)
        return np.vstack(feats).astype(np.float32)
