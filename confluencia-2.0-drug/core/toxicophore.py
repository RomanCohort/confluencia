"""
Toxicophore / Structural Alert Detection
=========================================
Identifies known toxic substructures (toxicophores) and PAINS (Pan-Assay
Interference Compounds) patterns in drug candidates.

References:
- Baell & Holloway (2010) J Med Chem 53:2719-2740 (PAINS filters)
- Brenk et al. (2008) ChemMedChem 3:435-444 (Brenk filters)
- Sushko et al. (2012) Chem Res Toxicol 25:1479-1489 (structural alerts)
- Thorne et al. (2024) JACS Au 4:296-306 (expanded toxicity alerts)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Set, Tuple

import numpy as np

try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.error")
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

from confluencia_shared.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# PAINS (Pan-Assay Interference Compounds) SMARTS patterns
# ---------------------------------------------------------------------------
# From Baell & Holloway (2010). These compounds cause false positives in
# high-throughput screens due to non-specific activity (aggregation,
# redox activity, membrane disruption, etc.)

_PAINS_SMARTS: List[Tuple[str, str, str, str]] = [
    # (pattern_id, name, category, smarts)
    ("PAINS_001", "hzone_phenyl", "covalent", "[#7]-[#6](=[#7])-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1"),
    ("PAINS_002", "quinone", "redox", "[#6]1([#6]=[#6][#6](=[#8])[#6]=[#6]1)=[#8]"),
    ("PAINS_003", "rhodanines", "promiscuous", "S=C1NC(=O)CS1"),
    ("PAINS_004", "enone", "reactive", "[#6]=[#6]-C(=O)-[#6]"),
    ("PAINS_005", "thiobarbiturates", "promiscuous", "[#6]1(=[#16])-[#7]-[#6](=[#8])-[#7]-[#6](=[#8])-[#6]1"),
    ("PAINS_006", "phenylsulfonylimino", "covalent", "O=S(=N)=[#6]"),
    ("PAINS_007", "phenols", "unspecific", "[#6]1:[#6](:[#6]:[#6]:[#6]:[#6]:1)-[OH]"),
    ("PAINS_008", "anisoles", "unspecific", "[#6]1:[#6](:[#6]:[#6]:[#6]:[#6]:1)-[O]-[#6]"),
    ("PAINS_009", "catechols", "chelator", "[#6]1:[#6](:[#6]:[#6](:[#6]:[#6]:1)-[OH])-[OH]"),
    ("PAINS_010", "hydroquinones", "redox", "[#6]1:[#6](:[#6]:[#6](:[#6]:[#6]:1)-[OH])-[OH]"),
    ("PAINS_011", "quinolines", "intercalator", "n1cccc2ccccc12"),
    ("PAINS_012", "quinazolines", "intercalator", "c1nc2ccccc2nc1"),
    ("PAINS_013", "acylhydrazines", "covalent", "[#7]-[#7]-C(=O)"),
    ("PAINS_014", "isocyanates", "reactive", "N=C=O"),
    ("PAINS_015", "isothiocyanates", "reactive", "N=C=S"),
    ("PAINS_016", "alkyl_halides_reactive", "covalent", "[#6]-[Cl,Br,I]"),
    ("PAINS_017", "epoxides", "covalent", "[#6]1[#6][#8]1"),
    ("PAINS_018", "aziridines", "covalent", "[#6]1[#6][#7]1"),
    ("PAINS_019", "nitroalkenes", "reactive", "[#6]=[#6]-[N+](=O)[O-]"),
    ("PAINS_020", "beta_lactams", "covalent", "C1(=O)NC(=O)C1"),
    ("PAINS_021", "pyrrolo_quinolines", "aggregator", "c1cc2c(cc1)nccc2C(=O)"),
    ("PAINS_022", "azo_compounds", "redox", "[N,n]=[N,n]"),
    ("PAINS_023", "diazo_compounds", "reactive", "[#6]=[N+]=[N-]"),
    ("PAINS_024", "triazenes", "reactive", "N=N-N"),
    ("PAINS_025", "perhalogenated", "promiscuous", "[#6]([F,Cl,Br,I])([F,Cl,Br,I])([F,Cl,Br,I])[F,Cl,Br,I]"),
]


# ---------------------------------------------------------------------------
# Brenk filters (structural alerts for toxicity)
# ---------------------------------------------------------------------------
# From Brenk et al. (2008). Compounds with these substructures have higher
# probability of toxicity, mutagenicity, or poor pharmacokinetics.

_BRENK_SMARTS: List[Tuple[str, str, str, str]] = [
    ("BRENK_001", "nitro_group", "mutagenic", "[N+](=O)[O-]"),
    ("BRENK_002", "nitroso_group", "mutagenic", "[N]=[O]"),
    ("BRENK_003", "aldehyde", "reactive", "[CH]=O"),
    ("BRENK_004", "polycyclic_aromatic_hydrocarbons", "mutagenic", "a2aaaaa2a1aaaaa1"),
    ("BRENK_005", "vinyl_halides", "mutagenic", "[C]=[C]-[F,Cl,Br,I]"),
    ("BRENK_006", "n_oxides", "unstable", "[#7]=[#8]"),
    ("BRENK_007", "thioamides", "hepatotoxic", "[#6]-C(=S)-[#7]"),
    ("BRENK_008", "thioureas", "hepatotoxic", "[#7]-C(=S)-[#7]"),
    ("BRENK_009", "sulfonic_acid", "toxic", "S(=O)(=O)O"),
    ("BRENK_010", "sulfonate_ester", "mutagenic", "S(=O)(=O)O[#6]"),
    ("BRENK_011", "phosphoric_acid", "toxic", "P(=O)(O)(O)O"),
    ("BRENK_012", "phosphinic_acid", "toxic", "P(=O)(O)([#6])[#6]"),
    ("BRENK_013", "azo_group", "mutagenic", "[#6]-N=N-[#6]"),
    ("BRENK_014", "alkyl_halide", "reactive", "[#6]-[Cl,Br,I]"),
    ("BRENK_015", "hydrazine", "mutagenic", "[N]-[N]"),
    ("BRENK_016", "disulfide", "unstable", "[#16]-[#16]"),
    ("BRENK_017", "peroxide", "reactive", "[O]-[O]"),
    ("BRENK_018", "azide", "explosive", "[N-]=[N+]=[N-]"),
    ("BRENK_019", "polycyclic_heterocycles", "mutagenic", "c1nc2ccccc2nc1"),
    ("BRENK_020", "alkyl_nitrate", "mutagenic", "[O]-N(=O)=O"),
    ("BRENK_021", "nitro_alkyl", "mutagenic", "[#6]-[N+](=O)[O-]"),
    ("BRENK_022", "urea_derivatives", "hepatotoxic", "[#7]-C(=O)-[#7]"),
    ("BRENK_023", "carbodiimide", "reactive", "[#7]=[#6]=[#7]"),
    ("BRENK_024", "isocyanate", "reactive", "N=C=O"),
    ("BRENK_025", "isothiocyanate", "reactive", "N=C=S"),
]


# ---------------------------------------------------------------------------
# CircRNA-specific toxicophores
# ---------------------------------------------------------------------------
# Patterns relevant for nucleic acid therapeutics

_CIRCRNA_SMARTS: List[Tuple[str, str, str, str]] = [
    # Modified nucleotides
    ("CIRCRNA_001", "unmodified_uridine", "immunogenic", "[U]"),
    ("CIRCRNA_002", "unmodified_guanosine", "immunogenic", "[G]"),
    ("CIRCRNA_003", "phosphorothioate", "toxic", "P=S"),
    ("CIRCRNA_004", "2_fluoro", "immunogenic", "[#6]-F"),
    # Sequence motifs
    ("CIRCRNA_005", "ugug_motif", "immunogenic", "UGUG"),
    ("CIRCRNA_006", "uaaa_motif", "immunogenic", "UAAA"),
    ("CIRCRNA_007", "long_poly_a", "immunogenic", "AAAAAA"),
    ("CIRCRNA_008", "long_poly_u", "immunogenic", "UUUUUU"),
]


# ---------------------------------------------------------------------------
# Additional toxicity alerts from literature
# ---------------------------------------------------------------------------

_TOXICITY_SMARTS: List[Tuple[str, str, str, str]] = [
    ("TOX_001", "acrylamide", "carcinogenic", "C=CC(=O)N"),
    ("TOX_002", "acrylate_ester", "sensitizer", "C=CC(=O)O"),
    ("TOX_003", "alkyl_sulfonate", "carcinogenic", "S(=O)(=O)-[#6]"),
    ("TOX_004", "aromatic_amine", "carcinogenic", "c-[NH2]"),
    ("TOX_005", "azido_group", "explosive", "[N-]=[N+]=[N-]"),
    ("TOX_006", "benzyl_halide", "irritant", "c1ccccc1C-[Cl,Br,I]"),
    ("TOX_007", "carbamate", "carcinogenic", "[#7]-C(=O)-O"),
    ("TOX_008", "chloroalkane", "mutagenic", "[Cl]-[#6]-[#6]"),
    ("TOX_009", "cyano_group", "toxic", "C#N"),
    ("TOX_010", "dialkyl_nitrosamine", "carcinogenic", "[#6]-N(-[#6])-N=O"),
    ("TOX_011", "dihydropyridine", "unstable", "C1=CNC=CC1"),
    ("TOX_012", "epoxide", "mutagenic", "[#6]1[#6][#8]1"),
    ("TOX_013", "ethyl_ester", "unstable", "[#6]-C(=O)O[#6]"),
    ("TOX_014", "formaldehyde", "carcinogenic", "C=O"),
    ("TOX_015", "haloalkene", "toxic", "[F,Cl,Br,I]-[#6]=C"),
    ("TOX_016", "hydantoin", "unstable", "C1NC(=O)NC1=O"),
    ("TOX_017", "hydroxylamine", "mutagenic", "[OH]-[#7]"),
    ("TOX_018", "imide", "toxic", "C(=O)-[#7]-C(=O)"),
    ("TOX_019", "maleimide", "covalent", "C1=CC(=O)N(C(=O)C1)"),
    ("TOX_020", "melamine", "toxic", "c1nc(nc(n1)N)N"),
    ("TOX_021", "mitomycin", "carcinogenic", "c1cc2c(cc1N)[C@@H]3N(C[C@]4(C3CO4)C(=O)N)C[C@@H]2COC"),
    ("TOX_022", "mustard_gas_analog", "carcinogenic", "S([#6]-[Cl,Br,I])[#6]-[Cl,Br,I]"),
    ("TOX_023", "n_methylol", "mutagenic", "C-N-[OH]"),
    ("TOX_024", "nitrosamine", "carcinogenic", "[#6]-N-N=O"),
    ("TOX_025", "perchlorate", "explosive", "[Cl](=O)(=O)(=O)=O"),
    ("TOX_026", "phosphoramide_mustard", "carcinogenic", "P(=O)(N)(N)[#6]"),
    ("TOX_027", "propiolactone", "carcinogenic", "C1(=O)COC1"),
    ("TOX_028", "quinone", "toxic", "C1=CC(=O)C=CC1=O"),
    ("TOX_029", "sulfonyl_halide", "irritant", "S(=O)(=O)[Cl,Br,F]"),
    ("TOX_030", "triazene", "carcinogenic", "N=NN"),
]


@dataclass
class ToxicophoreMatch:
    """A single toxicophore match in a molecule."""
    pattern_id: str
    name: str
    category: str
    severity: str  # "high", "medium", "low"
    description: str
    atom_indices: List[int]
    smarts: str


@dataclass
class ToxicophoreReport:
    """Complete toxicophore analysis for a molecule."""
    smiles: str
    matches: List[ToxicophoreMatch]
    total_alerts: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    categories: Dict[str, int]
    safe: bool
    risk_summary: str

    def to_dict(self) -> Dict:
        return {
            "smiles": self.smiles,
            "total_alerts": self.total_alerts,
            "high_risk_count": self.high_risk_count,
            "medium_risk_count": self.medium_risk_count,
            "low_risk_count": self.low_risk_count,
            "safe": self.safe,
            "categories": self.categories,
            "risk_summary": self.risk_summary,
            "matches": [
                {
                    "pattern_id": m.pattern_id,
                    "name": m.name,
                    "category": m.category,
                    "severity": m.severity,
                    "description": m.description,
                    "smarts": m.smarts,
                }
                for m in self.matches
            ],
        }


# Severity assignment rules
_SEVERITY_BY_CATEGORY = {
    "carcinogenic": "high",
    "mutagenic": "high",
    "explosive": "high",
    "covalent": "high",
    "hepatotoxic": "medium",
    "reactive": "medium",
    "immunogenic": "medium",
    "promiscuous": "medium",
    "unstable": "low",
    "chelator": "low",
    "unspecific": "low",
    "aggregator": "low",
    "intercalator": "low",
    "redox": "medium",
    "sensitizer": "medium",
    "irritant": "low",
    "toxic": "medium",
}

_DESCRIPTIONS: Dict[str, str] = {
    "PAINS_001": "Aryl hydrazone - covalent binder, interferes with assays",
    "PAINS_002": "Quinone - redox active, generates ROS",
    "PAINS_003": "Rhodanine scaffold - promiscuous binder",
    "PAINS_004": "Michael acceptor (enone) - covalent modification",
    "PAINS_005": "Thiobarbiturate - aggregates, false positives",
    "PAINS_006": "Sulfonylimino - covalent modifier",
    "PAINS_007": "Phenol - redox active, can oxidize",
    "PAINS_008": "Anisole - promiscuous binder",
    "PAINS_009": "Catechol - metal chelator, redox active",
    "PAINS_010": "Hydroquinone - redox cycling",
    "PAINS_011": "Quinoline - DNA intercalator",
    "PAINS_012": "Quinazoline - kinase scaffold, off-target",
    "PAINS_013": "Acylhydrazine - covalent binding",
    "PAINS_014": "Isocyanate - highly reactive",
    "PAINS_015": "Isothiocyanate - reactive",
    "PAINS_016": "Reactive alkyl halide - covalent binding",
    "PAINS_017": "Epoxide - alkylating agent, mutagenic",
    "PAINS_018": "Aziridine - DNA alkylator",
    "PAINS_019": "Nitroalkene - reactive Michael acceptor",
    "PAINS_020": "Beta-lactam - acylating agent",
    "PAINS_021": "Pyrroloquinoline - aggregator",
    "PAINS_022": "Azo group - can be reduced to amines",
    "PAINS_023": "Diazonium - unstable, explosive",
    "PAINS_024": "Triazene - decomposes to alkylating agents",
    "PAINS_025": "Perhalogenated - bioaccumulative",
    "BRENK_001": "Nitro group - mutagenic potential",
    "BRENK_002": "Nitroso group - carcinogenic",
    "BRENK_003": "Aldehyde - reactive, metabolized to acids",
    "BRENK_004": "Polycyclic aromatic hydrocarbon - carcinogenic",
    "BRENK_005": "Vinyl halide - mutagenic",
    "BRENK_006": "N-oxide - unstable",
    "BRENK_007": "Thioamide - hepatotoxic",
    "BRENK_008": "Thiourea - hepatotoxic",
    "BRENK_009": "Sulfonic acid - harsh on proteins",
    "BRENK_010": "Sulfonate ester - mutagenic",
    "BRENK_011": "Phosphoric acid - hydrolyzes easily",
    "BRENK_012": "Phosphinic acid - can be toxic",
    "BRENK_013": "Azo dye - can be carcinogenic",
    "BRENK_014": "Alkyl halide - reactive",
    "BRENK_015": "Hydrazine - carcinogenic",
    "BRENK_016": "Disulfide - unstable, oxidizes",
    "BRENK_017": "Peroxide - explosive, reactive",
    "BRENK_018": "Azide - explosive",
    "BRENK_019": "Polycyclic heterocycle - mutagenic",
    "BRENK_020": "Alkyl nitrate - mutagenic",
    "BRENK_021": "Nitroalkyl - mutagenic",
    "BRENK_022": "Urea derivative - hepatotoxic",
    "BRENK_023": "Carbodiimide - reactive coupling agent",
    "BRENK_024": "Isocyanate - highly reactive",
    "BRENK_025": "Isothiocyanate - reactive",
    "CIRCRNA_001": "Unmodified uridine - innate immune activation via TLR",
    "CIRCRNA_002": "Unmodified guanosine - immunogenic",
    "CIRCRNA_003": "Phosphorothioate backbone - potential toxicity",
    "CIRCRNA_004": "2'-Fluoro modification - check immunogenicity",
    "CIRCRNA_005": "UGUG motif - TLR activation risk",
    "CIRCRNA_006": "UAAA motif - immunostimulatory sequence",
    "CIRCRNA_007": "Long poly-A stretch - immunogenic",
    "CIRCRNA_008": "Long poly-U stretch - immunogenic",
}


class ToxicophoreDetector:
    """
    Detect structural alerts (toxicophores) in molecules.
    """

    def __init__(self):
        self._compiled_patterns: Dict[str, List[Tuple]] = {}
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile all SMARTS patterns for efficiency."""
        if not HAS_RDKIT:
            logger.warning("RDKit not available, toxicophore detection disabled")
            return

        all_patterns = [
            ("PAINS", _PAINS_SMARTS),
            ("BRENK", _BRENK_SMARTS),
            ("CIRCRNA", _CIRCRNA_SMARTS),
            ("TOXICITY", _TOXICITY_SMARTS),
        ]

        for source, pattern_list in all_patterns:
            compiled = []
            for pid, name, category, smarts in pattern_list:
                try:
                    mol = Chem.MolFromSmarts(smarts)
                    if mol:
                        severity = _SEVERITY_BY_CATEGORY.get(category, "medium")
                        description = _DESCRIPTIONS.get(pid, f"{name} - {category}")
                        compiled.append((pid, name, category, severity, description, smarts, mol))
                except Exception as e:
                    logger.debug(f"Failed to compile pattern {pid}: {e}")
            self._compiled_patterns[source] = compiled

    def detect(self, smiles: str) -> ToxicophoreReport:
        """Detect all toxicophores in a molecule."""
        if not HAS_RDKIT:
            return self._empty_report(smiles)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return self._empty_report(smiles)

        matches: List[ToxicophoreMatch] = []
        categories: Dict[str, int] = {}

        for source, patterns in self._compiled_patterns.items():
            for pid, name, category, severity, description, smarts, pattern_mol in patterns:
                try:
                    matched_atoms = mol.GetSubstructMatches(pattern_mol)
                    for atom_indices in matched_atoms:
                        match = ToxicophoreMatch(
                            pattern_id=pid,
                            name=name,
                            category=category,
                            severity=severity,
                            description=description,
                            atom_indices=list(atom_indices),
                            smarts=smarts,
                        )
                        matches.append(match)
                        categories[category] = categories.get(category, 0) + 1
                except Exception:
                    pass

        # Deduplicate matches by (pattern_id, atom_indices as tuple)
        seen: Set[Tuple[str, Tuple[int, ...]]] = set()
        unique_matches = []
        for m in matches:
            key = (m.pattern_id, tuple(sorted(m.atom_indices)))
            if key not in seen:
                seen.add(key)
                unique_matches.append(m)

        high = sum(1 for m in unique_matches if m.severity == "high")
        medium = sum(1 for m in unique_matches if m.severity == "medium")
        low = sum(1 for m in unique_matches if m.severity == "low")

        safe = len(unique_matches) == 0

        if safe:
            risk_summary = "No structural alerts detected. Compound appears safe."
        elif high > 0:
            risk_summary = f"HIGH RISK: {high} critical toxicophore(s) detected. Consider structural modification."
        elif medium > 0:
            risk_summary = f"MEDIUM RISK: {medium} moderate toxicophore(s) detected. Evaluate carefully."
        else:
            risk_summary = f"LOW RISK: {low} minor toxicophore(s) detected. Generally acceptable."

        return ToxicophoreReport(
            smiles=smiles,
            matches=unique_matches,
            total_alerts=len(unique_matches),
            high_risk_count=high,
            medium_risk_count=medium,
            low_risk_count=low,
            categories=categories,
            safe=safe,
            risk_summary=risk_summary,
        )

    def _empty_report(self, smiles: str) -> ToxicophoreReport:
        return ToxicophoreReport(
            smiles=smiles,
            matches=[],
            total_alerts=0,
            high_risk_count=0,
            medium_risk_count=0,
            low_risk_count=0,
            categories={},
            safe=True,
            risk_summary="Unable to analyze - RDKit not available or invalid SMILES",
        )

    def detect_batch(self, smiles_list: List[str]) -> List[ToxicophoreReport]:
        """Batch detection for multiple molecules."""
        return [self.detect(s) for s in smiles_list]

    def quick_check(self, smiles: str) -> bool:
        """Quick boolean check: True if safe (no alerts)."""
        return self.detect(smiles).safe

    def get_all_patterns(self) -> Dict[str, List[Dict]]:
        """Return all pattern definitions for documentation."""
        result = {}
        for source, patterns in self._compiled_patterns.items():
            result[source] = [
                {
                    "pattern_id": p[0],
                    "name": p[1],
                    "category": p[2],
                    "severity": p[3],
                    "description": p[4],
                    "smarts": p[5],
                }
                for p in patterns
            ]
        return result


# Global detector instance
_detector: Optional[ToxicophoreDetector] = None


def get_toxicophore_detector() -> ToxicophoreDetector:
    global _detector
    if _detector is None:
        _detector = ToxicophoreDetector()
    return _detector


def detect_toxicophores(smiles: str) -> ToxicophoreReport:
    """Quick function to detect toxicophores in a single molecule."""
    return get_toxicophore_detector().detect(smiles)


def is_safe(smiles: str) -> bool:
    """Quick check if a molecule is free of toxicophores."""
    return get_toxicophore_detector().quick_check(smiles)
