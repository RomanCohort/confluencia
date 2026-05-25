"""
rna_docking.py — RNA-Small Molecule Docking Prediction

Predicts RNA-drug molecule interactions for:
1. circRNA targeting by small molecules
2. RNA aptamer-drug binding affinity
3. Structure-based drug design for circRNA
4. Binding site identification on circRNA
5. Drug-induced structure modulation

Literature basis:
- Disney et al., 2014: RNA-targeted small molecules
- Donlic et al., 2018: RNA-ligand docking methods
- Morgan & Higgs, 2009: RNA structure prediction for docking
- Luo et al., 2020: Inforna - RNA motif-drug matching

Key concepts:
- RNA binding motifs: hairpins, internal loops, bulges
- Drug binding pockets: grooves, cavities, junctions
- Binding affinity: ΔG, KD predictions
- Structure modulation: drug-induced structural changes

Applications:
- circRNA stability modulation
- Immune pathway activation via RNA-targeting
- Antisense oligonucleotide (ASO) design
- siRNA/miRNA binding site prediction
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import re
from enum import Enum

# RNA structure motifs for drug binding
class RNAMotifType(Enum):
    HAIRPIN_LOOP = "hairpin_loop"       # Closed loop structure
    INTERNAL_LOOP = "internal_loop"     # Loop within stem
    BULGE = "bulge"                      # Unpaired bases in stem
    JUNCTION = "junction"               # Multi-way branch point
    STEM = "stem"                        # Double-stranded region
    APTAMER_SITE = "aptamer_site"       # Known aptamer binding


@dataclass
class BindingSite:
    """A potential drug binding site on RNA."""
    motif_type: RNAMotifType
    position_start: int
    position_end: int
    sequence: str                    # RNA sequence at site
    structure: str                   # Dot-bracket at site
    accessibility: float            # Binding accessibility score
    groove_type: str                 # major/minor/groove
    pocket_volume: float             # Estimated pocket volume


@dataclass
class DrugCandidate:
    """A small molecule drug candidate."""
    drug_id: str
    drug_name: str
    smiles: str                      # SMILES representation
    molecular_weight: float
    logp: float                      # Lipophilicity
    known_rna_targets: List[str]
    binding_mode: str                # intercalation/groove/site


@dataclass
class DockingResult:
    """Result of RNA-drug docking prediction."""
    binding_site: BindingSite
    drug: DrugCandidate
    binding_affinity: float          # ΔG in kcal/mol
    kd_estimate: float               # KD in nM
    binding_mode: str
    confidence: float                # Prediction confidence
    structural_impact: str           # Effect on RNA structure


@dataclass
class DockingFeatures:
    """Complete docking analysis features."""
    binding_sites: List[BindingSite]
    docking_results: List[DockingResult]
    best_hit: Optional[DockingResult]
    overall_druggability: float      # RNA drug targeting potential
    drug_candidates_ranked: List[str]  # Ranked drug IDs
    structure_modulation_sites: List[Tuple[int, str]]  # Sites for modulation
    aso_target_sites: List[int]      # Antisense oligo sites
    docking_method: str              # Method used


# Known RNA-binding small molecules
DRUG_CANDIDATES_DB: Dict[str, DrugCandidate] = {
    "ribavirin": DrugCandidate(
        drug_id="ribavirin",
        drug_name="Ribavirin",
        smiles="NC(=N)N[C@H]1[C@@H](O)[C@@H](O)[C@H](O)[C@@H]1O",
        molecular_weight=244.2,
        logp=-1.8,
        known_rna_targets=["RNA viruses", "hairpin loops"],
        binding_mode="groove",
    ),
    "bleomycin": DrugCandidate(
        drug_id="bleomycin",
        drug_name="Bleomycin",
        smiles="CC1=C(C(=O)N(N1)C[C@@H](O)[C@@H]2CNC(C2)C(=O)N...",
        molecular_weight=1415.6,
        logp=-4.2,
        known_rna_targets=["DNA/RNA intercalation"],
        binding_mode="intercalation",
    ),
    "neomycin_b": DrugCandidate(
        drug_id="neomycin_b",
        drug_name="Neomycin B",
        smiles="C1[C@H]([C@@H]([C@H]([C@@H]([C@H]1N)O)O)O...",
        molecular_weight=614.6,
        logp=-7.5,
        known_rna_targets=["RNA grooves", "internal loops"],
        binding_mode="groove",
    ),
    "tg003": DrugCandidate(
        drug_id="tg003",
        drug_name="TG003",
        smiles="CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
        molecular_weight=300.4,
        logp=3.0,
        known_rna_targets=["RNA polymerase inhibition"],
        binding_mode="site",
    ),
}


class RNADockingPredictor:
    """
    Predict RNA-small molecule docking.

    Methods:
    - Motif-based matching (Inforna-style)
    - Structure-based pocket identification
    - Affinity estimation from structure
    - ASO/siRNA target site prediction
    """

    def __init__(self, min_loop_size: int = 4, max_pocket_volume: float = 500.0):
        """
        Initialize docking predictor.

        Args:
            min_loop_size: Minimum loop size for binding site
            max_pocket_volume: Maximum pocket volume to consider
        """
        self.min_loop_size = min_loop_size
        self.max_pocket_volume = max_pocket_volume
        self.drug_db = DRUG_CANDIDATES_DB

    def analyze(
        self,
        sequence: str,
        dot_bracket: Optional[str] = None,
        target_drugs: Optional[List[str]] = None,
    ) -> DockingFeatures:
        """
        Analyze RNA for drug docking potential.

        Args:
            sequence: RNA sequence
            dot_bracket: Structure (if None, estimate)
            target_drugs: Specific drugs to test

        Returns:
            DockingFeatures with sites, results, druggability
        """
        seq = self._sanitize_sequence(sequence)

        if len(seq) < 100:
            return self._empty_features()

        # Estimate structure if not provided
        if dot_bracket is None:
            dot_bracket = self._estimate_structure(seq)

        # Identify binding sites
        sites = self._identify_binding_sites(seq, dot_bracket)

        # Perform docking predictions
        drugs_to_test = target_drugs or list(self.drug_db.keys())
        results = self._predict_docking(seq, sites, drugs_to_test)

        # Find best hit
        best = self._find_best_hit(results)

        # Compute overall druggability
        druggability = self._compute_druggability(sites, results)

        # Rank drug candidates
        ranked = self._rank_drugs(results)

        # Identify structure modulation sites
        modulation_sites = self._identify_modulation_sites(sites, results)

        # Identify ASO target sites
        aso_sites = self._identify_aso_sites(seq, sites)

        return DockingFeatures(
            binding_sites=sites,
            docking_results=results,
            best_hit=best,
            overall_druggability=druggability,
            drug_candidates_ranked=ranked,
            structure_modulation_sites=modulation_sites,
            aso_target_sites=aso_sites,
            docking_method="motif_based",
        )

    def _sanitize_sequence(self, sequence: str) -> str:
        """Convert DNA to RNA."""
        return sequence.upper().replace("T", "U")

    def _empty_features(self) -> DockingFeatures:
        """Return empty features for short sequences."""
        return DockingFeatures(
            binding_sites=[],
            docking_results=[],
            best_hit=None,
            overall_druggability=0.0,
            drug_candidates_ranked=[],
            structure_modulation_sites=[],
            aso_target_sites=[],
            docking_method="sequence_too_short",
        )

    def _estimate_structure(self, sequence: str) -> str:
        """Estimate dot-bracket structure."""
        length = len(sequence)
        gc = sum(1 for c in sequence if c in "GC") / length

        # Simple stem-loop model
        stem = int(length * gc * 0.2)
        loop = length - 2 * stem

        return "(" * stem + "." * loop + ")" * stem

    def _identify_binding_sites(self, seq: str, dot_bracket: str) -> List[BindingSite]:
        """
        Identify potential drug binding sites on RNA.

        Site types:
        - Hairpin loops (closed loops)
        - Internal loops (unpaired in stem)
        - Bulges (single unpaired)
        - Junctions (multi-way)
        """
        sites = []

        # Find hairpin loops
        hairpins = self._find_hairpin_loops(dot_bracket)
        for start, end in hairpins:
            loop_seq = seq[start:end]
            loop_struct = dot_bracket[start:end]
            loop_size = end - start

            if loop_size >= self.min_loop_size:
                accessibility = self._compute_accessibility(loop_struct)
                pocket_vol = loop_size * 10.0  # Angstrom^3 estimate

                sites.append(BindingSite(
                    motif_type=RNAMotifType.HAIRPIN_LOOP,
                    position_start=start,
                    position_end=end,
                    sequence=loop_seq,
                    structure=loop_struct,
                    accessibility=accessibility,
                    groove_type="minor",
                    pocket_volume=pocket_vol,
                ))

        # Find internal loops
        internal_loops = self._find_internal_loops(dot_bracket)
        for start, end in internal_loops:
            loop_seq = seq[start:end]
            loop_struct = dot_bracket[start:end]

            accessibility = self._compute_accessibility(loop_struct)

            sites.append(BindingSite(
                motif_type=RNAMotifType.INTERNAL_LOOP,
                position_start=start,
                position_end=end,
                sequence=loop_seq,
                structure=loop_struct,
                accessibility=accessibility,
                groove_type="major",
                pocket_volume=(end - start) * 12.0,
            ))

        # Find bulges
        bulges = self._find_bulges(dot_bracket)
        for pos, size in bulges:
            start, end = pos, pos + size

            sites.append(BindingSite(
                motif_type=RNAMotifType.BULGE,
                position_start=start,
                position_end=end,
                sequence=seq[start:end],
                structure=dot_bracket[start:end],
                accessibility=0.8,  # Bulges are accessible
                groove_type="minor",
                pocket_volume=size * 8.0,
            ))

        return sites

    def _find_hairpin_loops(self, dot_bracket: str) -> List[Tuple[int, int]]:
        """Find hairpin loop positions."""
        loops = []

        # Pattern: (... enclosed dots
        in_loop = False
        loop_start = 0

        for i, ch in enumerate(dot_bracket):
            if ch == "." and not in_loop:
                # Check if enclosed by parentheses
                prev = dot_bracket[i-1] if i > 0 else ""
                if prev == "(":
                    in_loop = True
                    loop_start = i
            elif ch != "." and in_loop:
                if ch == ")":
                    loops.append((loop_start, i))
                in_loop = False

        return loops

    def _find_internal_loops(self, dot_bracket: str) -> List[Tuple[int, int]]:
        """Find internal loop positions."""
        loops = []

        # Find regions of dots within stem regions
        stem_start = 0
        in_stem = False

        for i, ch in enumerate(dot_bracket):
            if ch == "(" and not in_stem:
                in_stem = True
                stem_start = i
            elif ch == "." and in_stem:
                # Dot in stem region
                loop_start = i
                while i < len(dot_bracket) and dot_bracket[i] == ".":
                    i += 1
                if dot_bracket[i] == "(" or dot_bracket[i] == ")":
                    loops.append((loop_start, i))

        return loops

    def _find_bulges(self, dot_bracket: str) -> List[Tuple[int, int]]:
        """Find bulge positions."""
        bulges = []

        # Single or few unpaired bases in stem
        i = 0
        while i < len(dot_bracket):
            if dot_bracket[i] == ".":
                # Check if in stem context
                prev = dot_bracket[i-1] if i > 0 else ""
                count = 0
                start = i
                while i < len(dot_bracket) and dot_bracket[i] == ".":
                    count += 1
                    i += 1

                next_ch = dot_bracket[i] if i < len(dot_bracket) else ""

                # Bulge: small dots within stem
                if count <= 3 and (prev == "(" or next_ch == ")"):
                    bulges.append((start, count))
            else:
                i += 1

        return bulges

    def _compute_accessibility(self, structure: str) -> float:
        """Compute binding accessibility score."""
        # All dots = fully accessible
        dots = structure.count(".")
        return dots / len(structure) if structure else 0.0

    def _predict_docking(
        self,
        seq: str,
        sites: List[BindingSite],
        drugs: List[str]
    ) -> List[DockingResult]:
        """
        Predict docking for each site-drug combination.

        Uses motif-based affinity estimation:
        - Hairpin + groove binder: moderate affinity
        - Internal loop + intercalator: high affinity
        - Bulge + site binder: variable
        """
        results = []

        for site in sites:
            for drug_id in drugs:
                drug = self.drug_db.get(drug_id)
                if not drug:
                    continue

                # Estimate binding affinity
                affinity = self._estimate_affinity(site, drug)
                kd = self._estimate_kd(affinity)
                confidence = self._compute_confidence(site, drug)
                impact = self._predict_structural_impact(site, drug)
                mode = self._determine_binding_mode(site, drug)

                results.append(DockingResult(
                    binding_site=site,
                    drug=drug,
                    binding_affinity=affinity,
                    kd_estimate=kd,
                    binding_mode=mode,
                    confidence=confidence,
                    structural_impact=impact,
                ))

        return results

    def _estimate_affinity(self, site: BindingSite, drug: DrugCandidate) -> float:
        """
        Estimate binding affinity (ΔG).

        Factors:
        - Motif type affinity
        - Drug properties (logP, MW)
        - Pocket volume match
        """
        # Base affinity by motif
        motif_affinity = {
            RNAMotifType.HAIRPIN_LOOP: -6.0,
            RNAMotifType.INTERNAL_LOOP: -8.0,
            RNAMotifType.BULGE: -5.0,
            RNAMotifType.JUNCTION: -10.0,
        }

        base = motif_affinity.get(site.motif_type, -5.0)

        # Drug property adjustments
        # Optimal logP for RNA binding: -2 to 2
        logp_penalty = abs(drug.logp) * 0.5

        # MW penalty (larger = harder to fit)
        mw_penalty = drug.molecular_weight / 500.0 * 1.0

        # Pocket volume match
        pocket_match = min(site.pocket_volume / drug.molecular_weight, 1.0)
        pocket_bonus = pocket_match * 2.0

        # Accessibility bonus
        access_bonus = site.accessibility * 3.0

        # Final affinity
        affinity = base - logp_penalty - mw_penalty + pocket_bonus + access_bonus

        return affinity

    def _estimate_kd(self, affinity: float) -> float:
        """
        Estimate KD from ΔG.

        ΔG = RT ln(KD)
        KD = exp(ΔG / RT)
        """
        RT = 0.616  # kcal/mol at 37°C

        # Convert to nM
        kd_molar = np.exp(affinity / RT)
        kd_nm = kd_molar * 1e9

        return kd_nm

    def _compute_confidence(self, site: BindingSite, drug: DrugCandidate) -> float:
        """Compute prediction confidence."""
        # Higher if drug known to bind similar motifs
        known = drug.known_rna_targets
        motif_name = site.motif_type.value

        confidence = 0.5  # Base

        if any(motif_name in target or "loop" in target for target in known):
            confidence += 0.3

        if site.accessibility > 0.7:
            confidence += 0.1

        return np.clip(confidence, 0.0, 1.0)

    def _predict_structural_impact(self, site: BindingSite, drug: DrugCandidate) -> str:
        """Predict how drug binding affects structure."""
        if drug.binding_mode == "intercalation":
            return "destabilize_stem"
        elif drug.binding_mode == "groove":
            return "stabilize_loop"
        else:
            return "minor_effect"

    def _determine_binding_mode(self, site: BindingSite, drug: DrugCandidate) -> str:
        """Determine binding mode for this site."""
        if site.motif_type == RNAMotifType.HAIRPIN_LOOP:
            return "loop_insertion"
        elif site.motif_type == RNAMotifType.INTERNAL_LOOP:
            return "groove_binding"
        elif site.motif_type == RNAMotifType.BULGE:
            return "bulge_recognition"
        else:
            return drug.binding_mode

    def _find_best_hit(self, results: List[DockingResult]) -> Optional[DockingResult]:
        """Find best docking hit."""
        if not results:
            return None

        # Best affinity with confidence > 0.5
        valid = [r for r in results if r.confidence > 0.5]
        if not valid:
            return None

        return max(valid, key=lambda r: -r.binding_affinity)

    def _compute_druggability(self, sites: List[BindingSite], results: List[DockingResult]) -> float:
        """Compute overall RNA druggability score."""
        if not sites:
            return 0.0

        # Factors:
        # 1. Number of binding sites
        site_factor = len(sites) / 10.0

        # 2. Accessibility
        avg_access = np.mean([s.accessibility for s in sites])

        # 3. Best affinity
        best_affinity = -abs(self._find_best_hit(results).binding_affinity) if results else 10.0
        affinity_factor = (best_affinity + 15.0) / 15.0

        druggability = site_factor * 0.3 + avg_access * 0.3 + affinity_factor * 0.4

        return np.clip(druggability, 0.0, 1.0)

    def _rank_drugs(self, results: List[DockingResult]) -> List[str]:
        """Rank drug candidates by affinity."""
        # Group by drug
        drug_scores = {}
        for r in results:
            if r.drug.drug_id not in drug_scores:
                drug_scores[r.drug.drug_id] = []
            drug_scores[r.drug.drug_id].append(r.binding_affinity)

        # Average affinity per drug
        avg_affinities = {
            drug_id: np.mean(scores)
            for drug_id, scores in drug_scores.items()
        }

        # Rank by affinity (most negative = best)
        ranked = sorted(avg_affinities.keys(), key=lambda d: avg_affinities[d])

        return ranked

    def _identify_modulation_sites(
        self,
        sites: List[BindingSite],
        results: List[DockingResult]
    ) -> List[Tuple[int, str]]:
        """Identify sites where drugs can modulate structure."""
        modulation = []

        for r in results:
            if "stabilize" in r.structural_impact or "destabilize" in r.structural_impact:
                modulation.append((
                    r.binding_site.position_start,
                    r.structural_impact
                ))

        return modulation

    def _identify_aso_sites(self, seq: str, sites: List[BindingSite]) -> List[int]:
        """
        Identify ASO (antisense oligonucleotide) target sites.

        ASO sites:
        - Accessible regions
        - Not in stable stems
        - Usually hairpin loops or bulges
        """
        aso_sites = []

        for site in sites:
            if site.accessibility > 0.6:
                # ASO can bind here
                aso_sites.append(site.position_start)

        return aso_sites


def compute_docking_score(features: DockingFeatures) -> Dict[str, float]:
    """Compute docking-related scores."""
    scores = {
        "druggability_score": features.overall_druggability,
        "binding_site_count": len(features.binding_sites),
        "best_affinity": features.best_hit.binding_affinity if features.best_hit else 0.0,
        "best_kd_nm": features.best_hit.kd_estimate if features.best_hit else 1e9,
    }

    # Drug targeting potential
    scores["drug_targeting_potential"] = features.overall_druggability * 0.7

    # ASO potential
    scores["aso_potential"] = len(features.aso_target_sites) / 10.0

    return scores


def design_rna_targeting_drug(
    sequence: str,
    target_motif: str = "hairpin",
    constraints: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Design RNA-targeting small molecule.

    Returns:
        Recommended drug, binding site, expected affinity
    """
    predictor = RNADockingPredictor()
    features = predictor.analyze(sequence)

    # Filter by target motif
    motif_map = {
        "hairpin": RNAMotifType.HAIRPIN_LOOP,
        "internal_loop": RNAMotifType.INTERNAL_LOOP,
        "bulge": RNAMotifType.BULGE,
    }

    target_type = motif_map.get(target_motif, RNAMotifType.HAIRPIN_LOOP)

    target_sites = [s for s in features.binding_sites if s.motif_type == target_type]

    if not target_sites:
        return {
            "recommendation": "No suitable binding sites found",
            "alternative_motifs": [s.motif_type.value for s in features.binding_sites],
        }

    # Find best result for target sites
    best_for_target = None
    for r in features.docking_results:
        if r.binding_site.motif_type == target_type:
            if best_for_target is None or r.binding_affinity < best_for_target.binding_affinity:
                best_for_target = r

    return {
        "recommended_drug": best_for_target.drug.drug_name if best_for_target else None,
        "target_site": {
            "position": best_for_target.binding_site.position_start if best_for_target else None,
            "sequence": best_for_target.binding_site.sequence if best_for_target else None,
        },
        "predicted_affinity": best_for_target.binding_affinity if best_for_target else None,
        "predicted_kd_nm": best_for_target.kd_estimate if best_for_target else None,
        "binding_mode": best_for_target.binding_mode if best_for_target else None,
        "structural_impact": best_for_target.structural_impact if best_for_target else None,
    }


# Convenience function
def predict_rna_docking(
    sequence: str,
    structure: Optional[str] = None,
) -> DockingFeatures:
    """Predict RNA-small molecule docking."""
    predictor = RNADockingPredictor()
    return predictor.analyze(sequence, structure)