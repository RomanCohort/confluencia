"""
rna_modifications.py — circRNA Modification Prediction

Predicts post-transcriptional modifications affecting:
1. m6A (N6-methyladenosine) - stability, immune recognition
2. IRES activity - circRNA translation potential
3. miRNA binding sites - ceRNA network analysis
4. RBP binding sites - protein interaction prediction
5. Evolutionary conservation - functional importance

Literature basis:
- Liu et al., 2022: m6A in circRNA affects immunogenicity
- Yang et al., 2017: circRNA translation via IRES
- Hansen et al., 2013: circRNA as miRNA sponge (ceRNA)
- Du et al., 2017: circRNA-RBP interactions
- Jeck et al., 2013: circRNA conservation patterns

Key concepts:
- m6A: DRACH motif (D=A/G/U, R=A/G, H=A/C/U)
- IRES: structured regions enabling cap-independent translation
- miRNA sponge: complementary binding, miRNA sequestration
- RBP: sequence/structure motifs for protein binding
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import re
from enum import Enum

# Modification types
class ModificationType(Enum):
    M6A = "m6a"                 # N6-methyladenosine
    M5C = "m5c"                 # 5-methylcytosine
    PSEUDOURIDINE = "psi"       # Pseudouridine
    ADAR_EDITING = "adar"       # A-to-I editing
    M6AM = "m6am"               # N6,2'-O-dimethyladenosine


@dataclass
class ModificationSite:
    """A post-transcriptional modification site."""
    mod_type: ModificationType
    position: int
    sequence_context: str       # Surrounding sequence
    probability: float          # Modification probability
    effect: str                 # Predicted effect
    enzyme: str                 # Predicted enzyme (METTL3, etc.)
    motif: str                  # Recognition motif


@dataclass
class IRESSite:
    """Internal Ribosome Entry Site."""
    position_start: int
    position_end: int
    sequence: str
    structure: str              # Required structure for IRES
    activity_score: float       # IRES activity prediction
    translation_potential: float # Protein production likelihood
    type: str                   # viral-like/cellular

    # NEW: Enhanced IRES features (literature-based)
    structural_motif: str = ""  # Y-shaped, H-type, pseudoknot, G-quadruplex
    itaf_binding_sites: List[str] = field(default_factory=list)  # ITAF proteins
    g_quadruplex_present: bool = False
    kozak_context_score: float = 0.0
    domain_count: int = 0       # Number of structural domains
    confidence_level: str = "low"  # high/medium/low


@dataclass
class MiRNABindingSite:
    """miRNA binding site."""
    position_start: int
    position_end: int
    circrna_seq: str            # circRNA segment
    miRNA_name: str             # miRNA ID (miR-21, etc.)
    miRNA_seed: str             # miRNA seed sequence
    complementarity: float      # Seed match score
    binding_energy: float       # Hybridization energy
    ceRNA_potential: float      # Sponge activity


@dataclass
class RBPBindingSite:
    """RNA-binding protein binding site."""
    position_start: int
    position_end: int
    sequence: str
    structure: str
    rbp_name: str               # RBP name (HuR, FMR1, etc.)
    motif: str                  # RBP recognition motif
    binding_score: float
    functional_role: str        # Predicted role


@dataclass
class ModificationFeatures:
    """Complete modification analysis."""
    m6a_sites: List[ModificationSite]
    ires_sites: List[IRESSite]
    miRNA_sites: List[MiRNABindingSite]
    rbp_sites: List[RBPBindingSite]
    adar_sites: List[ModificationSite]

    m6a_density: float          # m6A sites per kb
    translation_potential: float # Overall translation score
    ceRNA_activity: float       # miRNA sponge potential
    rbp_binding_density: float  # RBP sites per kb

    immunogenicity_modulation: float  # m6A effect on immunity
    stability_score: float      # Modification-based stability
    conservation_score: float   # Evolutionary conservation

    modification_method: str


# m6A motif: DRACH (D=A/G/U, R=A/G, H=A/C/U)
M6A_MOTIF_PATTERN = re.compile(r"[AGU][AG]AC[ACU]", re.IGNORECASE)

# Known miRNA seeds (top 20 oncogenic/regulatory)
MIRNA_DATABASE: Dict[str, str] = {
    "miR-21": "AUACUG",       # Oncogenic
    "miR-155": "UAAUGCU",     # Immune regulation
    "miR-122": "UGGAGUG",     # Liver-specific
    "miR-145": "CUCCAGUU",    # Tumor suppressor
    "miR-200a": "UAACACU",    # EMT regulation
    "miR-200b": "AAUAUAC",    # EMT regulation
    "miR-34a": "UGGCAGU",     # p53 pathway
    "miR-17": "AAAGUGC",      # Oncomir-1 cluster
    "miR-92a": "AUUGCAC",     # Oncomir-1 cluster
    "miR-let-7a": "AUACUAU",  # Tumor suppressor
    "miR-126": "UCGUACCG",    # Angiogenesis
    "miR-143": "UGAGAUGA",    # Tumor suppressor
    "miR-31": "GCAAGAU",      # Metastasis
    "miR-10b": "CACAAAU",     # Metastasis
    "miR-221": "ACUGGCA",     # Cell cycle
    "miR-222": "ACUGGCA",     # Cell cycle
}

# RBP motifs
RBP_DATABASE: Dict[str, Dict] = {
    "HuR": {"motif": "UUUU[AU]U", "role": "stabilization"},
    "FMR1": {"motif": "GGGAC[U]", "role": "translational_regulation"},
    "TDP43": {"motif": "GGUG", "role": "splicing"},
    "FUS": {"motif": "GGGG[AU]", "role": "splicing"},
    "IGF2BP1": {"motif": "CA[U]C[AU]", "role": "stabilization"},
    "YBX1": {"motif": "CAC[AU]C", "role": "translational_regulation"},
    "PTBP1": {"motif": "YYYYYY", "role": "splicing"},
}


class ModificationPredictor:
    """
    Predict circRNA post-transcriptional modifications.

    Methods:
    - Motif scanning for modification sites
    - Structure-based activity prediction
    - Database matching for miRNA/RBP
    """

    def __init__(self):
        """Initialize modification predictor."""
        self.m6a_pattern = M6A_MOTIF_PATTERN
        self.miRNA_db = MIRNA_DATABASE
        self.rbp_db = RBP_DATABASE

    def analyze(self, sequence: str, dot_bracket: Optional[str] = None) -> ModificationFeatures:
        """
        Analyze all modifications.

        Args:
            sequence: circRNA sequence
            dot_bracket: Structure (optional)

        Returns:
            ModificationFeatures with all modification types
        """
        seq = self._sanitize_sequence(sequence)
        length = len(seq)

        if length < 200:
            return self._empty_features()

        # Estimate structure if not provided
        if dot_bracket is None:
            dot_bracket = self._estimate_structure(seq)

        # Predict each modification type
        m6a_sites = self._predict_m6a_sites(seq)
        ires_sites = self._predict_ires_sites(seq, dot_bracket)
        miRNA_sites = self._predict_miRNA_sites(seq)
        rbp_sites = self._predict_rbp_sites(seq, dot_bracket)
        adar_sites = self._predict_adar_sites(seq)

        # Compute derived metrics
        m6a_density = len(m6a_sites) / (length / 1000)
        translation = self._compute_translation_potential(ires_sites)
        ceRNA = self._compute_ceRNA_activity(miRNA_sites)
        rbp_density = len(rbp_sites) / (length / 1000)

        immunogenicity_mod = self._compute_m6a_immunogenicity_effect(m6a_sites)
        stability = self._compute_modification_stability(m6a_sites, rbp_sites)
        conservation = self._estimate_conservation(seq, m6a_sites)

        return ModificationFeatures(
            m6a_sites=m6a_sites,
            ires_sites=ires_sites,
            miRNA_sites=miRNA_sites,
            rbp_sites=rbp_sites,
            adar_sites=adar_sites,
            m6a_density=m6a_density,
            translation_potential=translation,
            ceRNA_activity=ceRNA,
            rbp_binding_density=rbp_density,
            immunogenicity_modulation=immunogenicity_mod,
            stability_score=stability,
            conservation_score=conservation,
            modification_method="motif_based",
        )

    def _sanitize_sequence(self, sequence: str) -> str:
        """Convert DNA to RNA."""
        return sequence.upper().replace("T", "U")

    def _empty_features(self) -> ModificationFeatures:
        """Return empty features."""
        return ModificationFeatures(
            m6a_sites=[], ires_sites=[], miRNA_sites=[], rbp_sites=[], adar_sites=[],
            m6a_density=0.0, translation_potential=0.0, ceRNA_activity=0.0,
            rbp_binding_density=0.0, immunogenicity_modulation=0.0,
            stability_score=0.0, conservation_score=0.0,
            modification_method="sequence_too_short",
        )

    def _estimate_structure(self, seq: str) -> str:
        """Estimate structure."""
        length = len(seq)
        gc = sum(1 for c in seq if c in "GC") / length
        stem = int(length * gc * 0.2)
        return "(" * stem + "." * (length - 2 * stem) + ")" * stem

    def _predict_m6a_sites(self, seq: str) -> List[ModificationSite]:
        """
        Predict m6A modification sites.

        m6A motif: DRACH (D=A/G/U, R=A/G, A, C, H=A/C/U)
        Examples: GGACU, GAACA, AGACU
        """
        sites = []

        # Find all DRACH motifs
        for match in self.m6a_pattern.finditer(seq):
            pos = match.start()
            context = seq[pos-2:pos+7] if pos+7 <= len(seq) else seq[pos-2:]
            motif = match.group()

            # Compute probability (based on context)
            prob = self._compute_m6a_probability(seq, pos)

            # Predict effect
            effect = self._predict_m6a_effect(seq, pos)

            # Enzyme assignment
            enzyme = "METTL3/METTL14" if prob > 0.5 else "METTL16"

            sites.append(ModificationSite(
                mod_type=ModificationType.M6A,
                position=pos,
                sequence_context=context,
                probability=prob,
                effect=effect,
                enzyme=enzyme,
                motif=motif,
            ))

        return sites

    def _compute_m6a_probability(self, seq: str, pos: int) -> float:
        """Compute m6A probability at position."""
        # Factors:
        # 1. Motif strength (canonical vs variant)
        motif = seq[pos:pos+5]

        # Canonical motifs: GGACU, GAACU, AGACU have higher probability
        canonical = ["GGACU", "GAACU", "AGACU", "AAACU"]
        if motif in canonical:
            base_prob = 0.8
        else:
            base_prob = 0.5

        # 2. GC content around site (higher GC = more structured = more m6A)
        context = seq[pos-10:pos+10] if pos >= 10 else seq[:pos+10]
        gc = sum(1 for c in context if c in "GC") / len(context)
        gc_bonus = gc * 0.2

        return np.clip(base_prob + gc_bonus, 0.0, 1.0)

    def _predict_m6a_effect(self, seq: str, pos: int) -> str:
        """Predict effect of m6A modification."""
        # m6A effects:
        # - Decreased RIG-I recognition (Liu et al., 2022)
        # - Increased stability (m6A reader proteins)
        # - Enhanced translation (IRES-like effect)

        # Check context
        context = seq[pos-5:pos+5]

        # If near GU-rich region, affects RIG-I
        if "GU" in context or "UG" in context:
            return "reduced_RIG_I_recognition"

        # If near hairpin loop, affects stability
        if "GC" in context:
            return "stability_modulation"

        return "translation_enhancement"

    def _predict_ires_sites(self, seq: str, dot_bracket: str) -> List[IRESSite]:
        """
        Predict IRES (Internal Ribosome Entry Sites).

        Enhanced prediction based on Martinez-Salas et al., 2018:
        - Structural motifs: Y-shaped, H-type, pseudoknot, G-quadruplex
        - ITAF binding sites (PTB, hnRNP, La, PCBP2)
        - Polypyrimidine tract quality
        - Kozak context around AUG

        Literature:
            Martinez-Salas E et al., Wiley Interdiscip Rev RNA 2018
            Yang Y et al., Nat Commun 2018 (G-quadruplex)
            Weingarten-Gabbay S et al., Cell Rep 2016 (ITAFs)
        """
        sites = []

        # Look for structured regions >50nt
        in_stem = False
        stem_start = 0

        for i, ch in enumerate(dot_bracket):
            if ch == "(":
                if not in_stem:
                    stem_start = i
                    in_stem = True
            elif ch == "." and in_stem:
                # Check stem length
                stem_len = i - stem_start
                if stem_len > 20:
                    # Potential IRES region
                    region_start = stem_start - 10
                    region_end = i + 30

                    if region_start >= 0 and region_end <= len(seq):
                        region_seq = seq[region_start:region_end]
                        region_struct = dot_bracket[region_start:region_end]

                        # Check for polypyrimidine tract
                        pyrimidine = sum(1 for c in region_seq if c in "UC")
                        pyrimidine_ratio = pyrimidine / len(region_seq)

                        if pyrimidine_ratio > 0.4:
                            # Enhanced IRES analysis
                            structural_motif = self._detect_ires_structural_motif(region_struct, region_seq)
                            itaf_sites = self._detect_itaf_sites(region_seq)
                            g4_present = self._detect_g_quadruplex(region_seq)
                            kozak_score = self._score_kozak_context(region_seq)
                            domain_count = self._count_ires_domains(region_struct)
                            confidence = self._assess_ires_confidence(
                                pyrimidine_ratio, structural_motif, itaf_sites, g4_present
                            )

                            # Good IRES candidate
                            activity = (
                                pyrimidine_ratio * 0.3 +
                                stem_len / 50.0 * 0.2 +
                                len(itaf_sites) * 0.05 +
                                (1.0 if g4_present else 0.0) * 0.15 +
                                kozak_score * 0.2 +
                                (0.2 if structural_motif != "unknown" else 0.0)
                            )
                            trans_potential = activity * 0.8 + kozak_score * 0.2

                            ires_type = "cellular" if activity < 0.6 else "viral-like"

                            sites.append(IRESSite(
                                position_start=region_start,
                                position_end=region_end,
                                sequence=region_seq,
                                structure=region_struct,
                                activity_score=min(activity, 1.0),
                                translation_potential=min(trans_potential, 1.0),
                                type=ires_type,
                                structural_motif=structural_motif,
                                itaf_binding_sites=itaf_sites,
                                g_quadruplex_present=g4_present,
                                kozak_context_score=kozak_score,
                                domain_count=domain_count,
                                confidence_level=confidence,
                            ))

                in_stem = False

        return sites

    def _detect_ires_structural_motif(self, dot_bracket: str, sequence: str) -> str:
        """
        Detect IRES structural motif type.

        Known motifs:
        - Y-shaped: two stem-loops connected (((...)))(((...)))
        - H-type: hairpin with specific bulge (((..(...)..)))
        - Pseudoknot: non-nested base pairing (requires specialized detection)
        - G-quadruplex: GGG(N1-7)GGG pattern in sequence
        """
        # Check for Y-shaped (two adjacent hairpins)
        y_pattern = r"(\({10,}\.\.{5,}\){10,})(\.{5,})(\({10,}\.\.{5,}\){10,})"
        if re.search(y_pattern, dot_bracket):
            return "Y-shaped"

        # Check for H-type (hairpin with internal loop/bulge)
        h_pattern = r"\({5,}\.\.{3,}\(\.\.{3,}\)\.\.{3,}\){5,}"
        if re.search(h_pattern, dot_bracket):
            return "H-type"

        # Check for G-quadruplex in sequence (already checked separately)
        if self._detect_g_quadruplex(sequence):
            return "G-quadruplex"

        # Simple hairpin (stem-loop)
        if re.search(r"\({10,}\.\.{10,}\){10,}", dot_bracket):
            return "hairpin"

        return "unknown"

    def _detect_itaf_sites(self, sequence: str) -> List[str]:
        """
        Detect ITAF (IRES Trans-Acting Factor) binding sites.

        Known ITAFs (Weingarten-Gabbay et al., 2016):
        - PTB (Polypyrimidine Tract Binding): UCUU, UCUUC motifs
        - hnRNP A1: AU-rich elements
        - La protein: UUU sequences
        - PCBP2: C-rich sequences
        - HuR: U-rich regions
        - DAP5: AUG context
        """
        itafs = []

        # PTB sites
        if "UCUU" in sequence or "UCUUC" in sequence:
            itafs.append("PTB")

        # hnRNP A1 sites
        if "AUUA" in sequence or "AUUUA" in sequence:
            itafs.append("hnRNP_A1")

        # La protein sites
        if "UUU" in sequence:
            itafs.append("La")

        # PCBP2 sites
        c_rich_count = sum(1 for c in sequence if c == "C")
        if c_rich_count > len(sequence) * 0.4:
            itafs.append("PCBP2")

        # HuR sites
        u_count = sum(1 for c in sequence if c == "U")
        if u_count > len(sequence) * 0.3:
            itafs.append("HuR")

        return itafs

    def _detect_g_quadruplex(self, sequence: str) -> bool:
        """
        Detect potential G-quadruplex formation.

        G-quadruplex: GGG(N1-7)GGG(N1-7)GGG(N1-7)GGG
        Yang et al., 2018: G4 structures enhance IRES activity

        Returns:
            True if potential G-quadruplex detected
        """
        # Simplified pattern: 4+ G runs with 1-7 spacer
        g4_pattern = r"G{3,}([ACUG]{1,7}G{3,}){2,}[ACUG]{1,7}G{3,}"
        return bool(re.search(g4_pattern, sequence))

    def _score_kozak_context(self, sequence: str) -> float:
        """
        Score Kozak consensus context around start codon.

        Optimal Kozak: (A/G)CCAUGG
        Key positions:
        - -3: A or G (strong)
        - +4: G (strong)
        - -2: C (moderate)

        Returns:
            Score from 0 (weak) to 1 (strong)
        """
        score = 0.0

        # Find AUG codons
        aug_positions = []
        for i in range(len(sequence) - 2):
            if sequence[i:i+3] == "AUG":
                aug_positions.append(i)

        if not aug_positions:
            return 0.0

        # Score best Kozak context
        best_score = 0.0
        for aug_pos in aug_positions:
            current_score = 0.0

            # Check -3 position
            if aug_pos >= 3:
                pos_minus_3 = sequence[aug_pos - 3]
                if pos_minus_3 in "AG":
                    current_score += 0.3
                elif pos_minus_3 == "C":
                    current_score += 0.15

            # Check -2 position
            if aug_pos >= 2:
                pos_minus_2 = sequence[aug_pos - 2]
                if pos_minus_2 == "C":
                    current_score += 0.2

            # Check +4 position (after AUG)
            if aug_pos + 4 < len(sequence):
                pos_plus_4 = sequence[aug_pos + 4]
                if pos_plus_4 == "G":
                    current_score += 0.3

            best_score = max(best_score, current_score)

        return min(best_score, 1.0)

    def _count_ires_domains(self, dot_bracket: str) -> int:
        """
        Count structural domains in IRES region.

        Domain = independent stem-loop or structural unit.
        EMCV IRES: 4 domains
        HCV IRES: 3 domains
        """
        # Count stem regions separated by at least 5 unpaired bases
        domain_count = 0
        in_stem = False

        for ch in dot_bracket:
            if ch == "(":
                if not in_stem:
                    domain_count += 1
                    in_stem = True
            elif ch == "." and in_stem:
                in_stem = False

        return max(domain_count, 1)

    def _assess_ires_confidence(
        self,
        pyrimidine_ratio: float,
        structural_motif: str,
        itaf_sites: List[str],
        g4_present: bool,
    ) -> str:
        """
        Assess confidence level of IRES prediction.

        High confidence:
        - Strong polypyrimidine tract (>0.5)
        - Known structural motif
        - Multiple ITAF sites
        - G-quadruplex present
        """
        score = 0.0

        if pyrimidine_ratio > 0.5:
            score += 0.3
        elif pyrimidine_ratio > 0.4:
            score += 0.15

        if structural_motif in ["Y-shaped", "H-type", "G-quadruplex"]:
            score += 0.3
        elif structural_motif == "hairpin":
            score += 0.15

        score += len(itaf_sites) * 0.1

        if g4_present:
            score += 0.2

        if score >= 0.6:
            return "high"
        elif score >= 0.4:
            return "medium"
        else:
            return "low"

    def _predict_miRNA_sites(self, seq: str) -> List[MiRNABindingSite]:
        """
        Predict miRNA binding sites.

        miRNA binding:
        - 6-8 nt seed match (perfect or near-perfect)
        - 3' supplementary pairing
        - Site accessibility
        """
        sites = []

        for miRNA_name, seed in self.miRNA_db.items():
            # Find seed matches
            seed_len = len(seed)

            # Check all positions
            for i in range(len(seq) - seed_len):
                segment = seq[i:i+seed_len]

                # Compute complementarity
                complementarity = self._compute_seed_complementarity(seed, segment)

                if complementarity > 0.7:  # Good seed match
                    # Extend binding
                    full_site = seq[i:i+seed_len+10]

                    # Compute binding energy (simplified)
                    energy = self._estimate_binding_energy(seed, segment)

                    # ceRNA potential
                    ceRNA = complementarity * 0.6 + len(full_site) / 20.0 * 0.4

                    sites.append(MiRNABindingSite(
                        position_start=i,
                        position_end=i+seed_len+5,
                        circrna_seq=full_site,
                        miRNA_name=miRNA_name,
                        miRNA_seed=seed,
                        complementarity=complementarity,
                        binding_energy=energy,
                        ceRNA_potential=ceRNA,
                    ))

        return sites

    def _compute_seed_complementarity(self, seed: str, segment: str) -> float:
        """Compute seed match complementarity."""
        if len(seed) != len(segment):
            return 0.0

        matches = 0
        for s, m in zip(seed, segment):
            # RNA complementarity: A-U, G-C, G-U wobble
            if (s == "A" and m == "U") or (s == "U" and m == "A"):
                matches += 1
            elif (s == "G" and m == "C") or (s == "C" and m == "G"):
                matches += 1
            elif (s == "G" and m == "U") or (s == "U" and m == "G"):
                matches += 0.5  # Wobble pair

        return matches / len(seed)

    def _estimate_binding_energy(self, seed: str, segment: str) -> float:
        """Estimate miRNA-RNA binding energy."""
        # Simplified energy model
        # A-U: -1.1 kcal/mol
        # G-C: -2.4 kcal/mol
        # G-U: -1.0 kcal/mol

        energy = 0.0
        for s, m in zip(seed, segment):
            if (s == "A" and m == "U") or (s == "U" and m == "A"):
                energy -= 1.1
            elif (s == "G" and m == "C") or (s == "C" and m == "G"):
                energy -= 2.4
            elif (s == "G" and m == "U") or (s == "U" and m == "G"):
                energy -= 1.0

        return energy

    def _predict_rbp_sites(self, seq: str, dot_bracket: str) -> List[RBPBindingSite]:
        """
        Predict RBP binding sites.

        RBPs recognize:
        - Sequence motifs
        - Structure features
        - Combinations
        """
        sites = []

        for rbp_name, rbp_info in self.rbp_db.items():
            motif = rbp_info["motif"]
            role = rbp_info["role"]

            # Convert motif to regex
            # Y = pyrimidine (C or U)
            motif_regex = motif.replace("Y", "[CU]")

            # Find matches
            pattern = re.compile(motif_regex, re.IGNORECASE)
            for match in pattern.finditer(seq):
                pos = match.start()
                site_seq = seq[pos:pos+len(motif)+5]
                site_struct = dot_bracket[pos:pos+len(motif)+5]

                # Check structure accessibility
                accessibility = site_struct.count(".") / len(site_struct)

                binding_score = accessibility * 0.5 + 0.5

                sites.append(RBPBindingSite(
                    position_start=pos,
                    position_end=pos+len(motif),
                    sequence=site_seq,
                    structure=site_struct,
                    rbp_name=rbp_name,
                    motif=motif,
                    binding_score=binding_score,
                    functional_role=role,
                ))

        return sites

    def _predict_adar_sites(self, seq: str) -> List[ModificationSite]:
        """
        Predict ADAR editing sites.

        ADAR converts A to I (inosine) in dsRNA regions.
        Preference for AU mismatches in stems.
        """
        sites = []

        # Find dsRNA regions (inferred from sequence complementarity)
        # Simplified: look for repeated A's in potential stem context
        for i in range(len(seq) - 10):
            if seq[i] == "A":
                # Check if in potential stem context
                context = seq[i-5:i+5] if i >= 5 else seq[:i+5]
                u_content = sum(1 for c in context if c == "U")

                if u_content > 3:
                    # Potential ADAR site
                    prob = 0.4 + u_content * 0.05

                    sites.append(ModificationSite(
                        mod_type=ModificationType.ADAR_EDITING,
                        position=i,
                        sequence_context=context,
                        probability=prob,
                        effect="immunogenicity_modulation",  # A-to-I affects recognition
                        enzyme="ADAR1",
                        motif="AU_pair",
                    ))

        return sites

    def _compute_translation_potential(self, ires_sites: List[IRESSite]) -> float:
        """Compute overall translation potential."""
        if not ires_sites:
            return 0.0

        # Best IRES activity
        best_activity = max(s.activity_score for s in ires_sites)

        # Translation requires good IRES + accessibility
        return best_activity * 0.8

    def _compute_ceRNA_activity(self, miRNA_sites: List[MiRNABindingSite]) -> float:
        """Compute miRNA sponge potential."""
        if not miRNA_sites:
            return 0.0

        # Sum of binding potentials
        total_binding = sum(s.ceRNA_potential for s in miRNA_sites)

        # Normalize by number of sites
        return total_binding / max(len(miRNA_sites), 1) * 0.8

    def _compute_m6a_immunogenicity_effect(self, m6a_sites: List[ModificationSite]) -> float:
        """
        Compute m6A effect on immunogenicity.

        Liu et al., 2022: m6A reduces circRNA RIG-I recognition
        More m6A = less immunogenic
        """
        if not m6a_sites:
            return 0.0

        # Average probability
        avg_prob = np.mean([s.probability for s in m6a_sites])

        # Higher m6A = reduced immunogenicity
        reduction = avg_prob * 0.3

        return 1.0 - reduction  # Return remaining immunogenicity

    def _compute_modification_stability(
        self,
        m6a_sites: List[ModificationSite],
        rbp_sites: List[RBPBindingSite]
    ) -> float:
        """Compute modification-based stability."""
        # HuR and IGF2BP1 increase stability
        stabilizing_rbps = [s for s in rbp_sites if "stabilization" in s.functional_role]

        # m6A can have mixed effects
        m6a_effect = len(m6a_sites) * 0.02

        rbp_effect = len(stabilizing_rbps) * 0.05

        return np.clip(0.5 + m6a_effect + rbp_effect, 0.0, 1.0)

    def _estimate_conservation(self, seq: str, m6a_sites: List[ModificationSite]) -> float:
        """Estimate evolutionary conservation."""
        # m6A sites often conserved (functional)
        # Higher density = more likely functional = more conserved

        if not m6a_sites:
            return 0.3  # Baseline

        # Conservation increases with functional site density
        site_density = len(m6a_sites) / len(seq)

        return np.clip(0.3 + site_density * 50, 0.0, 1.0)


def compute_modification_score(features: ModificationFeatures) -> Dict[str, float]:
    """Compute modification-related scores."""
    return {
        "m6a_density": features.m6a_density,
        "translation_potential": features.translation_potential,
        "ceRNA_activity": features.ceRNA_activity,
        "rbp_binding_density": features.rbp_binding_density,
        "m6a_immunogenicity_effect": features.immunogenicity_modulation,
        "modification_stability": features.stability_score,
        "conservation_score": features.conservation_score,
        "total_m6a_sites": len(features.m6a_sites),
        "total_miRNA_sites": len(features.miRNA_sites),
        "total_rbp_sites": len(features.rbp_sites),
    }


def predict_modifications(sequence: str) -> ModificationFeatures:
    """Predict all modifications for circRNA."""
    predictor = ModificationPredictor()
    return predictor.analyze(sequence)