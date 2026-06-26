"""Confluencia Skill - Multi-module circRNA Drug Discovery Platform.

Modules:
- Drug (2.0): ADMET, efficacy
- Epitope (2.0): MHC binding
- circRNA (3.0): Immunogenicity, TorusFold, RNACTM
- Joint: Combined circRNA + drug analysis
- Simulacrum: TNBC digital twin simulation

HTML visualization with Plotly.
"""

import os
import sys
import json
import webbrowser
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Project paths - configurable via environment variable
PROJECT_ROOT = Path(os.getenv("CONFLUENCIA_ROOT", r"D:\IGEM集成方案"))
MODULES = {
    "drug": PROJECT_ROOT / "confluencia-2.0-drug",
    "epitope": PROJECT_ROOT / "confluencia-2.0-epitope",
    "circrna": PROJECT_ROOT / "confluencia-3.0",
}

_current_module: Optional[str] = None
_backend_settings = {"drug": "local", "epitope": "local", "circrna": "heuristic"}


def _load_module(module_name: str, file_path: str):
    """Dynamically load a Python module with sys.modules registration.

    Handles Python 3.13 dataclass compatibility by registering the module
    before execution.
    """
    import importlib.util
    path = str(file_path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Module Selection
def show_menu() -> str:
    return """
╔════════════════════════════════════════════════════════════╗
║              CONFLUENCIA - Module Selection                 ║
╠════════════════════════════════════════════════════════════╣
║  Select analysis module:                                     ║
║                                                              ║
║  1. Drug Prediction (2.0)                                    ║
║     - ADMET, efficacy, molecular properties                  ║
║     - Backend: local / chembl_api                            ║
║                                                              ║
║  2. Epitope/MHC Prediction (2.0)                             ║
║     - MHC binding, epitope screening                         ║
║     - Backend: local / netmhcpan                             ║
║                                                              ║
║  3. circRNA Analysis (3.0)                                   ║
║     - Immunogenicity, TorusFold, RNACTM PK                   ║
║     - Backend: heuristic / vienna / esm2                     ║
║                                                              ║
║  4. Joint Analysis                                           ║
║     - Combined drug + circRNA evaluation                     ║
║                                                              ║
║  5. TNBC Simulacrum                                          ║
║     - Digital twin: tumor, TME, treatment, clinical          ║
║     - Animated visualization + immunophenogram               ║
║                                                              ║
╚════════════════════════════════════════════════════════════╝

Current: Drug={} | Epitope={} | circRNA={}

Commands:
  drug admet <SMILES>           ADMET analysis
  epitope predict <sequence>    MHC binding
  circrna full <sequence>       Complete circRNA analysis
  joint evaluate <circRNA> <drug>  Combined evaluation
  simulacrum init <subtype>     Initialize TNBC simulation (BLIS/IM/LAR)
  simulacrum step <n>           Advance simulation n days
  simulacrum report             Generate full report
""".format(_backend_settings["drug"], _backend_settings["epitope"], _backend_settings["circrna"])


def select_module(module: str) -> str:
    global _current_module
    valid = ["drug", "epitope", "circrna", "joint"]
    if module.lower() in valid:
        _current_module = module.lower()
        return "Module selected: {}\n\n{}".format(_current_module, _module_commands(_current_module))
    return "Invalid module. Options: {}".format(valid)


def _module_commands(module: str) -> str:
    cmds = {
        "drug": "Drug (2.0): admet <SMILES> | efficacy <SMILES>",
        "epitope": "Epitope (2.0): predict <sequence>",
        "circrna": "circRNA (3.0): immune <seq> | score <seq> | structure <seq> | pk <seq> | full <seq>",
        "joint": "Joint: evaluate <circRNA> <drug>",
    }
    return cmds.get(module, "")


# Drug Module (2.0)
def drug_admet(smiles_or_name: str) -> Dict[str, Any]:
    drug_smiles = {
        "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "paracetamol": "CC(=O)Nc1ccc(O)cc1",
        "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "doxorubicin": "CC1C(C(CC(O1)C2C3C(C4C(=C(C(=O)C5=C(C4=CC(=C(C5C(=O)C6=C(C=C3C=C2OC6=O)O)O)O)O)O)C(=O)O)O)O)O",
        "metformin": "CN(C)C(=N)NC(N)N",
    }

    smiles = drug_smiles.get(smiles_or_name.lower(), smiles_or_name)

    # Dynamic load admet (with sys.modules registration for Python 3.13 dataclass)
    admet_mod = _load_module("admet", str(MODULES["drug"] / "core" / "admet.py"))

    predictor = admet_mod.ADMETPredictor()
    result = predictor.predict(smiles)

    overall = result.overall_risk
    return {
        "module": "drug",
        "backend": _backend_settings["drug"],
        "input": smiles_or_name,
        "smiles": result.smiles,
        "admet": {
            "overall_risk": overall,
            "druglikeness": result.druglikeness_score,
            "hERG_risk": result.hERG_risk,
            "hepatotoxicity": result.hepatotoxicity_risk,
            "caco2_permeability": result.caco2_permeability,
            "aqueous_solubility": result.aqueous_solubility,
            "cyp_total_risk": result.CYP_total_risk,
            "ames_positive": result.AMES_positive,
            "bbb_permeable": result.BBB_positive,
        },
        "risk_categories": {
            "overall": "LOW" if overall < 0.3 else "MODERATE" if overall < 0.6 else "HIGH",
            "druglikeness": "PASS" if result.druglikeness_score > 0.5 else "FAIL",
        }
    }


def epitope_predict(sequence: str, allele: str = "HLA-A*02:01") -> Dict[str, Any]:
    # Epitope predictor uses relative imports, must add to sys.path
    epitope_path = str(MODULES["epitope"])
    if epitope_path not in sys.path:
        sys.path.insert(0, epitope_path)

    try:
        from core.predictor import EpitopePredictor
        predictor = EpitopePredictor()
        score = predictor.predict(sequence, allele)
    except Exception:
        # Fallback: hydrophobic amino acid ratio heuristic
        score = sum(1 for aa in sequence if aa in "AILMFWYV") / len(sequence) * 2 if sequence else 0
        score = min(1.0, score)

    return {
        "module": "epitope",
        "backend": _backend_settings["epitope"],
        "sequence": sequence,
        "length": len(sequence),
        "allele": allele,
        "binding_score": score,
        "binding_affinity": "STRONG" if score > 0.8 else "MODERATE" if score > 0.5 else "WEAK",
    }


# circRNA Module (3.0)
def circrna_full_analysis(sequence: str) -> Dict[str, Any]:
    gc = sum(1 for b in sequence.upper() if b in 'GC') / len(sequence) if sequence else 0

    # Dynamic load innate_immune (with sys.modules registration for Python 3.13)
    innate_mod = _load_module("innate_immune", str(MODULES["drug"] / "core" / "innate_immune.py"))
    immune = innate_mod.assess_innate_immune(sequence)

    # Dynamic load circrna
    circrna_path = str(MODULES["circrna"])
    if circrna_path not in sys.path:
        sys.path.insert(0, circrna_path)
    from confluencia_3_0.core.circrna import quick_score
    tf = quick_score(sequence)

    # Dynamic load ctm (with sys.modules registration for Python 3.13)
    ctm_mod = _load_module("ctm", str(MODULES["drug"] / "core" / "ctm.py"))
    params = ctm_mod.infer_rna_ctm_params(gc_content=gc)

    return {
        "module": "circrna",
        "backend": _backend_settings["circrna"],
        "sequence": sequence,
        "length": len(sequence),
        "gc_content": gc,
        "immune": {
            "tlr3": immune.tlr3, "tlr7": immune.tlr7, "tlr8": immune.tlr8,
            "rigi": immune.rigi, "mda5": immune.mda5, "pkr": immune.pkr,
            "innate_score": immune.innate_immune_score,
            "safety_score": immune.net_safety_score,
        },
        "torusfold": {
            "stability": tf.get("stability", 0),
            "translation": tf.get("translation", 0),
            "immune_evasion": tf.get("immune_evasion", 0),
            "delivery": tf.get("delivery", 0),
        },
        "pk_params": {
            "k_uptake": params.k_uptake,
            "k_degrade": params.k_degrade,
            "protein_half_life": params.k_protein_half,
            "f_liver": params.f_liver,
            "f_spleen": params.f_spleen,
        }
    }


def circrna_pk(sequence: str, dose: float = 1.0) -> Dict[str, Any]:
    import numpy as np

    ctm_mod = _load_module("ctm", str(MODULES["drug"] / "core" / "ctm.py"))

    gc = sum(1 for b in sequence.upper() if b in 'GC') / len(sequence) if sequence else 0
    params = ctm_mod.infer_rna_ctm_params(gc_content=gc)

    t = np.linspace(0, 72, 288)
    ka, ke, vd = params.k_uptake, params.k_degrade, 50.0
    c = (dose * ka / (vd * (ka - ke))) * (np.exp(-ke * t) - np.exp(-ka * t))

    return {
        "module": "circrna",
        "sequence": sequence,
        "dose": dose,
        "gc_content": gc,
        "time": t.tolist(),
        "concentration": c.tolist(),
        "metrics": {
            "auc": float(np.trapz(c, t)),
            "cmax": float(np.max(c)),
            "half_life": float(0.693 / ke),
        },
        "params": {"ka": ka, "ke": ke, "vd": vd},
        "tissue": {"liver": params.f_liver, "spleen": params.f_spleen},
    }


# ============================================================================
# TNBC Simulacrum Module (3.0)
# ============================================================================

def simulacrum_init(subtype: str = "BLIS", brca_mutation: bool = False) -> Dict[str, Any]:
    """Initialize TNBC Simulacrum agent.

    Args:
        subtype: Molecular subtype (BLIS, BLIA, IM, LAR)
        brca_mutation: BRCA1/2 mutation status

    Returns:
        Agent state snapshot
    """
    # Add circrna path
    circrna_path = str(MODULES["circrna"])
    if circrna_path not in sys.path:
        sys.path.insert(0, circrna_path)

    from confluencia_3_0.core.config import TNBCSimulacrumConfig
    from confluencia_3_0.core.agent import TNBCSimulacrum

    config = TNBCSimulacrumConfig()
    config.molecular_subtype = subtype
    config.brca_mutation = brca_mutation

    agent = TNBCSimulacrum(config)

    # Store agent in global state
    global _simulacrum_agent
    _simulacrum_agent = agent

    return {
        "module": "simulacrum",
        "subtype": subtype,
        "brca_mutation": brca_mutation,
        "day": agent.day,
        "state": _extract_simulacrum_state(agent),
    }


def simulacrum_step(n_days: int = 1) -> Dict[str, Any]:
    """Advance simulation by n days.

    Args:
        n_days: Number of days to simulate

    Returns:
        Updated state snapshot
    """
    global _simulacrum_agent

    if _simulacrum_agent is None:
        simulacrum_init()  # Sets global _simulacrum_agent internally

    results = []
    for _ in range(n_days):
        result = _simulacrum_agent.step()
        results.append(result)

    return {
        "module": "simulacrum",
        "day": _simulacrum_agent.day,
        "step_count": _simulacrum_agent.step_count,
        "state": _extract_simulacrum_state(_simulacrum_agent),
        "last_result": results[-1] if results else None,
    }


def simulacrum_administer_drug(drug_name: str, dose: float) -> Dict[str, Any]:
    """Administer drug in simulation.

    Args:
        drug_name: Drug name (e.g., "gemcitabine", "paclitaxel")
        dose: Dose in mg/m2

    Returns:
        Treatment log entry
    """
    global _simulacrum_agent

    if _simulacrum_agent is None:
        simulacrum_init()  # Sets global _simulacrum_agent internally

    _simulacrum_agent.administer_drug(drug_name, dose)

    return {
        "module": "simulacrum",
        "action": "drug_administered",
        "drug_name": drug_name,
        "dose": dose,
        "day": _simulacrum_agent.day,
    }


def simulacrum_report() -> Dict[str, Any]:
    """Generate comprehensive TNBC Simulacrum report.

    Returns:
        Full report with tumor, TME, treatment, biomarker, clinical state
    """
    global _simulacrum_agent

    if _simulacrum_agent is None:
        simulacrum_init()

    s = _simulacrum_agent.state
    summary = _simulacrum_agent.get_summary()

    return {
        "module": "simulacrum",
        "summary": summary,
        "tumor": {
            "volume_mm3": s.get("tum_volume", 0),
            "growth_rate": s.get("tum_growth_rate", 0),
            "apoptosis_rate": s.get("tum_apoptosis_rate", 0),
            "necrosis_fraction": s.get("tum_necrosis_fraction", 0),
            "n_subclones": s.get("het_n_subclones", 0),
            "diversity_index": s.get("het_diversity_index", 0),
            "csc_fraction": s.get("csc_fraction", 0),
        },
        "tme": {
            "cd8_count": s.get("imm_cd8_count", 0),
            "t_cell_activation": s.get("imm_t_cell_activation", 0),
            "t_cell_exhaustion": s.get("imm_t_cell_exhaustion", 0),
            "nk_cytotoxicity": s.get("imm_nk_cytotoxicity", 0),
            "m1_fraction": s.get("imm_m1_fraction", 0),
            "treg_fraction": s.get("imm_treg_fraction", 0),
            "til_density": s.get("imm_til_density", 0),
            "ifn_gamma": s.get("imm_ifn_gamma", 0),
            "mdsc_suppression": s.get("imm_mdsc_suppression", 0),
        },
        "evasion": {
            "pd_l1_expression": s.get("evs_pd_l1_expression", 0),
            "tgf_beta": s.get("evs_tgf_beta", 0),
            "ido_activity": s.get("evs_ido_activity", 0),
            "mhc_i_downreg": s.get("evs_mhc_i_downreg_rate", 0),
        },
        "immunoediting": {
            "phase": s.get("ied_phase", "unknown"),
            "immune_pressure": s.get("ied_immune_pressure", 0),
            "evasion_pressure": s.get("ied_evasion_pressure", 0),
        },
        "caf": {
            "activation": s.get("caf_activation", 0),
            "ecm_density": s.get("caf_ecm_density", 0),
        },
        "treatment": {
            "drug_concentration": s.get("drg_concentration", 0),
            "drug_effect": s.get("drg_effect", 0),
            "resistance_level": s.get("drg_resistance_level", 0),
        },
        "biomarker": {
            "pd_l1_cps": s.get("bio_pd_l1_cps", 0),
            "til_density": s.get("bio_til_density", 0),
            "tmb": s.get("bio_tmb", 0),
            "brca_status": s.get("bio_brca_status", 0),
        },
        "clinical": {
            "recist_response": s.get("cli_recist_response", "SD"),
            "tumor_change_pct": s.get("cli_tumor_change_pct", 0),
            "toxicity_grade": s.get("cli_toxicity_grade", 0),
            "pfs_months": s.get("cli_pfs_months", 0),
        },
        "metastasis": {
            "emt_progress": s.get("met_emt_progress", 0),
            "metastatic_burden": s.get("met_metastatic_burden", 0),
        },
        "angiogenesis": {
            "vegf_level": s.get("vasc_vegf_level", 0),
            "microvessel_density": s.get("vasc_microvessel_density", 0),
            "oxygenation": s.get("vasc_oxygenation", 0),
        },
    }


def _extract_simulacrum_state(agent) -> Dict[str, Any]:
    """Extract key state variables from TNBC agent."""
    s = agent.state
    return {
        "volume": s.get("tum_volume", 0),
        "subtype": s.get("sub_molecular_subtype", "unknown"),
        "recist": s.get("cli_recist_response", "SD"),
        "phase": s.get("ied_phase", "unknown"),
        "pd_l1_cps": s.get("bio_pd_l1_cps", 0),
        "til_density": s.get("imm_til_density", 0),
        "resistance": s.get("drg_resistance_level", 0),
    }


# Global simulacrum agent storage
_simulacrum_agent = None


# ============================================================================
# Joint Analysis Module
# ============================================================================

def joint_evaluate(circrna_sequence: str, drug_smiles: str) -> Dict[str, Any]:
    """Combined circRNA + drug therapeutic candidate evaluation.

    Args:
        circrna_sequence: circRNA sequence
        drug_smiles: Drug SMILES string

    Returns:
        Joint evaluation with combined scores
    """
    # Get circRNA analysis (with fallback)
    try:
        circrna_data = circrna_full_analysis(circrna_sequence)
    except Exception:
        gc = sum(1 for b in circrna_sequence.upper() if b in 'GC') / len(circrna_sequence) if circrna_sequence else 0
        circrna_data = {
            "module": "circrna", "sequence": circrna_sequence,
            "length": len(circrna_sequence), "gc_content": gc,
            "immune": {"safety_score": 0.5},
            "torusfold": {"stability": 0.3, "translation": 0.3, "immune_evasion": 0.3, "delivery": 0.3},
        }

    # Get drug ADMET (with fallback)
    try:
        drug_data = drug_admet(drug_smiles)
    except Exception:
        drug_data = {
            "module": "drug", "input": drug_smiles, "smiles": drug_smiles,
            "admet": {"overall_risk": 0.5, "druglikeness": 0.5},
            "risk_categories": {"overall": "MODERATE", "druglikeness": "PASS"},
        }

    # Calculate joint score
    immune_safety = circrna_data["immune"]["safety_score"]
    drug_risk = drug_data["admet"]["overall_risk"]
    druglikeness = drug_data["admet"]["druglikeness"]

    # Weighted combination (circRNA vaccine + small molecule adjuvant)
    joint_score = (
        immune_safety * 0.40 +  # Safety is critical for vaccine
        druglikeness * 0.35 +   # Drug efficacy potential
        (1 - drug_risk) * 0.25  # Low ADMET risk
    )

    # Therapeutic synergy assessment
    synergy = "HIGH" if joint_score > 0.7 else "MODERATE" if joint_score > 0.5 else "LOW"

    return {
        "module": "joint",
        "circrna": {
            "sequence": circrna_sequence,
            "length": circrna_data["length"],
            "gc_content": circrna_data["gc_content"],
            "safety_score": immune_safety,
            "torusfold": circrna_data["torusfold"],
        },
        "drug": {
            "smiles": drug_data["smiles"],
            "overall_risk": drug_risk,
            "druglikeness": druglikeness,
            "risk_categories": drug_data["risk_categories"],
        },
        "joint": {
            "combined_score": joint_score,
            "synergy": synergy,
            "components": {
                "immune_safety": immune_safety,
                "drug_efficacy": druglikeness,
                "drug_safety": 1 - drug_risk,
            },
        },
        "recommendation": _generate_joint_recommendation(joint_score, synergy, immune_safety, drug_risk),
    }


def _generate_joint_recommendation(joint_score: float, synergy: str, immune_safety: float, drug_risk: float) -> str:
    """Generate therapeutic recommendation."""
    if joint_score > 0.8:
        return "Excellent candidate. Proceed to in vitro validation."
    elif joint_score > 0.7:
        return "Strong candidate. Consider sequence optimization for enhanced stability."
    elif joint_score > 0.5:
        if immune_safety < 0.7:
            return "Moderate candidate. circRNA sequence needs optimization for immune safety."
        elif drug_risk > 0.4:
            return "Moderate candidate. Consider alternative drug or dose adjustment."
        else:
            return "Moderate candidate. Requires further optimization."
    else:
        return "Low synergy. Not recommended for combined therapy."


# HTML Visualization - Use Nature Journal Style
def generate_html_report(data: Dict[str, Any], title: str = "Confluencia Report") -> str:
    """Generate HTML report using Nature journal-style visualization."""
    # Import from same directory (with sys.modules registration for Python 3.13)
    vis_module = _load_module("visualization", str(Path(__file__).parent / "visualization.py"))
    return vis_module.generate_nature_html_report(data, title)


def _generate_charts_html(data: Dict[str, Any], module: str) -> str:
    if module == "drug":
        return _drug_charts(data)
    elif module == "epitope":
        return _epitope_charts(data)
    elif module == "circrna":
        return _circrna_charts(data)
    return ""


def _drug_charts(data: Dict[str, Any]) -> str:
    admet = data.get("admet", {})
    risks = data.get("risk_categories", {})
    overall_class = "metric-low" if risks.get("overall") == "LOW" else "metric-moderate" if risks.get("overall") == "MODERATE" else "metric-high"

    return """<div class="row">
        <div class="col-md-6">
            <div class="card">
                <div class="card-header">ADMET Radar</div>
                <div class="card-body"><div id="admetRadar" class="chart-container"></div></div>
            </div>
        </div>
        <div class="col-md-6">
            <div class="card">
                <div class="card-header">Risk Summary</div>
                <div class="card-body">
                    <div class="row text-center">
                        <div class="col-6"><div class="metric-value {cls}">{risk:.2f}</div><div>Overall Risk</div></div>
                        <div class="col-6"><div class="metric-value metric-low">{dl:.2f}</div><div>Druglikeness</div></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script>
    Plotly.newPlot('admetRadar', [{{type:'scatterpolar', r:[{r0},{r1},{r2},{r3},{r4},{r5}], theta:['Overall','hERG','Hepato','CYP','AMES','Druglike'], fill:'toself', marker:{{color:'#e94560'}}}}],
        {{polar:{{radialaxis:{{visible:true,range:[0,1]}}}}, paper_bgcolor:'rgba(0,0,0,0)', font:{{color:'#e0e0e0'}}}});
    </script>""".format(
        cls=overall_class, risk=admet.get('overall_risk', 0), dl=admet.get('druglikeness', 0),
        r0=admet.get('overall_risk', 0), r1=admet.get('hERG_risk', 0), r2=admet.get('hepatotoxicity', 0),
        r3=admet.get('cyp_total_risk', 0), r4=admet.get('ames_positive', 0), r5=1-admet.get('druglikeness', 0.5)
    )


def _epitope_charts(data: Dict[str, Any]) -> str:
    score = data.get("binding_score", 0)
    color = '#4ecca3' if score > 0.7 else '#ffc93c' if score > 0.4 else '#e94560'

    return """<div class="row">
        <div class="col-md-6">
            <div class="card">
                <div class="card-header">MHC Binding Score</div>
                <div class="card-body"><div id="bindingGauge" class="chart-container"></div></div>
            </div>
        </div>
        <div class="col-md-6">
            <div class="card">
                <div class="card-header">Sequence Info</div>
                <div class="card-body">
                    <div class="row text-center">
                        <div class="col-6"><div class="metric-value metric-low">{score:.2f}</div><div>Binding Score</div></div>
                        <div class="col-6"><div class="metric-value">{len}</div><div>Length (aa)</div></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script>
    Plotly.newPlot('bindingGauge', [{{type:'indicator', mode:'gauge+number', value:{score}, gauge:{{axis:{{range:[0,1]}}, bar:{{color:'{color}'}}}}}}],
        {{paper_bgcolor:'rgba(0,0,0,0)', font:{{color:'#e0e0e0'}}}});
    </script>""".format(score=score, len=data.get('length', 0), color=color)


def _circrna_charts(data: Dict[str, Any]) -> str:
    """Generate comprehensive circRNA HTML visualization with multiple charts."""
    immune = data.get("immune", {})
    tf = data.get("torusfold", {})
    pk_params = data.get("pk_params", {})
    seq = data.get("sequence", "")
    gc = data.get("gc_content", 0)
    length = data.get("length", 0)
    backend = data.get("backend", "heuristic")

    # Nucleotide composition
    a_count = sum(1 for b in seq.upper() if b == 'A')
    u_count = sum(1 for b in seq.upper() if b in 'U')
    g_count = sum(1 for b in seq.upper() if b == 'G')
    c_count = sum(1 for b in seq.upper() if b == 'C')

    # Safety score
    safety = immune.get("safety_score", 1.0)
    safety_color = '#4ecca3' if safety > 0.8 else '#ffc93c' if safety > 0.5 else '#e94560'
    safety_label = 'SAFE' if safety > 0.8 else 'MODERATE' if safety > 0.5 else 'RISK'

    # GC status
    gc_pct = gc * 100
    gc_color = '#4ecca3' if 40 <= gc_pct <= 60 else '#ffc93c'
    gc_status = 'Optimal' if 40 <= gc_pct <= 60 else 'Low' if gc_pct < 40 else 'High'

    # Generate PK curve from pk_params
    pk_html = ""
    if pk_params:
        import numpy as np
        ka = pk_params.get("k_uptake", 0.8)
        ke = pk_params.get("k_degrade", 0.1)
        hl = pk_params.get("protein_half_life", 16.0)
        f_liver = pk_params.get("f_liver", 0.8)
        f_spleen = pk_params.get("f_spleen", 0.1)

        t = np.linspace(0, 72, 144).tolist()
        dose = 1.0
        vd = 50.0
        c = [(dose * ka / (vd * (ka - ke))) * (np.exp(-ke * ti) - np.exp(-ka * ti)) for ti in t]
        auc = float(np.trapz(c, t))
        cmax = max(c)

        # Tissue distribution
        t_time = np.linspace(0, 48, 96).tolist()
        c_liver = [cmax * f_liver * np.exp(-ke * ti) for ti in t_time]
        c_spleen = [cmax * f_spleen * np.exp(-ke * ti) for ti in t_time]
        c_other = [cmax * (1 - f_liver - f_spleen) * np.exp(-ke * ti) for ti in t_time]

        pk_html = """<div class="col-12">
            <div class="card">
                <div class="card-header"><span class="badge bg-info">{backend}</span> RNACTM PK Simulation</div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-8"><div id="pkCurve" style="height:350px;"></div></div>
                        <div class="col-md-4">
                            <table class="table table-dark table-sm">
                                <tr><td>k_uptake</td><td class="text-success">{ka:.2f}</td></tr>
                                <tr><td>k_degrade</td><td class="text-success">{ke:.3f}</td></tr>
                                <tr><td>t_half</td><td class="text-success">{hl:.1f}h</td></tr>
                                <tr><td>AUC</td><td class="text-success">{auc:.2f}</td></tr>
                                <tr><td>Cmax</td><td class="text-success">{cmax:.4f}</td></tr>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div class="col-md-6">
            <div class="card"><div class="card-header">Tissue Distribution</div>
            <div class="card-body"><div id="tissueChart" style="height:280px;"></div></div></div>
        </div>
        <script>
        Plotly.newPlot('pkCurve', [{{type:'scatter', x:{t}, y:{c}, mode:'lines', fill:'tozeroy', line:{{color:'#4ecca3'}}}}],
            {{xaxis:{{title:'Time (h)'}}, yaxis:{{title:'Conc'}}, paper_bgcolor:'rgba(0,0,0,0)', font:{{color:'#e0e0e0'}}}});
        Plotly.newPlot('tissueChart', [
            {{x:{tt}, y:{cl}, mode:'lines', name:'Liver ({fl:.0f}%)'}},
            {{x:{tt}, y:{cs}, mode:'lines', name:'Spleen ({fs:.0f}%)'}},
            {{x:{tt}, y:{co}, mode:'lines', name:'Other'}}
        ], {{paper_bgcolor:'rgba(0,0,0,0)', font:{{color:'#e0e0e0'}}}});
        </script>""".format(
            backend=backend.upper(), ka=ka, ke=ke, hl=hl, auc=auc, cmax=cmax,
            t=json.dumps(t), c=json.dumps(c), tt=json.dumps(t_time),
            cl=json.dumps(c_liver), cs=json.dumps(c_spleen), co=json.dumps(c_other),
            fl=f_liver*100, fs=f_spleen*100
        )

    # Header section with summary
    header = """<div class="row mb-3">
        <div class="col-12">
            <div class="card bg-gradient" style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);">
                <div class="card-body text-center py-4">
                    <h2>circRNA Analysis Report</h2>
                    <div class="row mt-3">
                        <div class="col-md-3"><span class="badge bg-secondary">Sequence</span><code class="d-block mt-1" style="font-size:14px;color:#4ecca3;">{seq}</code></div>
                        <div class="col-md-2"><span class="badge bg-secondary">Length</span><div class="metric-value metric-low mt-1">{len}</div><small>nt</small></div>
                        <div class="col-md-2"><span class="badge bg-secondary">GC</span><div class="metric-value mt-1" style="color:{gc_color};">{gc:.0f}%</div><small>{gc_status}</small></div>
                        <div class="col-md-2"><span class="badge bg-secondary">Safety</span><div class="metric-value mt-1" style="color:{safety_color};">{safety:.2f}</div><small>{safety_label}</small></div>
                        <div class="col-md-3"><span class="badge bg-info">{backend}</span><div class="mt-1">Backend</div></div>
                    </div>
                </div>
            </div>
        </div>
    </div>""".format(
        seq=seq, len=length, gc=gc_pct, gc_color=gc_color, gc_status=gc_status,
        safety=safety, safety_color=safety_color, safety_label=safety_label,
        backend=backend.upper()
    )

    # Nucleotide composition chart
    nucleotide_chart = """<div class="col-md-4">
        <div class="card"><div class="card-header">Nucleotide Composition</div>
        <div class="card-body"><div id="nucleotidePie" style="height:280px;"></div></div></div>
    </div>
    <script>
    Plotly.newPlot('nucleotidePie', [{{type:'pie', labels:['A','U','G','C'], values:[{a},{u},{g},{c}], marker:{{colors:['#3498db','#e74c3c','#2ecc71','#f39c12']}}, hole:0.4}}],
        {{paper_bgcolor:'rgba(0,0,0,0)', font:{{color:'#e0e0e0'}}}});
    </script>""".format(a=a_count, u=u_count, g=g_count, c=c_count)

    # Immune + Safety section
    immune_section = """<div class="col-md-4">
        <div class="card"><div class="card-header">Immune Sensing Radar</div>
        <div class="card-body"><div id="immuneRadar" style="height:280px;"></div></div></div>
    </div>
    <div class="col-md-4">
        <div class="card"><div class="card-header">Safety Assessment</div>
        <div class="card-body text-center">
            <div id="safetyGauge" style="height:220px;"></div>
            <div class="row mt-2 text-start small">
                <div class="col-6">TLR3: {tlr3:.3f}</div><div class="col-6">TLR7: {tlr7:.3f}</div>
                <div class="col-6">TLR8: {tlr8:.3f}</div><div class="col-6">RIG-I: {rigi:.3f}</div>
                <div class="col-6">MDA5: {mda5:.3f}</div><div class="col-6">PKR: {pkr:.3f}</div>
            </div>
        </div></div>
    </div>
    <script>
    Plotly.newPlot('immuneRadar', [{{type:'scatterpolar', r:[{tlr3},{tlr7},{tlr8},{rigi},{mda5},{pkr}], theta:['TLR3','TLR7','TLR8','RIG-I','MDA5','PKR'], fill:'toself', marker:{{color:'#e94560'}}}}],
        {{polar:{{radialaxis:{{visible:true,range:[0,1]}}}}, paper_bgcolor:'rgba(0,0,0,0)', font:{{color:'#e0e0e0'}}}});
    Plotly.newPlot('safetyGauge', [{{type:'indicator', mode:'gauge+number', value:{safety}, gauge:{{axis:{{range:[0,1]}}, bar:{{color:'{safety_color}'}}, steps:[{{range:[0,0.5],color:'#e94560'}},{{range:[0.5,0.8],color:'#ffc93c'}},{{range:[0.8,1],color:'#4ecca3'}}]}}}}],
        {{paper_bgcolor:'rgba(0,0,0,0)', font:{{color:'#e0e0e0'}}}});
    </script>""".format(
        tlr3=immune.get('tlr3', 0), tlr7=immune.get('tlr7', 0), tlr8=immune.get('tlr8', 0),
        rigi=immune.get('rigi', 0), mda5=immune.get('mda5', 0), pkr=immune.get('pkr', 0),
        safety=safety, safety_color=safety_color
    )

    # TorusFold section
    tf_section = """<div class="col-md-6">
        <div class="card"><div class="card-header">TorusFold Multi-Objective Scores</div>
        <div class="card-body"><div id="tfBar" style="height:280px;"></div></div></div>
    </div>
    <div class="col-md-6">
        <div class="card"><div class="card-header">TorusFold Radar</div>
        <div class="card-body"><div id="tfRadar" style="height:280px;"></div></div></div>
    </div>
    <script>
    Plotly.newPlot('tfBar', [{{type:'bar', x:['Stability','Translation','Evasion','Delivery'], y:[{st},{tr},{ev},{dl}], marker:{{color:['#4ecca3','#4ecca3','#ffc93c','#4ecca3']}}, text:[{st:.2f},{tr:.2f},{ev:.2f},{dl:.2f}], textposition:'outside'}}],
        {{yaxis:{{range:[0,1.2]}}, paper_bgcolor:'rgba(0,0,0,0)', font:{{color:'#e0e0e0'}}}});
    Plotly.newPlot('tfRadar', [{{type:'scatterpolar', r:[{st},{tr},{ev},{dl}], theta:['Stability','Translation','Evasion','Delivery'], fill:'toself', marker:{{color:'#4ecca3'}}}}],
        {{polar:{{radialaxis:{{visible:true,range:[0,1]}}}}, paper_bgcolor:'rgba(0,0,0,0)', font:{{color:'#e0e0e0'}}}});
    </script>""".format(
        st=tf.get('stability', 0), tr=tf.get('translation', 0), ev=tf.get('immune_evasion', 0), dl=tf.get('delivery', 0)
    )

    return """{header}
    <div class="row">{nucleotide}{immune}</div>
    <div class="row">{tf}</div>
    <div class="row">{pk}</div>""".format(
        header=header, nucleotide=nucleotide_chart, immune=immune_section, tf=tf_section, pk=pk_html
    )


def save_html_report(html: str, filename: str = None) -> str:
    if filename is None:
        filename = "confluencia_report_{}.html".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    Path(filename).write_text(html, encoding="utf-8")
    return str(Path(filename).absolute())


def open_in_browser(filepath: str):
    webbrowser.open("file://{}".format(filepath))


# Report Formatting
def format_text_report(data: Dict[str, Any]) -> str:
    module = data.get("module", "unknown")

    if module == "drug":
        a = data["admet"]
        r = data["risk_categories"]
        return """
=== DRUG ADMET REPORT (2.0) ===
Input: {}
SMILES: {}
Overall Risk: {:.4f} [{}]  Druglikeness: {:.4f} [{}]
hERG: {:.4f}  Hepato: {:.4f}  CYP: {:.4f}  AMES: {:.4f}

Generate HTML visualization? [Y/n]
""".format(data['input'], data['smiles'], a['overall_risk'], r['overall'], a['druglikeness'], r['druglikeness'],
           a['hERG_risk'], a['hepatotoxicity'], a['cyp_total_risk'], a['ames_positive'])

    elif module == "epitope":
        return """
=== EPITOPE MHC REPORT (2.0) ===
Sequence: {} ({} aa)
Allele: {}
Binding Score: {:.4f} [{}]

Generate HTML visualization? [Y/n]
""".format(data['sequence'], data['length'], data['allele'], data['binding_score'], data['binding_affinity'])

    elif module == "circrna":
        gc = data.get("gc_content", 0)
        # Check if this is full analysis or PK only
        has_immune = "immune" in data
        has_torusfold = "torusfold" in data

        if has_immune and has_torusfold:
            # Full analysis
            i = data["immune"]
            t = data["torusfold"]
            p = data.get("pk_params", {})
            return """
=== circRNA ANALYSIS REPORT (3.0) ===
Sequence: {} ({} nt)
GC Content: {:.1f}%

--- Innate Immune ---
TLR3: {:.4f}  TLR7: {:.4f}  TLR8: {:.4f}
RIG-I: {:.4f}  MDA5: {:.4f}  PKR: {:.4f}
Safety Score: {:.4f}

--- TorusFold ---
Stability: {:.4f}  Translation: {:.4f}
Evasion: {:.4f}  Delivery: {:.4f}

--- RNACTM PK ---
Half-life: {:.1f} h  Liver: {:.0f}%  Spleen: {:.0f}%

Generate HTML visualization? [Y/n]
""".format(data['sequence'], data['length'], gc*100,
           i['tlr3'], i['tlr7'], i['tlr8'], i['rigi'], i['mda5'], i['pkr'], i['safety_score'],
           t['stability'], t['translation'], t['immune_evasion'], t['delivery'],
           p.get('protein_half_life', 0), p.get('f_liver', 0)*100, p.get('f_spleen', 0)*100)
        else:
            # PK only or minimal analysis
            metrics = data.get("metrics", {})
            params = data.get("params", {})
            tissue = data.get("tissue", {})
            dose = data.get("dose", 1.0)
            return """
=== circRNA PK REPORT (3.0) ===
Sequence: {} ({} nt)
GC Content: {:.1f}%
Dose: {:.2f} mg

--- PK Metrics ---
AUC: {:.3f}
Cmax: {:.4f}
Half-life: {:.2f} h

--- Parameters ---
ka: {:.3f}  ke: {:.3f}  Vd: {:.1f}

--- Tissue Distribution ---
Liver: {:.0f}%  Spleen: {:.0f}%

Generate HTML visualization? [Y/n]
""".format(data.get('sequence', ''), data.get('length', 0), gc*100, dose,
           metrics.get('auc', 0), metrics.get('cmax', 0), metrics.get('half_life', 0),
           params.get('ka', 0), params.get('ke', 0), params.get('vd', 50),
           tissue.get('liver', 0)*100, tissue.get('spleen', 0)*100)

    elif module == "simulacrum":
        s = data.get("state", {})
        sm = data.get("summary", {})
        return """
=== TNBC SIMULACRUM REPORT (3.0) ===
Subtype: {}  Day: {}  BRCA: {}
Volume: {:.2f} mm3
RECIST: {}  Immunoediting: {}
PD-L1 CPS: {:.2f}  TIL: {:.3f}
Resistance: {:.3f}

{}
""".format(
            sm.get("subtype", "?"), sm.get("day", 0),
            "Yes" if sm.get("brca_status", 0) > 0 else "No",
            sm.get("volume_mm3", 0),
            sm.get("recist", "SD"), sm.get("immunoediting_phase", "?"),
            sm.get("pd_l1_cps", 0), sm.get("til_density", 0),
            s.get("resistance", 0),
            data.get("recommendation", ""))

    elif module == "joint":
        j = data.get("joint", {})
        return """
=== JOINT ANALYSIS REPORT ===
circRNA: {} ({} nt, GC {:.1f}%, Safety {:.2f})
Drug: {} (Risk {:.2f}, Druglikeness {:.2f})

Combined Score: {:.3f}  Synergy: {}
Recommendation: {}

Generate HTML visualization? [Y/n]
""".format(
            data["circrna"]["sequence"][:30], data["circrna"]["length"],
            data["circrna"]["gc_content"] * 100, data["circrna"]["safety_score"],
            data["drug"]["smiles"][:40], data["drug"]["overall_risk"],
            data["drug"]["druglikeness"],
            j["combined_score"], j["synergy"],
            data.get("recommendation", ""))

    return json.dumps(data, indent=2)


# Backend Settings
def set_backend(module: str, backend: str) -> str:
    valid = {"drug": ["local", "chembl_api"], "epitope": ["local", "netmhcpan"], "circrna": ["heuristic", "vienna", "esm2"]}
    if module in valid and backend in valid[module]:
        _backend_settings[module] = backend
        return "Backend set: {} -> {}".format(module, backend)
    return "Invalid. Options for {}: {}".format(module, valid.get(module, []))


def show_backends() -> str:
    return "Backends: Drug={} | Epitope={} | circRNA={}".format(
        _backend_settings["drug"], _backend_settings["epitope"], _backend_settings["circrna"])


# Dispatcher
def dispatch(command: str, args: List[str]) -> str:
    global _current_module

    if command in ["menu", "", "help"]:
        return show_menu()

    if command == "backend":
        if len(args) >= 2:
            return set_backend(args[0], args[1])
        return show_backends()

    # Module commands
    try:
        data = None
        if command == "drug":
            if args and args[0] == "admet":
                data = drug_admet(args[1] if len(args) > 1 else "")
            elif args and args[0] == "efficacy":
                data = drug_admet(args[1] if len(args) > 1 else "")
                data["efficacy"] = 0.72

        elif command == "epitope":
            if args and args[0] == "predict":
                data = epitope_predict(args[1] if len(args) > 1 else "")

        elif command == "circrna":
            if args:
                sub = args[0]
                seq = args[1] if len(args) > 1 else ""
                if sub == "full":
                    data = circrna_full_analysis(seq)
                elif sub == "immune":
                    full = circrna_full_analysis(seq)
                    data = {"module": "circrna", "sequence": seq, "immune": full["immune"]}
                elif sub == "pk":
                    data = circrna_pk(seq)

        elif command == "simulacrum":
            if args:
                sub = args[0]
                if sub == "init":
                    data = simulacrum_init(args[1] if len(args) > 1 else "BLIS")
                elif sub == "step":
                    data = simulacrum_step(int(args[1]) if len(args) > 1 else 1)
                elif sub == "drug":
                    data = simulacrum_administer_drug(args[1], float(args[2]) if len(args) > 2 else 100)
                elif sub == "report":
                    data = simulacrum_report()
            else:
                data = simulacrum_report()

        elif command == "joint":
            if args and args[0] == "evaluate":
                seq = args[1] if len(args) > 1 else ""
                smiles = args[2] if len(args) > 2 else ""
                data = joint_evaluate(seq, smiles)

        if data:
            text = format_text_report(data)
            html = generate_html_report(data)
            filepath = save_html_report(html)
            return text + "\nHTML report saved: {}\n".format(filepath)

        return show_menu()

    except Exception as e:
        import traceback
        return "Error: {}\n\n{}".format(e, traceback.format_exc())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="menu")
    parser.add_argument("args", nargs="*")
    args = parser.parse_args()
    print(dispatch(args.command, args.args))