"""Confluencia Streamlit Utilities - Central skill API integration.

Provides unified interface to confluencia skill API, handling:
- Environment variable configuration (CONFLUENCIA_ROOT)
- Python 3.13 dataclass compatibility
- Backend switching
- HTML report generation

All Streamlit pages should use these functions instead of direct imports.
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Environment variable configuration
PROJECT_ROOT = Path(os.getenv("CONFLUENCIA_ROOT", r"D:\IGEM集成方案"))
SKILL_PATH = Path(os.getenv("CONFLUENCIA_SKILL_PATH", r"C:\Users\LENOVO\.claude\skills\confluencia"))

# Add skill path to sys.path
if str(SKILL_PATH) not in sys.path:
    sys.path.insert(0, str(SKILL_PATH))

# Module directory names (normalized to use underscores for Python imports)
MODULE_NAMES = {
    "drug": "confluencia-2.0-drug",
    "epitope": "confluencia-2.0-epitope",
    "circrna": "confluencia-3.0",
}

# Backend settings (shared across all pages)
_backend_settings = {
    "drug": "local",
    "epitope": "local",
    "circrna": "heuristic",
}


def _load_skill_module() -> Any:
    """Load the skill __init__.py module with Python 3.13 compatibility."""
    import importlib.util

    skill_init = SKILL_PATH / "__init__.py"
    spec = importlib.util.spec_from_file_location("confluencia_skill", str(skill_init))
    mod = importlib.util.module_from_spec(spec)
    # Critical: register in sys.modules BEFORE exec for Python 3.13 dataclass
    sys.modules["confluencia_skill"] = mod
    spec.loader.exec_module(mod)
    return mod


# Global skill module (lazy loaded)
_skill_module = None


def get_skill_api() -> Any:
    """Get the skill API module (lazy loaded, cached)."""
    global _skill_module
    if _skill_module is None:
        _skill_module = _load_skill_module()
    return _skill_module


# ============================================================================
# Drug Module API
# ============================================================================

def drug_admet(smiles_or_name: str) -> Dict[str, Any]:
    """Run ADMET prediction via skill API.

    Args:
        smiles_or_name: SMILES string or drug name (aspirin, ibuprofen, etc.)

    Returns:
        ADMET result dict with overall_risk, druglikeness, etc.
    """
    api = get_skill_api()
    return api.drug_admet(smiles_or_name)


def get_drug_smiles_mapping() -> Dict[str, str]:
    """Get common drug name to SMILES mapping."""
    return {
        "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "paracetamol": "CC(=O)Nc1ccc(O)cc1",
        "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "doxorubicin": "CC1C(C(CC(O1)C2C3C(C4C(=C(C(=O)C5=C(C4=CC(=C(C5C(=O)C6=C(C=C3C=C2OC6=O)O)O)O)O)O)C(=O)O)O)O)O",
        "metformin": "CN(C)C(=N)NC(N)N",
        "paclitaxel": "CC1=C2C(=O)C[C@@H](O)[C@]3(O)C(C)=C(C[C@H](O)[C@H](NC(=O)c4ccccc4)[C@@H]5C(=O)N(C)C(=O)[C@H](O)[C@H](OC(=O)C(=O)Nc6ccc(OC)cc6)[C@@]7(O)C(=O)N(C)C(=O)[C@H](O)[C@@H](OC(=O)c8ccccc8)C(=O)[C@@H](C)C[C@@H]9OC[C@@H](O)[C@@H](C(=O)OC)C(=O)O1)C[C@H](C)C[C@@H]7O[C@@H]7C(=O)N(C)C(=O)[C@H](O)[C@@H](OC(=O)c%10ccccc%10)[C@@]5(O)C(=O)N(C)C(=O)[C@H](O)[C@@H](OC(=O)C(=O)Nc%11ccc(OC)cc%11)C(=O)[C@@H](C)C[C@H]3O2)C[C@H](OC(=O)C=Cc%12ccccc%12)C[C@@H]9OC(=O)C=Cc%13ccccc%13",
        "gemcitabine": "NC1=NC(=NC(=N1)[C@H]2C[C@H](O)[C@@H](CO)O2)N",
        "cisplatin": "N.N.Cl[Pt]Cl",
    }


# ============================================================================
# Epitope Module API
# ============================================================================

def epitope_predict(sequence: str, allele: str = "HLA-A*02:01") -> Dict[str, Any]:
    """Run MHC binding prediction via skill API.

    Args:
        sequence: Peptide sequence (8-11 aa typically)
        allele: HLA allele

    Returns:
        Binding prediction dict with binding_score, binding_affinity
    """
    api = get_skill_api()
    return api.epitope_predict(sequence, allele)


# ============================================================================
# circRNA Module API
# ============================================================================

def circrna_full_analysis(sequence: str) -> Dict[str, Any]:
    """Run complete circRNA analysis via skill API.

    Args:
        sequence: circRNA sequence (AUGC)

    Returns:
        Full analysis with immune, torusfold, pk_params
    """
    api = get_skill_api()
    return api.circrna_full_analysis(sequence)


def circrna_pk(sequence: str, dose: float = 1.0) -> Dict[str, Any]:
    """Run PK simulation for circRNA via skill API.

    Args:
        sequence: circRNA sequence
        dose: Dose in mg

    Returns:
        PK curve data with time, concentration, metrics
    """
    api = get_skill_api()
    return api.circrna_pk(sequence, dose)


# ============================================================================
# Simulacrum Module API
# ============================================================================

def simulacrum_init(subtype: str = "BLIS", brca_mutation: bool = False) -> Dict[str, Any]:
    """Initialize TNBC Simulacrum digital twin via skill API.

    Args:
        subtype: Molecular subtype (BLIS, BLIA, IM, LAR)
        brca_mutation: BRCA1/2 mutation status

    Returns:
        Initial agent state
    """
    api = get_skill_api()
    return api.simulacrum_init(subtype, brca_mutation)


def simulacrum_step(n_days: int = 1) -> Dict[str, Any]:
    """Advance simulation by n days via skill API.

    Args:
        n_days: Number of days to simulate

    Returns:
        Updated state snapshot
    """
    api = get_skill_api()
    return api.simulacrum_step(n_days)


def simulacrum_administer_drug(drug_name: str, dose: float) -> Dict[str, Any]:
    """Administer drug in simulation via skill API.

    Args:
        drug_name: Drug name
        dose: Dose in mg/m2

    Returns:
        Treatment log entry
    """
    api = get_skill_api()
    return api.simulacrum_administer_drug(drug_name, dose)


def simulacrum_report() -> Dict[str, Any]:
    """Generate full Simulacrum report via skill API.

    Returns:
        Comprehensive report with tumor, TME, clinical state
    """
    api = get_skill_api()
    return api.simulacrum_report()


# ============================================================================
# Joint Module API
# ============================================================================

def joint_evaluate(circrna_sequence: str, drug_smiles: str) -> Dict[str, Any]:
    """Run joint circRNA + drug evaluation via skill API.

    Args:
        circrna_sequence: circRNA sequence
        drug_smiles: Drug SMILES

    Returns:
        Joint evaluation with combined score and synergy
    """
    api = get_skill_api()
    return api.joint_evaluate(circrna_sequence, drug_smiles)


# ============================================================================
# Visualization
# ============================================================================

def generate_html_report(data: Dict[str, Any], title: str = "Confluencia Report") -> str:
    """Generate Nature journal-style HTML report via skill API.

    Args:
        data: Analysis result dict
        title: Report title

    Returns:
        HTML string
    """
    api = get_skill_api()
    return api.generate_html_report(data, title)


def save_html_report(html: str, directory: Optional[str] = None) -> str:
    """Save HTML report to file.

    Args:
        html: HTML content
        directory: Output directory (default: temp dir)

    Returns:
        Absolute path to saved file
    """
    if directory:
        output_dir = Path(directory)
    else:
        output_dir = Path(tempfile.gettempdir()) / "confluencia_reports"

    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / f"confluencia_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    filename.write_text(html, encoding="utf-8")
    return str(filename.absolute())


# ============================================================================
# Backend Settings
# ============================================================================

def set_backend(module: str, backend: str) -> str:
    """Set backend for a module.

    Args:
        module: Module name (drug, epitope, circrna)
        backend: Backend name

    Returns:
        Status message
    """
    global _backend_settings
    valid = {
        "drug": ["local", "chembl_api"],
        "epitope": ["local", "netmhcpan"],
        "circrna": ["heuristic", "vienna", "esm2"],
    }
    if module in valid and backend in valid[module]:
        _backend_settings[module] = backend
        # Also update skill API backend
        api = get_skill_api()
        api.set_backend(module, backend)
        return f"Backend set: {module} -> {backend}"
    return f"Invalid. Options for {module}: {valid.get(module, [])}"


def get_backend(module: str) -> str:
    """Get current backend for a module."""
    return _backend_settings.get(module, "local")


def get_all_backends() -> Dict[str, str]:
    """Get all backend settings."""
    return _backend_settings.copy()


# ============================================================================
# Legacy Compatibility (for gradual migration)
# ============================================================================

# These functions exist for backward compatibility with existing pages
# They internally call the skill API now

def get_admet():
    """Legacy: Get ADMET module. Now uses skill API."""
    # Just return the skill API for direct attribute access
    return get_skill_api()

def get_innate_immune():
    """Legacy: Get innate immune module. Now uses skill API."""
    return get_skill_api()

def get_ctm():
    """Legacy: Get CTM module. Now uses skill API."""
    return get_skill_api()

def get_predictor():
    """Legacy: Get epitope predictor. Now uses skill API."""
    return get_skill_api()

def get_circrna():
    """Legacy: Get circRNA module. Now uses skill API."""
    return get_skill_api()

def get_visualization():
    """Legacy: Get visualization module. Now uses skill API."""
    return get_skill_api()


# ============================================================================
# Utility Functions
# ============================================================================

def format_sequence(sequence: str, max_display: int = 50) -> str:
    """Format sequence for display (truncate if too long)."""
    if len(sequence) <= max_display:
        return sequence
    return sequence[:max_display] + "..."


def get_gc_content(sequence: str) -> float:
    """Calculate GC content of a sequence."""
    if not sequence:
        return 0.0
    return sum(1 for b in sequence.upper() if b in 'GC') / len(sequence)


def check_project_paths() -> Dict[str, bool]:
    """Check if all project module directories exist."""
    results = {}
    for name, path in MODULE_NAMES.items():
        full_path = PROJECT_ROOT / path
        results[name] = full_path.exists()
    return results


def get_project_info() -> Dict[str, Any]:
    """Get project configuration info."""
    return {
        "project_root": str(PROJECT_ROOT),
        "skill_path": str(SKILL_PATH),
        "modules": check_project_paths(),
        "backends": get_all_backends(),
    }