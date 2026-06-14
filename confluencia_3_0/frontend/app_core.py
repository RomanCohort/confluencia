"""TNBC Simulacrum Frontend Core - Shared Infrastructure

Provides:
- Page config and Tokyo Night dark theme CSS injection
- Agent initialization and session_state management
- Time-series history tracking (ring buffer)
- Lazy tab module imports
- Styled metric cards, RECIST/immunoediting badges
- Simulation control sidebar
"""
import streamlit as st
import pandas as pd
from typing import Dict, Any, Callable, List, Optional
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Tokyo Night palette
TN = {
    "bg": "#1a1b26", "surface": "#24283b", "border": "#414d68",
    "text": "#c0caf5", "muted": "#a9b1d6",
    "blue": "#7aa2f7", "green": "#9ece6a", "yellow": "#e0af68",
    "red": "#f7768e", "purple": "#bb9af7", "cyan": "#7dcfff",
    "orange": "#ff9e64", "teal": "#73daca",
}

# Default tracked keys for time-series (~40 keys)
DEFAULT_TRACKED_KEYS = [
    "tum_volume", "tum_growth_rate", "tum_apoptosis_rate", "tum_necrosis_fraction",
    "het_n_subclones", "het_diversity_index", "het_dominant_clone_fraction",
    "csc_fraction",
    "vasc_vegf_level", "vasc_microvessel_density", "vasc_oxygenation",
    "met_emt_progress", "met_metastatic_burden",
    "imm_cd8_count", "imm_t_cell_activation", "imm_t_cell_exhaustion",
    "imm_nk_cytotoxicity", "imm_m1_fraction", "imm_treg_fraction",
    "imm_til_density", "imm_ifn_gamma", "imm_mdsc_suppression",
    "evs_pd_l1_expression", "evs_tgf_beta", "evs_ido_activity",
    "ied_immune_pressure", "ied_evasion_pressure",
    "caf_activation", "caf_ecm_density",
    "drg_concentration", "drg_effect", "drg_kill_fraction", "drg_resistance_level",
    "bio_pd_l1_cps", "bio_til_density", "bio_ctdna_level", "bio_tmb",
    "cli_tumor_change_pct", "cli_pfs_months", "cli_toxicity_grade",
    "cfl_drug_prediction_score", "cfl_joint_score",
]

MAX_HISTORY = 1000


# ═══════════════════════════════════════════════════════════
# Page Configuration
# ═══════════════════════════════════════════════════════════

def page_config(title: str = "TNBC Simulacrum", icon: str = "🔬"):
    """Streamlit page configuration"""
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_global_styles():
    """Inject Tokyo Night dark theme CSS"""
    st.markdown(f"""
    <style>
    .stApp {{ background-color: {TN['bg']}; color: {TN['text']}; }}
    .stSidebar {{ background-color: #16161e; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {TN['surface']}; border-radius: 8px 8px 0 0; padding: 10px 20px;
        color: {TN['muted']};
    }}
    .stTabs [aria-selected="true"] {{ background-color: {TN['border']}; color: {TN['text']}; }}
    .metric-card {{
        background: linear-gradient(135deg, {TN['surface']} 0%, {TN['bg']} 100%);
        border: 1px solid {TN['border']}; border-radius: 12px; padding: 16px; margin: 8px 0;
    }}
    .stProgress > div > div > div {{ background-color: {TN['blue']}; }}

    /* RECIST badges */
    .recist-cr {{ background-color: {TN['green']}; color: #1a1b26; padding: 4px 12px; border-radius: 6px; font-weight: bold; }}
    .recist-pr {{ background-color: #73daca; color: #1a1b26; padding: 4px 12px; border-radius: 6px; font-weight: bold; }}
    .recist-sd {{ background-color: {TN['yellow']}; color: #1a1b26; padding: 4px 12px; border-radius: 6px; font-weight: bold; }}
    .recist-pd {{ background-color: {TN['red']}; color: #1a1b26; padding: 4px 12px; border-radius: 6px; font-weight: bold; }}

    /* Immunoediting phase badges */
    .ied-elimination {{ background-color: {TN['blue']}; color: #1a1b26; padding: 4px 12px; border-radius: 6px; font-weight: bold; }}
    .ied-equilibrium {{ background-color: {TN['yellow']}; color: #1a1b26; padding: 4px 12px; border-radius: 6px; font-weight: bold; }}
    .ied-escape {{ background-color: {TN['red']}; color: #1a1b26; padding: 4px 12px; border-radius: 6px; font-weight: bold; }}

    /* Subtype badges */
    .subtype-blis {{ background-color: {TN['red']}; color: #1a1b26; padding: 4px 12px; border-radius: 6px; font-weight: bold; }}
    .subtype-im {{ background-color: {TN['blue']}; color: #1a1b26; padding: 4px 12px; border-radius: 6px; font-weight: bold; }}
    .subtype-m {{ background-color: {TN['purple']}; color: #1a1b26; padding: 4px 12px; border-radius: 6px; font-weight: bold; }}
    .subtype-lar {{ background-color: {TN['teal']}; color: #1a1b26; padding: 4px 12px; border-radius: 6px; font-weight: bold; }}

    /* Section headers */
    .section-header {{ color: {TN['blue']}; border-bottom: 1px solid {TN['border']}; padding-bottom: 8px; margin-bottom: 16px; }}
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# Agent Initialization
# ═══════════════════════════════════════════════════════════

def get_agent():
    """Get or create TNBCSimulacrum from session_state.

    Initializes session_state keys:
      "agent": TNBCSimulacrum instance
      "history": list of per-step state snapshots
      "treatment_log": list of drug administration records
    """
    if "agent" not in st.session_state:
        from core.config import TNBCSimulacrumConfig
        from core.agent import TNBCSimulacrum
        config = TNBCSimulacrumConfig()
        agent = TNBCSimulacrum(config)
        st.session_state.agent = agent
        st.session_state.history = []
        st.session_state.treatment_log = []
        # Record initial snapshot
        _record_snapshot(agent, 0)
    return st.session_state.agent


def _record_snapshot(agent, day: int):
    """Record a single state snapshot into history"""
    snapshot = {"day": day}
    s = agent.state
    for key in DEFAULT_TRACKED_KEYS:
        val = s.get(key)
        if val is not None:
            snapshot[key] = val
    st.session_state.history.append(snapshot)
    # Ring buffer: drop oldest 200 if over limit
    if len(st.session_state.history) > MAX_HISTORY:
        st.session_state.history = st.session_state.history[200:]


def record_step(agent, keys_to_track: Optional[List[str]] = None):
    """Record current state snapshot after agent.step()"""
    _record_snapshot(agent, agent.day)


def get_history_df() -> pd.DataFrame:
    """Convert session_state history to pandas DataFrame"""
    history = st.session_state.get("history", [])
    if not history:
        return pd.DataFrame()
    return pd.DataFrame(history)


# ═══════════════════════════════════════════════════════════
# Lazy Import
# ═══════════════════════════════════════════════════════════

def import_tab_modules(tab_map: Dict[str, tuple]) -> Dict[str, Callable]:
    """Lazy import tab render functions.

    Args:
        tab_map: {key: (module_filename, render_function_name)}

    Returns:
        {key: render_function}
    """
    import importlib.util

    modules = {}
    tabs_dir = Path(__file__).parent / "tabs"

    for key, (module_name, func_name) in tab_map.items():
        module_path = tabs_dir / f"{module_name}.py"
        if module_path.exists():
            try:
                spec = importlib.util.spec_from_file_location(
                    f"frontend.tabs.{module_name}", str(module_path)
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                render_fn = getattr(mod, func_name, None)
                if render_fn is not None:
                    modules[key] = render_fn
            except Exception as e:
                pass

    return modules


# ═══════════════════════════════════════════════════════════
# Styled Components
# ═══════════════════════════════════════════════════════════

def metric_card(label: str, value: str, delta: str = "", color: str = ""):
    """Render a styled metric card"""
    delta_html = f'<span style="color:{TN["green"]};font-size:0.85em;">{delta}</span>' if delta else ""
    border_color = color or TN["blue"]
    st.markdown(f"""
    <div class="metric-card" style="border-left:3px solid {border_color};">
        <div style="color:{TN["muted"]};font-size:0.85em;">{label}</div>
        <div style="color:{TN["text"]};font-size:1.4em;font-weight:bold;">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def recist_badge(recist: str) -> str:
    """Return HTML badge for RECIST response"""
    recist = recist.upper()
    classes = {"CR": "recist-cr", "PR": "recist-pr", "SD": "recist-sd", "PD": "recist-pd"}
    cls = classes.get(recist, "recist-sd")
    return f'<span class="{cls}">{recist}</span>'


def ied_phase_badge(phase: str) -> str:
    """Return HTML badge for immunoediting phase"""
    classes = {
        "elimination": "ied-elimination",
        "equilibrium": "ied-equilibrium",
        "escape": "ied-escape",
    }
    cls = classes.get(phase, "ied-equilibrium")
    return f'<span class="{cls}">{phase.upper()}</span>'


def subtype_badge(subtype: str) -> str:
    """Return HTML badge for molecular subtype"""
    classes = {"BLIS": "subtype-blis", "IM": "subtype-im", "M": "subtype-m", "LAR": "subtype-lar"}
    cls = classes.get(subtype, "subtype-blis")
    return f'<span class="{cls}">{subtype}</span>'


def progress_bar(label: str, value: float, max_val: float = 1.0, color: str = ""):
    """Render a labeled progress bar"""
    frac = min(1.0, max(0.0, value / max_val)) if max_val > 0 else 0
    bar_color = color or TN["blue"]
    st.markdown(f"""
    <div style="margin:4px 0;">
        <div style="display:flex;justify-content:space-between;color:{TN["muted"]};font-size:0.8em;">
            <span>{label}</span><span>{value:.3f}</span>
        </div>
        <div style="background:{TN["border"]};border-radius:4px;height:6px;margin-top:2px;">
            <div style="background:{bar_color};border-radius:4px;height:6px;width:{frac*100:.1f}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def section_header(text: str):
    """Render a styled section header"""
    st.markdown(f'<div class="section-header" style="font-size:1.1em;font-weight:bold;">{text}</div>',
                unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# Simulation Controls Sidebar
# ═══════════════════════════════════════════════════════════

def render_simulation_controls(agent):
    """Render sidebar simulation controls"""
    with st.sidebar:
        st.markdown("### Simulation Controls")

        col1, col2 = st.columns(2)
        col1.metric("Day", agent.day)
        col2.metric("Steps", agent.step_count)

        st.divider()

        # Single step
        if st.button("Step 1 Day", use_container_width=True):
            agent.step()
            record_step(agent)
            st.rerun()

        # Multi-step
        n_days = st.number_input("Days to simulate", min_value=1, max_value=365, value=30)
        if st.button(f"Step {n_days} Days", use_container_width=True):
            with st.spinner(f"Simulating {n_days} days..."):
                for _ in range(n_days):
                    agent.step()
                    record_step(agent)
            st.rerun()

        st.divider()

        # Patient Configuration
        st.markdown("### Patient Config")
        subtype = st.selectbox("Molecular Subtype", ["BLIS", "IM", "M", "LAR"],
                               index=["BLIS", "IM", "M", "LAR"].index(
                                   agent.state.get("sub_molecular_subtype", "BLIS")))
        brca = st.checkbox("BRCA Mutation", value=agent.state.get("bio_brca_status", 0) > 0)

        if st.button("Reset Simulation", type="primary", use_container_width=True):
            from core.config import TNBCSimulacrumConfig
            from core.agent import TNBCSimulacrum
            config = TNBCSimulacrumConfig()
            config.molecular_subtype = subtype
            config.brca_mutation = brca
            st.session_state.agent = TNBCSimulacrum(config)
            st.session_state.history = []
            st.session_state.treatment_log = []
            _record_snapshot(st.session_state.agent, 0)
            st.rerun()

        st.divider()

        # Treatment log
        treatment_log = st.session_state.get("treatment_log", [])
        if treatment_log:
            st.markdown("### Recent Treatments")
            for entry in treatment_log[-5:]:
                st.caption(f"Day {entry['day']}: {entry['drug']} {entry['dose']:.0f} mg/m2")

        st.divider()

        # Quick state summary
        s = agent.state
        st.markdown("### Quick Summary")
        st.markdown(f"**Volume**: {s.get('tum_volume', 0):.2f} mm3")
        st.markdown(f"**RECIST**: {s.get('cli_recist_response', 'SD')}")
        st.markdown(f"**Phase**: {s.get('ied_phase', 'elimination')}")
        st.markdown(f"**Resistance**: {s.get('drg_resistance_level', 0):.3f}")
