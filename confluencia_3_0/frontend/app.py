"""TNBC Simulacrum - Interactive Frontend

Triple-Negative Breast Cancer simulation environment with
EventBus-first architecture, 4 molecular subtypes, and
Confluencia integration.

Usage:
    streamlit run frontend/app.py
"""
import streamlit as st
from frontend.app_core import (
    page_config, inject_global_styles, get_agent,
    get_history_df, import_tab_modules, render_simulation_controls,
)

# ═══════════════════════════════════════════════════════════
# Page Setup
# ═══════════════════════════════════════════════════════════

page_config("TNBC Simulacrum", "🔬")
inject_global_styles()

# ═══════════════════════════════════════════════════════════
# Agent & History
# ═══════════════════════════════════════════════════════════

agent = get_agent()
history_df = get_history_df()

# ═══════════════════════════════════════════════════════════
# Sidebar Controls
# ═══════════════════════════════════════════════════════════

render_simulation_controls(agent)

# ═══════════════════════════════════════════════════════════
# Title
# ═══════════════════════════════════════════════════════════

st.title("TNBC Simulacrum")
st.caption("Bio-inspired Triple-Negative Breast Cancer Simulation Environment")

# ═══════════════════════════════════════════════════════════
# Lazy Tab Imports
# ═══════════════════════════════════════════════════════════

TAB_MAP = {
    "Tumor Dashboard": ("tumor_dashboard", "render_tumor_dashboard"),
    "TME/Immune": ("tme_immune", "render_tme_immune"),
    "Treatment": ("treatment", "render_treatment"),
    "Biomarker": ("biomarker", "render_biomarker"),
    "Clinical": ("clinical", "render_clinical"),
    "Experiments": ("experiments", "render_experiments"),
    "Confluencia": ("confluencia", "render_confluencia"),
}

tab_modules = import_tab_modules(TAB_MAP)

# ═══════════════════════════════════════════════════════════
# Tab Layout
# ═══════════════════════════════════════════════════════════

tab_names = list(TAB_MAP.keys())
tabs = st.tabs(tab_names)

for tab, name in zip(tabs, tab_names):
    with tab:
        render_fn = tab_modules.get(name)
        if render_fn is not None:
            try:
                render_fn(agent, history_df)
            except Exception as e:
                st.error(f"Error rendering {name}: {e}")
                with st.expander("Traceback"):
                    import traceback
                    traceback.print_exc()
        else:
            st.warning(f"Tab module '{name}' not available")
