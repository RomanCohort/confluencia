"""Tumor Dashboard Tab

Displays: volume over time, growth rate, subtype, heterogeneity,
CSC, angiogenesis, metastasis, and TME schematic SVG.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Any

# Tokyo Night palette
TN = {
    "bg": "#1a1b26", "surface": "#24283b", "border": "#414d68",
    "text": "#c0caf5", "muted": "#a9b1d6",
    "blue": "#7aa2f7", "green": "#9ece6a", "yellow": "#e0af68",
    "red": "#f7768e", "purple": "#bb9af7", "cyan": "#7dcfff",
    "orange": "#ff9e64", "teal": "#73daca",
}

PLOTLY_LAYOUT = {
    "paper_bgcolor": TN["surface"],
    "plot_bgcolor": TN["bg"],
    "font": {"color": TN["text"], "size": 11},
    "xaxis": {"gridcolor": TN["border"], "title": "Day"},
    "yaxis": {"gridcolor": TN["border"]},
    "margin": {"l": 50, "r": 20, "t": 30, "b": 40},
    "legend": {"bgcolor": TN["surface"], "font": {"color": TN["muted"], "size": 10}},
}


def render_tumor_dashboard(agent: Any, history_df: pd.DataFrame):
    """Render the Tumor Dashboard tab"""
    s = agent.state

    # ── Row 1: Key Metrics ──
    from frontend.app_core import metric_card, subtype_badge, ied_phase_badge, section_header

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Tumor Volume", f"{s.get('tum_volume', 0):.2f} mm3",
                     color=TN["purple"])
    with col2:
        metric_card("Growth Rate", f"{s.get('tum_growth_rate', 0):.4f} /day",
                     color=TN["blue"])
    with col3:
        subtype = s.get("sub_molecular_subtype", "BLIS")
        st.markdown(f"""
        <div class="metric-card" style="border-left:3px solid {TN["teal"]};">
            <div style="color:{TN["muted"]};font-size:0.85em;">Molecular Subtype</div>
            <div style="font-size:1.4em;font-weight:bold;">{subtype_badge(subtype)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        phase = s.get("ied_phase", "elimination")
        st.markdown(f"""
        <div class="metric-card" style="border-left:3px solid {TN["cyan"]};">
            <div style="color:{TN["muted"]};font-size:0.85em;">Immunoediting</div>
            <div style="font-size:1.4em;font-weight:bold;">{ied_phase_badge(phase)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Row 2: Volume Chart + Info Cards ──
    col_left, col_right = st.columns([3, 2])

    with col_left:
        section_header("Tumor Volume Over Time")
        fig = go.Figure()
        if not history_df.empty and "tum_volume" in history_df.columns:
            fig.add_trace(go.Scatter(
                x=history_df["day"], y=history_df["tum_volume"],
                mode="lines", name="Volume",
                line=dict(color=TN["purple"], width=2),
                fill="tozeroy", fillcolor=f"rgba(187,154,247,0.1)",
            ))
        else:
            fig.add_trace(go.Scatter(x=[0], y=[s.get("tum_volume", 50)],
                                     mode="markers", name="Volume"))
        fig.update_layout(**PLOTLY_LAYOUT, yaxis_title="Volume (mm3)", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Heterogeneity card
        section_header("Heterogeneity")
        from frontend.app_core import progress_bar
        progress_bar("Subclones", s.get("het_n_subclones", 4), max_val=50, color=TN["orange"])
        progress_bar("Diversity Index", s.get("het_diversity_index", 0.5), max_val=2.0, color=TN["yellow"])
        progress_bar("Dominant Clone", s.get("het_dominant_clone_fraction", 0.6), max_val=1.0, color=TN["red"])
        progress_bar("Resistance Clones", s.get("het_resistance_clone_fraction", 0.0), max_val=1.0, color=TN["red"])

        st.markdown("")
        # CSC card
        section_header("Cancer Stem Cells")
        progress_bar("CSC Fraction", s.get("csc_fraction", 0.02), max_val=0.2, color=TN["yellow"])
        progress_bar("Self-Renewal", s.get("csc_self_renewal", 0.5), max_val=1.0, color=TN["green"])

    st.markdown("---")

    # ── Row 3: Angiogenesis + Metastasis + TME SVG ──
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        section_header("Angiogenesis")
        progress_bar("VEGF Level", s.get("vasc_vegf_level", 0.1), max_val=1.0, color=TN["red"])
        progress_bar("Microvessel Density", s.get("vasc_microvessel_density", 0.5), max_val=1.0, color=TN["red"])
        progress_bar("Oxygenation", s.get("vasc_oxygenation", 0.7), max_val=1.0, color=TN["cyan"])
        norm_window = s.get("vasc_normalization_window", 0)
        st.caption(f"Normalization Window: {norm_window:.0f} days")

    with col_b:
        section_header("Metastasis")
        progress_bar("EMT Progress", s.get("met_emt_progress", 0.0), max_val=1.0, color=TN["orange"])
        progress_bar("Metastatic Burden", s.get("met_metastatic_burden", 0.0), max_val=0.1, color=TN["red"])
        # Organ sites
        organ_sites = s.get("met_organ_sites", {})
        if organ_sites and isinstance(organ_sites, dict):
            st.caption("Organ Involvement:")
            for organ, val in organ_sites.items():
                if val > 0.001:
                    st.caption(f"  {organ}: {val:.4f}")

    with col_c:
        section_header("TME Schematic")
        from frontend.svgs.tme_schematic import render_tme_svg
        svg = render_tme_svg(s)
        st.markdown(svg, unsafe_allow_html=True)

    # ── Row 4: Growth Rate + Apoptosis Time-Series ──
    st.markdown("---")
    col_gr, col_ap = st.columns(2)

    with col_gr:
        section_header("Growth & Apoptosis Over Time")
        fig2 = go.Figure()
        if not history_df.empty:
            if "tum_growth_rate" in history_df.columns:
                fig2.add_trace(go.Scatter(
                    x=history_df["day"], y=history_df["tum_growth_rate"],
                    mode="lines", name="Growth Rate", line=dict(color=TN["green"], width=1.5),
                ))
            if "tum_apoptosis_rate" in history_df.columns:
                fig2.add_trace(go.Scatter(
                    x=history_df["day"], y=history_df["tum_apoptosis_rate"],
                    mode="lines", name="Apoptosis Rate", line=dict(color=TN["red"], width=1.5),
                ))
        fig2.update_layout(**PLOTLY_LAYOUT, yaxis_title="Rate (/day)", height=250)
        st.plotly_chart(fig2, use_container_width=True)

    with col_ap:
        section_header("Necrosis & Hypoxia Over Time")
        fig3 = go.Figure()
        if not history_df.empty:
            if "tum_necrosis_fraction" in history_df.columns:
                fig3.add_trace(go.Scatter(
                    x=history_df["day"], y=history_df["tum_necrosis_fraction"],
                    mode="lines", name="Necrosis", line=dict(color=TN["orange"], width=1.5),
                ))
            if "vasc_oxygenation" in history_df.columns:
                fig3.add_trace(go.Scatter(
                    x=history_df["day"], y=history_df["vasc_oxygenation"],
                    mode="lines", name="Oxygenation", line=dict(color=TN["cyan"], width=1.5),
                ))
        fig3.update_layout(**PLOTLY_LAYOUT, yaxis_title="Fraction", height=250)
        st.plotly_chart(fig3, use_container_width=True)
