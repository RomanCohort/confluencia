"""TME/Immune Tab

Displays: immune cell dynamics, immunoediting phase, evasion markers, CAF/ECM.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Any

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


def render_tme_immune(agent: Any, history_df: pd.DataFrame):
    """Render the TME/Immune tab"""
    s = agent.state
    from frontend.app_core import metric_card, ied_phase_badge, section_header, progress_bar

    # ── Row 1: Key Immune Metrics ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("CD8+ T Cells", f"{s.get('imm_cd8_count', 100):.0f} /mm3",
                     color=TN["green"])
    with col2:
        metric_card("T Cell Activation", f"{s.get('imm_t_cell_activation', 0.3):.3f}",
                     color=TN["blue"])
    with col3:
        metric_card("NK Cytotoxicity", f"{s.get('imm_nk_cytotoxicity', 0.3):.3f}",
                     color=TN["cyan"])
    with col4:
        metric_card("TIL Density", f"{s.get('imm_til_density', 0.2):.3f}",
                     color=TN["teal"])

    st.markdown("---")

    # ── Row 2: Immune Dynamics + Immunoediting ──
    col_left, col_right = st.columns([3, 2])

    with col_left:
        section_header("Immune Cell Dynamics Over Time")
        fig = go.Figure()
        if not history_df.empty:
            traces = [
                ("imm_cd8_count", "CD8+ Count", TN["green"]),
                ("imm_t_cell_activation", "T Cell Activation", TN["blue"]),
                ("imm_nk_cytotoxicity", "NK Cytotoxicity", TN["cyan"]),
            ]
            for key, name, color in traces:
                if key in history_df.columns:
                    fig.add_trace(go.Scatter(
                        x=history_df["day"], y=history_df[key],
                        mode="lines", name=name, line=dict(color=color, width=1.5),
                    ))
        fig.update_layout(**PLOTLY_LAYOUT, yaxis_title="Value", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Immunoediting Phase
        section_header("Immunoediting Phase")
        phase = s.get("ied_phase", "elimination")
        progress = s.get("ied_phase_progress", 0.0)
        st.markdown(f"<div style='text-align:center;font-size:1.5em;'>{ied_phase_badge(phase)}</div>",
                    unsafe_allow_html=True)
        progress_bar("Phase Progress", progress, max_val=1.0, color=TN["yellow"])

        # Pressure bars
        immune_p = s.get("ied_immune_pressure", 0.5)
        evasion_p = s.get("ied_evasion_pressure", 0.3)
        st.markdown("")
        st.markdown(f"**Immune Pressure**: {immune_p:.3f}")
        st.markdown(f"**Evasion Pressure**: {evasion_p:.3f}")

        # Pressure comparison bar
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=["Immune", "Evasion"], y=[immune_p, evasion_p],
            marker_color=[TN["blue"], TN["red"]],
            text=[f"{immune_p:.3f}", f"{evasion_p:.3f}"],
            textposition="outside",
        ))
        fig_bar.update_layout(
            paper_bgcolor=TN["surface"], plot_bgcolor=TN["bg"],
            font={"color": TN["text"], "size": 10},
            margin={"l": 20, "r": 20, "t": 10, "b": 30},
            height=150, showlegend=False,
            yaxis={"gridcolor": TN["border"]},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ── Row 3: Exhaustion + Suppression Time-Series ──
    col_exh, col_supp = st.columns(2)

    with col_exh:
        section_header("T Cell Exhaustion & Treg Suppression")
        fig2 = go.Figure()
        if not history_df.empty:
            if "imm_t_cell_exhaustion" in history_df.columns:
                fig2.add_trace(go.Scatter(
                    x=history_df["day"], y=history_df["imm_t_cell_exhaustion"],
                    mode="lines", name="Exhaustion", line=dict(color=TN["red"], width=1.5),
                ))
            if "imm_treg_fraction" in history_df.columns:
                fig2.add_trace(go.Scatter(
                    x=history_df["day"], y=history_df["imm_treg_fraction"],
                    mode="lines", name="Treg Fraction", line=dict(color=TN["orange"], width=1.5),
                ))
            if "imm_mdsc_suppression" in history_df.columns:
                fig2.add_trace(go.Scatter(
                    x=history_df["day"], y=history_df["imm_mdsc_suppression"],
                    mode="lines", name="MDSC Suppression", line=dict(color=TN["yellow"], width=1.5),
                ))
        fig2.update_layout(**PLOTLY_LAYOUT, yaxis_title="Level", height=250)
        st.plotly_chart(fig2, use_container_width=True)

    with col_supp:
        section_header("IFN-gamma & M1/M2 Polarization")
        fig3 = go.Figure()
        if not history_df.empty:
            if "imm_ifn_gamma" in history_df.columns:
                fig3.add_trace(go.Scatter(
                    x=history_df["day"], y=history_df["imm_ifn_gamma"],
                    mode="lines", name="IFN-gamma", line=dict(color=TN["blue"], width=1.5),
                ))
            if "imm_m1_fraction" in history_df.columns:
                fig3.add_trace(go.Scatter(
                    x=history_df["day"], y=history_df["imm_m1_fraction"],
                    mode="lines", name="M1 Fraction", line=dict(color=TN["green"], width=1.5),
                ))
        fig3.update_layout(**PLOTLY_LAYOUT, yaxis_title="Level", height=250)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # ── Row 4: Evasion Markers + CAF/ECM ──
    col_ev, col_caf = st.columns(2)

    with col_ev:
        section_header("Immune Evasion Markers")
        progress_bar("PD-L1 Expression", s.get("evs_pd_l1_expression", 0.15), max_val=1.0, color=TN["red"])
        progress_bar("MHC-I Downreg", s.get("evs_mhc_i_downreg", 0.1), max_val=1.0, color=TN["orange"])
        progress_bar("TGF-beta", s.get("evs_tgf_beta", 0.2), max_val=1.0, color=TN["yellow"])
        progress_bar("IDO Activity", s.get("evs_ido_activity", 0.1), max_val=1.0, color=TN["purple"])

        # Evasion time-series
        fig4 = go.Figure()
        if not history_df.empty:
            if "evs_pd_l1_expression" in history_df.columns:
                fig4.add_trace(go.Scatter(
                    x=history_df["day"], y=history_df["evs_pd_l1_expression"],
                    mode="lines", name="PD-L1", line=dict(color=TN["red"], width=1.5),
                ))
            if "evs_tgf_beta" in history_df.columns:
                fig4.add_trace(go.Scatter(
                    x=history_df["day"], y=history_df["evs_tgf_beta"],
                    mode="lines", name="TGF-beta", line=dict(color=TN["yellow"], width=1.5),
                ))
        fig4.update_layout(**PLOTLY_LAYOUT, yaxis_title="Level", height=200)
        st.plotly_chart(fig4, use_container_width=True)

    with col_caf:
        section_header("CAF / ECM")
        progress_bar("CAF Activation", s.get("caf_activation", 0.1), max_val=1.0, color=TN["orange"])
        progress_bar("ECM Density", s.get("caf_ecm_density", 0.3), max_val=1.0, color=TN["yellow"])
        progress_bar("ECM Stiffness", s.get("caf_ecm_stiffness", 0.2), max_val=1.0, color=TN["muted"])

        # CAF time-series
        fig5 = go.Figure()
        if not history_df.empty:
            if "caf_activation" in history_df.columns:
                fig5.add_trace(go.Scatter(
                    x=history_df["day"], y=history_df["caf_activation"],
                    mode="lines", name="CAF Activation", line=dict(color=TN["orange"], width=1.5),
                ))
            if "caf_ecm_density" in history_df.columns:
                fig5.add_trace(go.Scatter(
                    x=history_df["day"], y=history_df["caf_ecm_density"],
                    mode="lines", name="ECM Density", line=dict(color=TN["yellow"], width=1.5),
                ))
        fig5.update_layout(**PLOTLY_LAYOUT, yaxis_title="Level", height=200)
        st.plotly_chart(fig5, use_container_width=True)
