"""Biomarker Tab

Displays: biomarker tracking, subtype classification, resistance detection.
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


def render_biomarker(agent: Any, history_df: pd.DataFrame):
    """Render the Biomarker tab"""
    s = agent.state
    from frontend.app_core import metric_card, subtype_badge, section_header, progress_bar

    # ── Row 1: Key Biomarker Metrics ──
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        metric_card("PD-L1 CPS", f"{s.get('bio_pd_l1_cps', 0):.1f}", color=TN["red"])
    with col2:
        metric_card("TIL Density", f"{s.get('bio_til_density', 0.2):.3f}", color=TN["green"])
    with col3:
        metric_card("TMB", f"{s.get('bio_tmb', 5):.1f}", color=TN["purple"])
    with col4:
        brca = s.get("bio_brca_status", 0)
        brca_text = "Mutated" if brca > 0 else "Wild-type"
        brca_color = TN["red"] if brca > 0 else TN["green"]
        metric_card("BRCA Status", brca_text, color=brca_color)
    with col5:
        metric_card("ctDNA", f"{s.get('bio_ctdna_level', 0):.3f}", color=TN["orange"])
    with col6:
        metric_card("HR Status", s.get("bio_hr_status", "Negative"), color=TN["yellow"])

    st.markdown("---")

    # ── Row 2: Subtype Classification + Biomarker Radar ──
    col_sub, col_radar = st.columns(2)

    with col_sub:
        section_header("Molecular Subtype Classification")
        subtype = s.get("sub_molecular_subtype", "BLIS")
        st.markdown(f"<div style='text-align:center;font-size:2em;'>{subtype_badge(subtype)}</div>",
                    unsafe_allow_html=True)

        # Subtype characteristics
        subtype_info = {
            "BLIS": {"freq": "15-20%", "key": "Low immune, High Ki-67", "therapy": "Chemo + PARP (if BRCA+)"},
            "IM": {"freq": "20-25%", "key": "High PD-L1, High TIL", "therapy": "Chemo + Immune Checkpoint"},
            "M": {"freq": "30-35%", "key": "High proliferation", "therapy": "Chemo (taxane-based)"},
            "LAR": {"freq": "10-15%", "key": "AR+, Luminal-like", "therapy": "AR antagonist + Chemo"},
        }
        info = subtype_info.get(subtype, {})
        st.markdown(f"**Frequency**: {info.get('freq', 'N/A')}")
        st.markdown(f"**Key Features**: {info.get('key', 'N/A')}")
        st.markdown(f"**Therapy**: {info.get('therapy', 'N/A')}")

    with col_radar:
        section_header("Biomarker Profile")
        # Radar chart
        categories = ["PD-L1", "TIL", "TMB", "Ki-67", "AR", "BRCA"]
        values = [
            min(1, s.get("bio_pd_l1_cps", 0) / 50),
            min(1, s.get("bio_til_density", 0.2) * 3),
            min(1, s.get("bio_tmb", 5) / 20),
            min(1, s.get("tum_growth_rate", 0.027) * 30),
            min(1, s.get("bio_ar_expression", 0.1) * 3),
            min(1, s.get("bio_brca_status", 0)),
        ]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]], theta=categories + [categories[0]],
            fill="toself", fillcolor=f"rgba(122,162,247,0.2)",
            line=dict(color=TN["blue"], width=2), name="Current",
        ))
        fig.update_layout(
            polar=dict(
                bgcolor=TN["bg"],
                radialaxis=dict(
                    visible=True, range=[0, 1],
                    gridcolor=TN["border"], linecolor=TN["border"],
                    tickfont=dict(color=TN["muted"], size=9),
                ),
                angularaxis=dict(
                    gridcolor=TN["border"], linecolor=TN["border"],
                    tickfont=dict(color=TN["text"], size=10),
                ),
            ),
            paper_bgcolor=TN["surface"],
            font={"color": TN["text"]},
            height=300, showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Row 3: Biomarker Time-Series ──
    col_ts1, col_ts2 = st.columns(2)

    with col_ts1:
        section_header("PD-L1 & TIL Over Time")
        fig2 = go.Figure()
        if not history_df.empty:
            if "bio_pd_l1_cps" in history_df.columns:
                fig2.add_trace(go.Scatter(
                    x=history_df["day"], y=history_df["bio_pd_l1_cps"],
                    mode="lines", name="PD-L1 CPS", line=dict(color=TN["red"], width=1.5),
                ))
            if "bio_til_density" in history_df.columns:
                fig2.add_trace(go.Scatter(
                    x=history_df["day"], y=history_df["bio_til_density"],
                    mode="lines", name="TIL Density", line=dict(color=TN["green"], width=1.5),
                ))
        fig2.update_layout(**PLOTLY_LAYOUT, yaxis_title="Value", height=250)
        st.plotly_chart(fig2, use_container_width=True)

    with col_ts2:
        section_header("ctDNA & TMB Over Time")
        fig3 = go.Figure()
        if not history_df.empty:
            if "bio_ctdna_level" in history_df.columns:
                fig3.add_trace(go.Scatter(
                    x=history_df["day"], y=history_df["bio_ctdna_level"],
                    mode="lines", name="ctDNA", line=dict(color=TN["orange"], width=1.5),
                ))
            if "bio_tmb" in history_df.columns:
                fig3.add_trace(go.Scatter(
                    x=history_df["day"], y=history_df["bio_tmb"],
                    mode="lines", name="TMB", line=dict(color=TN["purple"], width=1.5),
                ))
        fig3.update_layout(**PLOTLY_LAYOUT, yaxis_title="Value", height=250)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # ── Row 4: Resistance Detection ──
    section_header("Resistance Detection")
    col_r1, col_r2, col_r3 = st.columns(3)

    with col_r1:
        resistance = s.get("drg_resistance_level", 0.0)
        res_color = TN["red"] if resistance > 0.3 else TN["yellow"] if resistance > 0.1 else TN["green"]
        metric_card("Resistance Level", f"{resistance:.3f}", color=res_color)
        if resistance > 0.3:
            st.markdown(f'<div style="color:{TN["red"]};font-weight:bold;text-align:center;">RESISTANCE DETECTED</div>',
                       unsafe_allow_html=True)

    with col_r2:
        res_clone_frac = s.get("het_resistance_clone_fraction", 0.0)
        metric_card("Resistant Clone Fraction", f"{res_clone_frac:.3f}", color=TN["orange"])

    with col_r3:
        n_subclones = s.get("het_n_subclones", 4)
        metric_card("Subclone Count", f"{n_subclones}", color=TN["purple"])

    # Resistance evolution chart
    fig4 = go.Figure()
    if not history_df.empty:
        if "drg_resistance_level" in history_df.columns:
            fig4.add_trace(go.Scatter(
                x=history_df["day"], y=history_df["drg_resistance_level"],
                mode="lines", name="Resistance Level", line=dict(color=TN["red"], width=2),
            ))
        if "het_resistance_clone_fraction" in history_df.columns:
            fig4.add_trace(go.Scatter(
                x=history_df["day"], y=history_df["het_resistance_clone_fraction"],
                mode="lines", name="Resistant Clone Fraction", line=dict(color=TN["orange"], width=1.5),
            ))
        fig4.add_hline(y=0.3, line_dash="dash", line_color=TN["yellow"],
                      annotation_text="Resistance Threshold")
    fig4.update_layout(**PLOTLY_LAYOUT, yaxis_title="Level", height=250)
    st.plotly_chart(fig4, use_container_width=True)
