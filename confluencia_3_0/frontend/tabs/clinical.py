"""Clinical Tab

Displays: RECIST response, survival estimates, toxicity profile.
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


def render_clinical(agent: Any, history_df: pd.DataFrame):
    """Render the Clinical tab"""
    s = agent.state
    from frontend.app_core import metric_card, recist_badge, section_header, progress_bar

    # ── Row 1: Key Clinical Metrics ──
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        recist = s.get("cli_recist_response", "SD")
        st.markdown(f"""
        <div class="metric-card" style="border-left:3px solid {TN["green"]};">
            <div style="color:{TN["muted"]};font-size:0.85em;">RECIST Response</div>
            <div style="font-size:1.4em;font-weight:bold;">{recist_badge(recist)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        change_pct = s.get("cli_tumor_change_pct", 0)
        change_color = TN["green"] if change_pct < -30 else TN["yellow"] if change_pct < 20 else TN["red"]
        metric_card("Tumor Change", f"{change_pct:.1f}%", color=change_color)

    with col3:
        pfs = s.get("cli_pfs_months", 0)
        metric_card("PFS", f"{pfs:.1f} months", color=TN["blue"])

    with col4:
        tox = s.get("cli_toxicity_grade", 0)
        tox_color = TN["green"] if tox <= 1 else TN["yellow"] if tox <= 2 else TN["red"]
        metric_card("Max Toxicity", f"Grade {tox}", color=tox_color)

    st.markdown("---")

    # ── Row 2: RECIST Tracking + Survival ──
    col_recist, col_surv = st.columns(2)

    with col_recist:
        section_header("Tumor Change % Over Time (RECIST 1.1)")
        fig = go.Figure()
        if not history_df.empty and "cli_tumor_change_pct" in history_df.columns:
            fig.add_trace(go.Scatter(
                x=history_df["day"], y=history_df["cli_tumor_change_pct"],
                mode="lines", name="Tumor Change %",
                line=dict(color=TN["blue"], width=2),
                fill="tozeroy", fillcolor=f"rgba(122,162,247,0.1)",
            ))
        # RECIST threshold lines
        fig.add_hline(y=-30, line_dash="dash", line_color=TN["green"],
                      annotation_text="PR (-30%)", annotation_font_color=TN["green"])
        fig.add_hline(y=20, line_dash="dash", line_color=TN["red"],
                      annotation_text="PD (+20%)", annotation_font_color=TN["red"])
        fig.add_hrect(y0=-100, y1=-30, fillcolor=f"rgba(158,206,106,0.05)", line_width=0)
        fig.add_hrect(y0=20, y1=100, fillcolor=f"rgba(247,118,142,0.05)", line_width=0)
        fig.update_layout(**PLOTLY_LAYOUT, yaxis_title="Change (%)", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col_surv:
        section_header("Survival Estimates")
        # PFS over time
        fig2 = go.Figure()
        if not history_df.empty and "cli_pfs_months" in history_df.columns:
            fig2.add_trace(go.Scatter(
                x=history_df["day"], y=history_df["cli_pfs_months"],
                mode="lines", name="PFS (months)",
                line=dict(color=TN["blue"], width=2),
            ))
        fig2.update_layout(**PLOTLY_LAYOUT, yaxis_title="Months", height=300)
        st.plotly_chart(fig2, use_container_width=True)

        # Additional info
        baseline = s.get("cli_baseline_volume", 50)
        nadir = s.get("cli_nadir_volume", 50)
        current = s.get("tum_volume", 50)
        st.caption(f"Baseline: {baseline:.1f} mm3 | Nadir: {nadir:.1f} mm3 | Current: {current:.1f} mm3")

    st.markdown("---")

    # ── Row 3: Toxicity Profile ──
    section_header("Toxicity Profile (CTCAE v5.0)")

    # Try to get detailed toxicity from drug registry
    try:
        from core.treatment.drug_pipeline.drug_registry import get_drug
        active_drug = s.get("drg_active_drug", "")
        drug_def = get_drug(active_drug) if active_drug else None
        tox_profile = drug_def.toxicity_profile if drug_def else {}
    except ImportError:
        tox_profile = {}

    # Default toxicity categories
    tox_categories = ["Neutropenia", "Cardiotoxicity", "Neuropathy", "Fatigue", "Nausea",
                      "Diarrhea", "Rash", "Anemia", "Immune Colitis", "Hypertension"]
    tox_map = {
        "neutropenia": "Neutropenia", "cardiotoxicity": "Cardiotoxicity",
        "neuropathy": "Neuropathy", "fatigue": "Fatigue", "nausea": "Nausea",
        "diarrhea": "Diarrhea", "rash": "Rash", "anemia": "Anemia",
        "immune_colitis": "Immune Colitis", "hypertension": "Hypertension",
    }

    # Build toxicity display
    cols = st.columns(5)
    for i, cat in enumerate(tox_categories):
        with cols[i % 5]:
            # Find grade from profile
            grade = 0
            for key, g in tox_profile.items():
                if tox_map.get(key) == cat:
                    grade = g
                    break

            # Color by grade
            if grade == 0:
                color = TN["green"]
            elif grade <= 2:
                color = TN["yellow"]
            elif grade == 3:
                color = TN["orange"]
            else:
                color = TN["red"]

            st.markdown(f"""
            <div class="metric-card" style="border-left:3px solid {color};padding:8px;">
                <div style="color:{TN["muted"]};font-size:0.75em;">{cat}</div>
                <div style="color:{color};font-size:1.2em;font-weight:bold;">Grade {grade}</div>
            </div>
            """, unsafe_allow_html=True)

    # Overall toxicity
    max_grade = s.get("cli_toxicity_grade", 0)
    st.markdown(f"**Overall Max Toxicity Grade**: {max_grade}")

    # Toxicity time-series
    fig3 = go.Figure()
    if not history_df.empty and "cli_toxicity_grade" in history_df.columns:
        fig3.add_trace(go.Scatter(
            x=history_df["day"], y=history_df["cli_toxicity_grade"],
            mode="lines", name="Max Toxicity Grade",
            line=dict(color=TN["red"], width=1.5),
        ))
        fig3.add_hline(y=3, line_dash="dash", line_color=TN["yellow"],
                      annotation_text="Grade 3 (Severe)")
    fig3.update_layout(**PLOTLY_LAYOUT, yaxis_title="CTCAE Grade", height=200)
    st.plotly_chart(fig3, use_container_width=True)
