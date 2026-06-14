"""Treatment Tab

Displays: drug administration panel, circRNA therapy, PK/PD curves, treatment log.
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


def render_treatment(agent: Any, history_df: pd.DataFrame):
    """Render the Treatment tab"""
    s = agent.state
    from frontend.app_core import metric_card, section_header, progress_bar

    # Load drug registry
    try:
        from core.treatment.drug_pipeline.drug_registry import (
            DRUG_REGISTRY, get_drug, list_drugs_by_class, list_all_drugs,
        )
        registry_available = True
    except ImportError:
        registry_available = False

    # ── Row 1: Drug Administration ──
    section_header("Drug Administration")

    col_admin, col_info = st.columns([2, 3])

    with col_admin:
        # Category selection
        category = st.radio("Drug Category", ["Chemo", "Immunotherapy", "Targeted", "Anti-angiogenic"],
                            horizontal=True)

        # Drug dropdown
        category_map = {
            "Chemo": "chemo", "Immunotherapy": "immunotherapy",
            "Targeted": "targeted", "Anti-angiogenic": "anti_angiogenic",
        }
        drug_class = category_map[category]

        if registry_available:
            drugs_in_class = list_drugs_by_class(drug_class)
            drug_names = [d.name for d in drugs_in_class]
            if not drug_names:
                drug_names = list_all_drugs()
        else:
            drug_names = {
                "chemo": ["doxorubicin", "paclitaxel", "carboplatin", "cisplatin"],
                "immunotherapy": ["atezolizumab", "pembrolizumab"],
                "targeted": ["olaparib", "ipatasertib", "enzalutamide"],
                "anti_angiogenic": ["bevacizumab"],
            }.get(drug_class, [])

        selected_drug = st.selectbox("Drug", drug_names)
        default_dose = 60.0
        if registry_available:
            drug_def = get_drug(selected_drug)
            if drug_def:
                default_dose = drug_def.dose_mg_m2

        dose = st.number_input("Dose (mg/m2)", value=default_dose, min_value=0.0, step=10.0)

        if st.button("Administer Drug", type="primary", use_container_width=True):
            agent.administer_drug(selected_drug, dose)
            st.session_state.treatment_log.append({
                "day": agent.day, "drug": selected_drug, "dose": dose, "route": "IV",
            })
            st.success(f"Administered {selected_drug} {dose:.0f} mg/m2 on Day {agent.day}")

    with col_info:
        # Drug info card
        if registry_available:
            drug_def = get_drug(selected_drug)
            if drug_def:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**{drug_def.name}** ({drug_def.drug_class})")
                    st.caption(f"Standard Dose: {drug_def.dose_mg_m2} mg/m2")
                    st.caption(f"Frequency: q{drug_def.frequency_days}d")
                    st.caption(f"Half-life: {drug_def.half_life_h:.1f} h")
                    st.caption(f"EC50: {drug_def.ec50} ng/mL")
                    st.caption(f"Emax: {drug_def.emax:.2f}")
                    st.caption(f"Hill coeff: {drug_def.hill_coeff:.1f}")
                with col_b:
                    if drug_def.receptor_targets:
                        st.markdown("**Receptor Targets:**")
                        for target, affinity in drug_def.receptor_targets.items():
                            st.caption(f"  {target}: {affinity:.2f}")
                    if drug_def.resistance_mechanisms:
                        st.markdown("**Resistance:**")
                        for mech in drug_def.resistance_mechanisms:
                            st.caption(f"  {mech}")
                    if drug_def.toxicity_profile:
                        st.markdown("**Toxicity:**")
                        for tox, grade in drug_def.toxicity_profile.items():
                            st.caption(f"  {tox}: Grade {grade}")
            else:
                st.info("Select a drug to see details")
        else:
            st.warning("Drug registry not available")

    st.markdown("---")

    # ── Row 2: circRNA Therapy ──
    with st.expander("circRNA Therapy", expanded=False):
        col_cfr1, col_cfr2 = st.columns(2)
        with col_cfr1:
            mechanism = st.selectbox("Mechanism", ["miRNA_sponge", "protein_coding", "immune_stimulation"])
            cfr_dose = st.number_input("circRNA Dose", value=1.0, min_value=0.0, step=0.1)
            cfr_target = st.text_input("Target (e.g., miR-21)", value="miR-21")

            if st.button("Add circRNA Therapy"):
                try:
                    agent.circrna_therapy.add_therapy(
                        mechanism=mechanism, dose=cfr_dose, target=cfr_target
                    )
                    st.success(f"Added {mechanism} therapy targeting {cfr_target}")
                except Exception as e:
                    st.error(f"Error: {e}")

        with col_cfr2:
            # Active circRNA therapies
            try:
                therapies = getattr(agent.circrna_therapy, '_therapies', [])
                if therapies:
                    st.markdown("**Active Therapies:**")
                    for t in therapies:
                        st.caption(f"  {t.mechanism} | dose={t.dose:.1f} | target={t.target}")
                else:
                    st.info("No active circRNA therapies")
            except Exception:
                st.info("circRNA therapy engine not available")

    st.markdown("---")

    # ── Row 3: PK/PD Curves ──
    col_pk, col_pd = st.columns(2)

    with col_pk:
        section_header("Drug Concentration Over Time")
        fig = go.Figure()
        if not history_df.empty and "drg_concentration" in history_df.columns:
            fig.add_trace(go.Scatter(
                x=history_df["day"], y=history_df["drg_concentration"],
                mode="lines", name="Concentration",
                line=dict(color=TN["blue"], width=2),
                fill="tozeroy", fillcolor=f"rgba(122,162,247,0.1)",
            ))
        fig.update_layout(**PLOTLY_LAYOUT, yaxis_title="Concentration (ng/mL)", height=280)
        st.plotly_chart(fig, use_container_width=True)

    with col_pd:
        section_header("Drug Effect & Kill Fraction")
        fig2 = go.Figure()
        if not history_df.empty:
            if "drg_effect" in history_df.columns:
                fig2.add_trace(go.Scatter(
                    x=history_df["day"], y=history_df["drg_effect"],
                    mode="lines", name="Effect", line=dict(color=TN["green"], width=1.5),
                ))
            if "drg_kill_fraction" in history_df.columns:
                fig2.add_trace(go.Scatter(
                    x=history_df["day"], y=history_df["drg_kill_fraction"],
                    mode="lines", name="Kill Fraction", line=dict(color=TN["red"], width=1.5),
                ))
        fig2.update_layout(**PLOTLY_LAYOUT, yaxis_title="Fraction", height=280)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ── Row 4: Resistance + Treatment Log ──
    col_res, col_log = st.columns([1, 2])

    with col_res:
        section_header("Resistance")
        resistance = s.get("drg_resistance_level", 0.0)
        resistance_color = TN["red"] if resistance > 0.3 else TN["yellow"] if resistance > 0.1 else TN["green"]
        metric_card("Resistance Level", f"{resistance:.3f}", color=resistance_color)
        if resistance > 0.3:
            st.markdown(f'<span style="color:{TN["red"]};font-weight:bold;">HIGH RESISTANCE</span>',
                       unsafe_allow_html=True)

        # Resistance time-series
        fig3 = go.Figure()
        if not history_df.empty and "drg_resistance_level" in history_df.columns:
            fig3.add_trace(go.Scatter(
                x=history_df["day"], y=history_df["drg_resistance_level"],
                mode="lines", name="Resistance", line=dict(color=TN["red"], width=1.5),
            ))
            fig3.add_hline(y=0.3, line_dash="dash", line_color=TN["yellow"],
                          annotation_text="Threshold")
        fig3.update_layout(**PLOTLY_LAYOUT, yaxis_title="Level", height=200)
        st.plotly_chart(fig3, use_container_width=True)

    with col_log:
        section_header("Treatment Log")
        treatment_log = st.session_state.get("treatment_log", [])
        if treatment_log:
            log_df = pd.DataFrame(treatment_log)
            st.dataframe(log_df, use_container_width=True, hide_index=True)
        else:
            st.info("No treatments administered yet")
