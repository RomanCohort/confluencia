"""Confluencia Integration Tab

Displays: integration status, drug prediction, PK simulation,
epitope prediction, joint evaluation.
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


def render_confluencia(agent: Any, history_df: pd.DataFrame):
    """Render the Confluencia tab"""
    s = agent.state
    from frontend.app_core import metric_card, section_header, progress_bar

    # ── Row 1: Integration Status ──
    section_header("Confluencia Integration Status")

    col_status, col_config = st.columns(2)

    with col_status:
        enabled = s.get("cfl_enabled", False)
        status_color = TN["green"] if enabled else TN["red"]
        status_text = "Connected" if enabled else "Disabled"
        metric_card("Integration Status", status_text, color=status_color)

        # Check bridge availability
        bridges = {}
        try:
            from core.confluencia.drug_bridge import DrugPredictionBridge
            bridges["Drug Prediction"] = True
        except ImportError:
            bridges["Drug Prediction"] = False
        try:
            from core.confluencia.pk_bridge import PKModelBridge
            bridges["PK Simulation"] = True
        except ImportError:
            bridges["PK Simulation"] = False
        try:
            from core.confluencia.epitope_bridge import EpitopePredictionBridge
            bridges["Epitope Prediction"] = True
        except ImportError:
            bridges["Epitope Prediction"] = False
        try:
            from core.confluencia.joint_bridge import JointEvaluationBridge
            bridges["Joint Evaluation"] = True
        except ImportError:
            bridges["Joint Evaluation"] = False

        for name, available in bridges.items():
            icon = "+" if available else "-"
            color = TN["green"] if available else TN["red"]
            st.markdown(f'<span style="color:{color};">{icon} {name}</span>', unsafe_allow_html=True)

    with col_config:
        st.markdown("**Configuration**")
        try:
            from core.config import ConfluenciaConfig
            cfg = agent.config.confluencia
            st.caption(f"Model: {cfg.drug_prediction_model}")
            st.caption(f"PK Model: {cfg.pk_model_type}")
            st.caption(f"Path: {cfg.confluencia_path or 'Not set'}")
            if cfg.joint_eval_weights:
                st.markdown("**Joint Eval Weights:**")
                for key, weight in cfg.joint_eval_weights.items():
                    st.caption(f"  {key}: {weight:.2f}")
        except Exception:
            st.info("Config not available")

    st.markdown("---")

    # ── Row 2: Drug Prediction + PK Simulation ──
    col_drug, col_pk = st.columns(2)

    with col_drug:
        section_header("Drug Prediction (MOE Ensemble)")
        smiles = st.text_input("SMILES", value="CC(=O)Oc1ccccc1C(=O)O", key="cfl_smiles")

        if st.button("Predict Drug Efficacy", key="cfl_predict"):
            try:
                from core.confluencia.drug_bridge import DrugPredictionBridge
                bridge = DrugPredictionBridge(agent.config.confluencia)
                result = bridge.predict(smiles)
                if result:
                    score = result.get("prediction_score", 0)
                    metric_card("Prediction Score", f"{score:.3f}", color=TN["blue"])
                    if "confidence" in result:
                        st.caption(f"Confidence: {result['confidence']:.3f}")
                else:
                    st.warning("Prediction returned no result")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

        # Show current score from state
        current_score = s.get("cfl_drug_prediction_score", 0)
        if current_score > 0:
            st.caption(f"Current prediction score: {current_score:.3f}")

    with col_pk:
        section_header("PK Simulation (RNACTM)")
        if st.button("Simulate PK", key="cfl_pk"):
            try:
                from core.confluencia.pk_bridge import PKModelBridge
                bridge = PKModelBridge(agent.config.confluencia)
                result = bridge.simulate()
                if result:
                    st.success("PK simulation completed")
                    if "concentration_profile" in result:
                        profile = result["concentration_profile"]
                        if isinstance(profile, dict) and "time" in profile and "conc" in profile:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=profile["time"], y=profile["conc"],
                                mode="lines", name="Concentration",
                                line=dict(color=TN["blue"], width=2),
                            ))
                            fig.update_layout(**PLOTLY_LAYOUT, height=250)
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("PK simulation returned no result")
            except Exception as e:
                st.error(f"PK simulation failed: {e}")

        pk_available = s.get("cfl_pk_simulation_available", False)
        st.caption(f"PK Simulation Available: {'Yes' if pk_available else 'No'}")

    st.markdown("---")

    # ── Row 3: Epitope Prediction + Joint Evaluation ──
    col_epi, col_joint = st.columns(2)

    with col_epi:
        section_header("Epitope Prediction (ESM2+Mamba)")
        peptide = st.text_input("Peptide Sequence", value="LLFGYPVYV", key="cfl_peptide")
        mhc_allele = st.selectbox("MHC Allele", ["HLA-A*02:01", "HLA-A*24:02", "HLA-B*07:02"],
                                   key="cfl_mhc")

        if st.button("Predict Epitope", key="cfl_epitope"):
            try:
                from core.confluencia.epitope_bridge import EpitopePredictionBridge
                bridge = EpitopePredictionBridge(agent.config.confluencia)
                result = bridge.predict(peptide, mhc_allele)
                if result:
                    binding = result.get("binding_score", 0)
                    metric_card("Binding Score", f"{binding:.3f}", color=TN["purple"])
                else:
                    st.warning("Epitope prediction returned no result")
            except Exception as e:
                st.error(f"Epitope prediction failed: {e}")

    with col_joint:
        section_header("Joint Evaluation")
        st.markdown("**Score Inputs**")
        clinical_score = st.slider("Clinical Score", 0.0, 1.0, 0.5, key="cfl_clinical")
        binding_score = st.slider("Binding Score", 0.0, 1.0, 0.5, key="cfl_binding")
        kinetics_score = st.slider("Kinetics Score", 0.0, 1.0, 0.5, key="cfl_kinetics")

        st.markdown("**Weights**")
        w_clinical = st.slider("Clinical Weight", 0.0, 1.0, 0.4, key="cfl_w_clinical")
        w_binding = st.slider("Binding Weight", 0.0, 1.0, 0.35, key="cfl_w_binding")
        w_kinetics = st.slider("Kinetics Weight", 0.0, 1.0, 0.25, key="cfl_w_kinetics")

        if st.button("Evaluate", key="cfl_evaluate"):
            total_w = w_clinical + w_binding + w_kinetics
            if total_w > 0:
                joint = (clinical_score * w_clinical + binding_score * w_binding +
                         kinetics_score * w_kinetics) / total_w
                metric_card("Joint Score", f"{joint:.3f}", color=TN["teal"])

                # Score breakdown bar
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=["Clinical", "Binding", "Kinetics", "Joint"],
                    y=[clinical_score, binding_score, kinetics_score, joint],
                    marker_color=[TN["blue"], TN["purple"], TN["cyan"], TN["teal"]],
                ))
                fig.update_layout(
                    paper_bgcolor=TN["surface"], plot_bgcolor=TN["bg"],
                    font={"color": TN["text"], "size": 10},
                    margin={"l": 20, "r": 20, "t": 10, "b": 30},
                    height=200, showlegend=False,
                    yaxis={"gridcolor": TN["border"], "range": [0, 1]},
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Weights must sum to > 0")

        # Current joint score from state
        current_joint = s.get("cfl_joint_score", 0)
        if current_joint > 0:
            st.caption(f"Current joint score: {current_joint:.3f}")
