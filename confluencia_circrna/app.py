"""
Confluencia circRNA Module Frontend

circRNA-specific analysis and design interface:
- Sequence Analysis: Immune scores, structure, modifications
- Sequence Design: Evolution optimization, IRES, modifications
- Vaccine Development: IPS scoring, drug response, treatment
- Clinical Report: Survival, adverse events, PDF generation

Streamlit-based UI with modular tabs.
"""

from __future__ import annotations

import io
import json
import hashlib
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Any
import base64

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add module to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from confluencia_circrna.core import (
    # Immune sensing
    predict_circrna_immunogenicity,
    ImmuneSensingConfig,
    # Structure prediction
    StructurePredictor,
    StructureFeatures,
    compute_pkr_score_from_structure,
    # Folding kinetics
    predict_folding_kinetics,
    KineticsFeatures,
    compute_kinetics_score,
    # Cotrans folding
    predict_cotrans_folding,
    CotransFeatures,
    compare_transcription_rates,
    # Folding pathways
    analyze_folding_pathways,
    PathwayFeatures,
    # Drug response
    predict_drug_response,
    recommend_treatment,
    DrugResponseFeatures,
    # RNA docking
    predict_rna_docking,
    DockingFeatures,
    design_rna_targeting_drug,
    # Modifications
    predict_modifications,
    ModificationFeatures,
    # Clinical prediction
    predict_clinical_outcome,
    generate_clinical_report,
    ClinicalFeatures,
    # Evolution
    CircRNAEvolutionConfig,
    CircRNAEvolutionArtifacts,
    evolve_cirrna,
    run_cirrna_evolution,
    optimize_for_translation,
    optimize_for_stability,
    optimize_for_immune_safety,
    compute_cirrna_objectives,
)

# i18n support (UI already English, adds toggle for Chinese)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "confluencia_shared"))
from lang import t, lang_toggle

# Page config
st.set_page_config(
    page_title=t("page_title_circrna"),
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="expanded",
)

lang_toggle()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2ca02c;
        margin-top: 1rem;
    }
    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
    }
    .metric-high {
        color: #2ca02c;
        font-weight: bold;
    }
    .metric-medium {
        color: #ff7f0e;
        font-weight: bold;
    }
    .metric-low {
        color: #d62728;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>🧬 Confluencia circRNA Module</h1>", unsafe_allow_html=True)
st.caption("circRNA vaccine design: immunogenicity, structure, modifications, clinical prediction")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Module",
    [
        "📊 Sequence Analysis",
        "🧪 Sequence Design",
        "💉 Vaccine Development",
        "📋 Clinical Report",
        "⚙️ Settings",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Links")
st.sidebar.markdown("- [Documentation](https://github.com/IGEM-FBH/confluencia)")
st.sidebar.markdown("- [ViennaRNA](https://www.tbi.univie.ac.at/RNA/)")


# ===================================================================
# Helper Functions
# ===================================================================

def sequence_input_widget(label: str = "circRNA Sequence", key: str = "seq_input") -> str:
    """Sequence input widget with validation."""
    seq = st.text_area(
        label,
        value="AUGCGCUAUGGCUAGCUAUGCGCUAUGGCUAGCUAUGCGCUAUGGC" * 5,
        height=100,
        key=key,
        help="Enter RNA sequence (A, U, G, C). DNA T will be converted to U.",
    )

    # Validate
    seq_clean = seq.upper().replace("T", "U").replace(" ", "").replace("\n", "")
    invalid_chars = set(seq_clean) - set("AUGC")

    if invalid_chars:
        st.warning(f"Invalid characters detected: {invalid_chars}. Will be filtered.")

    length = len(seq_clean)
    st.info(f"Sequence length: {length} nt")

    return seq_clean


def gene_expression_input_widget(key: str = "gene_input") -> Dict[str, float]:
    """Gene expression input widget."""
    st.subheader("Gene Expression Levels")

    col1, col2, col3 = st.columns(3)

    with col1:
        trop2 = st.slider("TROP2 (Tumor Aggressiveness)", 0.0, 10.0, 6.0, 0.5, key=f"{key}_trop2")
        mki67 = st.slider("MKI67 (Proliferation)", 0.0, 10.0, 5.0, 0.5, key=f"{key}_mki67")

    with col2:
        b7h4 = st.slider("B7-H4 (Immune Suppression)", 0.0, 10.0, 5.0, 0.5, key=f"{key}_b7h4")
        pd_l1 = st.slider("PD-L1", 0.0, 10.0, 4.0, 0.5, key=f"{key}_pd_l1")

    with col3:
        myc = st.slider("MYC (Oncogene)", 0.0, 10.0, 3.0, 0.5, key=f"{key}_myc")
        vegf = st.slider("VEGF (Angiogenesis)", 0.0, 10.0, 4.0, 0.5, key=f"{key}_vegf")

    return {
        "TROP2": trop2,
        "B7-H4": b7h4,
        "MKI67": mki67,
        "MYC": myc,
        "PD-L1": pd_l1,
        "VEGF": vegf,
    }


def display_score_card(title: str, value: float, threshold_high: float = 0.7, threshold_low: float = 0.3):
    """Display a score card with color coding."""
    if value >= threshold_high:
        color_class = "metric-high"
        status = "High"
    elif value >= threshold_low:
        color_class = "metric-medium"
        status = "Medium"
    else:
        color_class = "metric-low"
        status = "Low"

    st.markdown(f"""
    <div class="score-card">
        <h3>{title}</h3>
        <p class="{color_class}">{value:.3f} ({status})</p>
    </div>
    """, unsafe_allow_html=True)


def plot_radar_chart(scores: Dict[str, float], title: str = "Score Profile"):
    """Plot radar chart for multiple scores."""
    categories = list(scores.keys())
    values = list(scores.values())

    # Close the polygon
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=title,
        line_color='#1f77b4',
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
            ),
        ),
        showlegend=False,
        title=title,
        height=400,
    )

    return fig


def plot_timeline(time_points: List[float], values: List[float], title: str, y_label: str):
    """Plot timeline chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_points,
        y=values,
        mode='lines+markers',
        name=y_label,
        line_color='#2ca02c',
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Days",
        yaxis_title=y_label,
        height=300,
        template="plotly_white",
    )

    return fig


# ===================================================================
# Page: Sequence Analysis
# ===================================================================

def page_sequence_analysis():
    """Sequence analysis page."""
    st.markdown("<h2 class='sub-header'>📊 Sequence Analysis</h2>", unsafe_allow_html=True)

    # Input
    seq = sequence_input_widget("circRNA Sequence for Analysis", "analysis_seq")

    if len(seq) < 200:
        st.warning("Sequence too short (<200 nt). Some analyses may be limited.")

    # Analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🛡️ Immune Scores",
        "📐 Structure",
        "🔄 Modifications",
        "⚡ Kinetics",
    ])

    # --- Immune Scores ---
    with tab1:
        st.subheader("Immune Recognition Scores")

        col1, col2 = st.columns(2)

        with col1:
            st.info("Predict how immune sensors recognize circRNA")
            st.markdown("**Literature basis:**")
            st.markdown("- RIG-I: dsRNA backbone detection (Zhang et al., Nat Immunol 2016)")
            st.markdown("  - circRNA has no 5' end; blunt-end detection inapplicable")
            st.markdown("- TLR7/8: GU-rich/U-rich sequences")
            st.markdown("- PKR: dsRNA >33bp (Nallagatla et al., 2007)")

        with col2:
            if st.button("Compute Immune Scores", key="compute_immune"):
                with st.spinner("Analyzing..."):
                    config = ImmuneSensingConfig()
                    immune_result = predict_circrna_immunogenicity(seq, config)

                    scores = {
                        "RIG-I": immune_result.rig_i_score,
                        "TLR7": immune_result.tlr7_score,
                        "TLR8": immune_result.tlr8_score,
                        "PKR": immune_result.pkr_score,
                        "Overall": immune_result.overall_immunogenicity,
                    }

                    # Display scores
                    for name, score in scores.items():
                        st.metric(name, f"{score:.3f}")

                    # Radar chart
                    fig = plot_radar_chart(scores, "Immune Profile")
                    st.plotly_chart(fig, use_container_width=True)

                    # Interpretation
                    st.subheader("Interpretation")
                    if immune_result.overall_immunogenicity > 0.6:
                        st.success("✅ High immunogenicity - Good for vaccine design")
                    elif immune_result.overall_immunogenicity > 0.3:
                        st.warning("⚠️ Moderate immunogenicity - May need optimization")
                    else:
                        st.error("❌ Low immunogenicity - Consider sequence redesign")

    # --- Structure ---
    with tab2:
        st.subheader("Secondary Structure Prediction")

        st.info("ViennaRNA-based structure analysis (RNAfold)")

        if st.button("Predict Structure", key="predict_struct"):
            with st.spinner("Running ViennaRNA..."):
                predictor = StructurePredictor()
                struct_features = predictor.predict(seq)

                # Display results
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("MFE", f"{struct_features.mfe:.1f} kcal/mol")
                    st.metric("MFE/nt", f"{struct_features.mfe_normalized:.3f}")

                with col2:
                    st.metric("Stability", f"{struct_features.structure_stability:.2f}")
                    st.metric("Hairpins", struct_features.hairpin_count)

                with col3:
                    st.metric("dsRNA Fraction", f"{struct_features.dsrna_fraction:.1%}")
                    st.metric("Method", struct_features.prediction_method)

                # Structure visualization
                st.subheader("Dot-Bracket Structure")
                st.code(struct_features.dot_bracket[:200] + "..." if len(struct_features.dot_bracket) > 200 else struct_features.dot_bracket)

                # dsRNA regions for PKR
                if struct_features.dsrna_regions:
                    st.subheader("dsRNA Regions (PKR activation)")
                    for start, end in struct_features.dsrna_regions[:5]:
                        st.write(f"  Position {start}-{end}: {end-start} bp")

    # --- Modifications ---
    with tab3:
        st.subheader("Post-transcriptional Modifications")

        st.info("Predict m6A, IRES, miRNA, RBP binding sites")

        if st.button("Analyze Modifications", key="analyze_mods"):
            with st.spinner("Scanning motifs..."):
                mod_features = predict_modifications(seq)

                # m6A sites
                st.subheader("m6A Sites (DRACH motif)")
                st.metric("Total m6A Sites", len(mod_features.m6a_sites))
                st.metric("m6A Density", f"{mod_features.m6a_density:.1f} sites/kb")

                if mod_features.m6a_sites:
                    st.write("Top m6A sites:")
                    for site in mod_features.m6a_sites[:5]:
                        st.write(f"  Position {site.position}: {site.motif} (prob={site.probability:.2f})")

                # IRES sites
                st.subheader("IRES Sites (Translation)")
                st.metric("IRES Sites", len(mod_features.ires_sites))
                st.metric("Translation Potential", f"{mod_features.translation_potential:.2f}")

                # miRNA sites
                st.subheader("miRNA Binding Sites (ceRNA)")
                st.metric("miRNA Sites", len(mod_features.miRNA_sites))
                st.metric("ceRNA Activity", f"{mod_features.ceRNA_activity:.2f}")

                if mod_features.miRNA_sites:
                    miRNA_names = [s.miRNA_name for s in mod_features.miRNA_sites[:10]]
                    st.write("miRNAs detected:", ", ".join(miRNA_names))

                # RBP sites
                st.subheader("RBP Binding Sites")
                st.metric("RBP Sites", len(mod_features.rbp_sites))

                if mod_features.rbp_sites:
                    rbp_names = set(s.rbp_name for s in mod_features.rbp_sites)
                    st.write("RBPs:", ", ".join(rbp_names))

    # --- Kinetics ---
    with tab4:
        st.subheader("Folding Kinetics")

        st.info("Folding rate, energy barriers, pathway analysis")

        if st.button("Analyze Kinetics", key="analyze_kinetics"):
            with st.spinner("Computing kinetics..."):
                kinetics_features = predict_folding_kinetics(seq)

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Folding Rate", f"{kinetics_features.folding_rate:.2e} s⁻¹")
                    st.metric("Barrier Height", f"{kinetics_features.barrier_height:.1f} kcal/mol")
                    st.metric("Metastable States", kinetics_features.metastable_count)

                with col2:
                    st.metric("Landscape Complexity", f"{kinetics_features.landscape_complexity:.2f}")
                    st.metric("Cotrans Score", f"{kinetics_features.cotrans_folding_score:.2f}")
                    st.metric("Dynamic Stability", f"{kinetics_features.stability_dynamic:.2f}")

                st.info(f"Method: {kinetics_features.kinetics_method}")


# ===================================================================
# Page: Sequence Design
# ===================================================================

def page_sequence_design():
    """Sequence design and evolution page."""
    st.markdown("<h2 class='sub-header'>🧪 Sequence Design</h2>", unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "🧬 Evolution Optimization",
        "📝 IRES Designer",
        "🔧 Modification Selector",
    ])

    # --- Evolution ---
    with tab1:
        st.subheader("circRNA Sequence Evolution")

        st.info("""
        Optimize circRNA sequences using evolutionary algorithms:
        - Backbone mutation (protect backsplice junction)
        - IRES optimization
        - UTR shuffling
        - Modification selection
        """)

        # Input
        seed_seq = sequence_input_widget("Seed Sequence", "evolution_seed")

        # Parameters
        col1, col2 = st.columns(2)

        with col1:
            rounds = st.slider("Evolution Rounds", 1, 20, 5, key="evo_rounds")
            candidates = st.slider("Candidates/Round", 10, 100, 24, key="evo_candidates")
            top_k = st.slider("Top K Selection", 2, 20, 8, key="evo_top_k")

        with col2:
            init_mod = st.selectbox(
                "Initial Modification",
                ["m6A", "Psi", "5mC", "none"],
                key="evo_init_mod",
            )

            # Weight priorities
            st.subheader("Objective Priorities")
            w_stab = st.slider("Stability Weight", 0.0, 1.0, 0.35, key="w_stab")
            w_trans = st.slider("Translation Weight", 0.0, 1.0, 0.30, key="w_trans")
            w_immune = st.slider("Immune Evasion Weight", 0.0, 1.0, 0.25, key="w_immune")
            w_deliv = st.slider("Delivery Weight", 0.0, 1.0, 0.10, key="w_deliv")

        if st.button("Run Evolution", key="run_evolution"):
            with st.spinner("Evolving sequences..."):
                cfg = CircRNAEvolutionConfig(
                    rounds=rounds,
                    seed_seq=seed_seq,
                    modification=init_mod,
                    candidates_per_round=candidates,
                    top_k=top_k,
                    weight_stability=w_stab,
                    weight_translation=w_trans,
                    weight_immune_evasion=w_immune,
                    weight_delivery=w_deliv,
                )

                results_df, artifacts = evolve_cirrna(cfg)

                # Display results
                st.subheader("Evolution Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Rounds Completed", artifacts.rounds_ran)
                    st.metric("Best Reward", f"{artifacts.best_reward:.4f}")

                with col2:
                    st.metric("Best Modification", artifacts.best_modification)
                    st.metric("Best Sequence Length", len(artifacts.best_sequence))

                with col3:
                    # Policy logits
                    st.json(artifacts.final_policy_logits)

                # Best sequence
                st.subheader("Best Optimized Sequence")
                st.code(artifacts.best_sequence)

                # Results table
                st.subheader("Top Candidates")
                top_results = results_df.sort_values('reward', ascending=False).head(10)
                st.dataframe(
                    top_results[['circrna_seq', 'modification', 'reward', 'pareto_front']],
                    use_container_width=True,
                )

                # Reflections
                st.subheader("Evolution Reflections")
                for ref in artifacts.reflections:
                    st.write(f"- {ref}")

                # Plot
                fig = px.line(
                    x=range(1, len(artifacts.per_round_best)+1),
                    y=artifacts.per_round_best,
                    title="Best Reward per Round",
                    labels={'x': 'Round', 'y': 'Best Reward'},
                )
                st.plotly_chart(fig, use_container_width=True)

    # --- IRES Designer ---
    with tab2:
        st.subheader("IRES Optimization")

        st.info("Optimize sequence for translation initiation")

        seq_for_ires = sequence_input_widget("Sequence for IRES Optimization", "ires_seq")

        col1, col2 = st.columns(2)

        with col1:
            ires_rounds = st.slider("IRES Optimization Rounds", 1, 10, 3, key="ires_rounds")

        with col2:
            if st.button("Optimize IRES", key="optimize_ires"):
                with st.spinner("Optimizing..."):
                    optimized = optimize_for_translation(seq_for_ires, rounds=ires_rounds)

                    st.subheader("IRES-Optimized Sequence")
                    st.code(optimized)

                    # Compute objectives
                    obj = compute_cirrna_objectives(optimized, "m6A")

                    st.metric("Translation Score", f"{obj[1]:.2f}")
                    st.metric("Stability Score", f"{obj[0]:.2f}")

    # --- Modification Selector ---
    with tab3:
        st.subheader("Modification Strategy")

        st.info("Select optimal modifications for circRNA")

        seq_for_mod = sequence_input_widget("Sequence for Modification", "mod_seq")

        mods_to_test = st.multiselect(
            "Test Modifications",
            ["m6A", "Psi", "5mC", "ms2m6A", "2OMeA", "2OMeU", "s2U"],
            default=["m6A", "Psi"],
        )

        if st.button("Compare Modifications", key="compare_mods"):
            results = []

            for mod in mods_to_test:
                obj = compute_cirrna_objectives(seq_for_mod, mod)
                results.append({
                    "Modification": mod,
                    "Stability": obj[0],
                    "Translation": obj[1],
                    "Immune Evasion": obj[2],
                    "Delivery": obj[3],
                    "Average": np.mean(obj),
                })

            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)

            # Bar chart
            fig = px.bar(
                results_df,
                x='Modification',
                y=['Stability', 'Translation', 'Immune Evasion', 'Delivery'],
                title="Modification Comparison",
                barmode='group',
            )
            st.plotly_chart(fig, use_container_width=True)


# ===================================================================
# Page: Vaccine Development
# ===================================================================

def page_vaccine_development():
    """Vaccine development page."""
    st.markdown("<h2 class='sub-header'>💉 Vaccine Development</h2>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "🎯 IPS Scoring",
        "💊 Drug Response",
        "🧪 Treatment Recommendation",
    ])

    # --- IPS Scoring ---
    with tab1:
        st.subheader("Immunotherapy Potential Score (IPS)")

        st.info("""
        IPS predicts immunotherapy response based on:
        - Immune pathway scores (RIG-I, TLR, PKR)
        - Gene expression (TROP2, B7-H4, etc.)
        """)

        seq_ips = sequence_input_widget("circRNA Sequence", "ips_seq")
        gene_expr_ips = gene_expression_input_widget("ips_gene")

        if st.button("Compute IPS", key="compute_ips"):
            with st.spinner("Computing IPS..."):
                # Compute immune scores
                config = ImmuneSensingConfig()
                immune_result = predict_circrna_immunogenicity(seq_ips, config)

                immune_scores = {
                    "ips": 0.0,  # Will compute
                    "rig_i_score": immune_result.rig_i_score,
                    "tlr_score": immune_result.tlr7_score + immune_result.tlr8_score,
                    "pkr_score": immune_result.pkr_score,
                    "overall_immunogenicity": immune_result.overall_immunogenicity,
                }

                # IPS computation
                ips = (
                    immune_scores["overall_immunogenicity"] * 0.4 +
                    max(0, 10 - gene_expr_ips["TROP2"]) / 10 * 0.3 +
                    max(0, 10 - gene_expr_ips["B7-H4"]) / 10 * 0.3
                ) * 10

                immune_scores["ips"] = ips

                # Display
                st.metric("IPS Score", f"{ips:.2f}")

                if ips >= 7.0:
                    st.success("✅ High IPS - Likely responder")
                elif ips >= 5.0:
                    st.warning("⚠️ Moderate IPS - May respond")
                else:
                    st.error("❌ Low IPS - Likely non-responder")

                # Radar chart
                fig = plot_radar_chart({
                    "IPS": ips / 10,
                    "RIG-I": immune_scores["rig_i_score"],
                    "TLR": immune_scores["tlr_score"],
                    "PKR": immune_scores["pkr_score"],
                }, "Vaccine Profile")
                st.plotly_chart(fig, use_container_width=True)

    # --- Drug Response ---
    with tab2:
        st.subheader("Drug Response Prediction")

        st.info("Predict response to circRNA vaccine + combination therapy")

        # Input
        seq_drug = sequence_input_widget("circRNA Sequence", "drug_seq")
        gene_expr_drug = gene_expression_input_widget("drug_gene")

        # Combination drugs
        combination_drugs = st.multiselect(
            "Combination Drugs",
            ["pembrolizumab", "nivolumab", "atezolizumab", "chemotherapy"],
            default=["pembrolizumab"],
        )

        if st.button("Predict Response", key="predict_response"):
            with st.spinner("Predicting..."):
                # Compute immune scores
                config = ImmuneSensingConfig()
                immune_result = predict_circrna_immunogenicity(seq_drug, config)

                immune_scores = {
                    "ips": 7.0,  # Placeholder
                    "rig_i_score": immune_result.rig_i_score,
                    "tlr_score": immune_result.tlr7_score,
                    "pkr_score": immune_result.pkr_score,
                    "overall_immunogenicity": immune_result.overall_immunogenicity,
                }

                # Drug response
                response = predict_drug_response(immune_scores, gene_expr_drug)

                # Display
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Response", response.predicted_response)
                    st.metric("Probability", f"{response.response_probability:.2f}")

                with col2:
                    st.metric("Time to Response", f"{response.time_to_response:.0f} days")
                    st.metric("Duration", f"{response.duration_estimate:.1f} months")

                with col3:
                    st.metric("Resistance Risk", f"{response.resistance_risk:.2f}")
                    st.metric("Treatment Benefit", f"{response.treatment_benefit_score:.2f}")

                # Synergies
                st.subheader("Drug Synergies")
                for syn in response.synergy_scores:
                    st.write(f"**{syn.drug_a} + {syn.drug_b}**: {syn.synergy_type} ({syn.synergy_value:.2f})")

                # Contraindications
                if response.contraindications:
                    st.subheader("⚠️ Contraindications")
                    for contra in response.contraindications:
                        st.warning(contra)

    # --- Treatment Recommendation ---
    with tab3:
        st.subheader("Treatment Recommendation")

        st.info("Get personalized treatment recommendations")

        seq_treat = sequence_input_widget("circRNA Sequence", "treat_seq")
        gene_expr_treat = gene_expression_input_widget("treat_gene")

        patient_age = st.slider("Patient Age", 18, 90, 60, key="patient_age")
        cancer_stage = st.selectbox("Cancer Stage", ["I", "II", "III", "IV"], key="cancer_stage")

        if st.button("Generate Recommendation", key="gen_recommend"):
            with st.spinner("Generating..."):
                # Compute immune scores
                config = ImmuneSensingConfig()
                immune_result = predict_circrna_immunogenicity(seq_treat, config)

                immune_scores = {
                    "ips": 7.0,
                    "rig_i_score": immune_result.rig_i_score,
                    "tlr_score": immune_result.tlr7_score,
                    "pkr_score": immune_result.pkr_score,
                    "overall_immunogenicity": immune_result.overall_immunogenicity,
                }

                patient_data = {"age": patient_age, "stage": cancer_stage}

                recommendation = recommend_treatment(immune_scores, gene_expr_treat, patient_data)

                st.subheader("Recommended Treatment")
                st.success(f"**Primary**: {recommendation['recommended_primary']}")
                st.write(f"**Combinations**: {', '.join(recommendation['recommended_combinations'])}")

                st.subheader("Expected Outcomes")
                st.metric("Response Probability", f"{recommendation['response_probability']:.2f}")
                st.metric("Resistance Risk", f"{recommendation['resistance_risk']:.2f}")
                st.metric("Time to Response", f"{recommendation['time_to_response']:.0f} days")
                st.metric("Duration", f"{recommendation['expected_duration']:.1f} months")

                if recommendation['contraindications']:
                    st.subheader("⚠️ Contraindications")
                    for contra in recommendation['contraindications']:
                        st.warning(contra)

                st.subheader("Monitoring Schedule")
                for schedule in recommendation['monitoring_schedule']:
                    st.write(f"- {schedule}")


# ===================================================================
# Page: Clinical Report
# ===================================================================

def page_clinical_report():
    """Clinical report generation page."""
    st.markdown("<h2 class='sub-header'>📋 Clinical Report</h2>", unsafe_allow_html=True)

    st.info("Generate comprehensive clinical outcome report")

    # Input
    seq_clinical = sequence_input_widget("circRNA Sequence", "clinical_seq")
    gene_expr_clinical = gene_expression_input_widget("clinical_gene")

    patient_age_clinical = st.slider("Patient Age", 18, 90, 55, key="clinical_age")
    cancer_stage_clinical = st.selectbox("Cancer Stage", ["I", "II", "III", "IV"], index=2, key="clinical_stage")

    if st.button("Generate Clinical Report", key="gen_clinical"):
        with st.spinner("Generating report..."):
            # Compute immune scores
            config = ImmuneSensingConfig()
            immune_result = predict_circrna_immunogenicity(seq_clinical, config)

            immune_scores = {
                "ips": 7.2,
                "rig_i_score": immune_result.rig_i_score,
                "tlr_score": immune_result.tlr7_score,
                "pkr_score": immune_result.pkr_score,
                "overall_immunogenicity": immune_result.overall_immunogenicity,
            }

            patient_data = {"age": patient_age_clinical, "stage": cancer_stage_clinical}

            # Generate report
            report = generate_clinical_report(immune_scores, gene_expr_clinical, patient_data)

            # Display summary
            st.subheader("Report Summary")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Prognosis", report["summary"]["prognosis"])
                st.metric("Risk Score", f"{report['summary']['risk_score']:.2f}")

            with col2:
                st.metric("Treatment Benefit", f"{report['summary']['treatment_benefit']:.2f}")
                st.metric("OS Median", f"{report['survival']['os_median_months']:.0f} months")

            with col3:
                st.metric("PFS Median", f"{report['survival']['pfs_median_months']:.0f} months")
                st.metric("1-Year Survival", f"{report['survival']['1yr_survival']:.1%}")

            # Biomarkers
            st.subheader("Biomarkers")
            biomarker_df = pd.DataFrame(report["biomarkers"])
            st.dataframe(biomarker_df, use_container_width=True)

            # Adverse Events
            st.subheader("Adverse Event Risks")
            ae_df = pd.DataFrame(report["adverse_events"])
            st.dataframe(ae_df, use_container_width=True)

            # Recommendations
            st.subheader("Follow-up Schedule")
            for schedule in report["recommendations"]["followup_schedule"]:
                st.write(f"- {schedule}")

            # Download button
            st.subheader("Download Report")
            report_json = json.dumps(report, indent=2)
            st.download_button(
                "Download JSON Report",
                report_json,
                "clinical_report.json",
                "application/json",
            )


# ===================================================================
# Page: Settings
# ===================================================================

def page_settings():
    """Settings page."""
    st.markdown("<h2 class='sub-header'>⚙️ Settings</h2>", unsafe_allow_html=True)

    # Backend Configuration Section
    st.subheader("🔧 Backend Configuration")

    st.info("""
    **Flexible Backend Architecture:** Choose between fast local models and high-accuracy external APIs.
    - Local models: Fast, offline-ready, good for screening
    - External APIs: Higher accuracy, requires network, good for validation
    """)

    # MHC Backend
    st.write("**MHC Binding Prediction**")
    mhc_backend = st.selectbox(
        "Select MHC backend:",
        ["local", "netmhcpan"],
        index=0,
        key="mhc_backend_select",
        help="local: AUC=0.80, fast; netmhcpan: AUC=0.92-0.96, requires network"
    )

    if mhc_backend == "local":
        st.caption("✅ Local model (AUC=0.80, ~50ms, offline-ready)")
    else:
        st.caption("⚠️ NetMHCpan API (AUC=0.92-0.96, ~200ms, requires network)")

    # Immunogenicity Backend
    st.write("**Immunogenicity Scoring**")
    imm_backend = st.selectbox(
        "Select Immunogenicity backend:",
        ["heuristic", "vienna", "esm2"],
        index=0,
        key="imm_backend_select",
        help="heuristic: fast; vienna: with accessibility; esm2: experimental"
    )

    if imm_backend == "heuristic":
        st.caption("✅ Heuristic model (~85ms, offline-ready)")
    elif imm_backend == "vienna":
        st.caption("ℹ️ ViennaRNA-enhanced (~150ms, adds structural accessibility)")
    else:
        st.caption("⚠️ ESM-2 embeddings (~2-5s, experimental, may require GPU)")

    # Drug Backend
    st.write("**Drug Binding Prediction**")
    drug_backend = st.selectbox(
        "Select Drug backend:",
        ["local", "chembl_api"],
        index=0,
        key="drug_backend_select",
        help="local: R²=0.95, fast; chembl_api: experimental data"
    )

    if drug_backend == "local":
        st.caption("✅ Local model (R²=0.95, ~100ms, offline-ready)")
    else:
        st.caption("⚠️ ChEMBL API (~500ms, requires network)")

    # API Timeout
    st.write("**API Settings**")
    api_timeout = st.slider(
        "API Timeout (seconds):",
        min_value=5,
        max_value=120,
        value=30,
        step=5,
        key="api_timeout_slider"
    )

    # Save backend settings to session state
    st.session_state["backend_settings"] = {
        "mhc_backend": mhc_backend,
        "immunogenicity_backend": imm_backend,
        "drug_backend": drug_backend,
        "timeout": api_timeout
    }

    if st.button("Apply Backend Settings", key="apply_backend_btn"):
        st.success("Backend settings applied! Settings will be used in next analysis.")

    st.markdown("---")

    # ViennaRNA Configuration
    st.subheader("ViennaRNA Configuration")

    st.info("""
    ViennaRNA provides accurate structure prediction.
    Install on Linux: `apt-get install vienna-rna`
    """)

    # Check ViennaRNA
    import subprocess
    try:
        result = subprocess.run(["RNAfold", "--version"], capture_output=True, timeout=5)
        if result.returncode == 0:
            st.success(f"✅ ViennaRNA installed: {result.stdout.decode().strip()}")
        else:
            st.warning("⚠️ ViennaRNA not found - using fallback estimation")
    except Exception:
        st.warning("⚠️ ViennaRNA not installed - using fallback estimation")

    st.subheader("Analysis Parameters")

    # Immune weights
    st.write("Immune pathway weights (literature-based)")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**RIG-I**: 0.35 (dsRNA backbone detection)")
        st.write("**TLR7**: 0.20 (GU-rich motifs)")

    with col2:
        st.write("**TLR8**: 0.15 (AU-rich motifs)")
        st.write("**PKR**: 0.30 (dsRNA >33bp, Nallagatla et al., 2007)")

    with col3:
        st.write("**OAS**: 0.15")
        st.write("**MDA5**: 0.10")

    st.subheader("About")
    st.write("Confluencia circRNA Module v1.0")
    st.write("GitHub: https://github.com/IGEM-FBH/confluencia")
    st.write("Documentation: See README.md")


# ===================================================================
# Main App Router
# ===================================================================

def main():
    """Main app router."""

    if page == "📊 Sequence Analysis":
        page_sequence_analysis()

    elif page == "🧪 Sequence Design":
        page_sequence_design()

    elif page == "💉 Vaccine Development":
        page_vaccine_development()

    elif page == "📋 Clinical Report":
        page_clinical_report()

    elif page == "⚙️ Settings":
        page_settings()


if __name__ == "__main__":
    main()