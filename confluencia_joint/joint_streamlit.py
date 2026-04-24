"""
Unified Streamlit panel for Joint Drug-Epitope-PK Evaluation.

Provides a single-page UI with sidebar inputs, three-column score display,
PK curve tabs, and CSV batch evaluation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

_PROJECT = Path(__file__).resolve().parents[1]
for _dir in [
    _PROJECT / "confluencia-2.0-drug",
    _PROJECT / "confluencia-2.0-epitope",
    _PROJECT / "confluencia_shared",
]:
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------

from confluencia_joint import (
    JointInput,
    JointEvaluationEngine,
    JointEvaluationResult,
    JointScore,
    JointScoringEngine,
    ClinicalScore,
    BindingScore,
    KineticsScore,
    JointFusionLayer,
    FusionStrategy,
)


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

INJECTED_CSS = False


def _inject_styles():
    global INJECTED_CSS
    if INJECTED_CSS:
        return
    INJECTED_CSS = True
    st.markdown("""
    <style>
    .score-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        background: #fafafa;
    }
    .score-label { font-size: 13px; color: #666; margin: 0; }
    .score-value { font-size: 28px; font-weight: 700; margin: 4px 0; }
    .score-sub   { font-size: 12px; color: #888; margin: 0; }
    .rec-Go      { color: #1a7f37; font-weight: 700; font-size: 18px; }
    .rec-Cond    { color: #b5730a; font-weight: 700; font-size: 18px; }
    .rec-NoGo    { color: #c0392b; font-weight: 700; font-size: 18px; }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-clinical  { background: #e3f2fd; color: #1565c0; }
    .badge-binding   { background: #f3e5f5; color: #6a1b9a; }
    .badge-kinetics  { background: #e8f5e9; color: #2e7d32; }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_gauge(label: str, value: float, color: str = "#4a90d9") -> str:
    """Return HTML for a mini gauge bar."""
    pct = max(0, min(100, int(value * 100)))
    return f"""
    <div class="score-card">
      <p class="score-label">{label}</p>
      <div style="background:#e0e0e0;border-radius:4px;height:8px;width:100%;">
        <div style="background:{color};border-radius:4px;height:8px;width:{pct}%;transition:width 0.4s;"></div>
      </div>
      <p class="score-sub">{value:.3f}</p>
    </div>
    """


def _render_clinical_card(score: ClinicalScore) -> str:
    return _render_gauge(
        f"Clinical: Efficacy {score.efficacy:.2f} | "
        f"Binding {score.target_binding:.2f} | "
        f"Immune {score.immune_activation:.2f} | "
        f"Safety penalty {score.safety_penalty:.2f}",
        score.overall,
        "#1565c0",
    )


def _render_binding_card(score: BindingScore) -> str:
    color = {
        "strong_binder": "#2e7d32",
        "moderate_binder": "#b5730a",
        "weak_binder": "#c0392b",
        "non_binder": "#c0392b",
    }.get(score.mhc_affinity_class, "#666")
    return _render_gauge(
        f"Binding: {score.mhc_affinity_class} | Eff {score.epitope_efficacy:.2f} | "
        f"Uncertainty {score.uncertainty:.2f}",
        score.overall,
        color,
    )


def _render_kinetics_card(score: KineticsScore) -> str:
    return _render_gauge(
        f"Kinetics: Cmax={score.cmax:.2f} mg/L | Tmax={score.tmax:.1f}h | "
        f"HL={score.half_life:.1f}h | TI={score.therapeutic_index:.3f}",
        score.overall,
        "#2e7d32",
    )


def _render_recommendation(score: JointScore) -> str:
    rec_class = {
        "Go": "rec-Go",
        "Conditional": "rec-Cond",
        "No-Go": "rec-NoGo",
    }.get(score.recommendation, "")
    return f"""
    <div class="score-card" style="text-align:center;">
      <p class="score-label">COMPOSITE SCORE</p>
      <p class="score-value">{score.composite:.3f}</p>
      <p class="{rec_class}">{score.recommendation}</p>
      <p class="score-sub">{score.recommendation_reason}</p>
    </div>
    """


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_result(result: JointEvaluationResult):
    """Render a single JointEvaluationResult in Streamlit."""
    score = result.joint_score

    st.markdown("### Scores")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Clinical**", unsafe_allow_html=True)
        st.markdown(_render_clinical_card(score.clinical), unsafe_allow_html=True)
    with c2:
        st.markdown("**Binding**", unsafe_allow_html=True)
        st.markdown(_render_binding_card(score.binding), unsafe_allow_html=True)
    with c3:
        st.markdown("**Kinetics**", unsafe_allow_html=True)
        st.markdown(_render_kinetics_card(score.kinetics), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Composite")
    st.markdown(_render_recommendation(score), unsafe_allow_html=True)


def render_pk_curve(result: JointEvaluationResult):
    """Render PK/PD curve using Streamlit charts."""
    df = result.pk_curve
    if df is None or df.empty:
        st.warning("No PK curve data available.")
        return

    t = df["time_h"].values
    conc = df["pkpd_conc_mg_per_l"].values
    effect = df["pkpd_effect"].values

    tab1, tab2 = st.tabs(["Concentration", "PD Effect"])

    with tab1:
        chart_df = pd.DataFrame({"Time (h)": t, "Conc (mg/L)": conc})
        st.line_chart(chart_df.set_index("Time (h)"))

    with tab2:
        chart_df = pd.DataFrame({"Time (h)": t, "PD Effect": effect})
        st.line_chart(chart_df.set_index("Time (h)"))

    st.dataframe(df[["time_h", "pkpd_conc_mg_per_l", "pkpd_effect"]].head(50))


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

def run_streamlit():
    """Entry point for streamlit run."""
    _inject_styles()

    st.set_page_config(
        page_title="Confluencia 2.0 — Joint Drug-Epitope-PK",
        page_icon="🧬",
        layout="wide",
    )

    st.title("🧬 Confluencia 2.0 — Joint Drug-Epitope-PK Evaluation")
    st.markdown(
        "Three-dimensional assessment: **Clinical** (drug efficacy) + "
        "**Binding** (MHC-peptide) + **Kinetics** (PK/PD)."
    )
    st.divider()

    # --- Sidebar ---
    with st.sidebar:
        st.header("Input")

        mode = st.radio(
            "Evaluation mode",
            ["Single Sample", "Batch CSV"],
            help="Single sample: fill in the fields below. "
                 "Batch: upload a CSV with multiple rows.",
        )

        if mode == "Single Sample":
            _render_single_input()
        else:
            _render_batch_input()


# ---------------------------------------------------------------------------
# Sidebar sections
# ---------------------------------------------------------------------------

def _render_single_input():
    st.subheader("Molecule")

    smiles = st.text_area(
        "SMILES",
        placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O",
        help="Molecular structure in SMILES format",
    )

    epitope_seq = st.text_input(
        "Epitope Sequence",
        placeholder="e.g. SLYNTVATL",
        help="Peptide amino-acid sequence (1-letter codes)",
    ).strip().upper()

    mhc_allele = st.selectbox(
        "MHC Allele",
        [
            "HLA-A*01:01", "HLA-A*02:01", "HLA-A*02:03", "HLA-A*02:06",
            "HLA-A*03:01", "HLA-A*11:01", "HLA-A*23:01", "HLA-A*24:02",
            "HLA-A*26:01", "HLA-A*30:02", "HLA-A*31:01", "HLA-A*33:01",
            "HLA-A*68:01", "HLA-A*68:02",
            "HLA-B*07:02", "HLA-B*08:01", "HLA-B*15:01", "HLA-B*27:02",
            "HLA-B*27:05", "HLA-B*35:01", "HLA-B*39:01", "HLA-B*40:01",
            "HLA-B*44:02", "HLA-B*44:03", "HLA-B*45:01", "HLA-B*46:01",
            "HLA-B*48:01", "HLA-B*51:01", "HLA-B*53:01", "HLA-B*57:01",
            "HLA-B*58:01",
            "HLA-C*01:02", "HLA-C*02:02", "HLA-C*03:03", "HLA-C*04:01",
            "HLA-C*05:01", "HLA-C*06:02", "HLA-C*07:01", "HLA-C*07:02",
            "HLA-C*08:02", "HLA-C*12:03", "HLA-C*14:02", "HLA-C*15:02",
        ],
        index=1,  # HLA-A*02:01
    )

    st.subheader("Dosing")
    dose_mg = st.number_input("Dose (mg)", value=200.0, min_value=0.1, step=10.0)
    freq = st.number_input("Frequency (per day)", value=2.0, min_value=0.1, step=0.5)
    treatment_time = st.number_input("Treatment duration (h)", value=72.0, min_value=1.0, step=12.0)

    st.subheader("Immunological Context")
    circ_expr = st.slider("circRNA Expression", 0.0, 1.0, 0.0, help="circRNA expression level")
    ifn_score = st.slider("IFN Response Score", 0.0, 1.0, 0.0, help="Interferon response score")

    st.subheader("Advanced")
    epitope_backend = st.selectbox(
        "Epitope Model Backend",
        ["sklearn-moe", "hgb", "rf", "ridge"],
        index=0,
        help="Backend for MHC binding prediction",
    )
    pk_horizon = st.slider("PK Horizon (h)", 24, 168, 72, step=12)
    use_mhc = st.checkbox(
        "Use MHC features (recommended)",
        value=True,
        help="Include MHC pseudo-sequence encoding (979 dims) for improved binding prediction. "
             "Recommended: achieves AUC=0.917 vs 0.736 baseline.",
    )

    run_button = st.button("🚀 Run Evaluation", type="primary", use_container_width=True)

    # --- Run ---
    if run_button:
        if not smiles:
            st.error("Please enter a SMILES string.")
            return
        if not epitope_seq:
            st.error("Please enter an epitope sequence.")
            return

        inp = JointInput(
            smiles=smiles,
            epitope_seq=epitope_seq,
            mhc_allele=mhc_allele,
            dose_mg=dose_mg,
            freq_per_day=freq,
            treatment_time=treatment_time,
            circ_expr=circ_expr,
            ifn_score=ifn_score,
            group_id="single",
        )

        validation = inp.validate()
        if validation:
            st.error("Validation errors:\n" + "\n".join(f"  - {e}" for e in validation))
            return

        with st.spinner("Running joint evaluation..."):
            try:
                engine = JointEvaluationEngine(
                    epitope_backend=epitope_backend,
                    pk_horizon=pk_horizon,
                    use_mhc=use_mhc,
                )
                result = engine.evaluate_single(inp)
                st.session_state["joint_result"] = result
                st.session_state["joint_results"] = [result]
            except Exception as ex:
                st.exception(ex)
                return

    # Show results if available
    if "joint_result" in st.session_state:
        st.divider()
        result: JointEvaluationResult = st.session_state["joint_result"]
        render_result(result)

        st.divider()
        st.subheader("PK/PD Curve")
        render_pk_curve(result)

        # Export
        st.divider()
        csv_data = result.to_dataframe().to_csv(index=False)
        st.download_button(
            "📥 Download Result CSV",
            csv_data,
            file_name="joint_evaluation_result.csv",
            mime="text/csv",
        )


def _render_batch_input():
    st.subheader("Batch CSV Upload")
    st.markdown(
        "Expected columns: `smiles`, `epitope_seq`, `mhc_allele`, `dose`, "
        "`freq`, `treatment_time`\n\n"
        "Optional: `circ_expr`, `ifn_score`, `group_id`"
    )

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="CSV with one row per drug-epitope pair",
    )

    if not uploaded:
        return

    # Read CSV
    try:
        df = pd.read_csv(uploaded)
    except Exception as ex:
        st.error(f"Failed to read CSV: {ex}")
        return

    st.info(f"Loaded {len(df)} rows")
    st.dataframe(df.head())

    epitope_backend = st.selectbox(
        "Epitope Model",
        ["sklearn-moe", "hgb", "rf", "ridge"],
        index=0,
    )
    pk_horizon = st.slider("PK Horizon (h)", 24, 168, 72, step=12)
    use_mhc = st.checkbox(
        "Use MHC features (recommended)",
        value=True,
        help="Include MHC pseudo-sequence encoding (979 dims). AUC=0.917 vs 0.736 baseline.",
    )

    if st.button("🚀 Run Batch Evaluation", type="primary", use_container_width=True):
        with st.spinner(f"Evaluating {len(df)} rows..."):
            try:
                inputs = JointInput.from_dataframe(df)
            except Exception as ex:
                st.error(f"Failed to parse CSV: {ex}")
                return

            # Filter valid
            valid = [inp for inp in inputs if inp.is_valid()]
            invalid = [inp for inp in inputs if not inp.is_valid()]
            if invalid:
                st.warning(f"Skipped {len(invalid)} invalid rows")

            engine = JointEvaluationEngine(
                epitope_backend=epitope_backend,
                pk_horizon=pk_horizon,
                use_mhc=use_mhc,
            )
            results = engine.evaluate(valid)

            st.session_state["joint_results"] = results

            # Summary table
            summary_df = pd.concat([r.to_dataframe() for r in results], ignore_index=True)
            st.success(f"Evaluated {len(results)} / {len(inputs)} rows")
            st.dataframe(summary_df)

            # Download
            csv_all = summary_df.to_csv(index=False)
            st.download_button(
                "📥 Download All Results CSV",
                csv_all,
                file_name="joint_batch_results.csv",
                mime="text/csv",
            )

    # Show results
    if "joint_results" in st.session_state and st.session_state["joint_results"]:
        results: list = st.session_state["joint_results"]

        for i, result in enumerate(results):
            with st.expander(f"Row {i}: {result.input.epitope_seq} — {result.joint_score.recommendation}"):
                render_result(result)

                # PK curve
                with st.container():
                    st.subheader("PK/PD Curve")
                    render_pk_curve(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import subprocess
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", __file__,
        "--server.headless", "true",
    ], cwd=str(Path(__file__).parent))


if __name__ == "__streamlit_script__":
    run_streamlit()