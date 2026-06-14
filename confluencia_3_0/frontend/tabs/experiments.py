"""Experiments Tab

Displays: experiment runner, output figures, event bus monitor.
"""
import streamlit as st
import pandas as pd
from typing import Any
from pathlib import Path
import glob

TN = {
    "bg": "#1a1b26", "surface": "#24283b", "border": "#414d68",
    "text": "#c0caf5", "muted": "#a9b1d6",
    "blue": "#7aa2f7", "green": "#9ece6a", "yellow": "#e0af68",
    "red": "#f7768e", "purple": "#bb9af7", "cyan": "#7dcfff",
    "orange": "#ff9e64", "teal": "#73daca",
}


def render_experiments(agent: Any, history_df: pd.DataFrame):
    """Render the Experiments tab"""
    from frontend.app_core import section_header

    project_root = Path(__file__).parent.parent.parent
    experiments_dir = project_root / "experiments"
    output_dir = project_root / "output" / "figures"

    # ── Row 1: Experiment Runner ──
    section_header("Experiment Runner")

    # Discover experiment scripts
    exp_scripts = sorted(experiments_dir.glob("experiment_*.py")) if experiments_dir.exists() else []
    exp_names = [f.stem for f in exp_scripts]
    exp_labels = [name.replace("experiment_", "Exp: ").replace("_", " ").title() for name in exp_names]

    col_select, col_run = st.columns([3, 1])

    with col_select:
        selected_exp = st.selectbox("Select Experiment", exp_labels if exp_labels else ["No experiments found"])

    with col_run:
        st.markdown("<br>", unsafe_allow_html=True)
        run_clicked = st.button("Run Experiment", type="primary", use_container_width=True)

    if run_clicked and exp_names:
        idx = exp_labels.index(selected_exp)
        script_path = exp_scripts[idx]
        try:
            with st.spinner(f"Running {selected_exp}..."):
                import subprocess
                import sys
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True, text=True, timeout=300,
                    cwd=str(project_root),
                )
                if result.returncode == 0:
                    st.success(f"Experiment completed!")
                    with st.expander("Output"):
                        st.code(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
                else:
                    st.error(f"Experiment failed!")
                    with st.expander("Error"):
                        st.code(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        except subprocess.TimeoutExpired:
            st.error("Experiment timed out (5 min limit)")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")

    # ── Row 2: Output Figures ──
    section_header("Output Figures")

    if output_dir.exists():
        fig_files = sorted(output_dir.glob("*.png"))
        if fig_files:
            # Display in a grid (3 columns)
            cols_per_row = 3
            for i in range(0, len(fig_files), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(fig_files):
                        fig_path = fig_files[i + j]
                        with col:
                            st.image(str(fig_path), caption=fig_path.name, use_container_width=True)
        else:
            st.info("No output figures found. Run an experiment first.")
    else:
        st.info("Output directory not found. Run an experiment first.")

    st.markdown("---")

    # ── Row 3: Experiment Report ──
    section_header("Experiment Report")
    report_path = project_root / "output" / "experiment_report.md"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            report_content = f.read()
        st.markdown(report_content)
    else:
        st.info("No experiment report found. Run experiments to generate one.")

    st.markdown("---")

    # ── Row 4: Event Bus Monitor ──
    section_header("Event Bus Monitor")

    try:
        bus = agent.bus
        col_stats, col_recent = st.columns(2)

        with col_stats:
            st.markdown("**Event Bus Statistics**")
            stats = bus.get_stats() if hasattr(bus, "get_stats") else {}
            if stats:
                for key, val in stats.items():
                    st.caption(f"{key}: {val}")
            else:
                st.caption("No stats available")

            # Subscriber info
            subscribers = bus._subscribers if hasattr(bus, "_subscribers") else {}
            st.markdown(f"**Registered Events**: {len(subscribers)}")
            for event_type, subs in list(subscribers.items())[:10]:
                st.caption(f"  {event_type}: {len(subs)} subscriber(s)")

        with col_recent:
            st.markdown("**Recent Events**")
            event_log = bus._event_log if hasattr(bus, "_event_log") else []
            if event_log:
                for event in event_log[-10:]:
                    st.caption(f"[Day {getattr(event, 'step', '?')}] {getattr(event, 'type', str(event))}")
            else:
                st.caption("No events logged")
    except Exception:
        st.info("Event bus not available")
