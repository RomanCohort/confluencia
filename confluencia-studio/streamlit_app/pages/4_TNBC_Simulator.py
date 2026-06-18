"""TNBC Simulacrum Page - Streamlit UI.

Digital twin simulation with animated visualization.
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add paths - NOTE: confluencia_3_0 uses underscores!
PROJECT_ROOT = Path(r"D:\IGEM集成方案")
sys.path.insert(0, str(PROJECT_ROOT / "confluencia_3_0"))

# Add visualization module
import importlib.util
import sys as _sys
VIS_PATH = Path(r"C:\Users\LENOVO\.claude\skills\confluencia")
_spec = importlib.util.spec_from_file_location(
    "visualization",
    str(VIS_PATH / "visualization.py")
)
visualization = importlib.util.module_from_spec(_spec)
_sys.modules["visualization"] = visualization  # Critical fix for dataclass
_spec.loader.exec_module(visualization)
generate_nature_html_report = visualization.generate_nature_html_report

from datetime import datetime

st.set_page_config(page_title="TNBC Simulacrum - Confluencia", page_icon="🎮", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #1a1a2e 0%, #2d3a4f 100%); }
    .section-header {
        background: rgba(196,30,58,0.1);
        border-left: 4px solid #c41e3a;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
        color: #ecf0f1;
    }
    .phase-elimination { color: #27ae60; }
    .phase-equilibrium { color: #f39c12; }
    .phase-escape { color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section-header">
    <h2>🎮 TNBC Simulacrum 数字孪生</h2>
    <p style="color: #7f8c8d;">Triple-Negative Breast Cancer 肿瘤仿真系统</p>
</div>
""", unsafe_allow_html=True)

# Main input section - circRNA sequence for vaccine treatment
st.markdown("""
<div class="section-header">
    <h4>📋 circRNA 疫苗输入</h4>
    <p style="color: #7f8c8d;">如果选择 circRNA疫苗 治疗，请输入候选序列</p>
</div>
""", unsafe_allow_html=True)

circrna_vaccine_seq = st.text_area(
    "circRNA 疫苗序列 (用于免疫治疗模拟):",
    height=80,
    placeholder="例如: AUGCGCGCGUAUAGCGCGCG... (仅当选择 circRNA疫苗 时需要)"
)

st.markdown("**示例序列:**")
seq_examples = {
    "低免疫原性 (推荐)": "AUGCGCGCGUAUAGCGCGCGAUGCGCGCGUAUAGCGCGCG",
    "随机序列": "AUGAUCAAAAAAAGGGUAGCUUAUCAACGGAUC"
}
cols = st.columns(2)
for i, (name, seq) in enumerate(seq_examples.items()):
    with cols[i]:
        if st.button(name, key=f"tnbc_seq_{i}", use_container_width=True):
            circrna_vaccine_seq = seq
            st.rerun()

st.markdown("---")

# Sidebar - Simulation parameters
with st.sidebar:
    st.markdown("### ⚙️ 仿真参数")

    initial_volume = st.slider("初始肿瘤体积 (mm³)", 10, 200, 50)
    growth_rate = st.slider("生长速率", 0.01, 0.05, 0.027, format="%.3f")
    cd8_initial = st.slider("初始CD8+ T细胞", 50, 200, 100)
    subtype = st.selectbox("分子亚型", ["BLIS", "IM", "M", "LAR"])

    st.markdown("---")
    st.markdown("### 💉 治疗方案")

    treatment = st.multiselect(
        "选择治疗",
        ["化疗 (紫杉醇)", "免疫治疗 (PD-1)", "circRNA疫苗", "靶向治疗"],
        default=["化疗 (紫杉醇)"]
    )

    # Show circRNA input reminder if vaccine selected
    if "circRNA疫苗" in treatment:
        st.info("💡 请在主页面输入 circRNA 序列")

    treatment_start = st.slider("治疗开始时间 (步)", 0, 50, 10)

    st.markdown("---")
    st.markdown("### ⏱️ 仿真设置")

    n_steps = st.slider("仿真步数", 10, 100, 50)

    if st.button("▶️ 开始仿真", type="primary", use_container_width=True):
        st.session_state["sim_running"] = True
        st.session_state["sim_params"] = {
            "initial_volume": initial_volume,
            "growth_rate": growth_rate,
            "cd8_initial": cd8_initial,
            "subtype": subtype,
            "treatment": treatment,
            "treatment_start": treatment_start,
            "n_steps": n_steps,
            "circrna_vaccine_seq": circrna_vaccine_seq
        }

# Simulation state
sim_data = st.session_state.get("sim_params", {})
if st.session_state.get("sim_running"):
    with st.spinner("正在仿真..."):
        try:
            # Try to import state_schema, fallback to manual defaults
            try:
                from confluencia_3_0.core.state_schema import StateSchema
                schema = StateSchema()
                state = schema.init_defaults()
            except ImportError:
                # Manual default state
                state = {
                    "tum_volume": sim_data["initial_volume"],
                    "tum_growth_rate": sim_data["growth_rate"],
                    "tum_apoptosis_rate": 0.005,
                    "tum_proliferation_index": 0.3,
                    "imm_cd8_count": sim_data["cd8_initial"],
                    "imm_cd4_count": 150.0,
                    "imm_nk_count": 50.0,
                    "imm_t_cell_activation": 0.3,
                    "imm_t_cell_exhaustion": 0.1,
                    "imm_til_density": 0.2,
                    "drg_concentration": 0.0,
                    "drg_resistance_level": 0.0,
                    "evs_pd_l1_expression": 0.2,
                    "ied_phase": "elimination",
                    "sub_molecular_subtype": sim_data["subtype"]
                }

            # Override with user params
            state["tum_volume"] = sim_data["initial_volume"]
            state["tum_growth_rate"] = sim_data["growth_rate"]
            state["imm_cd8_count"] = sim_data["cd8_initial"]
            state["sub_molecular_subtype"] = sim_data["subtype"]

            # Calculate circRNA vaccine effect if sequence provided
            circrna_seq = sim_data.get("circrna_vaccine_seq", "")
            circrna_safety = 0.5  # default
            circrna_evasion = 0.3  # default
            if circrna_seq and "circRNA疫苗" in str(sim_data["treatment"]):
                circrna_clean = "".join(c for c in circrna_seq.upper() if c in "AUGC")
                if len(circrna_clean) > 0:
                    try:
                        # Use actual backend for immune assessment
                        from utils import get_innate_immune
                        innate_mod = get_innate_immune()
                        immune_result = innate_mod.assess_innate_immune(circrna_clean)
                        circrna_safety = immune_result.net_safety_score
                        circrna_evasion = immune_result.modification_evasion if hasattr(immune_result, 'modification_evasion') else 0.5
                        state["crna_immunogenicity_score"] = immune_result.innate_immune_score
                        state["crna_ips_score"] = circrna_safety * 8 + 1
                    except Exception as e:
                        # Fallback heuristic
                        gc = sum(1 for b in circrna_clean if b in "GC") / len(circrna_clean)
                        circrna_safety = max(0, 1.0 - gc * 0.8)
                        circrna_evasion = min(1.0, gc * 0.5 + 0.3)
                        state["crna_immunogenicity_score"] = 1.0 - circrna_safety
                        state["crna_ips_score"] = circrna_safety * 8 + 1

            # Run simulation
            history = []
            for step in range(sim_data["n_steps"]):
                # Apply treatment
                if step >= sim_data["treatment_start"]:
                    if "化疗" in str(sim_data["treatment"]):
                        state["drg_concentration"] = 0.5
                        state["tum_volume"] *= 0.95  # 5% kill
                    if "免疫" in str(sim_data["treatment"]):
                        state["imm_cd8_count"] += 5
                    if "circRNA疫苗" in str(sim_data["treatment"]):
                        # circRNA vaccine boosts immune response
                        if circrna_seq:
                            state["imm_cd8_count"] += int(8 * circrna_safety)
                            state["imm_t_cell_activation"] = min(1.0, state["imm_t_cell_activation"] + 0.03 * circrna_evasion)
                            state["imm_nk_cytotoxicity"] = min(1.0, state.get("imm_nk_cytotoxicity", 0.3) + 0.02)
                        else:
                            # No sequence provided, use default effect
                            state["imm_cd8_count"] += 3

                # Tumor growth
                state["tum_volume"] *= (1 + state["tum_growth_rate"] - state.get("tum_apoptosis_rate", 0.005))

                # Immune dynamics
                state["imm_t_cell_activation"] = min(1.0, state["imm_t_cell_activation"] + 0.02)
                state["imm_t_cell_exhaustion"] = min(1.0, state["imm_t_cell_exhaustion"] + 0.01)

                # Phase transition
                if state["tum_volume"] < sim_data["initial_volume"] * 0.9:
                    state["ied_phase"] = "elimination"
                elif state["tum_volume"] < sim_data["initial_volume"] * 1.5:
                    state["ied_phase"] = "equilibrium"
                else:
                    state["ied_phase"] = "escape"

                history.append({
                    "step": step,
                    "tum_volume": state["tum_volume"],
                    "imm_cd8_count": state["imm_cd8_count"],
                    "imm_t_cell_activation": state["imm_t_cell_activation"],
                    "ied_phase": state["ied_phase"]
                })

            final_data = {
                "module": "simulacrum",
                "state": state,
                "history": history,
                "step": sim_data["n_steps"]
            }

            st.session_state["sim_data"] = final_data
            st.session_state["sim_complete"] = True
            st.session_state["sim_running"] = False

            st.success(f"✅ 仿真完成！运行了 {sim_data['n_steps']} 步")

        except Exception as e:
            st.error(f"仿真出错: {e}")
            st.code(str(e))
            st.session_state["sim_running"] = False

# Display results
if st.session_state.get("sim_complete") and st.session_state.get("sim_data"):
    data = st.session_state["sim_data"]
    state = data["state"]
    history = data["history"]

    st.markdown("---")
    st.markdown("""
    <div class="section-header">
        <h3>📊 仿真结果</h3>
    </div>
    """, unsafe_allow_html=True)

    # Current state metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    phase = state.get("ied_phase", "unknown")
    phase_icons = {"elimination": "🟢", "equilibrium": "🟡", "escape": "🔴"}

    with col1:
        st.metric("肿瘤体积", f"{state['tum_volume']:.1f} mm³")

    with col2:
        st.metric(f"{phase_icons.get(phase, '⚪')} 免疫编辑阶段", phase.upper())

    with col3:
        st.metric("CD8+ T细胞", f"{int(state['imm_cd8_count'])}")

    with col4:
        recist = "PR" if state['tum_volume'] < sim_data["initial_volume"] * 0.7 else "SD" if state['tum_volume'] < sim_data["initial_volume"] * 1.2 else "PD"
        st.metric("RECIST响应", recist)

    with col5:
        st.metric("分子亚型", state.get("sub_molecular_subtype", "BLIS"))

    # Animated tumor growth
    st.markdown("#### 📈 肿瘤生长动态")

    steps = [h["step"] for h in history]
    volumes = [h["tum_volume"] for h in history]
    phases = [h["ied_phase"] for h in history]

    # Color by phase
    colors = []
    for p in phases:
        if p == "elimination":
            colors.append("#27ae60")
        elif p == "equilibrium":
            colors.append("#f39c12")
        else:
            colors.append("#e74c3c")

    tumor_fig = go.Figure(data=go.Scatter(
        x=steps, y=volumes, mode='lines+markers',
        marker=dict(color=colors, size=8),
        line=dict(color='#c41e3a', width=3)
    ))
    tumor_fig.update_layout(
        xaxis_title='仿真步数',
        yaxis_title='肿瘤体积 (mm³)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#ecf0f1',
        height=350
    )
    st.plotly_chart(tumor_fig, use_container_width=True)

    # Immune dynamics
    st.markdown("#### 🛡️ 免疫细胞动态")

    cd8_vals = [h["imm_cd8_count"] for h in history]
    activation_vals = [h.get("imm_t_cell_activation", 0.3) for h in history]

    immune_fig = make_subplots(rows=1, cols=2)

    immune_fig.add_trace(go.Scatter(
        x=steps, y=cd8_vals, mode='lines', name='CD8+ T细胞',
        line=dict(color='#27ae60', width=2)
    ), row=1, col=1)

    immune_fig.add_trace(go.Scatter(
        x=steps, y=activation_vals, mode='lines', name='T细胞激活',
        line=dict(color='#3498db', width=2)
    ), row=1, col=2)

    immune_fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#ecf0f1',
        height=350,
        showlegend=True
    )
    st.plotly_chart(immune_fig, use_container_width=True)

    # Phase timeline
    st.markdown("#### 📋 免疫编辑阶段时间轴")

    phase_counts = {}
    for p in phases:
        phase_counts[p] = phase_counts.get(p, 0) + 1

    phase_cols = st.columns(3)
    phase_colors_map = {"elimination": "#27ae60", "equilibrium": "#f39c12", "escape": "#e74c3c"}

    for i, (p, count) in enumerate(phase_counts.items()):
        with phase_cols[i]:
            st.markdown(f"""
            <div style="background: {phase_colors_map[p]}; padding: 10px; border-radius: 8px; text-align: center;">
                <div style="font-size: 1.5em; color: white;">{p.upper()}</div>
                <div style="color: white;">{count} 步 ({count/len(history)*100:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)

    # Treatment response
    st.markdown("#### 💉 治疗响应评估")

    if sim_data.get("treatment"):
        st.info(f"治疗方案: {', '.join(sim_data['treatment'])}")

        response_data = {
            "指标": ["RECIST响应", "肿瘤变化", "PFS预估", "免疫激活"],
            "数值": [recist,
                     f"{(state['tum_volume']-sim_data['initial_volume'])/sim_data['initial_volume']*100:+.1f}%",
                     f"{len(history) * 0.5:.1f} 月",
                     f"{state['imm_t_cell_activation']:.2f}"]
        }

        import pandas as pd
        st.dataframe(pd.DataFrame(response_data), use_container_width=True, hide_index=True)

    # Export
    st.markdown("---")
    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        if st.button("📄 生成动态HTML报告", use_container_width=True):
            html = generate_nature_html_report(data)
            filename = f"simulacrum_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            Path(filename).write_text(html, encoding="utf-8")
            st.success(f"✅ 报告已保存: {filename}")
            with open(filename, "r", encoding="utf-8") as f:
                st.download_button("⬇️ 下载动态报告", f.read(), file_name=filename, mime="text/html")

    with col_exp2:
        import json
        st.download_button(
            "⬇️ 下载仿真数据",
            json.dumps(data, indent=2, ensure_ascii=False),
            file_name=f"simulacrum_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Quick simulation (no sidebar params)
st.markdown("---")
st.markdown("#### ⚡ 快速演示")

if st.button("🔄 运行示例仿真 (50步)", use_container_width=True):
    # Default params
    st.session_state["sim_params"] = {
        "initial_volume": 50,
        "growth_rate": 0.027,
        "cd8_initial": 100,
        "subtype": "BLIS",
        "treatment": ["化疗 (紫杉醇)", "免疫治疗 (PD-1)"],
        "treatment_start": 10,
        "n_steps": 50
    }
    st.session_state["sim_running"] = True
    st.rerun()

# Navigation
st.markdown("---")
col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
with col_nav1:
    if st.button("🏠 返回首页", use_container_width=True):
        st.switch_page("Home.py")
with col_nav3:
    if st.button("📄 报告导出 →", use_container_width=True):
        st.switch_page("pages/5_Report_Export.py")