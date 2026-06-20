"""TNBC Simulator Page - Streamlit UI.

Digital twin simulation for Triple-Negative Breast Cancer.
Uses confluencia skill API for all backend computation - TNBCSimulacrum agent.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import json

from utils import (
    simulacrum_init, simulacrum_step, simulacrum_administer_drug, simulacrum_report,
    generate_html_report, save_html_report,
    circrna_full_analysis, drug_admet,
)

st.set_page_config(page_title="TNBC Simulator - Confluencia", page_icon="🎮", layout="wide")

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
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section-header">
    <h2>🎮 TNBC 数字孪生仿真器</h2>
    <p style="color: #7f8c8d;">三阴性乳腺癌数字孪生模拟 - 模拟肿瘤生长、免疫编辑、治疗响应</p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Simulation parameters
with st.sidebar:
    st.markdown("### 🎯 仿真配置")

    subtype = st.selectbox(
        "分子亚型",
        ["BLIS (基底样免疫抑制)", "BLIA (基底样免疫激活)", "IM (免疫调节型)", "LAR (管腔雄激素受体型)"],
        index=0
    )

    # Parse subtype code
    subtype_code = subtype.split()[0]

    brca_mutation = st.checkbox("BRCA1/2 突变", value=False)

    st.markdown("---")
    st.markdown("### 💊 治疗方案")

    treatment = st.multiselect(
        "选择治疗方式",
        ["化疗 (Gemcitabine)", "免疫治疗 (PD-1/PD-L1)", "circRNA疫苗", "联合治疗"],
        default=["化疗 (Gemcitabine)"]
    )

    treatment_start_day = st.slider("治疗开始日期 (天)", 0, 30, 5)

    st.markdown("---")
    st.markdown("### 🧬 circRNA疫苗配置")

    circrna_seq = st.text_area(
        "circRNA序列 (可选)",
        placeholder="AUGCGCGCGUAU...",
        height=80,
        help="用于circRNA疫苗治疗的序列"
    )

    st.markdown("---")
    st.markdown("### 📊 仿真时长")

    n_days = st.slider("仿真天数", 10, 365, 100)

    st.markdown("---")
    st.markdown("### 📚 TNBC知识")

    with st.expander("TNBC分子亚型"):
        st.markdown("""
        **三阴性乳腺癌 (TNBC)** 四种亚型：

        - **BLIS**: 基底样免疫抑制型，预后较差
        - **BLIA**: 基底样免疫激活型，免疫浸润高
        - **IM**: 免疫调节型，对免疫治疗敏感
        - **LAR**: 管腔雄激素受体型，AR靶向治疗
        """)

    with st.expander("免疫编辑三阶段"):
        st.markdown("""
        **肿瘤免疫编辑 (Immunoediting)** 三阶段：

        1. **清除 (Elimination)**: 免疫系统识别并清除肿瘤
        2. **平衡 (Equilibrium)**: 肿瘤与免疫系统动态平衡
        3. **逃逸 (Escape)**: 肛瘤逃避免疫监视，快速生长
        """)

# Initialize simulation
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    init_btn = st.button("🚀 初始化数字孪生", type="primary", use_container_width=True)

if init_btn:
    with st.spinner("初始化数字孪生..."):
        try:
            # Initialize via skill API
            init_data = simulacrum_init(subtype_code, brca_mutation)

            st.session_state["sim_initialized"] = True
            st.session_state["sim_subtype"] = subtype_code
            st.session_state["sim_day"] = 0
            st.session_state["sim_data"] = init_data
            st.session_state["sim_history"] = []

            st.success(f"数字孪生初始化成功！亚型: {subtype_code}")

        except Exception as e:
            import traceback
            st.error(f"初始化出错: {e}")
            with st.expander("详细错误信息"):
                st.code(traceback.format_exc())

# Run simulation step by step
if st.session_state.get("sim_initialized"):
    st.markdown("---")
    st.markdown("### 📈 仿真控制")

    col_step1, col_step2, col_step3 = st.columns(3)

    with col_step1:
        step_size = st.number_input("推进步数 (天)", 1, 50, 10)

    with col_step2:
        step_btn = st.button("▶️ 推进仿真", use_container_width=True)

    with col_step3:
        if treatment:
            drug_btn = st.button("💊 给药治疗", use_container_width=True)

    # Step simulation
    if step_btn:
        with st.spinner(f"推进 {step_size} 天..."):
            try:
                step_data = simulacrum_step(step_size)

                st.session_state["sim_day"] += step_size
                st.session_state["sim_data"] = step_data

                # Record history
                history_entry = {
                    "day": st.session_state["sim_day"],
                    "state": step_data.get("state", step_data),
                }
                st.session_state["sim_history"].append(history_entry)

                st.success(f"推进完成！当前: 第 {st.session_state['sim_day']} 天")

            except Exception as e:
                st.error(f"推进出错: {e}")

    # Administer drug
    if drug_btn:
        with st.spinner("给药..."):
            try:
                drug_name = "gemcitabine"
                dose = 1000.0

                drug_data = simulacrum_administer_drug(drug_name, dose)

                st.success(f"给药完成: {drug_name} {dose} mg/m2")

                # Record treatment
                st.session_state["sim_history"].append({
                    "day": st.session_state["sim_day"],
                    "event": "drug_administered",
                    "drug": drug_name,
                    "dose": dose,
                })

            except Exception as e:
                st.error(f"给药出错: {e}")

    # Display current state
    st.markdown("---")
    st.markdown("""
    <div class="section-header">
        <h3>📊 仿真状态</h3>
    </div>
    """, unsafe_allow_html=True)

    sim_data = st.session_state.get("sim_data", {})
    state = sim_data.get("state", sim_data.get("summary", {}))

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        volume = state.get("volume", state.get("volume_mm3", 0))
        st.metric("肿瘤体积", f"{volume:.1f} mm³")

    with col2:
        recist = state.get("recist", state.get("recist", "SD"))
        recist_color = "🟢" if recist in ["CR", "PR"] else "🟡" if recist == "SD" else "🔴"
        st.metric(f"{recist_color} RECIST", recist)

    with col3:
        phase = state.get("phase", state.get("immunoediting_phase", "?"))
        phase_color = "🟢" if phase == "elimination" else "🟡" if phase == "equilibrium" else "🔴"
        st.metric(f"{phase_color} 免疫编辑", phase)

    with col4:
        pd_l1 = state.get("pd_l1_cps", 0)
        st.metric("PD-L1 CPS", f"{pd_l1:.1f}")

    # Tumor growth curve
    st.markdown("#### 📉 肿瘤生长曲线")

    history = st.session_state.get("sim_history", [])
    if history:
        days = [h.get("day", 0) for h in history]
        volumes = [h.get("state", {}).get("volume", h.get("state", {}).get("volume_mm3", 0)) for h in history]

        # Add initial point if needed
        if not days or days[0] != 0:
            days = [0] + days
            volumes = [state.get("initial_volume", 100)] + volumes

        growth_fig = go.Figure()
        growth_fig.add_trace(go.Scatter(
            x=days, y=volumes, mode='lines+markers',
            line=dict(color='#c41e3a', width=3),
            marker=dict(size=8),
            name='肿瘤体积'
        ))

        # Mark treatment events
        treatment_days = [h.get("day") for h in history if h.get("event") == "drug_administered"]
        for td in treatment_days:
            idx = days.index(td) if td in days else -1
            if idx >= 0:
                growth_fig.add_trace(go.Scatter(
                    x=[td], y=[volumes[idx]],
                    mode='markers',
                    marker=dict(size=15, symbol='star', color='#f39c12'),
                    name='给药'
                ))

        growth_fig.update_layout(
            xaxis_title='天数',
            yaxis_title='肿瘤体积 (mm³)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ecf0f1',
            height=400
        )
        st.plotly_chart(growth_fig, use_container_width=True)
    else:
        st.info("推进仿真以生成曲线")

    # Generate report
    st.markdown("---")
    col_rep1, col_rep2 = st.columns(2)

    with col_rep1:
        if st.button("📄 生成完整报告", use_container_width=True):
            with st.spinner("生成报告..."):
                try:
                    report_data = simulacrum_report()
                    report_data["history"] = history

                    html = generate_html_report(report_data, f"TNBC {subtype_code} Report")
                    filepath = save_html_report(html)

                    with open(filepath, "r", encoding="utf-8") as f:
                        st.download_button("下载HTML报告", f.read(),
                                           file_name=Path(filepath).name, mime="text/html")
                    st.success("报告生成完成")

                except Exception as e:
                    st.error(f"报告生成出错: {e}")

    with col_rep2:
        full_data = {
            "module": "simulacrum",
            "subtype": subtype_code,
            "day": st.session_state.get("sim_day", 0),
            "state": state,
            "history": history,
        }
        st.download_button(
            "⬇️ 下载JSON数据",
            json.dumps(full_data, indent=2, ensure_ascii=False, default=str),
            file_name=f"tnbc_sim_{subtype_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Navigation
st.markdown("---")
col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
with col_nav1:
    if st.button("🏠 返回首页", use_container_width=True):
        st.switch_page("Home.py")
with col_nav3:
    if st.button("🔗 联合分析 →", use_container_width=True):
        st.switch_page("pages/6_Joint_Analysis.py")