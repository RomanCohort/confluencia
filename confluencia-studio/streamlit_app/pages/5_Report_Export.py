"""Report Export Page - Streamlit UI.

Generate and export analysis reports.
Uses confluencia skill API for report generation.
"""

import streamlit as st
from pathlib import Path
from datetime import datetime
import json

from utils import generate_html_report, save_html_report

st.set_page_config(page_title="Report Export - Confluencia", page_icon="📄", layout="wide")

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
    <h2>📄 报告导出</h2>
    <p style="color: #7f8c8d;">查看和导出分析结果</p>
</div>
""", unsafe_allow_html=True)

# Check for analysis data
has_circrna = st.session_state.get("analysis_done") and st.session_state.get("analysis_data")
has_drug = st.session_state.get("drug_done") and st.session_state.get("drug_data")
has_epitope = st.session_state.get("epitope_done") and st.session_state.get("epitope_data")
has_simulacrum = st.session_state.get("sim_initialized") and st.session_state.get("sim_data")
has_joint = st.session_state.get("joint_done") and st.session_state.get("joint_data")

if not (has_circrna or has_drug or has_epitope or has_simulacrum or has_joint):
    st.info("暂无分析数据。请先进行以下分析：")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("🔬 circRNA", use_container_width=True):
            st.switch_page("pages/1_CircRNA_Analysis.py")
    with col2:
        if st.button("💊 药物", use_container_width=True):
            st.switch_page("pages/2_Drug_Prediction.py")
    with col3:
        if st.button("🧬 表位", use_container_width=True):
            st.switch_page("pages/3_Epitope_Screening.py")
    with col4:
        if st.button("🎮 TNBC", use_container_width=True):
            st.switch_page("pages/4_TNBC_Simulator.py")
    with col5:
        if st.button("🔗 联合", use_container_width=True):
            st.switch_page("pages/6_Joint_Analysis.py")

else:
    st.markdown("#### 📊 分析记录")

    # Summary cards
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if has_circrna:
            data = st.session_state["analysis_data"]
            st.metric("circRNA", data.get("length", 0), delta="已完成")
        else:
            st.metric("circRNA", "--", delta="未完成")

    with col2:
        if has_drug:
            data = st.session_state["drug_data"]
            st.metric("药物", data.get("input", "--"), delta="已完成")
        else:
            st.metric("药物", "--", delta="未完成")

    with col3:
        if has_epitope:
            data = st.session_state["epitope_data"]
            st.metric("表位", f"{data.get('binding_score', 0):.2f}", delta="已完成")
        else:
            st.metric("表位", "--", delta="未完成")

    with col4:
        if has_simulacrum:
            data = st.session_state["sim_data"]
            state = data.get("state", data.get("summary", {}))
            st.metric("TNBC", f"D{st.session_state.get('sim_day', 0)}", delta="已完成")
        else:
            st.metric("TNBC", "--", delta="未完成")

    with col5:
        if has_joint:
            data = st.session_state["joint_data"]
            joint = data.get("joint", {})
            st.metric("联合", f"{joint.get('combined_score', 0):.2f}", delta="已完成")
        else:
            st.metric("联合", "--", delta="未完成")

    # Export options
    st.markdown("---")
    st.markdown("#### 📤 导出选项")

    st.markdown("**选择导出格式:**")
    export_format = st.radio(
        "",
        ["HTML 可视化报告", "JSON 数据文件", "完整打包 (HTML+JSON)"],
        horizontal=True
    )

    st.markdown("**选择导出内容:**")
    export_options = []
    if has_circrna:
        export_options.append("circRNA分析")
    if has_drug:
        export_options.append("药物预测")
    if has_epitope:
        export_options.append("表位筛选")
    if has_simulacrum:
        export_options.append("TNBC仿真")
    if has_joint:
        export_options.append("联合分析")

    export_content = st.multiselect(
        "",
        ["circRNA分析", "药物预测", "表位筛选", "TNBC仿真", "联合分析"],
        default=export_options
    )

    # Generate report
    st.markdown("")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("📦 生成报告包", type="primary", use_container_width=True):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            report_files = []

            if "circRNA分析" in export_content and has_circrna:
                data = st.session_state["analysis_data"]
                html = generate_html_report(data)
                filepath = save_html_report(html)
                report_files.append(("circRNA", filepath, data))

            if "药物预测" in export_content and has_drug:
                data = st.session_state["drug_data"]
                html = generate_html_report(data)
                filepath = save_html_report(html)
                report_files.append(("药物", filepath, data))

            if "表位筛选" in export_content and has_epitope:
                data = st.session_state["epitope_data"]
                html = generate_html_report(data)
                filepath = save_html_report(html)
                report_files.append(("表位", filepath, data))

            if "TNBC仿真" in export_content and has_simulacrum:
                data = st.session_state["sim_data"]
                data["history"] = st.session_state.get("sim_history", [])
                html = generate_html_report(data)
                filepath = save_html_report(html)
                report_files.append(("TNBC", filepath, data))

            if "联合分析" in export_content and has_joint:
                data = st.session_state["joint_data"]
                html = generate_html_report(data)
                filepath = save_html_report(html)
                report_files.append(("联合", filepath, data))

            if report_files:
                st.success(f"已生成 {len(report_files)} 个报告")

                st.markdown("#### 📁 下载文件")

                for module, html_path, data in report_files:
                    st.markdown(f"**{module} 分析报告**")
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        with open(html_path, "r", encoding="utf-8") as f:
                            st.download_button(f"HTML报告", f.read(),
                                               file_name=Path(html_path).name, mime="text/html")
                    with col_dl2:
                        st.download_button(f"JSON数据",
                                           json.dumps(data, indent=2, ensure_ascii=False, default=str),
                                           file_name=f"{module.lower()}_data_{timestamp}.json",
                                           mime="application/json")
            else:
                st.warning("请选择要导出的内容")

# Clear session data
st.markdown("---")
st.markdown("#### 🗑️ 数据管理")

col_clear1, col_clear2 = st.columns(2)
with col_clear1:
    if st.button("清除当前分析数据", use_container_width=True):
        keys_to_clear = ["analysis_data", "analysis_done", "drug_data", "drug_done",
                        "epitope_data", "epitope_done", "sim_data", "sim_initialized",
                        "sim_history", "joint_data", "joint_done"]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.success("已清除分析数据")
        st.rerun()

with col_clear2:
    if st.button("清除示例序列", use_container_width=True):
        for key in ["circrna_sequence", "drug_smiles_input", "epitope_sequence"]:
            if key in st.session_state:
                del st.session_state[key]
        st.success("已清除示例序列")

# Navigation
st.markdown("---")
col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
with col_nav1:
    if st.button("🏠 返回首页", use_container_width=True):
        st.switch_page("Home.py")
