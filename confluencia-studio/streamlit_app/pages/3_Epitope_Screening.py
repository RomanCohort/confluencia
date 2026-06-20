"""Epitope Screening Page - Streamlit UI.

MHC binding prediction for vaccine design.
Uses confluencia skill API for all backend computation.
"""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import json

from utils import (
    epitope_predict, generate_html_report, save_html_report,
    set_backend, get_backend,
)

st.set_page_config(page_title="Epitope Screening - Confluencia", page_icon="🧬", layout="wide")

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
    <h2>🧬 表位筛选</h2>
    <p style="color: #7f8c8d;">预测肽段与MHC分子的结合亲和力</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ 预测选项")

    allele = st.selectbox(
        "MHC 等位基因",
        ["HLA-A*02:01", "HLA-A*03:01", "HLA-B*07:02", "HLA-B*08:01",
         "HLA-C*07:01", "HLA-DRB1*01:01", "HLA-DRB1*04:01"]
    )

    backend = st.radio(
        "预测引擎",
        ["本地模型", "NetMHCpan API"]
    )

    # Map UI to backend setting
    backend_name = "local" if backend == "本地模型" else "netmhcpan"
    set_backend("epitope", backend_name)

    st.markdown("---")
    st.markdown("### 📚 MHC基础知识")

    with st.expander("什么是MHC？"):
        st.markdown("""
        **MHC (Major Histocompatibility Complex)** 是免疫系统识别抗原的关键分子。

        - **MHC I类**: HLA-A/B/C，呈递给CD8+ T细胞
        - **MHC II类**: HLA-DR/DQ/DP，呈递给CD4+ T细胞

        高结合亲和力 = 更好的免疫反应
        """)

    with st.expander("什么是表位？"):
        st.markdown("""
        **表位 (Epitope)** 是能被免疫系统识别的抗原片段。

        - **长度**: 通常8-11个氨基酸 (MHC I类)
        - **结合阈值**: IC50 < 50 nM 为强结合
        """)

# Input section
st.markdown("#### 📋 输入肽段序列")

sequence = st.text_input(
    "肽段序列 (氨基酸):",
    placeholder="例如: GILGFVFTL (流感病毒表位)",
    help="输入8-15个氨基酸序列"
)

st.markdown("**常见疫苗表位示例:**")
examples = {
    "流感病毒": "GILGFVFTL",
    "乙肝病毒": "FLGGFLVAP",
    "黑色素瘤": "EAAGIGILTV",
    "HIV病毒": "SLYNTVATL",
    "EB病毒": "GLCTLVAML"
}

cols = st.columns(5)
for i, (name, seq) in enumerate(examples.items()):
    with cols[i]:
        if st.button(name, key=f"ep_{i}", use_container_width=True):
            st.session_state["epitope_sequence"] = seq
            st.rerun()

# Restore from session state
if st.session_state.get("epitope_sequence"):
    sequence = st.session_state["epitope_sequence"]
    st.session_state["epitope_sequence"] = None

# Run prediction via skill API
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    run_btn = st.button("🚀 开始预测", type="primary", use_container_width=True, disabled=not sequence)

if run_btn and sequence:
    with st.spinner("预测中..."):
        try:
            data = epitope_predict(sequence, allele)

            st.session_state["epitope_data"] = data
            st.session_state["epitope_done"] = True

            backend_used = data.get("backend", "local")
            st.success(f"预测完成 (引擎: {backend_used})")

        except Exception as e:
            import traceback
            st.error(f"预测出错: {e}")
            with st.expander("详细错误信息"):
                st.code(traceback.format_exc())

# Display results
if st.session_state.get("epitope_done") and st.session_state.get("epitope_data"):
    data = st.session_state["epitope_data"]

    st.markdown("---")
    st.markdown("""
    <div class="section-header">
        <h3>📊 预测结果</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    score = data.get("binding_score", 0)
    aff_class = data.get("binding_affinity", "WEAK")

    with col1:
        st.metric("结合评分", f"{score:.3f}")

    with col2:
        aff_color = "🟢" if aff_class == "STRONG" else "🟡" if aff_class == "MODERATE" else "🔴"
        st.metric(f"{aff_color} 结合亲和力", aff_class)

    with col3:
        st.metric("MHC等位基因", data.get("allele", allele))

    with col4:
        st.metric("序列长度", f"{data.get('length', len(sequence))} aa")

    # Binding gauge
    st.markdown("#### 📊 结合评分仪表")

    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "#27ae60" if score > 0.7 else "#f39c12" if score > 0.4 else "#e74c3c"},
            'steps': [
                {'range': [0, 0.3], 'color': "#e74c3c"},
                {'range': [0.3, 0.7], 'color': "#f39c12"},
                {'range': [0.7, 1], 'color': "#27ae60"}
            ]
        },
        title={'text': "MHC结合评分"}
    ))
    gauge_fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#ecf0f1',
        height=350
    )
    st.plotly_chart(gauge_fig, use_container_width=True)

    # Interpretation
    st.markdown("#### 📝 结果解读")

    allele_name = data.get("allele", allele)
    if score > 0.8:
        st.success(f"""
        ✅ **强结合表位**
        - 该肽段与 {allele_name} 具有高亲和力结合
        - 预测IC50 < 50 nM
        - 推荐作为疫苗候选
        """)
    elif score > 0.5:
        st.warning(f"""
        ⚠️ **中等结合表位**
        - 该肽段与 {allele_name} 具有中等亲和力结合
        - 预测IC50 50-500 nM
        - 可作为备选候选
        """)
    else:
        st.error(f"""
        ❌ **弱/无结合表位**
        - 该肽段与 {allele_name} 结合亲和力低
        - 不推荐作为疫苗候选
        - 建议寻找替代表位
        """)

    # Export
    st.markdown("---")
    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        if st.button("📄 生成HTML报告", key="epi_html", use_container_width=True):
            html = generate_html_report(data)
            filepath = save_html_report(html)
            with open(filepath, "r", encoding="utf-8") as f:
                st.download_button("下载HTML报告", f.read(),
                                   file_name=Path(filepath).name, mime="text/html")

    with col_exp2:
        st.download_button(
            "⬇️ 下载JSON数据",
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            file_name=f"epitope_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Navigation
st.markdown("---")
col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
with col_nav1:
    if st.button("🏠 返回首页", use_container_width=True):
        st.switch_page("Home.py")
with col_nav3:
    if st.button("💊 药物预测 →", use_container_width=True):
        st.switch_page("pages/2_Drug_Prediction.py")
