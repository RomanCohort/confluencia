"""Drug Prediction Page - Streamlit UI.

ADMET properties, efficacy prediction, molecular optimization.
Uses confluencia skill API for all backend computation.
"""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import json

from utils import (
    drug_admet, generate_html_report, save_html_report,
    get_drug_smiles_mapping,
)

st.set_page_config(page_title="Drug Prediction - Confluencia", page_icon="💊", layout="wide")

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
    <h2>💊 药物 ADMET 预测</h2>
    <p style="color: #7f8c8d;">预测分子的吸收、分布、代谢、排泄、毒性属性</p>
</div>
""", unsafe_allow_html=True)

# Input section
col_input, col_examples = st.columns([2, 1])

with col_input:
    st.markdown("#### 📋 输入分子 (SMILES 或药物名)")

    smiles_input = st.text_input(
        "SMILES 字符串或药物名:",
        placeholder="例如: aspirin 或 CC(=O)Oc1ccccc1C(=O)O",
        help="不知道SMILES？直接输入药物名"
    )

with col_examples:
    st.markdown("#### 💊 常见药物")
    drug_examples = {
        "阿司匹林 (Aspirin)": "aspirin",
        "布洛芬 (Ibuprofen)": "ibuprofen",
        "对乙酰氨基酚 (Paracetamol)": "paracetamol",
        "咖啡因 (Caffeine)": "caffeine",
        "阿霉素 (Doxorubicin)": "doxorubicin",
        "二甲双胍 (Metformin)": "metformin"
    }

    cols = st.columns(2)
    for i, (name, smiles) in enumerate(drug_examples.items()):
        with cols[i % 2]:
            if st.button(name, key=f"drug_{i}", use_container_width=True):
                st.session_state["drug_smiles_input"] = smiles
                st.rerun()

    # Restore from session state
    if st.session_state.get("drug_smiles_input"):
        smiles_input = st.session_state["drug_smiles_input"]
        st.session_state["drug_smiles_input"] = None

# Analysis type
st.markdown("#### 🔬 分析类型")
analysis_type = st.radio(
    "",
    ["ADMET 属性", "疗效预测", "PK 模拟"],
    horizontal=True
)

# Run analysis via skill API
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    run_btn = st.button("🚀 开始预测", type="primary", use_container_width=True, disabled=not smiles_input)

if run_btn and smiles_input:
    with st.spinner("预测中..."):
        try:
            data = drug_admet(smiles_input)

            st.session_state["drug_data"] = data
            st.session_state["drug_done"] = True
            st.success("✅ 预测完成！")

        except Exception as e:
            import traceback
            st.error(f"预测出错: {e}")
            with st.expander("详细错误信息"):
                st.code(traceback.format_exc())

# Display results
if st.session_state.get("drug_done") and st.session_state.get("drug_data"):
    data = st.session_state["drug_data"]
    admet = data.get("admet", {})
    risks = data.get("risk_categories", {})

    st.markdown("---")
    st.markdown("""
    <div class="section-header">
        <h3>📊 预测结果</h3>
    </div>
    """, unsafe_allow_html=True)

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        risk_class = risks.get("overall", "N/A")
        risk_color = "🟢" if risk_class == "LOW" else "🟡" if risk_class == "MODERATE" else "🔴"
        st.metric(f"{risk_color} 总体风险", f"{admet.get('overall_risk', 0):.2f}", delta=risk_class)

    with col2:
        dl_status = risks.get("druglikeness", "N/A")
        dl_color = "🟢" if dl_status == "PASS" else "🔴"
        st.metric(f"{dl_color} 药物相似性", f"{admet.get('druglikeness', 0):.2f}", delta=dl_status)

    with col3:
        herg = admet.get("hERG_risk", 0)
        herg_class = "LOW" if herg < 0.3 else "MODERATE" if herg < 0.6 else "HIGH"
        st.metric("hERG 心脏毒性", f"{herg:.2f}", delta=herg_class)

    with col4:
        hepa = admet.get("hepatotoxicity", 0)
        st.metric("肝毒性", f"{hepa:.2f}")

    # ADMET radar
    st.markdown("#### 📊 ADMET 雷达图")

    radar_fig = go.Figure(data=go.Scatterpolar(
        r=[admet.get('overall_risk',0), admet.get('hERG_risk',0), admet.get('hepatotoxicity',0),
           admet.get('cyp_total_risk',0), admet.get('ames_positive',0), 1-admet.get('druglikeness',0)],
        theta=['总体风险', 'hERG', '肝毒性', 'CYP抑制', 'AMES致变', '药物相似性(逆)'],
        fill='toself',
        marker_color='#e94560'
    ))
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#ecf0f1',
        height=400
    )
    st.plotly_chart(radar_fig, use_container_width=True)

    # Detailed table
    st.markdown("#### 📋 详细属性")

    import pandas as pd
    detail_data = {
        "属性": ["总体风险", "药物相似性", "hERG 心脏毒性", "肝毒性",
                "Caco2 通透性", "水溶性", "CYP 抑制风险", "AMES 致变性", "血脑屏障"],
        "数值": [
            f"{admet.get('overall_risk',0):.3f}",
            f"{admet.get('druglikeness',0):.3f}",
            f"{admet.get('hERG_risk',0):.3f}",
            f"{admet.get('hepatotoxicity',0):.3f}",
            f"{admet.get('caco2_permeability',0):.3f}",
            f"{admet.get('aqueous_solubility',0):.3f}",
            f"{admet.get('cyp_total_risk',0):.3f}",
            "是" if admet.get('ames_positive',0) > 0.5 else "否",
            "是" if admet.get('bbb_permeable',0) > 0.5 else "否"
        ],
        "分级": [
            risks.get("overall", "N/A"),
            risks.get("druglikeness", "N/A"),
            "LOW" if admet.get('hERG_risk',0) < 0.3 else "MODERATE" if admet.get('hERG_risk',0) < 0.6 else "HIGH",
            "LOW" if admet.get('hepatotoxicity',0) < 0.3 else "MODERATE" if admet.get('hepatotoxicity',0) < 0.6 else "HIGH",
            "PASS" if admet.get('caco2_permeability',0) > 0.5 else "FAIL",
            "GOOD" if admet.get('aqueous_solubility',0) > 0.5 else "POOR",
            "LOW" if admet.get('cyp_total_risk',0) < 0.3 else "MODERATE" if admet.get('cyp_total_risk',0) < 0.6 else "HIGH",
            "POSITIVE" if admet.get('ames_positive',0) > 0.5 else "NEGATIVE",
            "PERMEABLE" if admet.get('bbb_permeable',0) > 0.5 else "IMPERMEABLE"
        ]
    }

    df = pd.DataFrame(detail_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Export
    st.markdown("---")
    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        if st.button("📄 生成HTML报告", key="drug_html", use_container_width=True):
            html = generate_html_report(data)
            filepath = save_html_report(html)
            with open(filepath, "r", encoding="utf-8") as f:
                st.download_button("下载HTML报告", f.read(),
                                   file_name=Path(filepath).name, mime="text/html")

    with col_exp2:
        st.download_button(
            "⬇️ 下载JSON数据",
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            file_name=f"drug_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Navigation
st.markdown("---")
col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
with col_nav1:
    if st.button("🏠 返回首页", use_container_width=True):
        st.switch_page("Home.py")
with col_nav3:
    if st.button("🔬 circRNA分析 →", use_container_width=True):
        st.switch_page("pages/1_CircRNA_Analysis.py")
