"""Drug Prediction Page - Streamlit UI.

ADMET properties, efficacy prediction, molecular optimization.
"""

import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"D:\IGEM集成方案")
sys.path.insert(0, str(PROJECT_ROOT / "confluencia-2.0-drug"))

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

import plotly.graph_objects as go
from datetime import datetime

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
                smiles_input = smiles
                st.rerun()

# Analysis type
st.markdown("#### 🔬 分析类型")
analysis_type = st.radio(
    "",
    ["ADMET 属性", "疗效预测", "PK 模拟"],
    horizontal=True
)

# Run analysis
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    run_btn = st.button("🚀 开始预测", type="primary", use_container_width=True, disabled=not smiles_input)

if run_btn and smiles_input:
    with st.spinner("预测中..."):
        try:
            # Use utility helper for Python 3.13 dataclass compatibility
            from utils import get_admet

            admet_mod = get_admet()

            # Map common drug names
            drug_smiles = {
                "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
                "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
                "paracetamol": "CC(=O)Nc1ccc(O)cc1",
                "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
                "doxorubicin": "CC1C(C(CC(O1)C2C3C(C4C(=C(C(=O)C5=C(C4=CC(=C(C5C(=O)C6=C(C=C3C=C2OC6=O)O)O)O)O)O)C(=O)O)O)O)O",
                "metformin": "CN(C)C(=N)NC(N)N"
            }

            smiles = drug_smiles.get(smiles_input.lower(), smiles_input)
            predictor = admet_mod.ADMETPredictor()
            result = predictor.predict(smiles)

            data = {
                "module": "drug",
                "backend": "local",
                "input": smiles_input,
                "smiles": result.smiles,
                "admet": {
                    "overall_risk": result.overall_risk,
                    "druglikeness": result.druglikeness_score,
                    "hERG_risk": result.hERG_risk,
                    "hepatotoxicity": result.hepatotoxicity_risk,
                    "caco2_permeability": result.caco2_permeability,
                    "aqueous_solubility": result.aqueous_solubility,
                    "cyp_total_risk": result.CYP_total_risk,
                    "ames_positive": result.AMES_positive,
                    "bbb_permeable": result.BBB_positive
                },
                "risk_categories": {
                    "overall": "LOW" if result.overall_risk < 0.3 else "MODERATE" if result.overall_risk < 0.6 else "HIGH",
                    "druglikeness": "PASS" if result.druglikeness_score > 0.5 else "FAIL"
                }
            }

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
    admet = data["admet"]
    risks = data["risk_categories"]

    st.markdown("---")
    st.markdown("""
    <div class="section-header">
        <h3>📊 预测结果</h3>
    </div>
    """, unsafe_allow_html=True)

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        risk_class = risks["overall"]
        risk_color = "🟢" if risk_class == "LOW" else "🟡" if risk_class == "MODERATE" else "🔴"
        st.metric(f"{risk_color} 总体风险", f"{admet['overall_risk']:.2f}", delta=risk_class)

    with col2:
        dl_color = "🟢" if risks["druglikeness"] == "PASS" else "🔴"
        st.metric(f"{dl_color} 药物相似性", f"{admet['druglikeness']:.2f}", delta=risks["druglikeness"])

    with col3:
        herg = admet["hERG_risk"]
        herg_class = "LOW" if herg < 0.3 else "MODERATE" if herg < 0.6 else "HIGH"
        st.metric("hERG 心脏毒性", f"{herg:.2f}", delta=herg_class)

    with col4:
        hepa = admet["hepatotoxicity"]
        st.metric("肝毒性", f"{hepa:.2f}")

    # ADMET radar
    st.markdown("#### 📊 ADMET 雷达图")

    radar_fig = go.Figure(data=go.Scatterpolar(
        r=[admet['overall_risk'], admet['hERG_risk'], admet['hepatotoxicity'],
           admet['cyp_total_risk'], admet['ames_positive'], 1-admet['druglikeness']],
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

    detail_data = {
        "属性": ["总体风险", "药物相似性", "hERG 心脏毒性", "肝毒性",
                "Caco2 通透性", "水溶性", "CYP 抑制风险", "AMES 致变性", "血脑屏障"],
        "数值": [
            f"{admet['overall_risk']:.3f}",
            f"{admet['druglikeness']:.3f}",
            f"{admet['hERG_risk']:.3f}",
            f"{admet['hepatotoxicity']:.3f}",
            f"{admet['caco2_permeability']:.3f}",
            f"{admet['aqueous_solubility']:.3f}",
            f"{admet['cyp_total_risk']:.3f}",
            "是" if admet['ames_positive'] > 0.5 else "否",
            "是" if admet['bbb_permeable'] > 0.5 else "否"
        ],
        "分级": [
            risks["overall"],
            risks["druglikeness"],
            "LOW" if admet['hERG_risk'] < 0.3 else "MODERATE" if admet['hERG_risk'] < 0.6 else "HIGH",
            "LOW" if admet['hepatotoxicity'] < 0.3 else "MODERATE" if admet['hepatotoxicity'] < 0.6 else "HIGH",
            "PASS" if admet['caco2_permeability'] > 0.5 else "FAIL",
            "GOOD" if admet['aqueous_solubility'] > 0.5 else "POOR",
            "LOW" if admet['cyp_total_risk'] < 0.3 else "MODERATE" if admet['cyp_total_risk'] < 0.6 else "HIGH",
            "POSITIVE" if admet['ames_positive'] > 0.5 else "NEGATIVE",
            "PERMEABLE" if admet['bbb_permeable'] > 0.5 else "IMPERMEABLE"
        ]
    }

    import pandas as pd
    df = pd.DataFrame(detail_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Export
    st.markdown("---")
    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        if st.button("📄 生成HTML报告", use_container_width=True):
            html = generate_nature_html_report(data)
            filename = f"drug_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            Path(filename).write_text(html, encoding="utf-8")
            st.success(f"✅ 报告已保存: {filename}")
            with open(filename, "r", encoding="utf-8") as f:
                st.download_button("⬇️ 下载HTML", f.read(), file_name=filename, mime="text/html")

    with col_exp2:
        import json
        st.download_button(
            "⬇️ 下载JSON数据",
            json.dumps(data, indent=2, ensure_ascii=False),
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