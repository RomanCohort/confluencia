"""Joint Analysis Page - Streamlit UI.

Combined circRNA + Drug evaluation for synergistic therapy design.
Uses confluencia skill API for all backend computation.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import json

from utils import (
    joint_evaluate, generate_html_report, save_html_report,
    get_drug_smiles_mapping,
)

st.set_page_config(page_title="Joint Analysis - Confluencia", page_icon="🔗", layout="wide")

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
    <h2>🔗 联合分析</h2>
    <p style="color: #7f8c8d;">circRNA疫苗 + 药物联合治疗方案评估</p>
</div>
""", unsafe_allow_html=True)

# Input section
col_circrna, col_drug = st.columns(2)

with col_circrna:
    st.markdown("#### 🧬 circRNA疫苗")

    circrna_input = st.text_area(
        "circRNA 序列:",
        placeholder="例如: AUGCGCGCGUAUAGCGCGCG...",
        height=100,
        help="输入 circRNA 疫苗序列"
    )

    # Examples
    st.markdown("**示例序列:**")
    circrna_examples = {
        "GC平衡": "AUGCGCGCGUAUAGCGCGCGAUGCGCGCGUAUAGCGCGCG",
        "高GC": "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC",
        "富AU": "AUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAU"
    }

    cols = st.columns(3)
    for i, (name, seq) in enumerate(circrna_examples.items()):
        with cols[i]:
            if st.button(name, key=f"circrna_{i}", use_container_width=True):
                st.session_state["joint_circrna"] = seq
                st.rerun()

    # Restore from session state
    if st.session_state.get("joint_circrna"):
        circrna_input = st.session_state["joint_circrna"]
        st.session_state["joint_circrna"] = None

with col_drug:
    st.markdown("#### 💊 化疗药物")

    drug_input = st.text_input(
        "药物名或 SMILES:",
        placeholder="例如: gemcitabine 或 SMILES字符串"
    )

    # Examples
    st.markdown("**示例药物:**")
    drug_examples = {
        "吉西他滨": "gemcitabine",
        "紫杉醇": "paclitaxel",
        "阿霉素": "doxorubicin",
        "顺铂": "cisplatin"
    }

    cols = st.columns(4)
    for i, (name, drug) in enumerate(drug_examples.items()):
        with cols[i]:
            if st.button(name, key=f"drug_{i}", use_container_width=True):
                st.session_state["joint_drug"] = drug
                st.rerun()

    # Restore from session state
    if st.session_state.get("joint_drug"):
        drug_input = st.session_state["joint_drug"]
        st.session_state["joint_drug"] = None

# Sidebar info
with st.sidebar:
    st.markdown("### 📚 联合治疗说明")

    with st.expander("为什么联合治疗？"):
        st.markdown("""
        **circRNA疫苗 + 化疗药物** 联合治疗优势：

        1. **免疫激活**: circRNA疫苗激活抗肿瘤免疫
        2. **直接杀伤**: 化疗药物直接杀死肿瘤细胞
        3. **协同效应**: 免疫系统清除化疗后残留病灶
        4. **降低耐药**: 多途径攻击降低肿瘤逃逸
        """)

    with st.expander("协同评分解释"):
        st.markdown("""
        **Combined Score** 联合评分考虑：

        - circRNA 免疫安全性
        - 药物 ADMET 风险
        - 治疗机制互补性
        - PK 参数匹配度

        > 0.7 = HIGH synergy (推荐)
        > 0.5 = MODERATE (可考虑)
        < 0.5 = LOW (不推荐)
        """)

# Run joint analysis
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    run_btn = st.button("🚀 开始联合评估", type="primary", use_container_width=True,
                        disabled=not (circrna_input and drug_input))

if run_btn and circrna_input and drug_input:
    with st.spinner("评估中..."):
        try:
            # Resolve SMILES from drug name
            drug_mapping = get_drug_smiles_mapping()
            drug_smiles = drug_mapping.get(drug_input.lower(), drug_input)

            # Clean circRNA sequence
            seq_clean = circrna_input.upper().replace(" ", "").replace("\n", "")
            seq_clean = "".join(c for c in seq_clean if c in "AUGC")

            data = joint_evaluate(seq_clean, drug_smiles)

            st.session_state["joint_data"] = data
            st.session_state["joint_done"] = True
            st.success("评估完成！")

        except Exception as e:
            import traceback
            st.error(f"评估出错: {e}")
            with st.expander("详细错误信息"):
                st.code(traceback.format_exc())

# Display results
if st.session_state.get("joint_done") and st.session_state.get("joint_data"):
    data = st.session_state["joint_data"]

    st.markdown("---")
    st.markdown("""
    <div class="section-header">
        <h3>📊 联合评估结果</h3>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    joint = data.get("joint", {})
    circrna = data.get("circrna", {})
    drug = data.get("drug", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        combined = joint.get("combined_score", 0)
        st.metric("联合评分", f"{combined:.3f}")

    with col2:
        synergy = joint.get("synergy", "LOW")
        syn_color = "🟢" if synergy == "HIGH" else "🟡" if synergy == "MODERATE" else "🔴"
        st.metric(f"{syn_color} 协同等级", synergy)

    with col3:
        circrna_safety = circrna.get("safety_score", circrna.get("immune", {}).get("safety_score", 0))
        st.metric("circRNA安全性", f"{circrna_safety:.2f}")

    with col4:
        drug_risk = drug.get("overall_risk", drug.get("admet", {}).get("overall_risk", 0))
        st.metric("药物风险", f"{drug_risk:.2f}")

    # Visualization
    st.markdown("#### 📈 综合评分可视化")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Radar chart comparing components
        radar_fig = go.Figure()

        # circRNA scores
        immune = circrna.get("immune", {})
        radar_fig.add_trace(go.Scatterpolar(
            r=[circrna_safety, immune.get("innate_score", 0.5), 1-immune.get("tlr_combined", 0.3),
               circrna.get("torusfold", {}).get("stability", 0.5), circrna.get("torusfold", {}).get("translation", 0.5)],
            theta=['安全性', '免疫评分', 'TLR抑制', '稳定性', '翻译效率'],
            fill='toself',
            name='circRNA',
            marker_color='#27ae60'
        ))

        # Drug scores
        admet = drug.get("admet", {})
        radar_fig.add_trace(go.Scatterpolar(
            r=[1-drug_risk, admet.get("druglikeness", 0.8), 1-admet.get("hERG_risk", 0.3),
               1-admet.get("hepatotoxicity", 0.3), 1-admet.get("cyp_total_risk", 0.3)],
            theta=['低风险', '药物性', '无心脏毒', '无肝毒', '无CYP抑制'],
            fill='toself',
            name='药物',
            marker_color='#3498db'
        ))

        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ecf0f1',
            height=400,
            showlegend=True
        )
        st.plotly_chart(radar_fig, use_container_width=True)

    with chart_col2:
        # Combined score bar
        bar_fig = go.Figure(data=[
            go.Bar(
                name='circRNA贡献',
                x=['联合评分'],
                y=[circrna_safety * 0.4],
                marker_color='#27ae60'
            ),
            go.Bar(
                name='药物贡献',
                x=['联合评分'],
                y=[(1-drug_risk) * 0.4],
                marker_color='#3498db'
            ),
            go.Bar(
                name='协同加成',
                x=['联合评分'],
                y=[combined - circrna_safety * 0.4 - (1-drug_risk) * 0.4],
                marker_color='#f39c12'
            )
        ])
        bar_fig.update_layout(
            barmode='stack',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ecf0f1',
            height=400,
            yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    # Recommendation
    st.markdown("#### 📝 治疗建议")

    recommendation = data.get("recommendation", "")
    if synergy == "HIGH":
        st.success(f"""
        ✅ **推荐联合治疗**

        {recommendation}

        - circRNA疫苗安全性高 (评分: {circrna_safety:.2f})
        - 药物风险低 (评分: {drug_risk:.2f})
        - 协同效应强，建议推进临床试验
        """)
    elif synergy == "MODERATE":
        st.warning(f"""
        ⚠️ **可考虑联合治疗**

        {recommendation}

        - 需优化 circRNA 序列或选择替代药物
        - 监控免疫反应和药物毒性
        """)
    else:
        st.error(f"""
        ❌ **不推荐联合治疗**

        {recommendation}

        - 建议单独评估各组分
        - 考虑其他治疗组合
        """)

    # Export
    st.markdown("---")
    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        if st.button("📄 生成HTML报告", key="joint_html", use_container_width=True):
            html = generate_html_report(data)
            filepath = save_html_report(html)
            with open(filepath, "r", encoding="utf-8") as f:
                st.download_button("下载HTML报告", f.read(),
                                   file_name=Path(filepath).name, mime="text/html")

    with col_exp2:
        st.download_button(
            "下载JSON数据",
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            file_name=f"joint_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Navigation
st.markdown("---")
col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
with col_nav1:
    if st.button("🏠 返回首页", use_container_width=True):
        st.switch_page("Home.py")
with col_nav3:
    if st.button("📄 报告导出 →", use_container_width=True):
        st.switch_page("pages/5_Report_Export.py")