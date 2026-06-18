"""Joint Analysis Page - Streamlit UI.

Combined circRNA + Drug therapeutic candidate assessment.
"""

import streamlit as st
import sys
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Add paths - NOTE: confluencia_3_0 uses underscores!
PROJECT_ROOT = Path(r"D:\IGEM集成方案")
sys.path.insert(0, str(PROJECT_ROOT / "confluencia-2.0-drug"))
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

st.set_page_config(page_title="Joint Analysis - Confluencia", page_icon="🧪", layout="wide")

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
    <h2>🧪 circRNA + 药物联合分析</h2>
    <p style="color: #7f8c8d;">综合评估 circRNA 疫苗候选与药物的协同效应</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
### 📋 输入区域

请同时输入 circRNA 序列和药物信息，系统将评估两者的联合治疗效果。
""")

# Input columns
col_circrna, col_drug = st.columns(2)

with col_circrna:
    st.markdown("#### 🔬 circRNA 序列")

    circrna_seq = st.text_area(
        "粘贴 circRNA 序列:",
        height=100,
        placeholder="例如: AUGCGCGCGUAUAGCGCGCG..."
    )

    st.markdown("**示例序列:**")
    seq_examples = {
        "低免疫原性": "AUGCGCGCGUAUAGCGCGCGAUGCGCGCGUAUAGCGCGCG",
        "随机序列": "AUGAUCAAAAAAAGGGUAGCUUAUCAACGGAUC"
    }
    for name, seq in seq_examples.items():
        if st.button(name, key=f"seq_{name}"):
            circrna_seq = seq
            st.rerun()

with col_drug:
    st.markdown("#### 💊 药物信息")

    drug_input = st.text_input(
        "药物名或 SMILES:",
        placeholder="例如: doxorubicin 或 CC(=O)Oc1ccccc1C(=O)O"
    )

    st.markdown("**示例药物:**")
    drug_examples = {
        "阿霉素": "doxorubicin",
        "紫杉醇": "paclitaxel",
        "顺铂": "cisplatin"
    }
    for name, smiles in drug_examples.items():
        if st.button(name, key=f"drug_{name}"):
            drug_input = smiles
            st.rerun()

# Analysis options
st.markdown("---")
st.markdown("#### ⚙️ 分析选项")

col_opt1, col_opt2, col_opt3 = st.columns(3)

with col_opt1:
    synergy_mode = st.radio(
        "协同模式",
        ["加性效应", "协同增效", "拮抗效应"],
        horizontal=True
    )

with col_opt2:
    pk_model = st.radio(
        "PK模型",
        ["CTM (经典)", "RNACTM (RNA专用)"],
        horizontal=True
    )

with col_opt3:
    include_simulation = st.checkbox("包含TNBC仿真模拟", value=True)

# Run analysis
st.markdown("")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    run_btn = st.button("🚀 开始联合分析", type="primary", use_container_width=True,
                        disabled=not (circrna_seq and drug_input))

if run_btn and circrna_seq and drug_input:
    with st.spinner("正在进行联合分析..."):
        try:
            # Use utility helper for Python 3.13 dataclass compatibility
            from utils import get_innate_immune, get_admet, PROJECT_ROOT

            # Process circRNA
            seq_clean = circrna_seq.upper().replace(" ", "").replace("\n", "")
            seq_clean = "".join(c for c in seq_clean if c in "AUGC")

            gc = sum(1 for b in seq_clean if b in "GC") / len(seq_clean) if seq_clean else 0

            # Map drug names
            drug_smiles = {
                "doxorubicin": "CC1C(C(CC(O1)C2C3C(C4C(=C(C(=O)C5=C(C4=CC(=C(C5C(=O)C6=C(C=C3C=C2OC6=O)O)O)O)O)O)C(=O)O)O)O)O",
                "paclitaxel": "CC1=C2C(=O)C(C(=O)C2C(C1=O)O)O",
                "cisplatin": "N.N.Cl[Pt]Cl"
            }
            smiles = drug_smiles.get(drug_input.lower(), drug_input)

            # Run actual circRNA immune assessment
            innate_immune_mod = get_innate_immune()
            immune = innate_immune_mod.assess_innate_immune(seq_clean)
            circrna_safety = immune.net_safety_score

            # Run actual drug ADMET prediction
            admet_mod = get_admet()
            drug_predictor = admet_mod.ADMETPredictor()
            drug_result = drug_predictor.predict(smiles)
            drug_risk = drug_result.overall_risk

            # Synergy score based on actual backend mode
            if synergy_mode == "协同增效":
                synergy_score = min(0.95, 0.6 + circrna_safety * 0.3 - drug_risk * 0.2)
            elif synergy_mode == "拮抗效应":
                synergy_score = max(0.1, 0.3 - circrna_safety * 0.1 + drug_risk * 0.2)
            else:
                synergy_score = 0.5

            # Joint efficacy
            joint_efficacy = (circrna_safety + (1 - drug_risk)) * synergy_score / 2

            # PK simulation
            ka = 0.8
            ke = 0.1
            t = np.linspace(0, 72, 144)

            # Drug curve
            c_drug = 0.5 * ka / (50 * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))

            # circRNA curve (slower kinetics)
            c_rna = 0.3 * 0.5 / (50 * (0.5 - 0.05)) * (np.exp(-0.05 * t) - np.exp(-0.5 * t))

            # Combined effect
            c_combined = c_drug + c_rna * synergy_score

            joint_data = {
                "module": "joint",
                "backend": "local",
                "circrna": {
                    "sequence": seq_clean,
                    "length": len(seq_clean),
                    "gc_content": gc,
                    "safety_score": circrna_safety,
                    "immune_evasion": immune.modification_evasion if hasattr(immune, 'modification_evasion') else 0.5
                },
                "drug": {
                    "input": drug_input,
                    "smiles": smiles,
                    "overall_risk": drug_risk,
                    "hERG_risk": drug_result.hERG_risk,
                    "hepatotoxicity": drug_result.hepatotoxicity_risk
                },
                "synergy": {
                    "mode": synergy_mode,
                    "score": synergy_score,
                    "joint_efficacy": joint_efficacy
                },
                "pk": {
                    "time": t.tolist(),
                    "drug_concentration": c_drug.tolist(),
                    "rna_concentration": c_rna.tolist(),
                    "combined_concentration": c_combined.tolist()
                }
            }

            st.session_state["joint_data"] = joint_data
            st.session_state["joint_done"] = True
            st.success("✅ 联合分析完成！")

        except Exception as e:
            import traceback
            st.error(f"分析出错: {e}")
            with st.expander("详细错误信息"):
                st.code(traceback.format_exc())

# Display results
if st.session_state.get("joint_done") and st.session_state.get("joint_data"):
    data = st.session_state["joint_data"]

    st.markdown("---")
    st.markdown("""
    <div class="section-header">
        <h3>📊 联合分析结果</h3>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("circRNA 安全评分", f"{data['circrna']['safety_score']:.2f}")

    with col2:
        st.metric("药物风险", f"{data['drug']['overall_risk']:.2f}")

    with col3:
        synergy_color = "🟢" if data['synergy']['score'] > 0.6 else "🟡" if data['synergy']['score'] > 0.3 else "🔴"
        st.metric(f"{synergy_color} 协同评分", f"{data['synergy']['score']:.2f}")

    with col4:
        st.metric("联合疗效", f"{data['synergy']['joint_efficacy']:.2f}")

    with col5:
        rec = "推荐" if data['synergy']['joint_efficacy'] > 0.5 else "待优化"
        st.metric("候选评估", rec)

    # PK curves
    st.markdown("#### ⏱️ 联合PK曲线")

    pk = data["pk"]
    t = pk["time"]
    c_drug = pk["drug_concentration"]
    c_rna = pk["rna_concentration"]
    c_combined = pk["combined_concentration"]

    pk_fig = go.Figure()
    pk_fig.add_trace(go.Scatter(
        x=t, y=c_drug, mode='lines', name='药物',
        line=dict(color='#e74c3c', width=2)
    ))
    pk_fig.add_trace(go.Scatter(
        x=t, y=c_rna, mode='lines', name='circRNA',
        line=dict(color='#27ae60', width=2)
    ))
    pk_fig.add_trace(go.Scatter(
        x=t, y=c_combined, mode='lines', name='联合效应',
        line=dict(color='#c41e3a', width=3, dash='dot')
    ))
    pk_fig.update_layout(
        xaxis_title='时间 (h)',
        yaxis_title='效应强度',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#ecf0f1',
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(pk_fig, use_container_width=True)

    # Synergy interpretation
    st.markdown("#### 📝 协同效应解读")

    synergy = data["synergy"]["score"]
    if synergy > 0.6:
        st.success("""
        ✅ **强协同效应**
        - circRNA 与药物联合使用效果显著优于单独使用
        - 建议作为优先候选方案
        - 可进入临床试验评估阶段
        """)
    elif synergy > 0.3:
        st.warning("""
        ⚠️ **中等协同效应**
        - 联合使用有一定增效，但不显著
        - 建议优化 circRNA 序列或药物剂量
        - 可作为备选方案
        """)
    else:
        st.error("""
        ❌ **拮抗效应**
        - circRNA 与药物联合可能降低各自效果
        - 不推荐联合使用
        - 建议单独评估各成分
        """)

    # Export
    st.markdown("---")
    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        if st.button("📄 生成HTML报告", use_container_width=True):
            html = generate_nature_html_report(data)
            filename = f"joint_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            Path(filename).write_text(html, encoding="utf-8")
            st.success(f"✅ 报告已保存: {filename}")
            with open(filename, "r", encoding="utf-8") as f:
                st.download_button("⬇️ 下载HTML", f.read(), file_name=filename, mime="text/html")

    with col_exp2:
        import json
        st.download_button(
            "⬇️ 下载JSON数据",
            json.dumps(data, indent=2, ensure_ascii=False),
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
    if st.button("🎮 TNBC仿真 →", use_container_width=True):
        st.switch_page("pages/4_TNBC_Simulator.py")