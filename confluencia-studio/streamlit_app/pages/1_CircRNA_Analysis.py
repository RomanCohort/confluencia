"""CircRNA Analysis Page - Streamlit UI.

Immunogenicity assessment, TorusFold scoring, PK simulation.
"""

import streamlit as st
import sys
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Add paths - NOTE: confluencia_3_0 uses underscores!
PROJECT_ROOT = Path(r"D:\IGEM集成方案")
sys.path.insert(0, str(PROJECT_ROOT / "confluencia-2.0-drug"))  # innate_immune, ctm
sys.path.insert(0, str(PROJECT_ROOT / "confluencia_3_0"))  # circrna module

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

st.set_page_config(page_title="circRNA Analysis - Confluencia", page_icon="🔬", layout="wide")

# Header
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
    .sequence-box {
        background: rgba(0,0,0,0.3);
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        font-family: 'Courier New', monospace;
        font-size: 1.1em;
        letter-spacing: 2px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section-header">
    <h2>🔬 circRNA 免疫原性分析</h2>
    <p style="color: #7f8c8d;">评估 circRNA 序列的免疫安全性、翻译效率、PK 参数</p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Analysis options
with st.sidebar:
    st.markdown("### ⚙️ 分析选项")

    analysis_type = st.radio(
        "选择分析类型",
        ["完整分析 (推荐)", "仅免疫原性", "仅PK模拟", "仅TorusFold评分"]
    )

    backend = st.select_slider(
        "计算精度",
        options=["快速 (heuristic)", "标准 (ViennaRNA)", "高精度 (ESM-2)"],
        value="标准 (ViennaRNA)"
    )

    st.markdown("---")
    st.markdown("### 📚 术语说明")

    with st.expander("什么是免疫原性？"):
        st.markdown("""
        **免疫原性 (Immunogenicity)** 指 circRNA 被免疫系统识别的能力。

        - **TLR3/7/8**: Toll样受体，识别RNA
        - **RIG-I/MDA5**: 细胞质RNA传感器
        - **PKR**: 蛋白激酶R，抗病毒反应

        **低免疫原性** = 更适合治疗应用
        """)

    with st.expander("什么是TorusFold？"):
        st.markdown("""
        **TorusFold** 是专门针对 circRNA 的结构预测模型。

        四维评分：
        - 📐 **稳定性**: 环状结构稳定性
        - 🔄 **翻译效率**: IRES驱动蛋白表达
        - 🛡️ **免疫逃逸**: 避免免疫识别
        - 📦 **递送效率**: 细胞摄取能力
        """)

    with st.expander("什么是RNACTM？"):
        st.markdown("""
        **RNACTM** 是六室药代动力学模型。

        模拟 circRNA 在体内的：
        - 血液浓度曲线
        - 组织分布（肝/脾/其他）
        - 半衰期预测
        """)

# Main input area
col_input, col_info = st.columns([2, 1])

with col_input:
    st.markdown("#### 📋 输入 circRNA 序列")

    sequence = st.text_area(
        "粘贴您的 circRNA 序列 (A, U, G, C):",
        height=150,
        placeholder="例如: AUGCGCGCGUAUAGCGCGCG...",
        help="支持任意长度的RNA序列，建议50-500nt"
    )

    # Example sequences
    st.markdown("**示例序列:**")
    examples = {
        "低免疫原性序列 (GC平衡)": "AUGCGCGCGUAUAGCGCGCGAUGCGCGCGUAUAGCGCGCG",
        "高GC含量序列": "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC",
        "富AU序列": "AUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAU",
        "随机序列": "AUGAUCAAAAAAAGGGUAGCUUAUCAACGGAUC"
    }

    example_cols = st.columns(4)
    for i, (name, seq) in enumerate(examples.items()):
        with example_cols[i]:
            if st.button(name.split(" ")[0], key=f"ex_{i}", use_container_width=True):
                sequence = seq
                st.rerun()

with col_info:
    st.markdown("#### 📊 序列信息")

    if sequence:
        seq_clean = sequence.upper().replace(" ", "").replace("\n", "")
        seq_clean = "".join(c for c in seq_clean if c in "AUGC")

        a_count = seq_clean.count("A")
        u_count = seq_clean.count("U")
        g_count = seq_clean.count("G")
        c_count = seq_clean.count("C")
        total = len(seq_clean)

        if total > 0:
            gc_content = (g_count + c_count) / total * 100

            st.metric("序列长度", f"{total} nt")
            st.metric("GC含量", f"{gc_content:.1f}%")

            # GC status
            if 40 <= gc_content <= 60:
                st.success("✅ GC含量适中 (40-60%)")
            elif gc_content < 40:
                st.warning("⚠️ GC含量偏低，可能影响稳定性")
            else:
                st.warning("⚠️ GC含量偏高，可能增加免疫原性")

            # Nucleotide composition
            st.markdown("**核苷酸组成:**")
            comp_fig = go.Figure(data=[go.Pie(
                labels=['A', 'U', 'G', 'C'],
                values=[a_count, u_count, g_count, c_count],
                marker_colors=['#3498db', '#e74c3c', '#27ae60', '#f39c12'],
                hole=0.4
            )])
            comp_fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ecf0f1',
                height=200,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(comp_fig, use_container_width=True)
    else:
        st.info("请输入序列以查看分析")

# Analysis button
st.markdown("")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    analyze_btn = st.button("🚀 开始分析", type="primary", use_container_width=True, disabled=not sequence)

# Run analysis
if analyze_btn and sequence:
    with st.spinner("正在分析中..."):
        try:
            # Use utility helper for Python 3.13 dataclass compatibility
            from utils import get_innate_immune, get_ctm, PROJECT_ROOT

            innate_immune_mod = get_innate_immune()
            ctm_mod = get_ctm()

            seq_clean = sequence.upper().replace(" ", "").replace("\n", "")
            seq_clean = "".join(c for c in seq_clean if c in "AUGC")

            gc = sum(1 for b in seq_clean if b in "GC") / len(seq_clean) if seq_clean else 0

            # Immune assessment
            immune = innate_immune_mod.assess_innate_immune(seq_clean)

            # TorusFold - use sys.path approach for package imports
            import sys
            conf_3_path = str(PROJECT_ROOT / "confluencia_3_0")
            if conf_3_path not in sys.path:
                sys.path.insert(0, conf_3_path)

            from core.circrna.torusfold_scorer import quick_score
            tf = quick_score(seq_clean)

            # PK params
            params = ctm_mod.infer_rna_ctm_params(gc_content=gc)

            # Prepare data
            analysis_data = {
                "module": "circrna",
                "backend": "vienna",
                "sequence": seq_clean,
                "length": len(seq_clean),
                "gc_content": gc,
                "immune": {
                    "tlr3": immune.tlr3,
                    "tlr7": immune.tlr7,
                    "tlr8": immune.tlr8,
                    "rigi": immune.rigi,
                    "mda5": immune.mda5,
                    "pkr": immune.pkr,
                    "innate_score": immune.innate_immune_score,
                    "safety_score": immune.net_safety_score
                },
                "torusfold": {
                    "stability": tf.get("stability", 0.3),
                    "translation": tf.get("translation", 0.3),
                    "immune_evasion": tf.get("immune_evasion", 0.5),
                    "delivery": tf.get("delivery", 0.3)
                },
                "pk_params": {
                    "k_uptake": params.k_uptake,
                    "k_degrade": params.k_degrade,
                    "protein_half_life": params.k_protein_half,
                    "f_liver": params.f_liver,
                    "f_spleen": params.f_spleen
                }
            }

            st.session_state["analysis_data"] = analysis_data
            st.session_state["analysis_done"] = True
            st.success("✅ 分析完成！")

        except Exception as e:
            import traceback
            st.error(f"分析出错: {e}")
            with st.expander("详细错误信息"):
                st.code(traceback.format_exc())

# Display results
if st.session_state.get("analysis_done") and st.session_state.get("analysis_data"):
    data = st.session_state["analysis_data"]

    st.markdown("---")
    st.markdown("""
    <div class="section-header">
        <h3>📊 分析结果</h3>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        safety = data["immune"]["safety_score"]
        safety_color = "🟢" if safety > 0.8 else "🟡" if safety > 0.5 else "🔴"
        st.metric(f"{safety_color} 安全评分", f"{safety:.2f}")

    with col2:
        innate = data["immune"]["innate_score"]
        st.metric("免疫评分", f"{innate:.3f}")

    with col3:
        hl = data["pk_params"]["protein_half_life"]
        st.metric("半衰期", f"{hl:.1f} h")

    with col4:
        liver = data["pk_params"]["f_liver"]
        st.metric("肝脏分布", f"{liver*100:.0f}%")

    # Visualizations
    st.markdown("")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("#### 🛡️ 免疫传感器雷达图")
        immune = data["immune"]
        radar_fig = go.Figure(data=go.Scatterpolar(
            r=[immune["tlr3"], immune["tlr7"], immune["tlr8"],
               immune["rigi"], immune["mda5"], immune["pkr"]],
            theta=['TLR3', 'TLR7', 'TLR8', 'RIG-I', 'MDA5', 'PKR'],
            fill='toself',
            marker_color='#c41e3a'
        ))
        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ecf0f1',
            height=350
        )
        st.plotly_chart(radar_fig, use_container_width=True)

        # Interpretation
        st.markdown("**解读:**")
        if safety > 0.8:
            st.success("✅ 低免疫原性，适合治疗应用")
        elif safety > 0.5:
            st.warning("⚠️ 中等免疫原性，建议序列优化")
        else:
            st.error("❌ 高免疫原性，可能触发强烈免疫反应")

    with col_chart2:
        st.markdown("#### 📐 TorusFold 四维评分")
        tf = data["torusfold"]

        bar_fig = go.Figure(data=[
            go.Bar(
                x=['稳定性', '翻译效率', '免疫逃逸', '递送效率'],
                y=[tf["stability"], tf["translation"], tf["immune_evasion"], tf["delivery"]],
                marker_color=['#27ae60', '#27ae60', '#f39c12', '#27ae60'],
                text=[f'{tf["stability"]:.2f}', f'{tf["translation"]:.2f}',
                      f'{tf["immune_evasion"]:.2f}', f'{tf["delivery"]:.2f}'],
                textposition='outside'
            )
        ])
        bar_fig.update_layout(
            yaxis=dict(range=[0, 1.2]),
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ecf0f1',
            height=350,
            showlegend=False
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    # PK Simulation
    st.markdown("#### ⏱️ RNACTM 药代动力学模拟")

    pk = data["pk_params"]
    ka, ke = pk["k_uptake"], pk["k_degrade"]

    t = np.linspace(0, 72, 144)
    dose, vd = 1.0, 50.0
    c = (dose * ka / (vd * (ka - ke))) * (np.exp(-ke * t) - np.exp(-ka * t))

    pk_fig = go.Figure()
    pk_fig.add_trace(go.Scatter(
        x=t, y=c, mode='lines', fill='tozeroy',
        line=dict(color='#27ae60', width=3),
        name='中央室浓度'
    ))
    pk_fig.update_layout(
        xaxis_title='时间 (h)',
        yaxis_title='浓度',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#ecf0f1',
        height=300
    )
    st.plotly_chart(pk_fig, use_container_width=True)

    # Export options
    st.markdown("---")
    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        if st.button("📄 生成HTML报告", use_container_width=True):
            html = generate_nature_html_report(data)
            filename = f"circrna_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = Path(filename)
            filepath.write_text(html, encoding="utf-8")

            st.success(f"✅ 报告已保存: {filename}")
            with open(filename, "r", encoding="utf-8") as f:
                st.download_button(
                    "⬇️ 下载HTML报告",
                    f.read(),
                    file_name=filename,
                    mime="text/html"
                )

    with col_exp2:
        import json
        json_data = json.dumps(data, indent=2, ensure_ascii=False)
        st.download_button(
            "⬇️ 下载JSON数据",
            json_data,
            file_name=f"circrna_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
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