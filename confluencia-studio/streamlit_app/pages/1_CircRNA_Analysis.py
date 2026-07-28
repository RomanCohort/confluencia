"""CircRNA Analysis Page - Streamlit UI.

Immunogenicity assessment, TorusFold scoring, PK simulation.
Uses confluencia skill API for all backend computation.
"""

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

from utils import (
    circrna_full_analysis, circrna_pk,
    generate_html_report, save_html_report,
    get_backend, get_gc_content, format_sequence,
    get_skill_api,
)

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

    # Map UI to backend setting
    backend_map = {
        "快速 (heuristic)": "heuristic",
        "标准 (ViennaRNA)": "vienna",
        "高精度 (ESM-2)": "esm2",
    }
    from utils import set_backend
    set_backend("circrna", backend_map.get(backend, "heuristic"))

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

    st.markdown("---")

    with st.sidebar.expander("TorusFold 3D Structure"):
        use_tf = st.checkbox("Enable 3D Structure Prediction", value=False,
                              help="勾选=用 TorusFold v2 真实推理；不勾选=合成预览（stem+loop 折叠，无需权重）")
        if use_tf:
            tf_model = st.text_input("v2 权重路径", value="models/torusfold_v2.pt",
                                      help="留空会用本地默认路径；填错会自动回退合成预览")
            tf_device = st.selectbox("Device", ["auto", "cuda", "cpu"], index=2)
            st.caption("⚠️ 真实推理需要 v2 格式权重；v1 残留/格式不对会自动 fallback 合成并在结果区提示")
        else:
            tf_model = ""
            tf_device = "cpu"
            st.info("3D 预览模式：合成 stem+loop 折叠结构（无需权重，秒出）。")

        # 存进 session_state，结果区生成 3D 时用
        st.session_state["use_tf_3d"] = use_tf
        st.session_state["tf_model_path"] = tf_model
        st.session_state["tf_device"] = tf_device

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
        "低免疫原性 (GC平衡)": "AUGCGCGCGUAUAGCGCGCGAUGCGCGCGUAUAGCGCGCG",
        "高GC含量": "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC",
        "富AU序列": "AUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAU",
        "随机序列": "AUGAUCAAAAAAAGGGUAGCUUAUCAACGGAUC"
    }

    example_cols = st.columns(4)
    for i, (name, seq) in enumerate(examples.items()):
        with example_cols[i]:
            if st.button(name.split(" ")[0], key=f"ex_{i}", use_container_width=True):
                st.session_state["circrna_sequence"] = seq
                st.rerun()

    # Restore from session state
    if st.session_state.get("circrna_sequence"):
        sequence = st.session_state["circrna_sequence"]
        st.session_state["circrna_sequence"] = None

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

            if 40 <= gc_content <= 60:
                st.success("GC含量适中 (40-60%)")
            elif gc_content < 40:
                st.warning("GC含量偏低，可能影响稳定性")
            else:
                st.warning("GC含量偏高，可能增加免疫原性")

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
    analyze_btn = st.button("开始分析", type="primary", use_container_width=True, disabled=not sequence)

# Run analysis via skill API
if analyze_btn and sequence:
    with st.spinner("正在分析中..."):
        try:
            seq_clean = sequence.upper().replace(" ", "").replace("\n", "")
            seq_clean = "".join(c for c in seq_clean if c in "AUGC")

            if analysis_type.startswith("完整"):
                data = circrna_full_analysis(seq_clean)
            elif analysis_type.startswith("仅PK"):
                data = circrna_pk(seq_clean)
            elif analysis_type.startswith("仅免疫"):
                full = circrna_full_analysis(seq_clean)
                data = {"module": "circrna", "sequence": seq_clean,
                        "length": len(seq_clean), "gc_content": full.get("gc_content", 0),
                        "immune": full.get("immune", {}), "backend": get_backend("circrna")}
            else:  # TorusFold only
                full = circrna_full_analysis(seq_clean)
                data = {"module": "circrna", "sequence": seq_clean,
                        "length": len(seq_clean), "gc_content": full.get("gc_content", 0),
                        "torusfold": full.get("torusfold", {}), "backend": get_backend("circrna")}

            st.session_state["analysis_data"] = data
            st.session_state["analysis_done"] = True
            st.success("分析完成！")

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
    immune = data.get("immune", {})
    tf = data.get("torusfold", {})
    pk = data.get("pk_params", {})
    metrics = data.get("metrics", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        safety = immune.get("safety_score", 0)
        safety_color = "🟢" if safety > 0.8 else "🟡" if safety > 0.5 else "🔴"
        st.metric(f"{safety_color} 安全评分", f"{safety:.2f}")

    with col2:
        innate = immune.get("innate_score", 0)
        st.metric("免疫评分", f"{innate:.3f}")

    with col3:
        hl = pk.get("protein_half_life", metrics.get("half_life", 0))
        st.metric("半衰期", f"{hl:.1f} h")

    with col4:
        liver = pk.get("f_liver", 0)
        st.metric("肝脏分布", f"{liver*100:.0f}%")

    # Immune Radar
    if immune:
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.markdown("#### 🛡️ 免疫传感器雷达图")
            radar_fig = go.Figure(data=go.Scatterpolar(
                r=[immune.get("tlr3",0), immune.get("tlr7",0), immune.get("tlr8",0),
                   immune.get("rigi",0), immune.get("mda5",0), immune.get("pkr",0)],
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

            if safety > 0.8:
                st.success("低免疫原性，适合治疗应用")
            elif safety > 0.5:
                st.warning("中等免疫原性，建议序列优化")
            else:
                st.error("高免疫原性，可能触发强烈免疫反应")

        with col_chart2:
            if tf:
                st.markdown("#### 📐 TorusFold 四维评分")
                bar_fig = go.Figure(data=[
                    go.Bar(
                        x=['稳定性', '翻译效率', '免疫逃逸', '递送效率'],
                        y=[tf.get("stability",0), tf.get("translation",0), tf.get("immune_evasion",0), tf.get("delivery",0)],
                        marker_color=['#27ae60', '#27ae60', '#f39c12', '#27ae60'],
                        text=[f'{tf.get("stability",0):.2f}', f'{tf.get("translation",0):.2f}',
                              f'{tf.get("immune_evasion",0):.2f}', f'{tf.get("delivery",0):.2f}'],
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
    if pk:
        st.markdown("#### ⏱️ RNACTM 药代动力学模拟")

        ka, ke = pk.get("k_uptake", 0.8), pk.get("k_degrade", 0.1)
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

    # 3D Structure Viewer (Mol* 双轨：合成预览 / TorusFold v2 真实推理)
    st.markdown("---")
    st.markdown("#### 🧬 circRNA 3D 结构预览")
    seq_clean = data.get("sequence", "")
    use_tf = st.session_state.get("use_tf_3d", False)
    tf_model = st.session_state.get("tf_model_path", "")
    tf_device = st.session_state.get("tf_device", "cpu")

    if seq_clean:
        # 数据源缓存：(sequence, use_tf, tf_model, tf_device) 作 key，避免重复推理
        cache_key = (seq_clean, use_tf, tf_model, tf_device)
        if st.session_state.get("_molstar_cache_key") != cache_key:
            with st.spinner("生成 3D 结构中..." if use_tf else "生成合成预览..."):
                try:
                    skill_api = get_skill_api()
                    if use_tf and tf_model:
                        html_3d = skill_api.generate_molstar_3d_html(
                            seq_clean, model="torusfold",
                            weights_path=tf_model, device=tf_device,
                        )
                        # 检查是否 fallback（真实路径 title 含 FALLBACK = 真推理失败）
                        if "SYNTHETIC-FALLBACK" in html_3d[:500]:
                            st.warning("⚠️ 真实推理失败，已回退合成预览（见 viewer 标题）")
                        else:
                            st.success("✅ 数据源：TorusFold v2 真实推理")
                    else:
                        html_3d = skill_api.generate_molstar_3d_html(
                            seq_clean, model="synthetic", device=tf_device,
                        )
                        st.info("ℹ️ 数据源：合成预览（stem+loop 折叠）。侧边栏勾选「Enable 3D」+ 填 v2 权重可看真实结构。")

                    st.session_state["_molstar_cache_html"] = html_3d
                    st.session_state["_molstar_cache_key"] = cache_key
                except Exception as e:
                    st.error(f"3D 结构生成失败: {e}")
                    st.session_state["_molstar_cache_html"] = None
                    st.session_state["_molstar_cache_key"] = None

        cached_html = st.session_state.get("_molstar_cache_html")
        if cached_html:
            components.html(cached_html, height=640, scrolling=False)
            st.caption(
                "结构长度 {} nt | HTML {:,} bytes | ".format(len(seq_clean), len(cached_html))
                + "浏览器内旋转/缩放/切 Coloring Scheme（confidence / PKR / m6A / TLR7 等）"
            )
    else:
        st.info("👆 输入序列并点击「开始分析」后，3D 结构会在这里显示")

    # Export
    st.markdown("---")
    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        if st.button("📄 生成HTML报告", key="html_btn", use_container_width=True):
            html = generate_html_report(data)
            filepath = save_html_report(html)
            with open(filepath, "r", encoding="utf-8") as f:
                st.download_button("下载HTML报告", f.read(),
                                   file_name=Path(filepath).name, mime="text/html")

    with col_exp2:
        import json
        st.download_button(
            "下载JSON数据",
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
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
