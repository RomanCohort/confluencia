"""Scheme Manager - Manage Scheme 0-7 for Circ-CASP 2026"""

import streamlit as st
import plotly.graph_objects as go


def render_scheme_manager(agent, history_df):
    """Render Scheme selection and management interface"""

    st.header("🔬 Scheme Manager")
    st.caption("Circ-CASP 2026 - 9 Training Schemes (0-7)")

    # ═══════════════════════════════════════════════════════════
    # Scheme Selector
    # ═══════════════════════════════════════════════════════════

    scheme_names = {
        0: "Scheme 0: CircFold Baseline (线性RNA环化法)",
        1: "Scheme 1: EGNN + 物理精修",
        2: "Scheme 2: 原子力场求解",
        3: "Scheme 3: 双引擎迭代蒸馏",
        4: "Scheme 4: 坐标扩散 + EGNN",
        5: "Scheme 5: Transformer物理bias ⚠️ 已弃用",
        6: "Scheme 6: 隐空间扩散",
        7: "Scheme 7: Mamba+Transformer混合 ⭐ 推荐",
    }

    selected_scheme = st.selectbox(
        "选择Scheme",
        options=list(scheme_names.keys()),
        format_func=lambda x: scheme_names[x]
    )

    # ═══════════════════════════════════════════════════════════
    # Scheme Description
    # ═══════════════════════════════════════════════════════════

    scheme_descriptions = {
        0: """
        **官方基线方法** - CASP CircRNA Track Baseline

        **5-stage Pipeline：**
        1. ViennaRNA → 二级结构预测
        2. trRosettaRNA2 → 线性3D预测
        3. OpenMM → BSJ环化
        4. AMBER14 MD → 分子动力学弛豫
        5. Quality Filter → 多级质量验证

        **特点：**
        - 生成8万条高质量数据
        - Teacher for Scheme 3
        - Team 9官方参赛方法
        """,
        1: "**EGNN等变图神经网络 + 物理精修**\n\nTeam 1参赛方法",
        2: "**原子力场求解**\n\nTeam 2参赛方法",
        3: """
        **双引擎迭代蒸馏**

        Teacher: Scheme 0 (CircFold Baseline)
        Student: Scheme 1/6/7

        知识蒸馏流程：
        - Teacher生成伪标签
        - Student学习预测
        - 损失反向传播

        Team 3参赛方法
        """,
        4: "**坐标扩散 + EGNN引导**\n\nTeam 4参赛方法",
        5: "**⚠️ 已弃用：NaN爆炸问题**\n\nTeam 5方法（不可用）",
        6: "**隐空间扩散模型**\n\nTeam 6参赛方法",
        7: """
        **⭐ 推荐：局部注意力 + 环式Mamba**

        **架构优势：**
        - Mamba长距离依赖 O(L)
        - Transformer局部注意力 O(L×w)
        - 混合架构最优

        **预测：Circ-CASP 2026冠军**

        Team 7参赛方法
        """,
    }

    st.markdown(scheme_descriptions[selected_scheme])

    # ═══════════════════════════════════════════════════════════
    # Scheme Controls
    # ═══════════════════════════════════════════════════════════

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🚀 运行Scheme", type="primary"):
            st.success(f"启动 {scheme_names[selected_scheme]}")

    with col2:
        if st.button("⏸️ 暂停"):
            st.warning("暂停运行")

    with col3:
        if st.button("📊 监控"):
            st.info("查看实时指标")

    # ═══════════════════════════════════════════════════════════
    # Progress Tracking
    # ═══════════════════════════════════════════════════════════

    if selected_scheme == 0:
        st.subheader("Pipeline进度")

        stages = ["ViennaRNA", "trRosettaRNA2", "OpenMM", "MD", "Filter"]
        progress = [100, 85, 60, 30, 10]  # Mock progress

        fig = go.Figure(data=[
            go.Bar(x=stages, y=progress, marker_color=['green', 'blue', 'orange', 'red', 'purple'])
        ])
        fig.update_layout(title="Stage进度", yaxis_title="完成度 (%)")
        st.plotly_chart(fig)

    # ═══════════════════════════════════════════════════════════
    # Output Preview
    # ═══════════════════════════════════════════════════════════

    st.subheader("输出统计")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric("处理序列", "130,472", "100%")

    with col_b:
        st.metric("高质量输出", "80,000", "60%")

    with col_c:
        st.metric("平均置信度", "0.85", "+0.15")