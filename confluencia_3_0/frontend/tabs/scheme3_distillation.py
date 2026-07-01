"""Scheme 3 Dual-Engine Distillation Visualization"""

import streamlit as st
import plotly.graph_objects as go


def render_scheme3_distillation(agent, history_df):
    """Render Scheme 3 dual-engine distillation interface"""

    st.header("🔄 Scheme 3: 双引擎蒸馏")
    st.caption("Teacher: Scheme 0 (CircFold Baseline) → Student: Scheme 1/6/7")

    # ═══════════════════════════════════════════════════════════
    # Teacher-Student Relationship
    # ═══════════════════════════════════════════════════════════

    st.subheader("知识蒸馏架构")

    # Teacher box
    st.info("""
    👨‍🏫 **Teacher: CircFold Baseline (Scheme 0)**

    - **类型**: Pipeline（固定流程）
    - **状态**: 不训练（冻结）
    - **输出**: 高质量伪标签（80k结构）
    - **质量**: 置信度 ≥ 0.70
    - **Team**: Team 9官方基线
    """)

    st.markdown("⬇️ **知识传递** ⬇️")

    # Student box
    st.success("""
    👨‍🎓 **Student: 可训练神经网络**

    - **可选**: Scheme 1 (EGNN) / Scheme 6 (GNN) / Scheme 7 (Mamba)
    - **状态**: 可训练（梯度更新）
    - **目标**: 学习Teacher预测模式
    - **优势**: 推理速度远快于Pipeline
    - **Team**: Team 3参赛方法
    """)

    # ═══════════════════════════════════════════════════════════
    # Distillation Flow
    # ═══════════════════════════════════════════════════════════

    st.subheader("蒸馏流程")

    flow_steps = [
        "1. Teacher生成伪标签（无梯度）",
        "2. Student预测序列",
        "3. 计算蒸馏损失（coords + confidence + BSJ）",
        "4. 反向传播（仅更新Student）",
        "5. 重复训练至收敛"
    ]

    for step in flow_steps:
        st.markdown(f"**{step}**")
        if flow_steps.index(step) < len(flow_steps) - 1:
            st.markdown("↓")

    # ═══════════════════════════════════════════════════════════
    # Loss Curves
    # ═══════════════════════════════════════════════════════════

    st.subheader("损失曲线对比")

    # Mock loss data
    epochs = list(range(1, 51))
    teacher_loss = [2.5] * 50  # Teacher不训练，固定
    student_loss = [5.0 - i*0.08 for i in range(50)]  # Student逐渐收敛

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=teacher_loss,
                             mode='lines', name='Teacher Loss (固定)'))
    fig.add_trace(go.Scatter(x=epochs, y=student_loss,
                             mode='lines', name='Student Loss (收敛)'))

    fig.update_layout(
        title="Teacher vs Student Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss"
    )

    st.plotly_chart(fig)

    # ═══════════════════════════════════════════════════════════
    # Knowledge Transfer Metrics
    # ═══════════════════════════════════════════════════════════

    st.subheader("知识转移指标")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("坐标损失", "0.85", "-15%")

    with col2:
        st.metric("置信度KL散度", "0.12", "-80%")

    with col3:
        st.metric("BSJ距离损失", "0.05", "-95%")

    # ═══════════════════════════════════════════════════════════
    # Student Selection
    # ═══════════════════════════════════════════════════════════

    st.subheader("Student模型选择")

    student_options = {
        "Scheme 1 (EGNN)": "EGNN等变图网络",
        "Scheme 6 (GNN)": "图神经网络扩散",
        "Scheme 7 (Mamba)": "Mamba+Transformer混合 ⭐ 推荐"
    }

    selected_student = st.selectbox(
        "选择Student架构",
        options=list(student_options.keys())
    )

    st.markdown(f"**架构**: {student_options[selected_student]}")

    # ═══════════════════════════════════════════════════════════
    # Distillation Controls
    # ═══════════════════════════════════════════════════════════

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🚀 开始蒸馏", type="primary"):
            st.success(f"启动蒸馏：Teacher (Scheme 0) → Student ({selected_student})")

    with col2:
        if st.button("📊 监控进度"):
            st.info("查看实时蒸馏曲线")

    # ═══════════════════════════════════════════════════════════
    # Advantages Display
    # ═══════════════════════════════════════════════════════════

    st.subheader("双引擎蒸馏优势")

    advantages = [
        "✅ Teacher质量保证（CASP官方基线）",
        "✅ Student推理速度提升（神经网络）",
        "✅ 知识传承（Pipeline → Model）",
        "✅ 泛化能力增强（从8万条数据学习）",
        "✅ BSJ准确率提升（物理约束 + 数据驱动）"
    ]

    for advantage in advantages:
        st.markdown(advantage)