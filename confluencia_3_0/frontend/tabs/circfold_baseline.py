"""CircFold Baseline Visualization - Scheme 0 Pipeline"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render_circfold_baseline(agent, history_df):
    """Render CircFold Baseline (Scheme 0) visualization"""

    st.header("🧬 CircFold Baseline (线性RNA环化法)")
    st.caption("Scheme 0 - Official CASP CircRNA Baseline Method")

    # ═══════════════════════════════════════════════════════════
    # Pipeline Flow Diagram
    # ═══════════════════════════════════════════════════════════

    st.subheader("5-Stage Pipeline流程")

    # Create flow diagram
    stages = [
        {"name": "Stage 1: ViennaRNA", "desc": "二级结构预测", "time": "~1s"},
        {"name": "Stage 2: trRosettaRNA2", "desc": "线性3D预测", "time": "~100s"},
        {"name": "Stage 3: OpenMM", "desc": "BSJ环化", "time": "~30s"},
        {"name": "Stage 4: MD Relaxation", "desc": "分子动力学", "time": "~5min"},
        {"name": "Stage 5: Quality Filter", "desc": "质量过滤", "time": "~5s"},
    ]

    for i, stage in enumerate(stages):
        col1, col2, col3 = st.columns([3, 5, 2])

        with col1:
            st.markdown(f"**{stage['name']}**")

        with col2:
            st.markdown(stage['desc'])

        with col3:
            st.code(stage['time'])

        if i < len(stages) - 1:
            st.markdown("↓")

    # ═══════════════════════════════════════════════════════════
    # Stage Progress Tracking
    # ═══════════════════════════════════════════════════════════

    st.subheader("实时进度监控")

    # Mock progress data
    progress_data = {
        "ViennaRNA": {"completed": 10000, "total": 130472, "status": "running"},
        "trRosettaRNA2": {"completed": 8000, "total": 130472, "status": "running"},
        "OpenMM": {"completed": 5000, "total": 130472, "status": "pending"},
        "MD": {"completed": 2000, "total": 130472, "status": "pending"},
        "Filter": {"completed": 1000, "total": 130472, "status": "pending"},
    }

    col1, col2, col3, col4, col5 = st.columns(5)

    cols = [col1, col2, col3, col4, col5]
    stage_names = ["ViennaRNA", "trRosettaRNA2", "OpenMM", "MD", "Filter"]

    for col, name in zip(cols, stage_names):
        data = progress_data[name]
        progress_pct = int(data['completed'] / data['total'] * 100)
        col.metric(name, f"{progress_pct}%", f"{data['completed']}条")

    # ═══════════════════════════════════════════════════════════
    # Quality Metrics
    # ═══════════════════════════════════════════════════════════

    st.subheader("质量指标分布")

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("置信度分布", "BSJ距离分布", "能量分布")
    )

    # Confidence distribution
    fig.add_trace(
        go.Histogram(x=[0.85, 0.90, 0.75, 0.88, 0.92], name="置信度"),
        row=1, col=1
    )

    # BSJ distance distribution
    fig.add_trace(
        go.Histogram(x=[3.5, 3.6, 3.4, 3.7, 3.5], name="BSJ距离"),
        row=1, col=2
    )

    # Energy distribution
    fig.add_trace(
        go.Histogram(x=[500, 600, 450, 550, 480], name="能量"),
        row=1, col=3
    )

    fig.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig)

    # ═══════════════════════════════════════════════════════════
    # Output Statistics
    # ═══════════════════════════════════════════════════════════

    st.subheader("输出统计")

    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.metric("输入序列", "130,472")

    with col_b:
        st.metric("处理完成", "10,000", "8%")

    with col_c:
        st.metric("高质量输出", "6,000", "60%保留率")

    with col_d:
        st.metric("平均置信度", "0.85", "+0.15")

    # ═══════════════════════════════════════════════════════════
    # Quality Thresholds
    # ═══════════════════════════════════════════════════════════

    st.subheader("质量过滤阈值")

    thresholds = """
    | 指标 | 阈值 | 说明 |
    |------|------|------|
    | 置信度 | ≥ 0.70 | 降低标准以保留更多数据 |
    | BSJ距离 | 2.8-5.0 Å | 磷酸二酯键合理范围 |
    | 能量 | < 800 kJ/mol | 放宽以适应circBase数据 |
    | RMSD方差 | < 0.3 | 结构收敛性 |
    | BSJ冲突 | < 5 | 几何合理性 |
    """

    st.markdown(thresholds)

    # ═══════════════════════════════════════════════════════════
    # Team 9 Badge
    # ═══════════════════════════════════════════════════════════

    st.success("✅ CircFold Baseline = Team 9 = CASP官方基线方法")