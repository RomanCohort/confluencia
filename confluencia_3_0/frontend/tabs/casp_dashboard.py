"""CASP Dashboard - Circ-CASP 2026 Competition Monitoring"""

import streamlit as st
import pandas as pd


def render_casp_dashboard(agent, history_df):
    """Render Circ-CASP 2026 competition dashboard"""

    st.header("🏆 Circ-CASP 2026 Dashboard")
    st.caption("Critical Assessment of circRNA Structure Prediction")

    # ═══════════════════════════════════════════════════════════
    # Competition Overview
    # ═══════════════════════════════════════════════════════════

    st.markdown("""
    **参赛队伍：13支**
    - 正式赛道：9支（Scheme 0-7）
    - 神仙打架赛道：3支（外部成熟方法）
    - 随机数赛道：1支
    """)

    # ═══════════════════════════════════════════════════════════
    # Main Track Participants
    # ═══════════════════════════════════════════════════════════

    st.subheader("正式赛道")

    teams_data = {
        "Team": ["Team 1", "Team 2", "Team 3", "Team 4", "Team 5",
                 "Team 6", "Team 7", "Team 8", "Team 9"],
        "单位": ["吉林大学CS"] * 9,
        "方法": ["EGNN+物理精修", "原子力场求解", "双引擎蒸馏", "坐标扩散+EGNN",
                "Transformer物理bias", "隐空间扩散", "Mamba+Transformer",
                "稀疏配对引导", "线性RNA环化"],
        "Scheme": ["S1", "S2", "S3", "S4", "S5 ⚠️", "S6", "S7 ⭐", "S8", "S0 ✅"],
        "状态": ["已实现", "已实现", "已实现", "已实现", "已弃用",
                "已实现", "推荐", "已实现", "官方基线"]
    }

    df = pd.DataFrame(teams_data)
    st.dataframe(df, use_container_width=True)

    # ═══════════════════════════════════════════════════════════
    # Team 9 Special Badge
    # ═══════════════════════════════════════════════════════════

    st.success("✅ Team 9 = Scheme 0 = CircFold Baseline（线性RNA环化法）")
    st.markdown("""
    **Team 9特殊地位：**
    - 🏅 **官方基线** - CASP基准方法
    - 📊 **数据生成** - 为其他队伍提供8万条训练数据
    - 👨‍🏫 **Teacher** - 为Team 3提供知识蒸馏
    - 🔧 **Pipeline** - 5-stage物理优化流程
    """)

    # ═══════════════════════════════════════════════════════════
    # Expert Track
    # ═══════════════════════════════════════════════════════════

    st.subheader("神仙打架赛道")

    expert_teams = {
        "Team": ["Team 10", "Team 11", "Team 12"],
        "来源": ["浙江大学", "山东大学", "维也纳大学"],
        "方法": ["isRNAcirc", "trRosettaRNA2（环化）", "ViennaRNA-Circ"],
        "状态": ["外部成熟方法", "已集成Stage 2", "已集成Stage 1"]
    }

    st.table(pd.DataFrame(expert_teams))

    # ═══════════════════════════════════════════════════════════
    # Random Baseline
    # ═══════════════════════════════════════════════════════════

    st.subheader("随机数赛道")
    st.code("Team 13: {114514, 67, 886}")

    # ═══════════════════════════════════════════════════════════
    # Predicted Rankings
    # ═══════════════════════════════════════════════════════════

    st.subheader("预测获胜排名 🏆")

    rankings = [
        ("🥇 第1名", "Team 7", "Mamba长距离依赖 + 环式优化"),
        ("🥈 第2名", "Team 3", "Team 9作为Teacher，知识蒸馏优势"),
        ("🥉 第3名", "Team 9", "官方基线，物理优化保证质量"),
        ("第4名", "Team 8", "稀疏配对引导，BSJ准确率高"),
        ("第5名", "Team 10", "成熟外部方法（isRNAcirc）"),
    ]

    for medal, team, reason in rankings:
        st.markdown(f"**{medal} - {team}**: {reason}")

    # ═══════════════════════════════════════════════════════════
    # Scheme-Team Mapping
    # ═══════════════════════════════════════════════════════════

    st.subheader("Scheme-Team对应关系")

    mapping = """
    | Scheme | Team | 方法 | 状态 |
    |--------|------|------|------|
    | 0 | Team 9 | 线性RNA环化 | 官方基线 ✅ |
    | 1 | Team 1 | EGNN + 物理 | 已实现 |
    | 2 | Team 2 | 原子力场 | 已实现 |
    | 3 | Team 3 | 双引擎蒸馏 | 已实现（Teacher: Team 9） |
    | 4 | Team 4 | 坐标扩散 | 已实现 |
    | 5 | Team 5 | Transformer | 已弃用 ⚠️ |
    | 6 | Team 6 | 隐空间扩散 | 已实现 |
    | 7 | Team 7 | Mamba+Transformer | 推荐 ⭐ |
    | 8 | Team 8 | 稀疏配对 | 已实现 |
    """

    st.markdown(mapping)