"""
Confluencia circRNA Module — Streamlit Application
===================================================
基于 drug 2.0 架构的 circRNA 预测模块。

运行：
    streamlit run confluencia-circrna-encoder/app.py

功能：
- circRNA 序列输入
- RNA-FM 编码
- 免疫原性预测
- 多任务评分 (类似 drug 模块)
- 基因表达参数调整
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Optional

import streamlit as st
import pandas as pd
import numpy as np

# Add project paths
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT.parent))

from core.encoder import CircRNAEncoder, CircRNAEncoderConfig
from core.predictor import CircRNAPredictor
from core.scoring import CompositeScorer, ReportScorer, calculate_ips_score
from core.features import build_sequence_features, get_default_gene_expression

# Page config (mirrors drug module)
st.set_page_config(
    page_title="Confluencia circRNA",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; color: #1f77b4; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.3rem; color: #ff7f0e; margin-top: 1rem; }
    .score-high { background: #28a745; color: white; padding: 0.3rem 0.6rem; border-radius: 0.3rem; }
    .score-medium { background: #ffc107; color: black; padding: 0.3rem 0.6rem; border-radius: 0.3rem; }
    .score-low { background: #dc3545; color: white; padding: 0.3rem 0.6rem; border-radius: 0.3rem; }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state (mirrors drug module)."""
    if "predictor" not in st.session_state:
        st.session_state.predictor = None
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "last_results" not in st.session_state:
        st.session_state.last_results = None


def mock_predict(sequence: str, gene_expr: Dict) -> Dict:
    """Mock prediction for demo."""
    import random
    random.seed(hash(sequence) % 10000)

    seq_feats = build_sequence_features(sequence)
    gc = seq_feats["gc_content"]
    entropy = seq_feats["entropy"]

    trop2 = gene_expr.get("TROP2", 7.0)
    gene_factor = trop2 / 15.0

    return {
        "immunotherapy_score": 0.35 + gc * 0.25 + random.uniform(-0.1, 0.1),
        "tumor_killing_index": 0.3 + entropy / 2.0 + random.uniform(-0.1, 0.1),
        "overall_immunogenicity": 0.4 + gc * 0.2 + gene_factor * 0.2,
        "immune_cycle_score": 0.3 + gene_factor * 0.3,
        "tme_score": 0.35 + random.uniform(-0.1, 0.15),
        "therapeutic_window": 0.45 + gc * 0.15,
        "tide_score": 0.5 - gene_factor * 0.15,
        "ips": (3 + gc * 3 + gene_factor * 2) + random.uniform(-0.5, 0.5),
        "rig_i_score": 0.3 + gc * 0.15,
        "tlr_score": 0.25 + seq_feats["au_content"] * 0.2,
        "pkr_score": 0.28 + entropy / 4.0,
        "trained_model_risk": 0.35 + random.uniform(-0.1, 0.1),
        "prob_likely_non_responder": random.uniform(0.1, 0.35),
        "prob_intermediate": random.uniform(0.25, 0.45),
        "prob_likely_responder": random.uniform(0.15, 0.35),
        "predicted_response": random.choice(["likely_non_responder", "intermediate", "likely_responder"]),
    }


def render_sidebar():
    """Render sidebar (mirrors drug module layout)."""
    with st.sidebar:
        st.markdown("### 🧬 模型设置")

        use_real_model = st.checkbox("使用真实模型", value=False)
        model_path = st.text_input("模型路径", "data/models/best.pt")

        if st.button("加载模型", type="primary"):
            if use_real_model:
                try:
                    st.session_state.predictor = CircRNAPredictor(model_path=model_path)
                    st.session_state.model_loaded = True
                    st.success("✅ 模型已加载")
                except Exception as e:
                    st.error(f"❌ 加载失败: {e}")
            else:
                st.session_state.predictor = None
                st.session_state.model_loaded = False
                st.info("使用模拟预测")

        st.markdown("---")
        st.markdown("### 🧪 基因表达参数")

        gene_expr = {}
        col1, col2 = st.columns(2)
        with col1:
            gene_expr["TROP2"] = st.slider("TROP2", 0.0, 15.0, 7.2)
            gene_expr["NECTIN4"] = st.slider("NECTIN4", 0.0, 12.0, 5.1)
            gene_expr["LIV-1"] = st.slider("LIV-1", 0.0, 10.0, 3.5)
        with col2:
            gene_expr["B7-H4"] = st.slider("B7-H4", 0.0, 12.0, 6.0)
            gene_expr["MKI67"] = st.slider("MKI67", 0.0, 15.0, 8.0)
            gene_expr["MYC"] = st.slider("MYC", 0.0, 10.0, 4.5)

        st.markdown("---")
        st.markdown("### 🔗 双模态联动")

        st.markdown("""
        - [药物模块](../confluencia-2.0-drug/app.py)
        - [表位模块](../confluencia-2.0-epitope/app.py)
        """)

        if st.button("🔄 重置参数"):
            st.session_state.clear()
            st.rerun()

        return gene_expr


def render_sequence_input():
    """Render sequence input section."""
    st.markdown('<p class="sub-header">📝 circRNA 序列输入</p>', unsafe_allow_html=True)

    input_method = st.radio("输入方式", ["单序列", "批量CSV", "示例序列"], horizontal=True)

    sequence = ""

    if input_method == "单序列":
        sequence = st.text_area(
            "circRNA 序列 (RNA格式, 使用 U)",
            value="AUCCAAAAGCGGGGUAUUUGCACUUCCCUUAAUCCAUAAGGGCUUUUGCCGCGUGUUAGAGGAAGCUAUCCCACACUUGUGUAUGGCAUCUUCCCCCUCAGCCUCCCUCGUGUCGUACUAUACGAUCAUUUAAAGAAAGAUAUUUGGGAUGGAGACGCAUGAUUCAUGGCUAGUUCGGAGAGCGAACGGCGGAGGCCUAGGUGAUAUUCAGGAGGAUAUGG",
            height=120,
        )

    elif input_method == "批量CSV":
        csv_file = st.file_uploader("上传 CSV", type=["csv"])
        if csv_file:
            try:
                df = pd.read_csv(csv_file)
                st.write(f"已加载 {len(df)} 条序列")
                st.dataframe(df.head(3))
                sequence = df["sequence"].iloc[0] if "sequence" in df.columns else ""
                st.session_state.batch_df = df
            except Exception as e:
                st.error(f"解析失败: {e}")

    else:
        examples = {
            "高GC示例": "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC",
            "高AU示例": "AAAAAAAAUUUUUUUUAAAAAAAAUUUUUUUUAAAAAAAAUUUUUUUUAAAAAAAAUUUUUUUUAAAA",
            "平衡示例": "AUGCGAUGCGAUGCGAUGCGAUGCGAUGCGAUGCGAUGCGAUGCGAUGCGAUGCGAUGCGAUGCG",
        }
        selected = st.selectbox("选择示例", list(examples.keys()))
        sequence = examples[selected]

    return sequence, input_method


def render_sequence_info(sequence: str):
    """Render sequence info panel."""
    if not sequence:
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        seq_feats = build_sequence_features(sequence)
        st.markdown("**序列统计**")
        st.metric("长度", f"{seq_feats['length']} nt")
        st.metric("GC含量", f"{seq_feats['gc_content']:.1%}")
        st.metric("熵值", f"{seq_feats['entropy']:.2f}")

    with col2:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 3))
        bases = ["A", "U", "G", "C"]
        counts = [
            seq_feats["a_count"],
            seq_feats["u_count"],
            seq_feats["g_count"],
            seq_feats["c_count"],
        ]
        ax.bar(bases, counts, color=["#ff9999", "#99ff99", "#9999ff", "#ffff99"])
        ax.set_ylabel("数量")
        st.pyplot(fig)


def render_prediction_results(results: Dict, gene_expr: Dict):
    """Render prediction results (mirrors drug module layout)."""
    st.markdown('<p class="sub-header">🔮 预测结果</p>', unsafe_allow_html=True)

    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)

    imm = results.get("overall_immunogenicity", 0.5)
    with col1:
        color_class = "score-high" if imm >= 0.6 else ("score-medium" if imm >= 0.4 else "score-low")
        st.markdown(f"**免疫原性**: <span class='{color_class}'>{imm:.2f}</span>", unsafe_allow_html=True)

    with col2:
        tk = results.get("tumor_killing_index", 0.5)
        st.metric("肿瘤杀伤", f"{tk:.2f}")

    with col3:
        ips = results.get("ips", 0)
        st.metric("IPS评分", f"{ips:.1f}/10")

    with col4:
        response = results.get("predicted_response", "intermediate")
        response_map = {
            "likely_responder": "✅ Likely应答",
            "intermediate": "⚠️ 中等",
            "likely_non_responder": "❌ Likely无应答",
        }
        st.metric("预测应答", response_map.get(response, response))

    # Detailed scores tabs
    tab1, tab2, tab3 = st.tabs(["综合评分", "免疫激活", "风险评估"])

    with tab1:
        composite_keys = [
            "immunotherapy_score", "tumor_killing_index", "overall_immunogenicity",
            "immune_cycle_score", "tme_score", "therapeutic_window", "tide_score",
        ]
        scores = [results.get(k, 0.5) for k in composite_keys]

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.barh(composite_keys, scores, color="#1f77b4")
        ax.set_xlim(0, 1)
        for i, v in enumerate(scores):
            ax.text(v + 0.02, i, f"{v:.2f}", va="center")
        ax.set_xlabel("分数")
        st.pyplot(fig)

    with tab2:
        innate_keys = ["rig_i_score", "tlr_score", "pkr_score"]
        col1, col2, col3 = st.columns(3)
        for i, (col, key) in enumerate(zip([col1, col2, col3], innate_keys)):
            val = results.get(key, 0.3)
            name = {"rig_i_score": "RIG-I", "tlr_score": "TLR", "pkr_score": "PKR"}[key]
            col.metric(name, f"{val:.2f}")

    with tab3:
        risk = results.get("trained_model_risk", 0.3)
        tide = results.get("tide_score", 0.5)
        col1, col2 = st.columns(2)
        col1.metric("模型风险", f"{risk:.2f}")
        col2.metric("TIDE逃逸", f"{tide:.2f}")

    # Summary
    st.markdown("---")
    st.markdown("**📋 预测摘要**")

    imm_level = "高" if imm >= 0.6 else ("中" if imm >= 0.4 else "低")

    summary = f"""
    | 指标 | 结果 |
    |------|------|
    | 免疫原性 | {imm_level} ({imm:.2f}) |
    | IPS评分 | {ips:.1f}/10 |
    | 肿瘤杀伤 | {tk:.2f} |
    | 预测应答 | {response} |
    | TROP2表达 | {gene_expr.get('TROP2', 0)} |
    | NECTIN4表达 | {gene_expr.get('NECTIN4', 0)} |
    """
    st.markdown(summary)

    # Recommendation
    if imm >= 0.6:
        st.success("✅ 高免疫原性：适合用于免疫治疗研究")
    elif imm >= 0.4:
        st.warning("⚠️ 中等免疫原性：建议联合治疗方案")
    else:
        st.error("❌ 低免疫原性：需要序列优化或佐剂增强")


def render_batch_results(df: pd.DataFrame):
    """Render batch prediction results."""
    st.markdown('<p class="sub-header">📊 批量预测结果</p>', unsafe_allow_html=True)

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总数", len(df))
    col2.metric("高免疫原性", int((df["overall_immunogenicity"] >= 0.6).sum()))
    col3.metric("中等", int((df["overall_immunogenicity"] >= 0.4).sum()))
    col4.metric("低", int((df["overall_immunogenicity"] < 0.4).sum()))

    # Distribution
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(df["overall_immunogenicity"], bins=20, color="#1f77b4", edgecolor="white")
    ax.set_xlabel("免疫原性分数")
    ax.set_ylabel("数量")
    st.pyplot(fig)

    # Table
    st.dataframe(df.head(20))

    # Export
    if st.button("📥 导出CSV"):
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("下载", csv, "circrna_predictions.csv")


def main():
    """Main application (mirrors drug module structure)."""
    init_session_state()

    # Header
    st.markdown('<p class="main-header">🧬 Confluencia circRNA Module</p>', unsafe_allow_html=True)
    st.markdown("**circRNA 序列免疫原性与疗效预测 (基于 drug 2.0 架构)**")

    # Sidebar
    gene_expr = render_sidebar()

    # Sequence input
    sequence, input_method = render_sequence_input()

    # Sequence info
    render_sequence_info(sequence)

    # Prediction button
    st.markdown("---")

    col1, col2 = st.columns([1, 4])
    with col1:
        predict_btn = st.button("🚀 开始预测", type="primary")

    if predict_btn and sequence:
        with st.spinner("正在预测..."):
            time.sleep(0.3)

            predictor = st.session_state.predictor
            if predictor and st.session_state.model_loaded:
                try:
                    results = predictor.predict(sequence, gene_expr)
                except Exception as e:
                    st.warning(f"模型预测失败: {e}")
                    results = mock_predict(sequence, gene_expr)
            else:
                results = mock_predict(sequence, gene_expr)

            st.session_state.last_results = results

    # Display results
    if st.session_state.last_results:
        render_prediction_results(st.session_state.last_results, gene_expr)

    # Batch prediction
    if input_method == "批量CSV" and hasattr(st.session_state, "batch_df"):
        if st.button("🚀 批量预测"):
            df = st.session_state.batch_df
            sequences = df["sequence"].tolist()

            results_df = pd.DataFrame([mock_predict(s, gene_expr) for s in sequences])
            results_df["sequence_id"] = range(len(sequences))
            results_df["sequence"] = [s[:50] + "..." for s in sequences]

            st.session_state.batch_results = results_df

        if hasattr(st.session_state, "batch_results"):
            render_batch_results(st.session_state.batch_results)


if __name__ == "__main__":
    main()