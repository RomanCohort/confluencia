"""Confluencia Studio - Streamlit Web UI for Non-Programmers.

Easy-to-use interface for:
- circRNA immunogenicity analysis
- Drug ADMET prediction
- TNBC Simulacrum simulation
- Clinical report generation

Usage:
    streamlit run Home.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add confluencia paths - NOTE: confluencia_3_0 uses underscores!
PROJECT_ROOT = Path(r"D:\IGEM集成方案")
sys.path.insert(0, str(PROJECT_ROOT / "confluencia-2.0-drug"))
sys.path.insert(0, str(PROJECT_ROOT / "confluencia-2.0-epitope"))
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

st.set_page_config(
    page_title="Confluencia Studio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Nature journal style
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #2d3a4f 100%);
    }
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #2c3e50 100%);
        border-left: 5px solid #c41e3a;
        padding: 20px;
        border-radius: 0 8px 8px 0;
        color: #ecf0f1;
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        color: #27ae60;
    }
    .metric-label {
        color: #7f8c8d;
        font-size: 0.85em;
    }
    .safe { color: #27ae60; }
    .warning { color: #f39c12; }
    .danger { color: #e74c3c; }
    .info-box {
        background: rgba(41,128,185,0.1);
        border-left: 4px solid #2980b9;
        padding: 10px 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/dna.png", width=80)
    st.title("🧬 Confluencia Studio")
    st.markdown("---")
    st.markdown("### 分析模块")
    st.markdown("""
    - 🏠 **首页** - 模块选择
    - 🔬 **circRNA 分析** - 免疫原性评估
    - 💊 **药物预测** - ADMET属性
    - 🧬 **表位筛选** - MHC结合预测
    - 🎮 **TNBC仿真** - 数字孪生
    - 📄 **报告导出** - 生成报告
    """)
    st.markdown("---")
    st.markdown(f"**版本**: 3.0.0")
    st.markdown(f"**日期**: {datetime.now().strftime('%Y-%m-%d')}")

# Main content
st.markdown("""
<div class="main-header">
    <h1>🧬 Confluencia Studio</h1>
    <p style="font-style: italic; color: #7f8c8d;">
        circRNA 药物发现平台 - 无需编程的图形化分析工具
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("")

# Module cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 3em;">🔬</div>
        <h3 style="color: #ecf0f1;">circRNA 分析</h3>
        <p style="color: #7f8c8d;">免疫原性评估<br>TorusFold评分<br>PK模拟</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("进入 circRNA 分析", key="btn_circrna", use_container_width=True):
        st.switch_page("pages/1_CircRNA_Analysis.py")

with col2:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 3em;">💊</div>
        <h3 style="color: #ecf0f1;">药物预测</h3>
        <p style="color: #7f8c8d;">ADMET属性<br>疗效预测<br>分子优化</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("进入药物预测", key="btn_drug", use_container_width=True):
        st.switch_page("pages/2_Drug_Prediction.py")

with col3:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 3em;">🧬</div>
        <h3 style="color: #ecf0f1;">表位筛选</h3>
        <p style="color: #7f8c8d;">MHC结合预测<br>表位评分<br>免疫表型</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("进入表位筛选", key="btn_epitope", use_container_width=True):
        st.switch_page("pages/3_Epitope_Screening.py")

st.markdown("")

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 3em;">🎮</div>
        <h3 style="color: #ecf0f1;">TNBC 仿真</h3>
        <p style="color: #7f8c8d;">数字孪生<br>肿瘤动态<br>治疗响应</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("进入TNBC仿真", key="btn_tnbc", use_container_width=True):
        st.switch_page("pages/4_TNBC_Simulator.py")

with col5:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 3em;">🧪</div>
        <h3 style="color: #ecf0f1;">联合分析</h3>
        <p style="color: #7f8c8d;">circRNA+药物<br>协同评估<br>候选筛选</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("进入联合分析", key="btn_joint", use_container_width=True):
        st.switch_page("pages/6_Joint_Analysis.py")

with col6:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 3em;">📄</div>
        <h3 style="color: #ecf0f1;">报告导出</h3>
        <p style="color: #7f8c8d;">HTML报告<br>临床报告<br>数据打包</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("生成报告", key="btn_report", use_container_width=True):
        st.switch_page("pages/5_Report_Export.py")

# Quick start guide
st.markdown("---")
st.markdown("""
<div class="info-box">
    <h4>🚀 快速开始指南</h4>
    <ol>
        <li><b>粘贴序列</b> - 在 circRNA 分析页面输入您的 circRNA 序列</li>
        <li><b>点击分析</b> - 选择分析类型，点击"开始分析"按钮</li>
        <li><b>查看报告</b> - 系统自动生成可视化 HTML 报告</li>
        <li><b>导出数据</b> - 可下载 JSON 数据用于进一步研究</li>
    </ol>
</div>
""", unsafe_allow_html=True)

# Recent activity placeholder
st.markdown("### 📊 最近分析记录")
st.info("暂无分析记录。开始您的第一次分析吧！")