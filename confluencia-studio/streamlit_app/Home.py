"""Confluencia Studio - Home Page.

Integrated circRNA drug discovery platform.
All modules use confluencia skill API for backend computation.
"""

import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Confluencia Studio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #1a1a2e 0%, #2d3a4f 100%); }
    .main-header {
        background: linear-gradient(135deg, rgba(196,30,58,0.2) 0%, rgba(45,58,79,0.8) 100%);
        border-radius: 15px;
        padding: 30px;
        margin-bottom: 20px;
        text-align: center;
    }
    .module-card {
        background: rgba(0,0,0,0.3);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
        transition: all 0.3s ease;
    }
    .module-card:hover {
        border-color: #c41e3a;
        transform: translateY(-5px);
    }
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
<div class="main-header">
    <h1>🧬 Confluencia Studio</h1>
    <p style="font-size: 1.2em; color: #bdc3c7;">
        Integrated circRNA Drug Discovery Platform
    </p>
    <p style="color: #7f8c8d;">
        circRNA Immunogenicity | Drug ADMET | Epitope Screening | TNBC Digital Twin
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Project info
with st.sidebar:
    st.markdown("### 📁 项目信息")

    from utils import get_project_info
    info = get_project_info()

    st.markdown(f"""
    **根目录:** `{Path(info['project_root']).name}`

    **模块状态:**
    - Drug 2.0: {'✅' if info['modules'].get('drug') else '❌'}
    - Epitope 2.0: {'✅' if info['modules'].get('epitope') else '❌'}
    - circRNA 3.0: {'✅' if info['modules'].get('circrna') else '❌'}

    **当前后端:**
    - Drug: {info['backends'].get('drug', 'local')}
    - Epitope: {info['backends'].get('epitope', 'local')}
    - circRNA: {info['backends'].get('circrna', 'heuristic')}
    """)

    st.markdown("---")
    st.markdown("### 🔗 快速链接")

    if st.button("📚 使用文档", use_container_width=True):
        st.info("查看项目 README 获取详细使用说明")

    if st.button("⚙️ 环境变量配置", use_container_width=True):
        st.markdown("""
        **环境变量:**
        - `CONFLUENCIA_ROOT`: 项目根目录
        - `CONFLUENCIA_SKILL_PATH`: skill 路径

        **示例:**
        ```
        export CONFLUENCIA_ROOT=/path/to/IGEM集成方案
        ```
        """)

# Module cards
st.markdown("""
<div class="section-header">
    <h3>🔬 分析模块</h3>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="module-card">
        <h4>💊 Drug Prediction (2.0)</h4>
        <p style="color: #bdc3c7;">药物 ADMET 预测</p>
        <ul style="color: #95a5a6;">
            <li>吸收、分布、代谢预测</li>
            <li>毒性评估 (hERG, 肝毒性)</li>
            <li>药物相似性评分</li>
            <li>Backend: local / ChEMBL API</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    if st.button("打开药物预测", key="btn_drug", use_container_width=True):
        st.switch_page("pages/2_Drug_Prediction.py")

with col2:
    st.markdown("""
    <div class="module-card">
        <h4>🔬 circRNA Analysis (3.0)</h4>
        <p style="color: #bdc3c7;">circRNA 免疫原性分析</p>
        <ul style="color: #95a5a6;">
            <li>先天免疫激活评估</li>
            <li>TorusFold 结构评分</li>
            <li>RNACTM 药代动力学</li>
            <li>Backend: heuristic / Vienna / ESM-2</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    if st.button("打开circRNA分析", key="btn_circrna", use_container_width=True):
        st.switch_page("pages/1_CircRNA_Analysis.py")

with col3:
    st.markdown("""
    <div class="module-card">
        <h4>🧬 Epitope Screening (2.0)</h4>
        <p style="color: #bdc3c7;">MHC 结合预测</p>
        <ul style="color: #95a5a6;">
            <li>MHC I/II 结合亲和力</li>
            <li>疫苗表位筛选</li>
            <li>多等位基因覆盖</li>
            <li>Backend: local / NetMHCpan</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    if st.button("打开表位筛选", key="btn_epitope", use_container_width=True):
        st.switch_page("pages/3_Epitope_Screening.py")

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("""
    <div class="module-card">
        <h4>🎮 TNBC Simulacrum</h4>
        <p style="color: #bdc3c7;">三阴性乳腺癌数字孪生</p>
        <ul style="color: #95a5a6;">
            <li>四种分子亚型模拟</li>
            <li>免疫编辑动态</li>
            <li>治疗方案仿真</li>
            <li>RECIST 响应预测</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    if st.button("打开TNBC仿真", key="btn_simulacrum", use_container_width=True):
        st.switch_page("pages/4_TNBC_Simulator.py")

with col5:
    st.markdown("""
    <div class="module-card">
        <h4>🔗 Joint Analysis</h4>
        <p style="color: #bdc3c7;">联合治疗评估</p>
        <ul style="color: #95a5a6;">
            <li>circRNA疫苗 + 化疗药物</li>
            <li>协同效应评估</li>
            <li>PK 参数匹配</li>
            <li>治疗建议生成</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    if st.button("打开联合分析", key="btn_joint", use_container_width=True):
        st.switch_page("pages/6_Joint_Analysis.py")

with col6:
    st.markdown("""
    <div class="module-card">
        <h4>📄 Report Export</h4>
        <p style="color: #bdc3c7;">报告导出中心</p>
        <ul style="color: #95a5a6;">
            <li>HTML 可视化报告</li>
            <li>JSON 数据导出</li>
            <li>批量报告生成</li>
            <li>Nature 期刊风格</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    if st.button("打开报告导出", key="btn_report", use_container_width=True):
        st.switch_page("pages/5_Report_Export.py")

# Feature highlights
st.markdown("---")
st.markdown("""
<div class="section-header">
    <h3>✨ 核心特性</h3>
</div>
""", unsafe_allow_html=True)

features = [
    ("🎯 统一 API", "所有模块通过 confluencia skill API 统一调用，确保一致性和可维护性"),
    ("🔄 多后端支持", "支持本地计算和远程 API，可根据精度需求切换"),
    ("📊 Nature 风格可视化", "HTML 报告采用 Nature 期刊风格，适合学术展示"),
    ("🐍 Python 3.13 兼容", "完整支持 Python 3.13 dataclass，无需降级"),
    ("🖥️ 跨平台", "支持 Windows/Linux/macOS，通过环境变量配置路径"),
    ("🧪 TNBC 数字孪生", "基于真实临床数据的 TNBC 仿真模型"),
]

for title, desc in features:
    st.markdown(f"**{title}** - {desc}")

# Version info
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d;">
    <p>Confluencia Studio v3.0 | Powered by Claude Code Skill API</p>
    <p>Python 3.13 | Streamlit | Plotly | Nature Journal Style</p>
</div>
""", unsafe_allow_html=True)
