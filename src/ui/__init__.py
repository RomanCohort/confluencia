"""src.ui -- Confluencia Streamlit UI 模块。

包含从 frontend.py 提取的模块化 UI 组件：
- docking_ui: 分子对接训练/预测/筛选
- dleps_ui: DLEPS 药物功效预测
- multiscale_ui: 多尺度 GNN-PINN 建模
"""
from src.ui.constants import APP_TITLE, UPLOAD_TYPES, _PROJECT_ROOT

__all__ = [
    "APP_TITLE",
    "UPLOAD_TYPES",
    "_PROJECT_ROOT",
]