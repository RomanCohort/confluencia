from __future__ import annotations

import base64
import csv
import functools
import gc
import io
import importlib
import json

import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype


class _LazyModule:
    """Lazy-load a module on first attribute access to reduce startup memory."""

    def __init__(self, module_name: str) -> None:
        self._module_name = module_name
        self._module = None

    def _load(self):
        if self._module is None:
            self._module = importlib.import_module(self._module_name)
        return self._module

    def __getattr__(self, item: str):
        return getattr(self._load(), item)


class _LazySymbol:
    """Resolve a symbol from deferred import context when first used."""

    def __init__(self, key: str, loader) -> None:
        self._key = key
        self._loader = loader

    def _resolve(self):
        return self._loader()[self._key]

    def __call__(self, *args, **kwargs):
        return self._resolve()(*args, **kwargs)

    def __getattr__(self, item: str):
        return getattr(self._resolve(), item)


@functools.lru_cache(maxsize=1)
def _load_gnn_symbols() -> Dict[str, Any]:
    try:
        from .gnn import mol_to_graph as _mol_to_graph, SimpleGNN as _SimpleGNN, EnhancedGNN as _EnhancedGNN
        from .gnn_sensitivity import sensitivity_masking as _sensitivity_masking, example_model_fn_factory as _example_model_fn_factory
        from .multiscale import MultiScaleModel as _MultiScaleModel
        from .pinn import pinn_loss as _pinn_loss
    except Exception:
        from src.gnn import mol_to_graph as _mol_to_graph, SimpleGNN as _SimpleGNN, EnhancedGNN as _EnhancedGNN
        from src.gnn_sensitivity import sensitivity_masking as _sensitivity_masking, example_model_fn_factory as _example_model_fn_factory
        from src.multiscale import MultiScaleModel as _MultiScaleModel
        from src.pinn import pinn_loss as _pinn_loss

    return {
        "mol_to_graph": _mol_to_graph,
        "SimpleGNN": _SimpleGNN,
        "EnhancedGNN": _EnhancedGNN,
        "sensitivity_masking": _sensitivity_masking,
        "example_model_fn_factory": _example_model_fn_factory,
        "MultiScaleModel": _MultiScaleModel,
        "pinn_loss": _pinn_loss,
    }


@functools.lru_cache(maxsize=1)
def _load_rdkit_symbols() -> Dict[str, Any]:
    from rdkit import Chem as _Chem
    from rdkit.Chem import Draw as _Draw, AllChem as _AllChem
    from rdkit.Chem import Descriptors as _Descriptors, rdMolDescriptors as _rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D as _rdMolDraw2D

    return {
        "Chem": _Chem,
        "Draw": _Draw,
        "AllChem": _AllChem,
        "Descriptors": _Descriptors,
        "rdMolDescriptors": _rdMolDescriptors,
        "rdMolDraw2D": _rdMolDraw2D,
    }


@functools.lru_cache(maxsize=1)
def _load_heavy_ui_symbols() -> Dict[str, Any]:
    """Load heavy UI/runtime modules only when related pages are opened."""
    try:
        from .rl_sampling import AtomPolicyNet as _AtomPolicyNet, sample_atoms as _sample_atoms, reinforce_update as _reinforce_update
        from .drug.crawler import (
            crawl_pubchem_activity_dataset as _crawl_pubchem_activity_dataset,
            crawl_multiscale_training_dataset as _crawl_multiscale_training_dataset,
        )
        from .data_aug_denoise.ui import data_aug_denoise_ui as _data_aug_denoise_ui
    except Exception:
        from src.rl_sampling import AtomPolicyNet as _AtomPolicyNet, sample_atoms as _sample_atoms, reinforce_update as _reinforce_update
        from src.drug.crawler import (
            crawl_pubchem_activity_dataset as _crawl_pubchem_activity_dataset,
            crawl_multiscale_training_dataset as _crawl_multiscale_training_dataset,
        )
        from src.data_aug_denoise.ui import data_aug_denoise_ui as _data_aug_denoise_ui

    return {
        "AtomPolicyNet": _AtomPolicyNet,
        "sample_atoms": _sample_atoms,
        "reinforce_update": _reinforce_update,
        "crawl_pubchem_activity_dataset": _crawl_pubchem_activity_dataset,
        "crawl_multiscale_training_dataset": _crawl_multiscale_training_dataset,
        "data_aug_denoise_ui": _data_aug_denoise_ui,
    }


torch = _LazyModule("torch")
mol_to_graph = _LazySymbol("mol_to_graph", _load_gnn_symbols)
SimpleGNN = _LazySymbol("SimpleGNN", _load_gnn_symbols)
EnhancedGNN = _LazySymbol("EnhancedGNN", _load_gnn_symbols)
sensitivity_masking = _LazySymbol("sensitivity_masking", _load_gnn_symbols)
example_model_fn_factory = _LazySymbol("example_model_fn_factory", _load_gnn_symbols)
MultiScaleModel = _LazySymbol("MultiScaleModel", _load_gnn_symbols)
pinn_loss = _LazySymbol("pinn_loss", _load_gnn_symbols)
Chem = _LazySymbol("Chem", _load_rdkit_symbols)
Draw = _LazySymbol("Draw", _load_rdkit_symbols)
AllChem = _LazySymbol("AllChem", _load_rdkit_symbols)
Descriptors = _LazySymbol("Descriptors", _load_rdkit_symbols)
rdMolDescriptors = _LazySymbol("rdMolDescriptors", _load_rdkit_symbols)
rdMolDraw2D = _LazySymbol("rdMolDraw2D", _load_rdkit_symbols)
AtomPolicyNet = _LazySymbol("AtomPolicyNet", _load_heavy_ui_symbols)
sample_atoms = _LazySymbol("sample_atoms", _load_heavy_ui_symbols)
reinforce_update = _LazySymbol("reinforce_update", _load_heavy_ui_symbols)
crawl_pubchem_activity_dataset = _LazySymbol("crawl_pubchem_activity_dataset", _load_heavy_ui_symbols)
crawl_multiscale_training_dataset = _LazySymbol("crawl_multiscale_training_dataset", _load_heavy_ui_symbols)
data_aug_denoise_ui = _LazySymbol("data_aug_denoise_ui", _load_heavy_ui_symbols)

# Ensure project src/ is on sys.path so `from src...` imports work when running this file directly
_THIS_FILE = Path(__file__).resolve()
# Be robust: ensure the parent folders that contain `src/` are on sys.path.
# Common layout: <project-root>/src/frontend.py -> project-root should be on sys.path
_PROJECT_ROOT = _THIS_FILE.parents[1]
_CANDIDATE_PATHS = [
    _PROJECT_ROOT,      # directory that contains `src/` (新建文件夹)
    _PROJECT_ROOT.parent,  # possible workspace root (IGEM集成方案)
]
for p in _CANDIDATE_PATHS:
    try:
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    except Exception:
        pass

from PIL import Image
import io as _io
import streamlit.components.v1 as components

# Allow running this file directly (or via streamlit) from outside the project root.
# When executed as a script, Python puts this file's directory (./src) on sys.path,
# which breaks absolute imports like `from src...`.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Try package-relative imports first (when run as module), fallback to absolute imports.
try:
    from .common.dataset_fetch import concat_tables
    from .common.agent_api import call_openai_chat, call_raw_json, safe_parse_json
    from .common.notify import load_feishu_config_from_env, send_feishu_webhook
    from .common.plotting import save_regression_diagnostic_plots
    from .common.training import EarlyStopping, build_scheduler

    from .epitope.featurizer import SequenceFeatures
    from .epitope.predictor import EpitopeModelBundle, build_model, infer_env_cols, make_xy, predict_one, train_bundle
    from .epitope.training_eval import evaluate_epitope_from_csv, train_epitope_from_csv
    from .epitope.sensitivity import (
        group_importance,
        sensitivity_from_bundle,
        top_features,
        wetlab_takeaway,
        sensitivity_report,
        format_sensitivity_report,
    )
    from .common.optim.hyperopt import run_hyper_search
    from .ui.common import (
        inject_global_styles as _inject_global_styles,
        render_performance_sidebar,
        render_whitebox_panel,
        preview_rows as _preview_rows,
        preview_df as _preview_df,
    )
except Exception:
    from src.common.dataset_fetch import concat_tables
    from src.common.agent_api import call_openai_chat, call_raw_json, safe_parse_json
    from src.common.notify import load_feishu_config_from_env, send_feishu_webhook
    from src.common.plotting import save_regression_diagnostic_plots
    from src.common.training import EarlyStopping, build_scheduler

    from src.epitope.featurizer import SequenceFeatures
    from src.epitope.predictor import EpitopeModelBundle, build_model, infer_env_cols, make_xy, predict_one, train_bundle
    from src.epitope.training_eval import evaluate_epitope_from_csv, train_epitope_from_csv
    from src.epitope.sensitivity import (
        group_importance,
        sensitivity_from_bundle,
        top_features,
        wetlab_takeaway,
        sensitivity_report,
        format_sensitivity_report,
    )
    from src.common.optim.hyperopt import run_hyper_search
    from src.ui.common import (
        inject_global_styles as _inject_global_styles,
        render_performance_sidebar,
        render_whitebox_panel,
        preview_rows as _preview_rows,
        preview_df as _preview_df,
    )


APP_TITLE = "confluencia:IGEM-FBH 虚拟筛选前端"
IGEM_FBH_URL = os.getenv("IGEM_FBH_URL", "https://igem.org")


# --- 序列编码工具函数 -------------------------------------------------
AMINO_ACIDS = list("ARNDCQEGHILKMFPSTWYV")


try:
    from .epitope.encoding import (
        load_aaindex_from_csv,
        one_hot_encode,
        sequence_to_aaindex,
        continuous_onehot_encode,
    )
except Exception:
    from src.epitope.encoding import (
        load_aaindex_from_csv,
        one_hot_encode,
        sequence_to_aaindex,
        continuous_onehot_encode,
    )

# ---------------------------------------------------------------------


def _feedback_log_path() -> Path:
    log_dir = _PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "feedback.csv"


def set_feedback_context(module: str, page: str, extra: Optional[Dict[str, Any]] = None) -> None:
    ctx: Dict[str, Any] = {
        "module": str(module),
        "page": str(page),
    }
    if extra:
        ctx.update(extra)
    st.session_state["_feedback_context"] = ctx


def update_feedback_context(extra: Dict[str, Any]) -> None:
    ctx = st.session_state.get("_feedback_context")
    if not isinstance(ctx, dict):
        ctx = {}
    ctx.update(extra)
    st.session_state["_feedback_context"] = ctx


def _append_feedback_row(row: Dict[str, Any]) -> None:
    path = _feedback_log_path()

    base_fields = [
        "ts_utc",
        "module",
        "page",
        "rating",
        "helpful",
        "expected",
        "comment",
        "contact",
        "context_json",
    ]

    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=base_fields, extrasaction="ignore")
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in base_fields})


def release_runtime_memory(*, clear_streamlit_cache: bool = False) -> None:
    """Best-effort memory release hook for long Streamlit sessions."""
    if clear_streamlit_cache:
        try:
            st.cache_data.clear()
        except Exception:
            pass
        try:
            st.cache_resource.clear()
        except Exception:
            pass

    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    gc.collect()

def render_feedback_sidebar() -> None:
    with st.sidebar.expander("用户反馈", expanded=False):
        st.markdown("如需反馈，请发送邮件至：**18806370529@163.com**")
        ctx = st.session_state.get("_feedback_context") if isinstance(st.session_state.get("_feedback_context"), dict) else {}
        with st.form("feedback_form"):
            rating = st.slider("满意度（1-5）", min_value=1, max_value=5, value=3, step=1)
            helpful = st.selectbox("是否解决问题", options=["是", "部分", "否"], index=1)
            expected = st.text_input("期望结果（可选）", value="")
            comment = st.text_area("问题描述/建议", value="", height=100)
            contact = st.text_input("联系方式（可选）", value="")
            include_ctx = st.checkbox("附带当前页面上下文", value=True)
            submitted = st.form_submit_button("提交反馈")

        if include_ctx and isinstance(ctx, dict) and ctx:
            with st.expander("当前上下文预览", expanded=False):
                st.json(ctx)

        if submitted:
            row = {
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "module": str(ctx.get("module", "")),
                "page": str(ctx.get("page", "")),
                "rating": int(rating),
                "helpful": str(helpful),
                "expected": str(expected),
                "comment": str(comment),
                "contact": str(contact),
                "context_json": json.dumps(ctx, ensure_ascii=False) if include_ctx and isinstance(ctx, dict) else "",
            }
            _append_feedback_row(row)
            st.success("已提交反馈，感谢支持！")

        log_path = _feedback_log_path()
        if log_path.exists():
            st.download_button(
                "下载反馈日志",
                data=log_path.read_bytes(),
                file_name=log_path.name,
                mime="text/csv",
                key="feedback_log_download",
            )


def render_user_guidance_sidebar() -> None:
    with st.sidebar.expander("新手指南", expanded=False):
        st.markdown(
            """
**傻瓜式使用指南**  
不需要写代码，不需要懂机器学习。准备一份 CSV 就能完成预测。  

**三步完成预测**  
1) 选模块：药效预测（SMILES）或 表位预测（序列）。  
2) 上传 CSV：第一行是列名，目标列是数值。  
3) 有标签先训练，没标签用批量筛选导出结果。  

**一分钟上手（最短路径）**  
- 打开“药效预测/表位预测”里的“单条预测”。  
- 输入一条数据，点“预测”，立刻出结果。  
- 如果想批量：切到“批量筛选”，上传 CSV 一键导出。  

**CSV 模板要点**  
- 必填列：药效= `smiles`；表位= `sequence`。  
- 目标列：你要预测的数值列名（如 `efficacy`/`fluorescence`）。  
- 条件列（可选）：如 `dose`、`freq`、`route`（必须是数字）。  
- 不要有空列、不要混入单位或中文。  

**数据获取（分子生成与爬虫）**  
- 分子生成：先准备“种子 SMILES”（CSV 的 `smiles` 列或手动列表），生成候选并导出 CSV，核心列是 `smiles` 与 `score`。  
- 生成结果可附带性质列：如 QED、MW、logP、TPSA，用于二次筛选。  
- 爬虫采集：支持公开数据库（如 PubChem）或 URL/本地 CSV 合并；先小批量测试，再统一列名与数值格式。  
- 合规提醒：只抓取公开可用数据，尊重网站条款与频率限制。  
- 生成示例：`python src/drug_cli.py generate --data data/example_drug.csv --smiles-col smiles --out data/gen_out.csv --top-k 200 --with-props`  
- 爬虫示例：`python src/drug_cli.py crawl --site pubchem --start-cid 1 --n 200 --out data/pubchem_activity.csv`  

**原理解释（分子生成与爬虫）**  
- 分子生成：以“种子分子指纹”为起点，进行变异/交叉搜索，按打分函数（模型预测或 QED 等）排序。  
- 典型输出：`rank`、`smiles`、`score`，可选性质列（QED/MW/logP/TPSA）。  
- 爬虫采集：通过 API/网页查询条件拉取条目，规范化为表格列（如 `cid`、`smiles`、`activity_score`）。  
- 生成与爬虫的作用：扩大候选与训练样本范围，提高筛选覆盖度。  

**小词典**  
- SMILES：分子的简写公式（如 CCO）。  
- 序列：氨基酸字母串（如 SIINFEKL）。  
- 目标列：你想预测的数值列名。  

**最常见的 4 个错误**  
1) 列名写错（大小写、空格）。  
2) 目标列不是数值。  
3) 模型混用（药效/表位）。  
4) CSV 里混入单位或中文。  

**怎么判断结果好不好**  
- 训练后看“绘图”页：点图接近对角线表示拟合较好。  
- 批量筛选后：按预测值排序，先看 Top-N。  

**结果解释（怎么看预测值）**  
- 预测值是模型对“目标列”的估计，含义由你的目标列定义（如活性、信号强度）。  
- 越大/越小不一定更好：请以实验目标为准（例如“越小越好”的毒性指标）。  
- 单条预测只适合快速验证流程；真正筛选请用批量结果对比排序。  
- 若有条件列（dose/freq/route），同一分子在不同条件下会有不同预测值。  
- 不确定时：先用样例数据训练，再用已知样本验证方向是否正确。  

**结果解释原理（为什么会有这个值）**  
- 模型先把 SMILES/序列/条件转换成数值特征，再学习它们与目标列之间的映射关系。  
- 预测值本质是“在当前输入特征下，目标列的期望估计”。  
- 训练数据分布决定输出范围：输入越接近训练集，结果通常越稳定。  
- 若出现异常大/小值，可能是外推结果：建议先检查输入是否合法，再用已知样本做对照。  

**没有模型怎么办**  
- 用样例 CSV 先训练一个模型，再用它做预测。  
- 或者只做“单条预测”，快速验证流程。  

**训练介绍（原理 + 实操）**  
**训练原理（用一句话理解）**  
模型会学习“输入特征 -> 目标数值”的映射，也就是从 SMILES/序列/条件中总结规律。  

**训练前要准备什么**  
- 一份 CSV：包含输入列 + 目标列。  
- 目标列必须是数值（例如 `efficacy`、`fluorescence`）。  
- 条件列可选，但必须是数值（如 `dose`、`freq`）。  

**训练步骤（详细版）**  
1) 进入“模型训练”页，选择对应模块。  
2) 上传带目标列的 CSV，确认列名无误。  
3) 选择模型类型（如 `hgb`/`rf`/`mlp`）。  
4) 设置评估方式（交叉验证或留出）。  
5) 点击“训练”，等待完成。  
6) 下载模型文件，用于“单条预测/批量筛选”。  

**训练结果怎么看**  
- 进入“绘图”页：散点越接近对角线越好。  
- 误差越小越好：MAE、RMSE 越低越好。  
- R2 越接近 1 越好。  

**常见问题与修正**  
- 结果很差：检查目标列是否正确、是否含单位/中文。  
- 过拟合：换更简单模型（`hgb/rf`）或减少特征。  
- 训练报错：多为列名不匹配或数据类型问题。  
"""
        )

        # 新增：近期功能简要（保留原文，不覆盖）
        with st.expander("其他功能", expanded=False):
            st.markdown(
                """
- 原子/片段敏感性分析（GNN）：可视化每个原子对预测的影响，支持遮蔽对比（masking）。  
- 多尺度耦合（GNN + PINN）：将分子嵌入与物理约束结合，用于过程曲线与 PDE 系数估计（CoeffNet）。  
- 强化学习采样（AtomPolicyNet）：用于候选分子片段优先采样，提高搜索效率。  
- 表格数据增强与去噪：VAE 风格生成器用于扩增或去噪训练集（UI 支持导出增强数据）。  
- 白盒解释与绘图诊断：训练后直接生成残差图、重要性报告与可下载图片。  
- Agent/助手接口：集成辅助对话（OpenAI API）以协助数据准备与结果解释。  
"""
            )

        # 详细说明每项新增功能（保留原文）
        with st.expander("功能详情：原子/片段敏感性（GNN）", expanded=False):
            st.markdown(
                """
功能说明：计算并可视化每个原子或片段对模型预测的贡献（热力图 + 列表）。

使用方法：在“多尺度分析 / 原子敏感性”页输入 `smiles`，选择 GNN 模型并运行；可导出原子贡献表（CSV）与分子图像（PNG）。

原理简述：通过遮蔽（masking）或计算模型输出对输入嵌入的梯度，评估每个原子/边对输出的影响分数。常用方法包括敏感性差分与梯度归一化。

注意事项：结果依赖模型训练分布与规范化，外推分子可能产生误导性高贡献点；建议与化学性质（如 QED）联合判断。
"""
            )

        with st.expander("功能详情：多尺度耦合（GNN + PINN）", expanded=False):
            st.markdown(
                """
功能说明：将分子级别的 GNN 嵌入与物理信息神经网络（PINN）联合训练，输出过程曲线与 PDE/动力学系数估计（如 D、Vmax、Km）。

使用方法：在“多尺度”页提供 SMILES、初始条件与观测数据，启用 PINN 选项并设置训练轮数；短训模式可在本地演示，批量/云端模式推荐 GPU。 

原理简述：在损失函数中加入 PDE 残差项，使网络在拟合数据的同时满足物理方程约束，从而提高外推稳定性与可解释性。CoeffNet 将分子嵌入映射到 PDE 系数空间。

注意事项：训练时间与硬件要求较高；若观测数据稀疏，建议先做局部短训或启用正则项。  
"""
            )

        with st.expander("功能详情：强化学习采样（AtomPolicyNet）", expanded=False):
            st.markdown(
                """
功能说明：使用策略网络在分子构建过程中优先采样高得分片段，提高候选集质量和多样性。

使用方法：在“采样/生成”页选择 RL 策略，设置预算（episodes、steps、top-k），启动采样并导出候选 CSV（包含 `smiles` 与策略得分）。

原理简述：基于策略梯度或 REINFORCE 更新策略网络，使其在构建分子时偏向获得更高的外部奖励（如训练模型预测分数或合成可行性）。

注意事项：奖励函数设计关键；不稳定训练可能导致模式坍缩，应启用熵或多种奖励组合以保持多样性。  
"""
            )

        with st.expander("功能详情：表格数据增强与去噪（VAE）", expanded=False):
            st.markdown(
                """
功能说明：对小样本或噪声数据使用 VAE/生成模型进行增强或去噪，支持导出增强后的 CSV 以用于训练或预筛选。

使用方法：进入“数据增强与去噪”页面，上传原始表格，选择增强模式（合成 / 重构去噪），预览结果并导出。可设置增强倍率与保守阈值。 

原理简述：VAE 学习表格数据的潜在分布，通过在潜在空间采样生成新样本或用解码器重构降低噪声。结合条件化生成可保留重要属性。 

注意事项：生成数据不能替代真实标签；用于扩充时建议保留部分真实数据用于验证。  
"""
            )

        with st.expander("功能详情：白盒解释与绘图诊断", expanded=False):
            st.markdown(
                """
功能说明：训练完成后自动或手动生成残差图、特征重要性（Permutation/SHAP/Group importance）、以及导出报告与 PNG 图像。 

使用方法：训练完成页面点击“生成白盒报告”，或在模型评估页选择“导出诊断图”。支持下载 PDF/PNG/CSV。 

原理简述：通过模型自带的特征重要性方法（如树模型的 gain）、Permutation importance 或基于样本的贡献度（group_importance）来解释模型决策路径。 

注意事项：解释方法各有偏差，建议结合多种指标与实际化学/生物知识交叉验证。  
"""
            )

        with st.expander("功能详情：Agent / 助手接口（OpenAI 集成）", expanded=False):
            st.markdown(
                """
功能说明：内置交互式助手，用于列名映射、数据清洗建议、生成示例脚本与快速诊断（需配置 API Key）。

使用方法：打开侧边或独立助手面板，粘贴样本数据或描述任务；可请求“生成清洗脚本”或“列名映射建议”，并将建议下载为脚本。 

原理简述：助手通过调用对话式模型（OpenAI）对自然语言或示例数据做解析并生成建议文本/代码；本质是大模型的提示工程与自动化文本生成。 

注意事项：需要设置环境变量 `OPENAI_API_KEY`；不要上传敏感/受限数据到外部 API；模型建议需人工复核。  
"""
            )

        st.markdown("**模块原理与功能（点击展开）**")
        with st.expander("药效预测（SMILES + 条件 -> 疗效）", expanded=False):
            st.markdown(
                """
- 原理：把分子结构（SMILES）转成数值特征，再和实验条件一起输入模型，预测疗效数值。  
- 适合：有分子结构、想估计活性/效果的场景。  
- 能做：单条预测、批量筛选、训练模型、绘图诊断。  
"""
            )
        with st.expander("表位预测（序列 + 条件 -> 信号/疗效）", expanded=False):
            st.markdown(
                """
- 原理：把氨基酸序列转成特征（长度、组成、理化性质等），与条件一起预测效果。  
- 适合：有肽段序列、想筛出更可能有效的序列。  
- 能做：单条预测、批量筛选、敏感性分析、训练与绘图。  
"""
            )
        with st.expander("对接预测（SMILES + 蛋白序列 -> 对接分数）", expanded=False):
            st.markdown(
                """
- 原理：用交叉注意力把分子与蛋白序列的信息对齐，输出对接/结合评分。  
- 适合：同时有小分子与靶标蛋白序列，想快速比较潜在结合强弱。  
- 能做：单条预测、批量筛选、训练对接模型（有数据时）。  
"""
            )
        with st.expander("数据增强与去噪（表格 VAE）", expanded=False):
            st.markdown(
                """
- 原理：用生成模型学习表格分布，生成更“平滑”的数据或补充样本。  
- 适合：样本量小、噪声多、需要扩充数据时。  
- 能做：数据增强、重构去噪、导出增强后的 CSV。  
"""
            )
        with st.expander("模型训练与评估", expanded=False):
            st.markdown(
                """
- 原理：用你的标注数据拟合模型，并用验证指标评估效果。  
- 适合：有目标列（数值）且想定制自己的模型。  
- 能做：训练、保存模型、绘图诊断（散点/残差）。  
"""
            )
        with st.expander("多尺度分析（GNN-PINN）", expanded=False):
            st.markdown(
                """
- 原理：分子级别的图神经网络提取信息，物理方程约束宏观过程。  
- 适合：研究/解释导向的探索，不是日常必用。  
- 能做（功能较多）：  
    - 原子敏感性分析：找出对结果影响最大的原子/片段。  
    - Mask 对比曲线：比较“遮蔽原子前后”的预测变化。  
    - PINN 本地短训：快速演示微观到宏观的耦合。  
    - 云端 PINN 批量任务：对多条 SMILES 批量提交并汇总结果。  
    - CoeffNet：从分子嵌入预测 PDE 系数（如 D/Vmax/Km）。  
    - E(3)-等变 GNN：在三维几何上更稳定的结构建模。  
    - 物理势消息：LJ/电势等物理项引入到消息传递。  
    - 预测目标扩展：ADMET 相关指标与结合评分（可选）。  

- 多尺度分析可预测
    - 原子敏感性得分：每个原子对输出的影响权重。  
    - PINN 过程曲线：随时间/空间的响应变化。  
    - PDE 系数：D / Vmax / Km（若启用 CoeffNet 或提供参数）。  
    - 结合相关评分：Binding affinity（可选）。  
    - ADMET 指标：如 MolLogP、ESOL、TPSA/MW（可选）。  
"""
            )

        with st.expander("常用参数说明（点开看）", expanded=False):
            st.markdown(
                """
**数据与列名**  
- `smiles`：分子结构列（必填于药效/对接）。  
- `sequence`：氨基酸序列列（必填于表位）。  
- `protein`：蛋白序列列（对接需要）。  
- `target`：目标列名（你要预测的数值列）。  

**预测与筛选**  
- `model`：训练好的模型文件（.joblib/.pkl/.pt）。  
- `candidates`：候选 CSV（用于批量筛选）。  
- `out` / `out_col`：输出文件与预测列名。  
- `param key=value`：条件参数（如 dose=10、freq=2）。  

**训练常见参数**  
- `model_type`：模型类型（如 hgb、rf、gbr、mlp）。  
- `cv` / `split`：交叉验证或留出比例。  
- `seed`：随机种子（用于可复现）。  

**对接训练参数**  
- `smiles_col` / `protein_col`：指定列名。  
- `epochs`：训练轮数（越大越慢）。  
- `batch_size`：批大小（越大越快但更吃内存）。  
- `lr`：学习率（过大易不稳定）。  

**结果解读**  
- 预测值是连续数值：越大不一定越好，取决于你的目标列含义。  
- 批量筛选后先排序，再看 Top-N。  
"""
            )

        with st.expander("参数词典（点开看）", expanded=False):
            st.markdown(
                """
**训练相关**  
- `epochs`：训练轮数，越大越慢，过大可能过拟合。  
- `batch_size`：每次训练的样本数，越大越快但更吃内存。  
- `lr`：学习率，过大不稳定，过小收敛慢。  
- `weight_decay`：权重衰减，防止过拟合。  
- `dropout`：随机失活比例，防止过拟合。  

**数据与列名**  
- `smiles`：分子结构列。  
- `sequence`：氨基酸序列列。  
- `protein`：蛋白序列列。  
- `target`：目标列（你要预测的数值）。  
- `env_cols`：条件列名列表（如剂量/频次）。  

**筛选与输出**  
- `candidates`：候选数据表。  
- `out`：输出文件路径。  
- `out_col`：输出预测列名。  

**对接相关**  
- `smiles_col` / `protein_col`：指定列名。  
- `hidden`：模型隐藏维度。  
- `use_lstm`：是否加入 LSTM 结构。  
- `distill_weight`：蒸馏权重（教师模型影响比例）。  

**多尺度相关（GNN-PINN）**  
- `steps`：GNN 消息传递步数。  
- `readout`：图读出方式（mean/attention）。  
- `D`：扩散系数。  
- `Vmax`：最大反应速率。  
- `Km`：米氏常数。  
"""
            )

        with st.expander("差分进化与序列编码快速提示", expanded=False):
            st.markdown(
                """
- 差分进化（DE）：前端“单条预测”面板已支持在数值型环境变量空间中进行全局搜索，设置每个 env 的上下界后点击“运行差分进化建议环境”即可获得建议参数与对应预测值。CLI 对应命令：`epitope_cli suggest-env` / `drug_cli suggest-env`。
- 使用场景：设计实验条件、寻找对模型输出影响最大的参数组合或进行快速候选优化。
- 序列编码（AAIndex / one-hot）：工具位于 `src/epitope/encoding.py`，提供 `one_hot_encode`, `load_aaindex_from_csv`, `sequence_to_aaindex`, `continuous_onehot_encode`。在需要把序列转为连续特征或拼接 one-hot 时使用。
- 小贴士：DE 为随机全局优化，建议先用小预算（`max_iter`~50，`pop_size`~5*D）快速验证，再扩大搜索；为可复现可设随机种子。
"""
            )

        with st.expander("模块模型介绍（点开看）", expanded=False):
            st.markdown(
                """
**药效预测常见模型**  
- `hgb`：直方图梯度提升，速度快、对小样本友好，推荐首选。  
- `rf`：随机森林，稳健、对噪声不敏感，适合基线模型。  
- `gbr`：梯度提升回归，能拟合非线性，但更容易过拟合。  
- `mlp`：小型神经网络，数据量较大时更有优势。  
- 选型建议：小样本优先 `hgb/rf`，大样本可尝试 `mlp`。  

**表位预测常见模型**  
- 传统回归/分类模型：基于序列特征（长度、组成、理化性质）。  
- 小样本优先用轻量模型，稳定、易解释。  
- 若有更多数据，可尝试更复杂模型提升上限。  

**对接预测模型**  
- 交叉注意力模型：同时看分子与蛋白序列，对齐后输出对接分数。  
- 需要 `smiles` + `protein` 两列，缺一不可。  
- 有 GPU 更快；无 GPU 也能跑但更慢。  

**数据增强与去噪模型**  
- 表格 VAE：学习数据分布后生成或重构表格。  
- 适合扩充样本或平滑噪声；不要替代真实标签。  

**多尺度分析模型**  
- GNN：分子层面的结构表征（Simple/Enhanced/PhysicsMessage）。  
- SimpleGNN：速度快，适合快速验证。  
- EnhancedGNN：带注意力，表征能力更强。  
- PhysicsMessageGNN：加入物理势（LJ/电势）信息，解释性更好。  
- PINN：用物理方程约束宏观过程，偏研究探索用途。  
- 推荐先用默认参数跑通，再逐步调参。  

**结果指标怎么理解**  
- MAE：平均绝对误差，越小越好，直观稳定。  
- RMSE：均方根误差，对大误差更敏感。  
- R2：拟合优度，越接近 1 越好。  
"""
            )

        with st.expander("模型选择速查（点开看）", expanded=False):
            st.markdown(
                """
**新手快速选择流程**  
1) 数据量 < 1000：先选 `hgb` 或 `rf`。  
2) 数据量 >= 1000 且特征丰富：尝试 `gbr` 或 `mlp`。  
3) 只想快出结果：`hgb`。  
4) 对稳定性要求高：`rf`。  

**优缺点对照（药效预测常用）**  
| 模型 | 优点 | 风险/代价 | 适用场景 |  
|---|---|---|---|  
| hgb | 快、稳、对小样本友好 | 过度调参反而不稳 | 快速基线/小样本 |  
| rf | 抗噪、默认强 | 模型体积大 | 稳健筛选 |  
| gbr | 非线性拟合强 | 易过拟合 | 有一定数据量 |  
| mlp | 上限高 | 需更多数据、易波动 | 大样本/非线性 |  

**对接模型选择**  
- 只有 CPU：能跑但慢，建议少量预测。  
- 有 GPU：适合批量筛选。  
"""
            )

        st.markdown("**一键样例下载**")
        sample_drug = _PROJECT_ROOT / "data" / "example_drug.csv"
        sample_epitope = _PROJECT_ROOT / "data" / "example_epitope.csv"
        sample_drug_unlabeled = _PROJECT_ROOT / "data" / "example_drug_unlabeled.csv"
        sample_epitope_unlabeled = _PROJECT_ROOT / "data" / "example_epitope_unlabeled.csv"

        if sample_drug.exists():
            st.download_button("药效训练样例", data=sample_drug.read_bytes(), file_name=sample_drug.name, mime="text/csv")
        else:
            st.caption("未找到药效训练样例")
        if sample_epitope.exists():
            st.download_button("表位训练样例", data=sample_epitope.read_bytes(), file_name=sample_epitope.name, mime="text/csv")
        else:
            st.caption("未找到表位训练样例")
        if sample_drug_unlabeled.exists():
            st.download_button("药效筛选样例", data=sample_drug_unlabeled.read_bytes(), file_name=sample_drug_unlabeled.name, mime="text/csv")
        else:
            st.caption("未找到药效筛选样例")
        if sample_epitope_unlabeled.exists():
            st.download_button("表位筛选样例", data=sample_epitope_unlabeled.read_bytes(), file_name=sample_epitope_unlabeled.name, mime="text/csv")
        else:
            st.caption("未找到表位筛选样例")

        st.markdown(
            f"""
**高级功能**：去“多尺度分析 / 模型训练 / 数据处理”页。  
**反馈入口**：侧边栏“用户反馈”。  
**IGEM-FBH 官方网页**：[{IGEM_FBH_URL}]({IGEM_FBH_URL})
"""
        )


def render_acknowledgement_sidebar() -> None:
    with st.sidebar.expander("鸣谢", expanded=False):
        st.markdown(
            """
感谢以下团队与社区的支持与贡献：  
- IGEM 团队：项目组织与反馈。
- DLEPS 团队：方法论与算法支持。
- GitHub、Kaggle 等开源社区与数据生态。  
"""
        )


def render_multiscale_sidebar() -> None:
    run = False
    train_demo = False
    train_epochs = int(st.session_state.get("ms_train_epochs", 8))

    tabs = st.tabs(["基础", "物理势/等变", "预测目标", "模型上传", "运行训练", "RL 采样", "显示设置"])

    with tabs[0]:
        st.caption("输入分子 SMILES，计算原子敏感性并演示 PINN 微观→宏观耦合示例。")
        crawled_ms = st.session_state.get("multiscale_crawl_df")
        if isinstance(crawled_ms, pd.DataFrame) and "smiles" in crawled_ms.columns and len(crawled_ms) > 0:
            st.caption("可从爬虫抓取的多尺度数据中选择 SMILES")
            options = [s for s in crawled_ms["smiles"].astype(str).tolist() if s]
            sel = st.selectbox("选择 SMILES（来自爬虫）", options=options, index=0, key="ms_crawl_smiles_select")
            smiles = st.text_input("SMILES", value=str(sel), key="ms_smiles")
            # 自动填充 PDE 参数（若存在）
            try:
                row = crawled_ms[crawled_ms["smiles"].astype(str) == str(sel)].iloc[0]
                if "D" in crawled_ms.columns:
                    st.session_state["ms_default_D"] = float(row.get("D", st.session_state.get("ms_default_D", 0.1)))
                if "Vmax" in crawled_ms.columns:
                    st.session_state["ms_default_Vmax"] = float(row.get("Vmax", st.session_state.get("ms_default_Vmax", 0.5)))
                if "Km" in crawled_ms.columns:
                    st.session_state["ms_default_Km"] = float(row.get("Km", st.session_state.get("ms_default_Km", 0.1)))
            except Exception:
                pass
        else:
            smiles = st.text_input("SMILES", value=st.session_state.get("ms_smiles", "CCO"), key="ms_smiles")
        steps = st.number_input("GNN 消息传递步数", min_value=1, max_value=8, value=int(st.session_state.get("ms_steps", 3)), step=1, key="ms_steps")
        hidden = st.number_input("GNN 隐藏维度", min_value=8, max_value=256, value=64, step=8, key="ms_hidden")
        gnn_dropout = st.slider("GNN Dropout", min_value=0.0, max_value=0.5, value=0.1, step=0.05, key="ms_gnn_dropout")
        model_type = st.selectbox("GNN 类型", options=["SimpleGNN", "EnhancedGNN", "PhysicsMessageGNN", "E(3)-EquivariantGNN"], index=1 if st.session_state.get("ms_use_enhanced", True) else 0, key="ms_model_type")
        gat_heads = st.number_input("GAT 头数 (仅 EnhancedGNN)", min_value=1, max_value=8, value=4, step=1, key="ms_gat_heads")
        use_physics = st.checkbox("在 GAT 中使用物理调制 (距离/角度)", value=True, key="ms_use_physics")
        enable_coeff = st.checkbox("启用 coeff_net（分子嵌入 -> PDE 系数）", value=False, key="ms_enable_coeff")
        coeff_hidden = st.number_input("CoeffNet 隐藏维度", min_value=8, max_value=256, value=64, step=8, key="ms_coeff_hidden")
        readout_type = st.selectbox("Readout 类型", options=["mean", "attention"], index=0, key="ms_readout")
        st.markdown("**PDE 默认参数（可用于 PINN 训练/演示）**")
        dcol1, dcol2, dcol3 = st.columns(3)
        with dcol1:
            st.number_input("D", min_value=0.0, value=float(st.session_state.get("ms_default_D", 0.1)), step=0.01, key="ms_default_D")
        with dcol2:
            st.number_input("Vmax", min_value=0.0, value=float(st.session_state.get("ms_default_Vmax", 0.5)), step=0.05, key="ms_default_Vmax")
        with dcol3:
            st.number_input("Km", min_value=0.0, value=float(st.session_state.get("ms_default_Km", 0.1)), step=0.01, key="ms_default_Km")

    with tabs[1]:
        st.caption("物理势参数（仅在 PhysicsMessageGNN 时生效）")
        potential_type = st.selectbox("potential_type", options=["auto", "lennard", "electrostatic"], index=0, key="ms_potential_type")
        lj_epsilon = st.number_input("LJ epsilon", min_value=0.0, max_value=10.0, value=0.1, format="%g", key="ms_lj_eps")
        lj_sigma = st.number_input("LJ sigma (Å)", min_value=0.1, max_value=10.0, value=3.5, format="%g", key="ms_lj_sigma")
        dielectric = st.number_input("电介常数", min_value=1.0, max_value=1000.0, value=80.0, format="%g", key="ms_dielectric")
        st.markdown("---")
        st.caption("E(3)-等变 GNN 参数（用于 E(3)-EquivariantGNN）")
        eg_layers = st.number_input("EGNN 层数", min_value=1, max_value=12, value=3, step=1, key="ms_eg_layers")
        eg_hidden = st.number_input("EGNN 隐藏维度", min_value=8, max_value=512, value=64, step=8, key="ms_eg_hidden")
        st.caption("说明: LJ epsilon 控制势深度（越大吸引/排斥越强），sigma 为粒子尺寸标度（Å）。'auto' 会把 LJ 作为软项混入特征相似性衰减；'lennard' 则以 LJ 为主；'electrostatic' 目前为距离衰减近似（若提供原子电荷可扩展为 Coulomb）。")

    with tabs[2]:
        prediction_targets = st.multiselect(
            "预测目标（可多选）",
            options=[
                "原子敏感性",
                "Binding affinity (protein-ligand)",
                "ADMET: MolLogP",
                "ADMET: Solubility (ESOL)",
                "ADMET: TPSA & MW",
                "PDE coefficients (CoeffNet)",
            ],
            default=st.session_state.get("ms_prediction_targets", ["原子敏感性"]),
            key="ms_prediction_targets",
        )

    with tabs[3]:
        st.caption("模型上传（可选）：提供训练好的模型以获得更可靠的预测。")
        st.markdown("**ADMET 模型说明**: 上传的回归模型应以 joblib 保存（`.pkl`/`.joblib`），并以以下描述向量为输入顺序: [MolLogP, MolWt, RotatableBonds, TPSA, AromaticProportion]。若模型是 sklearn 风格，将使用 `n_features_in_` 做快速校验。")
        admet_model_up = st.file_uploader("上传 ADMET 回归模型 (joblib: .pkl/.joblib)", type=["pkl", "joblib"], key="ms_admet_model_up")
        st.markdown("---")
        st.caption("对接模型上传（可选）：上传 PyTorch `.pth/.pt` 文件以用自定义权重进行预测")
        pl_model_up = st.file_uploader("上传 对接 模型 (.pth/.pt, PyTorch)", type=["pth", "pt"], key="ms_pl_model_up")
        st.caption("若上传模型与默认架构不完全匹配，可选择隐藏维度并启用自动匹配尝试（会用 `strict=False` 加载匹配的参数）。")
        pl_hidden_choice = st.selectbox("PL 模型隐藏维度 (用于尝试构建模型以加载 state_dict)", options=[32, 64, 128, 256], index=1, key="ms_pl_hidden")
        pl_autofit = st.checkbox("尝试自动匹配并部分加载 state_dict (strict=False)", value=True, key="ms_pl_autofit")
        st.caption("示例：蛋白口袋 CSV 样例下载")
        if st.button("下载蛋白口袋样例 CSV", key="ms_download_prot_example"):
            import io as _io

            sample = pd.DataFrame(
                {
                    "element": ["C", "N", "O", "C"],
                    "x": [0.0, 1.2, -0.8, 0.5],
                    "y": [0.0, 0.1, -1.0, 1.5],
                    "z": [0.0, -0.2, 0.5, -0.7],
                }
            )
            buf = _io.StringIO()
            sample.to_csv(buf, index=False)
            buf.seek(0)
            st.download_button("下载示例 CSV", data=buf.getvalue(), file_name="protein_pocket_example.csv", mime="text/csv")

    with tabs[4]:
        run = st.button("运行多尺度分析", key="ms_run")
        train_demo = st.button("训练并演示 PINN（本地短训）", key="ms_train_demo")
        train_epochs = st.number_input("PINN 训练轮数（示例）", min_value=1, max_value=200, value=8, step=1, key="ms_train_epochs")
        pinn_lr = st.number_input("PINN 学习率", min_value=1e-6, max_value=1e-1, value=1e-3, format="%g", key="ms_pinn_lr")
        pinn_weight_decay = st.number_input("PINN weight_decay", min_value=0.0, max_value=1e-2, value=1e-4, format="%.6f", key="ms_pinn_wd")
        pinn_lr_schedule = st.selectbox("PINN 学习率调度", options=["cosine", "step", "none"], index=0, key="ms_pinn_sched")
        pinn_step_size = st.number_input("PINN 阶梯步长", min_value=1, max_value=200, value=20, step=1, key="ms_pinn_step")
        pinn_gamma = st.number_input("PINN 阶梯衰减系数", min_value=0.1, max_value=0.99, value=0.5, step=0.05, key="ms_pinn_gamma")
        pinn_min_lr = st.number_input("PINN 最小学习率", min_value=1e-8, max_value=1e-3, value=1e-6, format="%.8f", key="ms_pinn_minlr")
        pinn_early_pat = st.number_input("PINN 早停耐心", min_value=1, max_value=200, value=10, step=1, key="ms_pinn_pat")
        pinn_max_grad = st.number_input("PINN 梯度裁剪上限", min_value=0.0, max_value=100.0, value=5.0, step=0.5, key="ms_pinn_clip")
        pinn_dropout = st.slider("PINN Dropout", min_value=0.0, max_value=0.5, value=0.1, step=0.05, key="ms_pinn_dropout")

        # --- PDE 配置与自定义物理注册 ---
        st.markdown("**PDE 配置与自定义物理（可注册到当前 MultiScaleModel）**")
        pde_choice = st.selectbox(
            "选择内置 PDE 或 自定义",
            options=["默认: Diffusion + Michaelis-Menten", "Heat equation (dC/dt = D ∇²C)", "Poisson (steady) (-D ∇²u = f)", "Burgers 1D", "自定义 Python 残差代码"],
            index=0,
            key="ms_pde_choice",
        )
        custom_code = None
        uploaded_pde = None
        if pde_choice == "自定义 Python 残差代码":
            st.caption("请在下面粘贴定义 `residual_fn(model, pts, mol_emb, **kwargs)` 的 Python 代码，可另定义 `coeff_fn(mol_emb)`。")
            custom_code = st.text_area("残差函数代码", value=st.session_state.get("ms_custom_pde_code", "# def residual_fn(model, pts, mol_emb, **kwargs):\n#     return torch.zeros(pts.shape[0])"), height=240, key="ms_custom_pde_text")
            st.session_state["ms_custom_pde_code"] = custom_code
            uploaded_pde = st.file_uploader("或上传包含 residual_fn 的 Python 文件 (.py)", type=["py"], key="ms_custom_pde_up")
        else:
            st.caption("可选择内置 PDE，随后点击 '注册物理' 将其绑定到当前 MultiScaleModel。")

        if st.button("注册物理到当前模型", key="ms_register_physics"):
            msm = st.session_state.get("ms_model")
            if msm is None:
                st.error("当前没有已构建的 MultiScaleModel。请先运行多尺度分析以构建模型实例。")
            else:
                import types
                import torch as _torch
                import inspect as _inspect

                chosen_res = None
                chosen_coeff = None
                try:
                    # prefer built-in implementations from src.pinn
                    from src.pinn import default_residual, heat_residual, poisson_residual, burgers_residual

                    if pde_choice == "默认: Diffusion + Michaelis-Menten":
                        chosen_res = default_residual
                    elif pde_choice == "Heat equation (dC/dt = D ∇²C)":
                        chosen_res = heat_residual
                    elif pde_choice == "Poisson (steady) (-D ∇²u = f)":
                        chosen_res = poisson_residual
                    elif pde_choice == "Burgers 1D":
                        chosen_res = burgers_residual
                    elif pde_choice == "自定义 Python 残差代码":
                        # try file upload first
                        user_ns = {"torch": _torch}
                        if uploaded_pde is not None:
                            code = uploaded_pde.read().decode("utf-8")
                        else:
                            code = custom_code or ""
                        exec(compile(code, "<user_pde>", "exec"), user_ns)
                        if "residual_fn" in user_ns and callable(user_ns["residual_fn"]):
                            chosen_res = user_ns["residual_fn"]
                        else:
                            st.error("未在自定义代码中找到名为 residual_fn 的可调用函数。")
                        if "coeff_fn" in user_ns and callable(user_ns["coeff_fn"]):
                            chosen_coeff = user_ns["coeff_fn"]
                    else:
                        st.error("未知 PDE 选项")

                    if chosen_res is not None:
                        msm.register_physics(residual_fn=chosen_res, coeff_fn=chosen_coeff)
                        st.success("已将物理残差函数注册到当前 MultiScaleModel（会话内生效）。")
                        # persist choice for UX
                        st.session_state["ms_registered_pde"] = pde_choice
                    else:
                        st.warning("未注册物理：未选择或解析失败。")
                except Exception as e:
                    st.error(f"注册物理失败: {e}")

                # add a short test/train button to run a tiny demo and show loss curve
                if st.button("测试物理短训（8步）", key="ms_test_physics"):
                    msm_local = st.session_state.get("ms_model")
                    if msm_local is None or getattr(msm_local, 'pinn', None) is None:
                        st.error("当前没有可用的 PINN 模型，请先构建 PINN。")
                    else:
                        try:
                            coll = _torch.rand((64, msm_local.pinn.net[0].in_features - (msm_local.pinn.net[-1].out_features if hasattr(msm_local.pinn.net[-1], 'out_features') else 1)))
                        except Exception:
                            coll = _torch.rand((64, 2))
                        Dv = float(st.session_state.get("ms_default_D", 0.1))
                        Vmaxv = float(st.session_state.get("ms_default_Vmax", 0.5))
                        Kmv = float(st.session_state.get("ms_default_Km", 0.1))
                        optim = _torch.optim.AdamW(msm_local.pinn.parameters(), lr=1e-3)
                        losses = []
                        for i in range(8):
                            loss = msm_local.pinn_step(optim, coll, msm_local.encode_molecule(st.session_state.get("ms_smiles", "CCO")).detach(), Dv, Vmaxv, Kmv)
                            losses.append(loss)
                        st.line_chart(losses)

        st.caption("对接模型训练（PL)")
        st.caption("使用云端任务训练对接模型，返回模型文件或 ID。")
        pl_out = st.text_input("模型保存路径", value="models/pl_model.pth", key="ms_pl_out")
        pl_epochs = st.number_input("训练轮数", min_value=1, max_value=500, value=20, step=1, key="ms_pl_epochs")
        pl_batch = st.number_input("batch_size", min_value=1, max_value=256, value=8, step=1, key="ms_pl_batch")
        pl_lr = st.number_input("learning_rate", min_value=1e-6, max_value=1e-1, value=1e-3, format="%g", key="ms_pl_lr")
        pl_weight_decay = st.number_input("weight_decay", min_value=0.0, max_value=1e-2, value=1e-4, format="%.6f", key="ms_pl_wd")
        pl_lr_schedule = st.selectbox("学习率调度", options=["cosine", "step", "none"], index=0, key="ms_pl_sched")
        pl_step_size = st.number_input("阶梯步长", min_value=1, max_value=200, value=20, step=1, key="ms_pl_step")
        pl_gamma = st.number_input("阶梯衰减系数", min_value=0.1, max_value=0.99, value=0.5, step=0.05, key="ms_pl_gamma")
        pl_min_lr = st.number_input("最小学习率", min_value=1e-8, max_value=1e-3, value=1e-6, format="%.8f", key="ms_pl_minlr")
        pl_early_pat = st.number_input("早停耐心", min_value=1, max_value=200, value=10, step=1, key="ms_pl_pat")
        pl_max_grad = st.number_input("梯度裁剪上限", min_value=0.0, max_value=100.0, value=5.0, step=0.5, key="ms_pl_clip")
        pl_dropout = st.slider("dropout", min_value=0.0, max_value=0.5, value=0.1, step=0.05, key="ms_pl_dropout")
        pl_use_lstm = st.checkbox("加入 LSTM", value=False, key="ms_pl_lstm")
        pl_lstm_hidden = st.number_input("LSTM 隐藏维度", min_value=32, max_value=512, value=128, step=32, key="ms_pl_lstm_hidden")
        pl_lstm_layers = st.number_input("LSTM 层数", min_value=1, max_value=4, value=1, step=1, key="ms_pl_lstm_layers")
        pl_lstm_bi = st.checkbox("双向 LSTM", value=True, key="ms_pl_lstm_bi")
        pl_cuda = st.checkbox("使用 CUDA（如可用）", value=False, key="ms_pl_cuda")
        pl_teacher_up = st.file_uploader("上传教师模型 .pth/.pt（可选）", type=["pth", "pt"], key="ms_pl_teacher")
        pl_teacher_path = st.text_input("教师模型路径（可选，云/本地可读）", value="", key="ms_pl_teacher_path")
        pl_distill_weight = st.slider("蒸馏权重", min_value=0.0, max_value=1.0, value=0.2, step=0.05, key="ms_pl_distill")
        pl_enable_ema = st.checkbox("启用参数级 EMA 教师（训练期间更新）", value=False, key="ms_pl_enable_ema")
        if pl_enable_ema:
            pl_ema_decay = st.number_input("EMA 衰减系数", min_value=0.0, max_value=0.9999, value=0.99, step=0.001, format="%.4f", key="ms_pl_ema_decay")
        else:
            pl_ema_decay = 0.99
        if st.button("开始训练对接模型", key="ms_pl_train"):
            cloud_cfg = _get_cloud_cfg()
            if cloud_cfg.get("enabled"):
                payload = {
                    "epochs": int(pl_epochs),
                    "batch": int(pl_batch),
                    "lr": float(pl_lr),
                    "out_name": str(Path(pl_out).name),
                    "lj_eps": float(st.session_state.get("ms_lj_eps", 0.1)),
                    "lj_sigma": float(st.session_state.get("ms_lj_sigma", 3.5)),
                    "dielectric": float(st.session_state.get("ms_dielectric", 80.0)),
                    "cuda": bool(pl_cuda),
                    "weight_decay": float(pl_weight_decay),
                    "lr_schedule": str(pl_lr_schedule),
                    "step_size": int(pl_step_size),
                    "gamma": float(pl_gamma),
                    "min_lr": float(pl_min_lr),
                    "early_patience": int(pl_early_pat),
                    "max_grad_norm": float(pl_max_grad),
                    "dropout": float(pl_dropout),
                    "use_lstm": bool(pl_use_lstm),
                    "lstm_hidden": int(pl_lstm_hidden),
                    "lstm_layers": int(pl_lstm_layers),
                    "lstm_bi": bool(pl_lstm_bi),
                    "distill_weight": float(pl_distill_weight),
                }
                if str(pl_teacher_path).strip():
                    payload["teacher_path"] = str(pl_teacher_path).strip()
                if pl_teacher_up is not None:
                    payload["teacher_b64"] = base64.b64encode(pl_teacher_up.getvalue()).decode("ascii")
                if bool(pl_enable_ema):
                    payload["use_ema"] = True
                    payload["ema_decay"] = float(pl_ema_decay)
                res = _submit_cloud_job("pl_train", payload, cloud_cfg)
                if res.get("ok"):
                    st.success("已提交云端训练任务")
                    resp_json = res.get("json") if isinstance(res, dict) else None
                    if isinstance(resp_json, dict) and resp_json.get("model_b64"):
                        try:
                            model_bytes = base64.b64decode(str(resp_json.get("model_b64")))
                            fname = str(resp_json.get("out_name") or "pl_model.pth")
                            st.download_button(
                                "下载云端训练模型",
                                data=model_bytes,
                                file_name=fname,
                                mime="application/octet-stream",
                                key="ms_pl_download_cloud",
                            )
                        except Exception:
                            pass
            else:
                try:
                    import subprocess
                    import tempfile

                    out_path = Path(pl_out)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    cmd = [
                        sys.executable,
                        str(_PROJECT_ROOT / "src" / "pl_train.py"),
                        "--epochs",
                        str(int(pl_epochs)),
                        "--batch",
                        str(int(pl_batch)),
                        "--lr",
                        str(float(pl_lr)),
                        "--weight_decay",
                        str(float(pl_weight_decay)),
                        "--lr_schedule",
                        str(pl_lr_schedule),
                        "--step_size",
                        str(int(pl_step_size)),
                        "--gamma",
                        str(float(pl_gamma)),
                        "--min_lr",
                        str(float(pl_min_lr)),
                        "--early_patience",
                        str(int(pl_early_pat)),
                        "--max_grad_norm",
                        str(float(pl_max_grad)),
                        "--dropout",
                        str(float(pl_dropout)),
                        "--lstm_hidden",
                        str(int(pl_lstm_hidden)),
                        "--lstm_layers",
                        str(int(pl_lstm_layers)),
                        "--out",
                        str(out_path),
                        "--lj_eps",
                        str(float(st.session_state.get("ms_lj_eps", 0.1))),
                        "--lj_sigma",
                        str(float(st.session_state.get("ms_lj_sigma", 3.5))),
                        "--dielectric",
                        str(float(st.session_state.get("ms_dielectric", 80.0))),
                    ]
                    teacher_path_arg = str(pl_teacher_path).strip()
                    if teacher_path_arg:
                        cmd.extend(["--teacher_path", str(teacher_path_arg)])
                    elif pl_teacher_up is not None:
                        tmp_dir = _PROJECT_ROOT / "tmp"
                        tmp_dir.mkdir(parents=True, exist_ok=True)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth", dir=str(tmp_dir)) as tf:
                            tf.write(pl_teacher_up.getvalue())
                            teacher_path = tf.name
                        cmd.extend(["--teacher_path", str(teacher_path)])
                    if bool(pl_enable_ema):
                        cmd.extend(["--use_ema"]) 
                        cmd.extend(["--ema_decay", str(float(pl_ema_decay))])
                    cmd.extend(["--distill_weight", str(float(pl_distill_weight))])
                    if bool(pl_use_lstm):
                        cmd.append("--use_lstm")
                    if bool(pl_lstm_bi):
                        cmd.append("--lstm_bi")
                    if bool(pl_cuda):
                        cmd.append("--cuda")
                    with st.spinner("训练中..."):
                        subprocess.run(cmd, check=True)
                    st.success("对接模型训练完成")
                    if out_path.exists():
                        st.download_button(
                            "下载训练好的模型",
                            data=out_path.read_bytes(),
                            file_name=out_path.name,
                            mime="application/octet-stream",
                            key="ms_pl_download",
                        )
                except Exception as e:
                    st.error(f"训练失败: {e}")

        st.markdown("---")
        st.caption("批量提交云端 PINN 训练（使用爬虫多尺度数据）")
        crawled_ms = st.session_state.get("multiscale_crawl_df")
        max_submit = st.number_input("最多提交条数", min_value=1, max_value=200, value=10, step=1, key="ms_batch_max")
        if st.button("批量提交云端 PINN 训练", key="ms_batch_submit"):
            if not isinstance(crawled_ms, pd.DataFrame) or len(crawled_ms) == 0:
                st.error("没有可用的多尺度爬虫数据")
            elif "smiles" not in crawled_ms.columns:
                st.error("爬虫数据缺少 smiles 列")
            else:
                cloud_cfg = _get_cloud_cfg()
                if not cloud_cfg.get("enabled"):
                    st.error("未启用云算力")
                else:
                    df_ms = crawled_ms.copy()
                    df_ms = df_ms[df_ms["smiles"].astype(str).str.len() > 0]
                    df_ms = df_ms.head(int(max_submit))
                    total = len(df_ms)
                    if total == 0:
                        st.error("可提交数据为空")
                    else:
                        prog = st.progress(0)
                        ok_count = 0
                        results: List[Dict[str, Any]] = []
                        for i, row in df_ms.iterrows():
                            smiles_val = str(row.get("smiles"))
                            try:
                                d_val = float(row.get("D", st.session_state.get("ms_default_D", 0.1)))
                            except Exception:
                                d_val = float(st.session_state.get("ms_default_D", 0.1))
                            try:
                                vmax_val = float(row.get("Vmax", st.session_state.get("ms_default_Vmax", 0.5)))
                            except Exception:
                                vmax_val = float(st.session_state.get("ms_default_Vmax", 0.5))
                            try:
                                km_val = float(row.get("Km", st.session_state.get("ms_default_Km", 0.1)))
                            except Exception:
                                km_val = float(st.session_state.get("ms_default_Km", 0.1))

                            payload = {
                                "smiles": smiles_val,
                                "steps": int(steps),
                                "hidden": int(hidden),
                                "model_type": str(model_type),
                                "enable_coeff": bool(enable_coeff),
                                "coeff_hidden": int(coeff_hidden),
                                "readout_type": str(readout_type),
                                "train_epochs": int(train_epochs),
                                "lj_eps": float(st.session_state.get("ms_lj_eps", 0.1)),
                                "lj_sigma": float(st.session_state.get("ms_lj_sigma", 3.5)),
                                "dielectric": float(st.session_state.get("ms_dielectric", 80.0)),
                                "use_physics": bool(st.session_state.get("ms_use_physics", True)),
                                "pinn_lr": float(pinn_lr),
                                "pinn_weight_decay": float(pinn_weight_decay),
                                "pinn_lr_schedule": str(pinn_lr_schedule),
                                "pinn_step_size": int(pinn_step_size),
                                "pinn_gamma": float(pinn_gamma),
                                "pinn_min_lr": float(pinn_min_lr),
                                "pinn_early_pat": int(pinn_early_pat),
                                "pinn_max_grad": float(pinn_max_grad),
                                "pinn_dropout": float(pinn_dropout),
                                "D": float(d_val),
                                "Vmax": float(vmax_val),
                                "Km": float(km_val),
                            }
                            res = _submit_cloud_job("ms_pinn_train", payload, cloud_cfg)
                            if res.get("ok"):
                                ok_count += 1
                            res_json = res.get("json") if isinstance(res, dict) else None
                            last_loss = None
                            if isinstance(res_json, dict):
                                losses = res_json.get("pinn_losses")
                                if isinstance(losses, list) and losses:
                                    last_loss = losses[-1]
                            results.append(
                                {
                                    "smiles": smiles_val,
                                    "ok": bool(res.get("ok")),
                                    "status": int(res.get("status", 0) or 0),
                                    "last_loss": last_loss,
                                    "message": res.get("text") if isinstance(res.get("text"), str) else "",
                                }
                            )
                            prog.progress(int((ok_count + 1) / max(1, total) * 100))
                        st.success(f"提交完成：{ok_count}/{total}")
                        st.session_state["ms_batch_results"] = results

        results = st.session_state.get("ms_batch_results")
        if isinstance(results, list) and results:
            st.markdown("**批量提交结果汇总**")
            result_df = pd.DataFrame(results)
            st.dataframe(result_df, use_container_width=True)
            st.download_button(
                "下载汇总 CSV",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="ms_pinn_batch_summary.csv",
                mime="text/csv",
                key="ms_batch_summary_download",
            )

    with tabs[5]:
        st.caption("基于原子敏感性训练简单策略（REINFORCE），用于建议关键原子采样")
        rl_k = st.number_input("每次采样原子数 k", min_value=1, max_value=8, value=1, step=1, key="ms_rl_k")
        rl_steps = st.number_input("策略训练步数", min_value=0, max_value=200, value=20, step=1, key="ms_rl_steps")
        rl_lr = st.number_input("策略学习率", min_value=1e-5, max_value=1.0, value=1e-3, format="%g", key="ms_rl_lr")
        rl_penalty = st.number_input("惩罚系数（对负奖励）", min_value=0.0, max_value=10.0, value=1.0, step=0.1, key="ms_rl_penalty")
        rl_run = st.button("运行 RL 采样并训练策略", key="ms_rl_run")

    with tabs[6]:
        st.caption("图像显示设置（用于多尺度展示）")
        st.number_input("图像宽度 (px)", min_value=200, max_value=1600, value=int(st.session_state.get("ms_img_width", 640)), step=10, key="ms_img_width")
        st.number_input("图像高度 (px)", min_value=150, max_value=1200, value=int(st.session_state.get("ms_img_height", 480)), step=10, key="ms_img_height")
        st.checkbox("使用列宽展示图像（会忽略像素宽度）", value=bool(st.session_state.get("ms_use_col_width", True)), key="ms_use_col_width")

    if run:
            cloud_cfg = _get_cloud_cfg()
            # if cloud enabled, submit job and display results returned by cloud
            if cloud_cfg.get("enabled"):
                try:
                    payload = {
                        "smiles": smiles,
                        "steps": int(steps),
                        "hidden": int(hidden),
                        "model_type": str(model_type),
                        "use_physics": bool(st.session_state.get("ms_use_physics", True)),
                        "gnn_dropout": float(gnn_dropout),
                        "readout_type": str(st.session_state.get("ms_readout", "mean")),
                        "lj_eps": float(st.session_state.get("ms_lj_eps", 0.1)),
                        "lj_sigma": float(st.session_state.get("ms_lj_sigma", 3.5)),
                        "dielectric": float(st.session_state.get("ms_dielectric", 80.0)),
                    }
                    with st.spinner("提交云端多尺度计算任务..."):
                        res = _submit_cloud_job("multiscale", payload, cloud_cfg)
                    if not res.get("ok"):
                        st.error(f"云端计算失败 (HTTP {res.get('status')}): {res.get('text')}")
                    else:
                        j = res.get("json") or {}
                        scores = j.get("scores")
                        pinn_losses = j.get("pinn_losses")
                        svg_b64 = j.get("svg_b64") or j.get("png_b64")
                        if scores:
                            st.session_state["ms_last_scores"] = scores
                            st.write("原子敏感性（云端返回）:")
                            try:
                                import pandas as _pd

                                df = _pd.DataFrame([{"atom_index": k, "sensitivity": float(v)} for k, v in scores.items()])
                                st.dataframe(df.sort_values("sensitivity", ascending=False))
                            except Exception:
                                st.write(scores)
                        if pinn_losses is not None:
                            st.write("PINN 训练损失（云端）:", pinn_losses)
                        if svg_b64:
                            import base64 as _b64

                            try:
                                img_bytes = _b64.b64decode(svg_b64)
                                st.image(
                                    img_bytes,
                                    use_column_width=bool(st.session_state.get("ms_use_col_width", True)),
                                    width=int(st.session_state.get("ms_img_width", 640)),
                                )
                            except Exception:
                                st.write("收到云端图片，但无法显示")
                        st.success("云端多尺度计算完成（结果来自远端）。")
                        return
                except Exception as e:
                    st.error(f"提交云端任务失败: {e}")

            # otherwise run locally (fallback)
            try:
                st.info("正在解析 SMILES 并构建图...")
                X, A, mol = mol_to_graph(smiles)
            except Exception as e:
                st.error(f"SMILES 解析失败: {e}")
                return

            # quick ADMET / descriptor estimates (RDKit-based) and optional protein uploader for docking demo
            try:
                preds = st.session_state.get("ms_prediction_targets", [])
                if any(p.startswith("ADMET") or p == "ADMET: MolLogP" for p in preds):
                    desc = compute_admet_quick(smiles)
                    if desc:
                        st.write("ADMET 快速估算:", desc)
                    # if user uploaded an ADMET regressor (joblib), use it
                    admet_file = st.session_state.get("ms_admet_model_up")
                    if admet_file is not None:
                        try:
                            import io as _io

                            buf = _io.BytesIO(admet_file.read())
                            model = joblib.load(buf)
                            # feature ordering must match training; we use the quick descriptor order
                            feat = [
                                desc.get("MolLogP", 0.0),
                                desc.get("MolWt", 0.0),
                                desc.get("RotatableBonds", 0.0),
                                desc.get("TPSA", 0.0),
                                desc.get("AromaticProportion", 0.0),
                            ]

                            # validation: sklearn-like models often expose n_features_in_
                            if hasattr(model, 'n_features_in_'):
                                try:
                                    if int(model.n_features_in_) != len(feat):
                                        st.warning(f"上传的模型期望输入维度为 {model.n_features_in_}，但当前描述符长度为 {len(feat)}；已跳过模型预测。请确认训练时使用的特征顺序与本说明一致。")
                                        raise RuntimeError("feature-dim-mismatch")
                                except Exception:
                                    pass

                            try:
                                pred = model.predict([feat])
                                st.write("ADMET 模型预测 (模型输出):", pred[0] if hasattr(pred, '__len__') else float(pred))
                            except Exception as e:
                                st.warning(f"使用上传的 ADMET 模型预测失败: {e}")
                        except Exception as e:
                            st.warning(f"加载上传的 ADMET 模型失败: {e}")

                if "Binding affinity (protein-ligand)" in preds:
                    st.caption("若选择对接预测，请上传蛋白口袋原子坐标 CSV (cols: element,x,y,z)")
                    prot_up = st.file_uploader("Protein pocket CSV (element,x,y,z)", type=["csv"], key="ms_prot_csv")
                    if prot_up is not None:
                        try:
                            dfp = pd.read_csv(prot_up)
                            # normalize column names
                            cols_low = [c.lower() for c in dfp.columns]
                            dfp.columns = cols_low
                            if not set(["element", "x", "y", "z"]).issubset(set(cols_low)):
                                st.error("CSV 需包含列: element,x,y,z (不区分大小写)")
                            else:
                                atom_types = dfp['element'].astype(str).tolist()
                                coords = dfp[["x", "y", "z"]].to_numpy(dtype=float)
                                try:
                                    from src.pl_interaction import example_predict, ProteinLigandInteractionModel, protein_from_coords

                                    # compute protein features (X_p, A_p)
                                    X_p, A_p, dmat_p = protein_from_coords(atom_types, coords)

                                    # try using uploaded PyTorch model (.pth/.pt) if provided
                                    pl_file = st.session_state.get("ms_pl_model_up")
                                    model_used = False
                                    if pl_file is not None:
                                        try:
                                            import io as _io

                                            data = pl_file.read()
                                            lig_X_tmp, lig_A_tmp, _ = mol_to_graph(smiles)
                                            lig_in_dim = lig_X_tmp.shape[1]
                                            prot_in_dim = X_p.shape[1]
                                            # build model with user-selected hidden dim
                                            hid = int(st.session_state.get("ms_pl_hidden", 64))
                                            model = ProteinLigandInteractionModel(lig_in_dim, prot_in_dim, hidden=hid)
                                            try:
                                                state = torch.load(_io.BytesIO(data), map_location="cpu")
                                            except Exception:
                                                state = None

                                            if isinstance(state, dict) and state is not None:
                                                if 'model_state_dict' in state:
                                                    sd = state['model_state_dict']
                                                elif 'state_dict' in state:
                                                    sd = state['state_dict']
                                                else:
                                                    sd = state
                                                try:
                                                    try:
                                                        model.load_state_dict(sd)
                                                    except Exception:
                                                        # attempt partial load: keep only keys present and matching shape
                                                        target_sd = model.state_dict()
                                                        filtered = {}
                                                        for k, v in sd.items():
                                                            if k in target_sd:
                                                                try:
                                                                    if getattr(v, 'shape', None) == getattr(target_sd[k], 'shape', None):
                                                                        filtered[k] = v
                                                                except Exception:
                                                                    # skip mismatched
                                                                    pass
                                                        model.load_state_dict(filtered, strict=False)
                                                    model.eval()
                                                    lig_X_t = torch.from_numpy(lig_X_tmp).float()
                                                    lig_A_t = torch.from_numpy(lig_A_tmp).float()
                                                    prot_X_t = torch.from_numpy(X_p).float()
                                                    prot_A_t = torch.from_numpy(A_p).float()
                                                    prot_coords_t = torch.from_numpy(coords).float()
                                                    with torch.no_grad():
                                                        score = model(lig_X_t, lig_A_t, None, prot_X_t, prot_A_t, None, prot_coords=prot_coords_t)
                                                    st.write("Binding affinity (uploaded model):", float(score))
                                                    model_used = True
                                                except Exception:
                                                    model_used = False
                                        except Exception:
                                            model_used = False

                                    if not model_used:
                                        # fallback demo predictor
                                        score = example_predict(smiles, atom_types, coords)
                                        st.write("Binding affinity (demo score):", float(score))
                                except Exception as e:
                                    st.error(f"对接预测失败: {e}")
                        except Exception as e:
                            st.error(f"解析上传的蛋白 CSV 失败: {e}")
            except Exception:
                pass

            in_dim = X.shape[1] if X.ndim > 1 else 0
            if in_dim == 0:
                st.error("无法从分子中提取原子特征")
                return

            st.info("构建 GNN 并计算原子敏感性（掩码扰动）...")
            if model_type == "EnhancedGNN":
                gnn = EnhancedGNN(in_dim, hidden_dim=hidden, steps=steps, gat_heads=int(gat_heads), use_physics=bool(use_physics), dropout=float(gnn_dropout))
            elif model_type == "PhysicsMessageGNN":
                from src.gnn import PhysicsMessageGNN

                gnn = PhysicsMessageGNN(in_dim, hidden_dim=hidden, steps=steps, potential_type=st.session_state.get("ms_potential_type", "auto"), lj_epsilon=float(st.session_state.get("ms_lj_eps", 0.1)), lj_sigma=float(st.session_state.get("ms_lj_sigma", 3.5)), dielectric=float(st.session_state.get("ms_dielectric", 80.0)), dropout=float(gnn_dropout))
            elif model_type == "E(3)-EquivariantGNN":
                from src.gnn import EGNN

                gnn = EGNN(in_dim, hidden_dim=int(st.session_state.get("ms_eg_hidden", 64)), n_layers=int(st.session_state.get("ms_eg_layers", 3)))
            else:
                gnn = SimpleGNN(in_dim, hidden_dim=hidden, steps=steps, dropout=float(gnn_dropout))

            # simple linear probe for sensitivity (untrained) - user can replace with trained regressor
            model_fn, probe_module = example_model_fn_factory(hidden)

            try:
                scores = sensitivity_masking(smiles, gnn, model_fn)
            except Exception as e:
                st.error(f"敏感性计算失败: {e}")
                return

            # 保存最近一次运行的得分，SMILES 由输入控件保持
            st.session_state["ms_last_scores"] = scores

            st.write("原子敏感性（归一化）:")
            df = None
            try:
                import pandas as _pd

                df = _pd.DataFrame([{"atom_index": k, "sensitivity": float(v)} for k, v in scores.items()])
                st.dataframe(df.sort_values("sensitivity", ascending=False))
            except Exception:
                st.write(scores)

            st.info("构建多尺度模型并演示 PINN 训练（小样例）...")
            msm = MultiScaleModel(gnn, readout=st.session_state.get("ms_readout", "mean"))
            mol_emb = msm.encode_molecule(smiles).detach()
            # If user requested PDE coeff prediction, show CoeffNet output when available
            preds = st.session_state.get("ms_prediction_targets", [])
            if "PDE coefficients (CoeffNet)" in preds:
                if enable_coeff and getattr(msm, 'coeff_net', None) is not None:
                    try:
                        with torch.no_grad():
                            inp = mol_emb.unsqueeze(0) if mol_emb.dim() == 1 else mol_emb
                            coeff_out = msm.coeff_net(inp).squeeze(0).cpu().numpy()
                        st.write("CoeffNet 预测的 PDE 系数:", coeff_out.tolist())
                    except Exception as e:
                        st.warning(f"CoeffNet 推断失败: {e}")
                else:
                    st.info("未启用或未构建 CoeffNet；请勾选 '启用 coeff_net' 并重新运行。")
            msm.build_pinn(spatial_dim=1, mol_emb_dim=mol_emb.shape[0], hidden=64, dropout=float(pinn_dropout))
            if enable_coeff:
                msm.build_coeff_net(mol_emb_dim=mol_emb.shape[0], hidden=int(coeff_hidden))

            # 保存模型实例以响应后续点击事件（保存在会话状态，进程内有效）
            st.session_state["ms_model"] = msm

            # GNN 权重保存与加载（可用于 PhysicsMessageGNN/其他 GNN）
            try:
                st.markdown('---')
                st.caption('模型权重管理：保存或加载当前 GNN 权重')
                if st.button('下载当前 GNN 权重', key='ms_download_gnn'):
                    try:
                        import io as _io

                        buf = _io.BytesIO()
                        torch.save(msm.gnn.state_dict(), buf)
                        buf.seek(0)
                        st.download_button('下载 GNN 权重 (.pth)', data=buf.read(), file_name=f'gnn_weights_{int(time.time())}.pth', mime='application/octet-stream')
                    except Exception as e:
                        st.warning(f'导出 GNN 权重失败: {e}')

                up = st.file_uploader('上传 GNN 权重文件 (.pth) 并加载到当前模型', type=['pth', 'pt'], key='ms_upload_gnn')
                if up is not None:
                    try:
                        import io as _io

                        byte_data = up.read()
                        state = torch.load(_io.BytesIO(byte_data), map_location='cpu')
                        # load either full checkpoint dict or state_dict directly
                        if isinstance(state, dict) and any(k in state for k in ['gnn', 'model_state_dict', 'state_dict']):
                            if 'gnn' in state:
                                gnn_state = state['gnn']
                            elif 'model_state_dict' in state:
                                gnn_state = state['model_state_dict']
                            elif 'state_dict' in state:
                                gnn_state = state['state_dict']
                            else:
                                gnn_state = state
                        else:
                            gnn_state = state
                        msm.gnn.load_state_dict(gnn_state)
                        st.success('已加载 GNN 权重到当前模型（仅进程内生效，建议保存完整 checkpoint）。')
                    except Exception as e:
                        st.error(f'加载权重失败: {e}')
            except Exception:
                pass

            # tiny collocation set and short train loop for demo
            collocation = torch.rand((64, 2))  # x,t
            D = float(st.session_state.get("ms_default_D", 0.1))
            Vmax = float(st.session_state.get("ms_default_Vmax", 0.5))
            Km = float(st.session_state.get("ms_default_Km", 0.1))
            optimizer = torch.optim.AdamW(msm.pinn.parameters(), lr=float(pinn_lr), weight_decay=float(pinn_weight_decay))
            losses = []
            for epoch in range(8):
                loss = msm.pinn_step(optimizer, collocation, mol_emb, D, Vmax, Km)
                losses.append(loss)

            st.write("PINN 训练损失（示例）：", losses)
            st.success("多尺度分析完成（演示级，结果受模型初始化影响）。")

            # if user requested training demo, run train_pinn and offer checkpoint download
            if train_demo:
                cloud_cfg = _get_cloud_cfg()
                if cloud_cfg.get("enabled"):
                    payload = {
                        "smiles": str(smiles),
                        "steps": int(steps),
                        "hidden": int(hidden),
                        "model_type": str(model_type),
                        "enable_coeff": bool(enable_coeff),
                        "coeff_hidden": int(coeff_hidden),
                        "readout_type": str(readout_type),
                        "train_epochs": int(train_epochs),
                        "lj_eps": float(st.session_state.get("ms_lj_eps", 0.1)),
                        "lj_sigma": float(st.session_state.get("ms_lj_sigma", 3.5)),
                        "dielectric": float(st.session_state.get("ms_dielectric", 80.0)),
                        "use_physics": bool(st.session_state.get("ms_use_physics", True)),
                        "pinn_lr": float(pinn_lr),
                        "pinn_weight_decay": float(pinn_weight_decay),
                        "pinn_lr_schedule": str(pinn_lr_schedule),
                        "pinn_step_size": int(pinn_step_size),
                        "pinn_gamma": float(pinn_gamma),
                        "pinn_min_lr": float(pinn_min_lr),
                        "pinn_early_pat": int(pinn_early_pat),
                        "pinn_max_grad": float(pinn_max_grad),
                        "pinn_dropout": float(pinn_dropout),
                        "D": float(st.session_state.get("ms_default_D", 0.1)),
                        "Vmax": float(st.session_state.get("ms_default_Vmax", 0.5)),
                        "Km": float(st.session_state.get("ms_default_Km", 0.1)),
                    }
                    res = _submit_cloud_job("ms_pinn_train", payload, cloud_cfg)
                    if res.get("ok"):
                        st.success("已提交云端 PINN 训练任务")
                        j = res.get("json") or {}
                        losses = j.get("pinn_losses") if isinstance(j, dict) else None
                        if isinstance(losses, list) and losses:
                            st.write("云端 PINN 训练曲线")
                            st.line_chart({"loss": losses})
                            df_hist = pd.DataFrame({"epoch": list(range(1, len(losses) + 1)), "loss": losses})
                            st.dataframe(df_hist, use_container_width=True)
                else:
                    coeff_fn = msm.coeff_net if (enable_coeff and getattr(msm, 'coeff_net', None) is not None) else None
                    collocation_size = 64
                    optimizer = torch.optim.AdamW(msm.pinn.parameters(), lr=float(pinn_lr), weight_decay=float(pinn_weight_decay))
                    scheduler = build_scheduler(
                        optimizer,
                        str(pinn_lr_schedule),
                        epochs=int(train_epochs),
                        step_size=int(pinn_step_size),
                        gamma=float(pinn_gamma),
                        min_lr=float(pinn_min_lr),
                    )
                    stopper = EarlyStopping(patience=int(pinn_early_pat), mode="min")
                    losses = []

                    loss_slot = st.empty()
                    chart = st.line_chart([], height=200)
                    progress_bar = st.progress(0)
                    heat_slot = st.empty()

                    try:
                        for ep in range(int(train_epochs)):
                            pts = torch.rand((collocation_size, 2))
                            optimizer.zero_grad()
                            loss = pinn_loss(msm.pinn, pts, mol_emb.detach(), D=0.1, Vmax=0.5, Km=0.1, coeff_fn=coeff_fn)
                            loss.backward()
                            if float(pinn_max_grad) > 0:
                                torch.nn.utils.clip_grad_norm_(msm.pinn.parameters(), max_norm=float(pinn_max_grad))
                            optimizer.step()
                            if scheduler is not None:
                                scheduler.step()
                            losses.append(float(loss.item()))

                            loss_slot.write(f"Epoch {ep+1}/{int(train_epochs)}  loss={losses[-1]:.6g}")
                            chart.add_rows(pd.DataFrame({"loss": [losses[-1]]}))
                            progress_bar.progress(int((ep + 1) / int(train_epochs) * 100))

                            if (ep % max(1, int(int(train_epochs) / 4))) == 0 or ep == int(train_epochs) - 1:
                                try:
                                    import matplotlib.pyplot as plt

                                    nx = 64
                                    nt = 32
                                    xs = torch.linspace(0.0, 1.0, nx)
                                    ts = torch.linspace(0.0, 1.0, nt)
                                    grid_x, grid_t = torch.meshgrid(xs, ts, indexing='xy')
                                    pts_grid = torch.stack([grid_x.reshape(-1), grid_t.reshape(-1)], dim=1)
                                    with torch.no_grad():
                                        mol_b = mol_emb.detach().unsqueeze(0).repeat(pts_grid.shape[0], 1)
                                        inp = torch.cat([pts_grid, mol_b], dim=1)
                                        vals = msm.pinn(inp).cpu().numpy().reshape(nt, nx)

                                    fig, ax = plt.subplots(figsize=(5, 3))
                                    im = ax.imshow(vals, origin='lower', aspect='auto', extent=[0, 1, 0, 1], cmap='viridis')
                                    ax.set_xlabel('x')
                                    ax.set_ylabel('t')
                                    ax.set_title('PINN Prediction C(x,t)')
                                    fig.colorbar(im, ax=ax)
                                    heat_slot.pyplot(fig)
                                    plt.close(fig)
                                except Exception:
                                    pass

                            if stopper.step(float(loss.item())):
                                break

                        df_hist = pd.DataFrame({
                            "epoch": list(range(1, len(losses) + 1)),
                            "loss": losses,
                        })
                        st.dataframe(df_hist, use_container_width=True)

                        st.success("短训完成")
                    except Exception as e:
                        st.error(f"短训失败: {e}")

                    ck_path = _PROJECT_ROOT / 'tmp' / f"ms_checkpoint_{int(time.time())}.pt"
                    ck_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        msm.save(str(ck_path))
                        with open(ck_path, 'rb') as fh:
                            data = fh.read()
                        st.download_button("下载模型 checkpoint", data=data, file_name=ck_path.name, mime='application/octet-stream')
                    except Exception as e:
                        st.warning(f"无法保存 checkpoint: {e}")

            # RL sampling and policy training (based on sensitivity scores)
            if rl_run:
                cloud_cfg = _get_cloud_cfg()
                if cloud_cfg.get("enabled"):
                    scrs = scores if isinstance(scores, dict) else st.session_state.get('ms_last_scores', {})
                    payload = {
                        "smiles": str(smiles),
                        "rl_k": int(rl_k),
                        "rl_steps": int(rl_steps),
                        "rl_lr": float(rl_lr),
                        "scores": scrs if isinstance(scrs, dict) else {},
                        "steps": int(steps),
                        "hidden": int(hidden),
                        "model_type": str(model_type),
                    }
                    res = _submit_cloud_job("ms_rl_train", payload, cloud_cfg)
                    if res.get("ok"):
                        st.success("已提交云端 RL 训练任务")
                else:
                    try:
                        scrs = scores if isinstance(scores, dict) else st.session_state.get('ms_last_scores', {})
                        if not scrs:
                            st.warning('未能获取原子敏感性，先运行多尺度分析以生成 scores。')
                        else:
                            in_dim = gnn.hidden_dim if hasattr(gnn, 'hidden_dim') else getattr(gnn, 'hidden', None)
                            policy = AtomPolicyNet(in_dim)
                            opt = torch.optim.Adam(policy.parameters(), lr=float(rl_lr))

                            n_atoms = len(scrs)
                            rewards_vec = torch.tensor([float(scrs.get(i, 0.0)) for i in range(n_atoms)])

                            node_X, node_A, _ = mol_to_graph(smiles)
                            x = torch.from_numpy(node_X).float()
                            adj = torch.from_numpy(node_A).float()
                            node_emb = gnn(x, adj)

                            rl_hist = {"loss": [], "reward": [], "penalty": []}
                            for step in range(int(rl_steps)):
                                idxs, logp = sample_atoms(policy, node_emb, int(rl_k))
                                if isinstance(idxs, torch.Tensor) and idxs.numel() > 0:
                                    r = rewards_vec[idxs]
                                    penalty = torch.relu(-r) * float(rl_penalty)
                                    m = reinforce_update(
                                        opt,
                                        logp,
                                        r,
                                        baseline=float(rewards_vec.mean()),
                                        penalty=penalty,
                                        return_metrics=True,
                                    )
                                    if isinstance(m, dict):
                                        rl_hist["loss"].append(float(m.get("loss", 0.0)))
                                        rl_hist["reward"].append(float(m.get("reward_mean", 0.0)))
                                        rl_hist["penalty"].append(float(m.get("penalty_mean", 0.0)))

                            with torch.no_grad():
                                final_logits = policy(node_emb)
                                final_probs = torch.softmax(final_logits, dim=0).cpu().numpy()
                            topk = int(min(int(rl_k), len(final_probs)))
                            top_idx = final_probs.argsort()[-topk:][::-1].tolist()
                            st.success("RL 训练完成")
                            st.write("Top-k 原子索引:", top_idx)
                            if rl_hist["loss"]:
                                st.write("奖励/惩罚/损失趋势（每步）")
                                st.line_chart(
                                    {
                                        "loss": rl_hist["loss"],
                                        "reward": rl_hist["reward"],
                                        "penalty": rl_hist["penalty"],
                                    }
                                )
                    except Exception as e:
                        st.error(f"RL 训练失败: {e}")
                        st.write('RL 推荐原子索引 (top-k):', top_idx)
                        st.write('对应敏感性:', [scrs.get(i, 0.0) for i in top_idx])
                        # show probs bar
                        import pandas as _pd

                        st.bar_chart(_pd.DataFrame({'prob': final_probs}))


            # Attention readout fine-tune using sensitivity scores as supervision
            if st.session_state.get("ms_readout", "mean") == "attention" and getattr(msm, 'attn_readout', None) is not None:
                st.markdown("---")
                st.caption("微调 Attention Readout（使用当前原子敏感性作为监督信号）")
                tf_epochs = st.number_input("微调轮数", min_value=1, max_value=200, value=10, step=1, key="ms_attn_epochs")
                tf_lr = st.number_input("微调学习率", min_value=1e-6, max_value=1.0, value=1e-3, format="%g", key="ms_attn_lr")
                if st.button("微调 Attention 并保存", key="ms_attn_finetune"):
                    try:
                        scrs = scores if isinstance(scores, dict) else st.session_state.get('ms_last_scores', {})
                        if not scrs:
                            st.warning('未能获取原子敏感性，无法微调。请先运行多尺度分析以生成 scores。')
                        else:
                            # prepare node embeddings and target distribution
                            X_np, A_np, _ = mol_to_graph(smiles)
                            x = torch.from_numpy(X_np).float()
                            adj = torch.from_numpy(A_np).float()
                            node_emb = gnn(x, adj)

                            # target probs from scores
                            n_atoms = node_emb.shape[0]
                            target_vals = torch.tensor([float(scrs.get(i, 0.0)) for i in range(n_atoms)])
                            if target_vals.sum() == 0:
                                target_probs = torch.ones_like(target_vals) / float(max(1, n_atoms))
                            else:
                                target_probs = torch.softmax(target_vals, dim=0)

                            opt = torch.optim.Adam(msm.attn_readout.parameters(), lr=float(tf_lr))
                            loss_fn = torch.nn.KLDivLoss(reduction='batchmean')

                            loss_slot = st.empty()
                            for ep in range(int(tf_epochs)):
                                opt.zero_grad()
                                logits = msm.attn_readout.score(node_emb).squeeze(-1)
                                pred_log_probs = torch.log_softmax(logits, dim=0)
                                loss = loss_fn(pred_log_probs.unsqueeze(0), target_probs.unsqueeze(0))
                                loss.backward()
                                opt.step()
                                loss_slot.write(f"Attn finetune epoch {ep+1}/{int(tf_epochs)} loss={float(loss.item()):.6g}")

                            # save attn state dict to tmp and offer download
                            attn_path = _PROJECT_ROOT / 'tmp' / f"attn_readout_{int(time.time())}.pth"
                            attn_path.parent.mkdir(parents=True, exist_ok=True)
                            torch.save(msm.attn_readout.state_dict(), str(attn_path))
                            with open(attn_path, 'rb') as fh:
                                data = fh.read()
                            st.download_button("下载 Attention Readout 权重", data=data, file_name=attn_path.name, mime='application/octet-stream')
                            st.success('微调并保存完成')
                    except Exception as e:
                        st.error(f"微调失败: {e}")


def atom_heatmap_image(smiles: str, scores: dict, size: Tuple[int, int] = (400, 300)) -> Optional[Image.Image]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)
        # map scores to colors (red high, blue low)
        vals = [scores.get(i, 0.0) for i in range(mol.GetNumAtoms())]
        if sum(vals) == 0:
            cmap = {i: (0.8, 0.8, 0.8) for i in range(len(vals))}
        else:
            import matplotlib.pyplot as plt
            norm = plt.Normalize(min(vals), max(vals))
            cmap_map = plt.get_cmap("RdYlBu_r")
            cmap = {i: tuple(int(255 * c) for c in cmap_map(norm(v))[:3]) for i, v in enumerate(vals)}

        highlight_atoms = [i for i in range(mol.GetNumAtoms())]
        highlight_colors = {i: tuple([c/255.0 for c in cmap[i]]) for i in highlight_atoms}
        drawer = Draw.MolDraw2DCairo(size[0], size[1])
        opts = drawer.drawOptions()
        for i in highlight_atoms:
            opts.atomHighlights[i] = highlight_colors[i]
        Draw.rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=highlight_atoms, highlightAtomColors=highlight_colors)
        drawer.FinishDrawing()
        png = drawer.GetDrawingText()
        img = Image.open(_io.BytesIO(png))
        return img
    except Exception:
        return None


def atom_heatmap_svg(smiles: str, scores: dict, width: int = 480, height: int = 360) -> Optional[str]:
    """Return an HTML string containing an SVG with interactive atom tooltips and a colorbar.

    Falls back to None on any failure.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)

        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = drawer.drawOptions()
        opts.padding = 0.05
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)

        # attempt to get atom drawing coordinates
        atom_coords = []
        n = mol.GetNumAtoms()
        for i in range(n):
            try:
                p = drawer.GetDrawCoords(i)
                atom_coords.append((float(p.x), float(p.y)))
            except Exception:
                atom_coords = []
                break

        base_svg = drawer.GetDrawingText()
        drawer.FinishDrawing()

        if not atom_coords:
            return None

        # build color mapping
        vals = [scores.get(i, 0.0) for i in range(n)]
        vmin = min(vals)
        vmax = max(vals)
        if vmin == vmax:
            vmax = vmin + 1e-6

        def to_rgb(v):
            import matplotlib.pyplot as plt

            cmap = plt.get_cmap("RdYlBu_r")
            norm = plt.Normalize(vmin, vmax)
            c = cmap(norm(v))
            return int(c[0]*255), int(c[1]*255), int(c[2]*255)

        circles_svg = ""
        for i, (x, y) in enumerate(atom_coords):
            r = 8
            col = to_rgb(vals[i])
            color = f"rgb({col[0]},{col[1]},{col[2]})"
            tooltip = f"atom {i}: {vals[i]:.4f}"
            circles_svg += f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" fill-opacity="0.8" stroke="#222" stroke-width="0.6" data-tip="{tooltip}" data-atom="{i}"></circle>'

        # colorbar as small SVG: vertical gradient
        grad_id = "gbar"
        colorbar_svg = f'''
        <defs>
          <linearGradient id="{grad_id}" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="rgb{to_rgb(vmax)}"/>
            <stop offset="100%" stop-color="rgb{to_rgb(vmin)}"/>
          </linearGradient>
        </defs>
        <rect x="{width-70}" y="20" width="20" height="200" fill="url(#{grad_id})" stroke="#000" />
        <text x="{width-40}" y="30" font-size="12">{vmax:.3f}</text>
        <text x="{width-40}" y="220" font-size="12">{vmin:.3f}</text>
        '''

        # assemble HTML with base SVG and overlay circles + tooltip script
        html = (
            '<div style="position:relative; width:' + str(width) + 'px; height:' + str(height) + 'px">'
            + base_svg
            + '<svg width="' + str(width) + '" height="' + str(height) + '" style="position:absolute; left:0; top:0; pointer-events:none;">'
            + circles_svg
            + colorbar_svg
            + '</svg>'
            + '<div id="tip" style="position:absolute; display:none; background:#fff; padding:6px; border:1px solid #333; border-radius:4px; font-size:12px; pointer-events:none;"></div>'
            + '</div>'
            + '<script>'
            + "const tip=document.getElementById('tip');"
            + "function showTip(evt, text){ tip.style.display='block'; tip.style.left=(evt.clientX+10)+'px'; tip.style.top=(evt.clientY+10)+'px'; tip.innerText=text; }"
            + "function hideTip(){ tip.style.display='none'; }"
            + "const svgs=document.getElementsByTagName('svg'); for(const s of svgs){ s.style.pointerEvents='auto'; }"
            + "document.querySelectorAll('circle[data-atom]').forEach(c=>{ c.style.cursor='pointer'; c.addEventListener('mouseover', (e)=>{ showTip(e, c.getAttribute('data-tip')); }); c.addEventListener('mouseout', hideTip); });"
            + '</script>'
        )
        return html
    except Exception:
        return None


def compute_admet_quick(smiles: str) -> Dict[str, float]:
    """Quick RDKit-based ADMET-like estimates: logP, MW, TPSA, RotB and ESOL logS (Delaney-style)."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        mw = float(Descriptors.MolWt(mol))
        logp = float(Descriptors.MolLogP(mol))
        rotb = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
        tpsa = float(rdMolDescriptors.CalcTPSA(mol))
        n_arom = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        arom_prop = float(n_arom / mol.GetNumAtoms()) if mol.GetNumAtoms() > 0 else 0.0

        # Delaney ESOL-style estimate (simple linear model)
        esol_logS = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rotb - 0.74 * arom_prop

        return {
            "MolLogP": logp,
            "MolWt": mw,
            "RotatableBonds": rotb,
            "TPSA": tpsa,
            "AromaticProportion": arom_prop,
            "ESOL_logS": esol_logS,
        }
    except Exception:
        return {}


def multiscale_ui() -> None:
    st.header("多尺度 GNN-PINN 建模")
    st.write("使用原子级 GNN -> GAT -> PINN 的演示流水线。")
    st.subheader("参数与运行")
    render_multiscale_sidebar()

    # also show recent run results if any
    scores = None
    if "ms_last_scores" in st.session_state:
        scores = st.session_state["ms_last_scores"]
    if scores:
        # 尝试生成交互式 SVG，若失败回退到静态图像
        svg_html = None
        try:
            svg_html = atom_heatmap_svg(
                st.session_state.get("ms_smiles", "CCO"),
                scores,
                width=int(st.session_state.get("ms_img_width", 640)),
                height=int(st.session_state.get("ms_img_height", 480)),
            )
        except Exception:
            svg_html = None

        if svg_html:
            components.html(svg_html, height=int(st.session_state.get("ms_img_height", 480)))
        else:
            img = atom_heatmap_image(
                st.session_state.get("ms_smiles", "CCO"),
                scores,
                size=(int(st.session_state.get("ms_img_width", 640)), int(st.session_state.get("ms_img_height", 480))),
            )
            if img is not None:
                st.image(
                    img,
                    caption="原子敏感性热图",
                    use_column_width=bool(st.session_state.get("ms_use_col_width", True)),
                    width=int(st.session_state.get("ms_img_width", 640)),
                )
            else:
                st.write("无法绘制分子图（可能缺少 RDKit 或 2D 坐标）。")

    # 处理 URL 查询参数中可能的 atom 选择
    # `st.query_params` replaces deprecated `st.experimental_get_query_params()`
    params = st.query_params
    sel = None
    if "ms_atom" in params:
        try:
            # st.query_params returns lists for each key similar to the experimental API
            sel = int(params.get("ms_atom")[0])
        except Exception:
            sel = None

    if sel is not None:
        st.markdown(f"**已选择原子**: {sel}")
        # show atom basic info
        try:
            smiles = st.session_state.get("ms_smiles", None)
            if not smiles:
                st.warning("请先运行多尺度分析以生成模型/得分，然后点击原子。")
            else:
                X_np, A_np, mol = mol_to_graph(smiles)
                atom = mol.GetAtomWithIdx(sel)
                atom_info = {
                    "symbol": atom.GetSymbol(),
                    "degree": atom.GetDegree(),
                    "formal_charge": atom.GetFormalCharge(),
                    "aromatic": atom.GetIsAromatic(),
                }
                st.json(atom_info)
                # show sensitivity if available
                scores = st.session_state.get("ms_last_scores", {})
                if sel in scores:
                    st.write(f"敏感性: {scores[sel]:.4f}")

                # if model exists, offer mask-and-evaluate
                msm = st.session_state.get("ms_model", None)
                if msm is None or msm.pinn is None:
                    st.info("无 PINN 模型（请先运行多尺度分析以建立 PINN）。")
                else:
                    st.markdown("---")
                    st.caption("可视化 PINN 在空间切片上的原始/掩码输出差异")
                    t_val = st.number_input("时间 t", value=0.1, step=0.01, key=f"ms_t_{sel}")
                    x_min = st.number_input("x_min", value=0.0, step=0.1, key=f"ms_xmin_{sel}")
                    x_max = st.number_input("x_max", value=1.0, step=0.1, key=f"ms_xmax_{sel}")
                    n_pts = st.number_input("采样点数", min_value=10, max_value=500, value=101, step=10, key=f"ms_n_{sel}")
                    if st.button("Mask 原子并绘制空间切片", key=f"mask_plot_{sel}"):
                        # If cloud enabled, prefer cloud evaluation for mask-plot
                        cloud_cfg = _get_cloud_cfg()
                        if cloud_cfg.get("enabled"):
                            try:
                                payload = {
                                    "smiles": smiles,
                                    "atom_idx": int(sel),
                                    "x_min": float(x_min),
                                    "x_max": float(x_max),
                                    "n_pts": int(n_pts),
                                    "t_val": float(t_val),
                                    "steps": int(steps),
                                    "hidden": int(hidden),
                                    "model_type": str(model_type),
                                    "use_physics": bool(st.session_state.get("ms_use_physics", True)),
                                    "gnn_dropout": float(gnn_dropout),
                                    "readout_type": str(st.session_state.get("ms_readout", "mean")),
                                    "lj_eps": float(st.session_state.get("ms_lj_eps", 0.1)),
                                    "lj_sigma": float(st.session_state.get("ms_lj_sigma", 3.5)),
                                    "dielectric": float(st.session_state.get("ms_dielectric", 80.0)),
                                }
                                with st.spinner("提交云端 mask-plot 任务..."):
                                    res = _submit_cloud_job("mask_plot", payload, cloud_cfg)
                                if not res.get("ok"):
                                    st.error(f"云端计算失败 (HTTP {res.get('status')}): {res.get('text')}")
                                else:
                                    j = res.get("json") or {}
                                    x_arr = j.get("x")
                                    y_orig = j.get("y_orig")
                                    y_mask = j.get("y_mask")
                                    diff = j.get("diff")
                                    img_b64 = j.get("img_b64")
                                    if img_b64:
                                        import base64 as _b64

                                        try:
                                            img_bytes = _b64.b64decode(img_b64)
                                            st.image(
                                                img_bytes,
                                                use_column_width=bool(st.session_state.get("ms_use_col_width", True)),
                                                width=int(st.session_state.get("ms_img_width", 640)),
                                            )
                                        except Exception:
                                            st.write("收到云端图片，但无法显示")
                                    elif x_arr and y_orig and y_mask:
                                        import matplotlib.pyplot as plt
                                        import numpy as _np

                                        fig, ax = plt.subplots(figsize=(6, 3))
                                        ax.plot(_np.array(x_arr), _np.array(y_orig), label='原始')
                                        ax.plot(_np.array(x_arr), _np.array(y_mask), label='掩码')
                                        ax.plot(_np.array(x_arr), _np.array(diff), label='掩码-原始', linestyle='--')
                                        ax.set_xlabel('x')
                                        ax.set_ylabel('PINN 输出 C(x,t)')
                                        ax.legend()
                                        st.pyplot(fig)
                                        st.write(f"差值区间: {_np.min(diff):.4e} 到 {_np.max(diff):.4e}")
                                    else:
                                        st.write("云端返回结果格式不完整")
                            except Exception as e:
                                st.error(f"提交云端任务失败: {e}")
                                # fallback to local below

                        # local fallback (if cloud disabled or cloud failed)
                        try:
                            xs = torch.linspace(float(x_min), float(x_max), int(n_pts)).unsqueeze(1)
                            ts = torch.full((xs.shape[0], 1), float(t_val))
                            baseline_x = torch.cat([xs, ts], dim=1)  # shape (N, 2)

                            emb_orig = msm.encode_molecule(smiles).detach()
                            inp = torch.cat([baseline_x, emb_orig.unsqueeze(0).repeat(baseline_x.shape[0], 1)], dim=1)
                            with torch.no_grad():
                                y_orig = msm.pinn(inp).detach().cpu().numpy()

                            emb_mask = msm.encode_with_mask(smiles, [sel]).detach()
                            inp2 = torch.cat([baseline_x, emb_mask.unsqueeze(0).repeat(baseline_x.shape[0], 1)], dim=1)
                            with torch.no_grad():
                                y_mask = msm.pinn(inp2).detach().cpu().numpy()

                            diff = y_mask - y_orig

                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(6, 3))
                            ax.plot(xs.squeeze().numpy(), y_orig, label='原始')
                            ax.plot(xs.squeeze().numpy(), y_mask, label='掩码')
                            ax.plot(xs.squeeze().numpy(), diff, label='掩码-原始', linestyle='--')
                            ax.set_xlabel('x')
                            ax.set_ylabel('PINN 输出 C(x,t)')
                            ax.legend()
                            st.pyplot(fig)
                            st.write(f"差值区间: {diff.min():.4e} 到 {diff.max():.4e}")
                        except Exception as e:
                            st.error(f"绘制失败: {e}")
        except Exception as e:
            st.error(f"读取原子信息失败: {e}")


_UPLOAD_TYPES = [
    "csv",
    "tsv",
    "txt",
    "xlsx",
    "xls",
    "json",
    "jsonl",
    "ndjson",
    "parquet",
    "pq",
    "feather",
    "arrow",
]


def _cloud_request(
    method: str,
    url: str,
    *,
    headers: Dict[str, str],
    payload: Optional[Dict[str, Any]],
    timeout: int,
) -> Dict[str, Any]:
    t0 = time.time()
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
    else:
        req = urllib.request.Request(url, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            text = body.decode("utf-8", errors="ignore")
            try:
                j = json.loads(text)
            except Exception:
                j = None
            return {
                "ok": True,
                "status": resp.status,
                "text": text,
                "json": j,
                "elapsed_ms": int((time.time() - t0) * 1000),
                "bytes": int(len(body)),
            }
    except urllib.error.HTTPError as e:
        body = e.read() if hasattr(e, "read") else b""
        text = body.decode("utf-8", errors="ignore")
        return {
            "ok": False,
            "status": int(getattr(e, "code", 0)),
            "text": text,
            "json": None,
            "elapsed_ms": int((time.time() - t0) * 1000),
            "bytes": int(len(body)),
        }
    except Exception as e:
        return {
            "ok": False,
            "status": 0,
            "text": str(e),
            "json": None,
            "elapsed_ms": int((time.time() - t0) * 1000),
            "bytes": 0,
        }


def _cloud_encode_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return {
        "filename": "data.csv",
        "content_b64": base64.b64encode(csv_bytes).decode("ascii"),
        "content_type": "text/csv",
        "bytes": int(len(csv_bytes)),
    }


def _cloud_encode_bytes(data: bytes, filename: str, content_type: str) -> Dict[str, Any]:
    return {
        "filename": str(filename),
        "content_b64": base64.b64encode(data).decode("ascii"),
        "content_type": str(content_type),
        "bytes": int(len(data)),
    }


def _cloud_payload_bytes(payload: Dict[str, Any]) -> int:
    total = 0
    if not isinstance(payload, dict):
        return total
    for v in payload.values():
        if isinstance(v, dict) and "bytes" in v:
            try:
                total += int(v.get("bytes", 0))
            except Exception:
                pass
        elif isinstance(v, dict):
            total += _cloud_payload_bytes(v)
    return total


def _cloud_ui() -> Dict[str, Any]:
    with st.sidebar.expander("云算力接口", expanded=False):
        enabled = st.checkbox("启用云算力", value=bool(st.session_state.get("cloud_enabled", False)), key="cloud_enabled")
        base_url = st.text_input("云服务地址", value=str(st.session_state.get("cloud_base_url", "")), key="cloud_base_url")
        api_key = st.text_input("API Key", value=str(st.session_state.get("cloud_api_key", "")), type="password", key="cloud_api_key")
        timeout = st.number_input("请求超时(秒)", min_value=1, max_value=600, value=int(st.session_state.get("cloud_timeout", 60)), key="cloud_timeout")
        retry = st.number_input("失败重试次数", min_value=0, max_value=5, value=int(st.session_state.get("cloud_retry", 1)), key="cloud_retry")
        retry_backoff = st.number_input("重试间隔(秒)", min_value=0.0, max_value=10.0, value=float(st.session_state.get("cloud_retry_backoff", 0.8)), step=0.2, key="cloud_retry_backoff")
        show_resp = st.checkbox("显示云端响应详情", value=bool(st.session_state.get("cloud_show_resp", False)), key="cloud_show_resp")
        extra_headers_text = st.text_area("额外Header(JSON)", value=str(st.session_state.get("cloud_headers", "{}")), height=100, key="cloud_headers")
        health_path = st.text_input("健康检查路径", value=str(st.session_state.get("cloud_health_path", "/health")), key="cloud_health_path")
        job_path = st.text_input("任务提交路径", value=str(st.session_state.get("cloud_job_path", "/jobs")), key="cloud_job_path")
        test_btn = st.button("连接测试", key="cloud_test")

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            extra_headers = json.loads(extra_headers_text or "{}")
            if isinstance(extra_headers, dict):
                for k, v in extra_headers.items():
                    headers[str(k)] = str(v)
        except Exception as e:
            st.error(f"额外Header解析失败：{e}")

        if test_btn:
            if not base_url:
                st.error("请先填写云服务地址")
            else:
                url = base_url.rstrip("/") + str(health_path)
                res = _cloud_request("GET", url, headers=headers, payload=None, timeout=int(timeout))
                if res["ok"]:
                    st.success(f"连接成功 (HTTP {res['status']})")
                    if res.get("text"):
                        st.code(res["text"], language="json")
                else:
                    st.error(f"连接失败 (HTTP {res['status']}): {res['text']}")

    cfg = {
        "enabled": bool(enabled),
        "base_url": str(base_url),
        "headers": headers,
        "timeout": int(timeout),
        "retry": int(retry),
        "retry_backoff": float(retry_backoff),
        "show_resp": bool(show_resp),
        "job_path": str(job_path),
    }
    st.session_state["_cloud_cfg"] = cfg
    return cfg


def _get_cloud_cfg() -> Dict[str, Any]:
    cfg = st.session_state.get("_cloud_cfg")
    if isinstance(cfg, dict):
        return cfg
    return {
        "enabled": False,
        "base_url": "",
        "headers": {"Content-Type": "application/json"},
        "timeout": 60,
        "retry": 1,
        "retry_backoff": 0.8,
        "show_resp": False,
        "job_path": "/jobs",
    }


def _submit_cloud_job(task: str, payload: Dict[str, Any], cloud_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not cloud_cfg.get("enabled"):
        st.error("未启用云算力")
        return {"ok": False, "status": 0, "text": "cloud disabled", "json": None}
    base_url = str(cloud_cfg.get("base_url", "")).strip()
    if not base_url:
        st.error("未配置云服务地址")
        return {"ok": False, "status": 0, "text": "missing base_url", "json": None}
    url = base_url.rstrip("/") + str(cloud_cfg.get("job_path", "/jobs"))
    req_payload = {"task": task, "payload": payload}
    retries = int(cloud_cfg.get("retry", 0))
    backoff = float(cloud_cfg.get("retry_backoff", 0.8))
    last = None
    for i in range(retries + 1):
        last = _cloud_request(
            "POST",
            url,
            headers=cast(Dict[str, str], cloud_cfg.get("headers", {})),
            payload=req_payload,
            timeout=int(cloud_cfg.get("timeout", 60)),
        )
        if last.get("ok"):
            return last
        status = int(last.get("status", 0) or 0)
        if status and status < 500:
            break
        if i < retries:
            time.sleep(backoff)
    return last or {"ok": False, "status": 0, "text": "unknown error", "json": None}


def _cloud_try_download_csv(resp_json: Dict[str, Any], *, default_name: str) -> None:
    data_b64 = resp_json.get("result_csv_b64") or resp_json.get("csv_b64")
    if not data_b64:
        return
    try:
        csv_bytes = base64.b64decode(data_b64)
        st.download_button("下载云端结果 CSV", data=csv_bytes, file_name=default_name, mime="text/csv")
    except Exception as e:
        st.info(f"解析云端CSV失败：{e}")


def _cloud_submit_section(
    task: str,
    payload: Dict[str, Any],
    *,
    button_label: str,
    key: str,
    download_name: Optional[str] = None,
    model_id_state_key: Optional[str] = None,
) -> None:
    cloud_cfg = _get_cloud_cfg()
    payload_bytes = _cloud_payload_bytes(payload)
    if payload_bytes > 0:
        st.caption(f"云端负载大小: {payload_bytes / (1024 * 1024):.2f} MB")
    if st.button(button_label, key=key):
        with st.spinner("提交云端任务..."):
            res = _submit_cloud_job(task, payload, cloud_cfg)
        if res["ok"]:
            elapsed = res.get("elapsed_ms", 0)
            st.success(f"云端任务已提交 (HTTP {res['status']})，耗时 {elapsed} ms")
            resp_json = res.get("json")
            if isinstance(resp_json, dict):
                if model_id_state_key:
                    model_id = resp_json.get("model_id")
                    if model_id:
                        st.session_state[model_id_state_key] = str(model_id)
                        st.info(f"已更新云端模型ID: {model_id}")
                if cloud_cfg.get("show_resp"):
                    with st.expander("云端响应详情", expanded=False):
                        st.json(resp_json)
                if download_name:
                    _cloud_try_download_csv(resp_json, default_name=download_name)
            elif res.get("text"):
                if cloud_cfg.get("show_resp"):
                    with st.expander("云端响应详情", expanded=False):
                        st.code(res["text"], language="json")
        else:
            elapsed = res.get("elapsed_ms", 0)
            st.error(f"云端任务提交失败 (HTTP {res['status']}), 耗时 {elapsed} ms: {res['text']}")


def _resolve_model_payload(uploaded: Any, local_path: Optional[str], *, content_type: str) -> Optional[Dict[str, Any]]:
    if uploaded is not None:
        name = getattr(uploaded, "name", "model")
        return _cloud_encode_bytes(uploaded.getvalue(), name, content_type)
    if local_path:
        try:
            data = Path(local_path).read_bytes()
            return _cloud_encode_bytes(data, Path(local_path).name, content_type)
        except Exception as e:
            st.error(f"读取本地模型失败：{e}")
            return None
    return None


def _strip_leading_comments(text: str) -> tuple[str, int]:
    if not text:
        return text, 0
    markers = ("#", "//", ";", "%")
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        s = raw.strip()
        if not s:
            i += 1
            continue
        if any(s.startswith(m) for m in markers):
            i += 1
            continue
        break
    if i == 0:
        return text, 0
    return "\n".join(lines[i:]), i


def _resolve_dleps_data_dir(project_root: Path) -> Path:
    env = os.getenv("DLEPS_DATA_DIR")
    if env:
        env_path = Path(env)
        if env_path.exists():
            return env_path

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        meipass_data = Path(meipass) / "data"
        if meipass_data.exists():
            return meipass_data

    repo_data = project_root / "DLEPS-main" / "DLEPS-main" / "data"
    if repo_data.exists():
        return repo_data

    cwd_data = Path.cwd() / "data"
    if cwd_data.exists():
        return cwd_data

    return repo_data


def dleps_ui() -> None:
    """Integrated DLEPS UI using the bundled DLEPS code under DLEPS-main."""
    st.subheader("DLEPS 药物功效预测前端")
    st.write("输入 SMILES 字符串，选择疾病基因签名，预测富集评分。")

    def _get_dleps_class():
        try:
            from dleps_predictor import DLEPS  # type: ignore

            return DLEPS
        except ModuleNotFoundError as e:
            st.error(
                "模型依赖未安装，无法执行预测/训练。\n"
                f"缺少模块：{e.name}\n\n"
                "说明：DLEPS 依赖 TensorFlow/Keras 等大型依赖。"
                "如果你使用的是 Python 3.13，TensorFlow 可能没有对应的官方轮子，"
                "建议使用 Python 3.11/3.12 的虚拟环境再安装 tensorflow。"
            )
            return None
        except Exception as e:
            st.error(f"导入 DLEPS 失败：{e}")
            return None

    # Ensure DLEPS code path is importable
    dleps_code = _PROJECT_ROOT / "DLEPS-main" / "DLEPS-main" / "code"
    if not dleps_code.exists():
        st.error(f"未找到 DLEPS 代码目录：{dleps_code}")
        return

    if str(dleps_code) not in sys.path:
        sys.path.insert(0, str(dleps_code))

    # Resolve DLEPS data dir
    dleps_data = _resolve_dleps_data_dir(_PROJECT_ROOT)
    if not dleps_data.exists():
        st.error(
            "未找到 data 文件夹。请确认项目结构包含 data/，并从项目仓库运行该应用。\n"
            f"期望路径：{dleps_data}"
        )
        return

    # Base files upload (optional)
    st.subheader("基础文件管理（可选）")
    with st.expander("上传/覆盖模型基础文件"):
        st.write(
            "可在这里上传必需文件：vae.hdf5、denseweight.h5、benchmark.csv、gene_info.txt，"
            "以及可选的 DLEPS_30000_tune_gvae10000.h5。"
        )
        base_files = [
            ("vae.hdf5", ["h5", "hdf5"]),
            ("denseweight.h5", ["h5", "hdf5"]),
            ("benchmark.csv", ["csv"]),
            ("gene_info.txt", ["txt", "csv"]),
            ("DLEPS_30000_tune_gvae10000.h5", ["h5", "hdf5"]),
        ]
        for name, exts in base_files:
            uploaded = st.file_uploader(f"上传 {name}", type=exts, key=f"dleps_base_{name}")
            if uploaded is not None:
                try:
                    dleps_data.mkdir(parents=True, exist_ok=True)
                    (dleps_data / name).write_bytes(uploaded.getvalue())
                    st.success(f"已保存：{dleps_data / name}")
                except Exception as e:
                    st.error(f"保存 {name} 失败：{e}")

        st.write("当前 data/ 内基础文件状态：")
        required = [
            dleps_data / "vae.hdf5",
            dleps_data / "denseweight.h5",
            dleps_data / "benchmark.csv",
            dleps_data / "gene_info.txt",
        ]
        missing = [p.name for p in required if not p.exists()]
        if missing:
            st.warning("缺少：" + ", ".join(missing))
        else:
            st.success("基础文件齐全")

    # Discover diseases from dleps_data
    diseases: Dict[str, tuple[str, str]] = {}
    try:
        for f in dleps_data.iterdir():
            if f.name.endswith("_up"):
                down = dleps_data / (f.name.replace("_up", "_down"))
                if down.exists():
                    diseases[f.stem.replace("_up", "")] = (str(f), str(down))
    except Exception:
        diseases = {}

    # Test tools (mock data)
    st.subheader("测试工具（模拟数据）")
    with st.expander("生成示例疾病签名/填充示例 SMILES"):
        st.write("用于前端自测（不保证生物学意义）。")
        sample_name = st.text_input("示例疾病名称", value="SAMPLE", key="dleps_sample_name")
        sample_smiles = st.text_area(
            "示例 SMILES（每行一个）",
            value="CCO\nCC(=O)O\nCCN\nC1=CC=CC=C1",
            height=120,
            key="dleps_sample_smiles",
        )
        if st.button("生成示例疾病签名并填充输入", key="dleps_make_sample"):
            try:
                up_path = dleps_data / f"{sample_name}_up"
                down_path = dleps_data / f"{sample_name}_down"
                pd.Series([1, 2, 3, 4, 5]).to_csv(up_path, index=False, header=False)
                pd.Series([6, 7, 8, 9, 10]).to_csv(down_path, index=False, header=False)
                st.session_state["dleps_smiles_text"] = sample_smiles
                st.session_state["dleps_select_disease"] = sample_name
                st.success(f"已生成示例疾病签名：{sample_name}（up/down 文件已写入 data/）")
            except Exception as e:
                st.error(f"生成示例数据失败：{e}")

    # Add new disease
    st.subheader("添加新疾病")
    with st.expander("展开以添加新疾病"):
        new_name = st.text_input("输入新疾病名称（英文，无空格）", key="dleps_new_disease")
        up_file = st.file_uploader("上传上调基因文件（CSV，无表头，每行一个基因ID）", type=["csv"], key="dleps_up")
        down_file = st.file_uploader("上传下调基因文件（CSV，无表头，每行一个基因ID）", type=["csv"], key="dleps_down")
        if st.button("保存新疾病", key="dleps_save_disease"):
            if not new_name:
                st.error("请输入疾病名称。")
            elif up_file is None or down_file is None:
                st.error("请上传上调和下调基因文件。")
            else:
                try:
                    up_path = dleps_data / f"{new_name}_up"
                    down_path = dleps_data / f"{new_name}_down"
                    pd.read_csv(up_file, header=None).to_csv(up_path, index=False, header=False)
                    pd.read_csv(down_file, header=None).to_csv(down_path, index=False, header=False)
                    st.success(f"新疾病 '{new_name}' 已添加！请刷新页面以在疾病列表中看到它。")
                    st.rerun()
                except Exception as e:
                    st.error(f"保存失败：{e}")

    # Input SMILES
    st.subheader("输入 SMILES")
    input_method = st.radio("选择输入方式", ("手动输入", "上传 CSV 文件"), key="dleps_input_mode")
    smiles_list: list[str] = []
    if input_method == "手动输入":
        txt = st.text_area("粘贴 SMILES 字符串（每行一个）", height=200, key="dleps_smiles_text")
        if txt:
            smiles_list = [s.strip() for s in txt.splitlines() if s.strip()]
    else:
        uploaded = st.file_uploader("上传 CSV 文件（需包含 'SMILES' 列）", type=["csv"], key="dleps_smiles_csv")
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                if "SMILES" in df.columns:
                    smiles_list = df["SMILES"].dropna().astype(str).tolist()
                else:
                    st.error("CSV 文件必须包含 'SMILES' 列。")
            except Exception as e:
                st.error(f"读取 CSV 失败：{e}")

    st.subheader("选择疾病")
    selected = ""
    if not diseases:
        st.warning("未找到疾病签名文件（*_up / *_down）。可以先用“测试工具”生成示例，或上传自定义签名。")
    else:
        selected = st.selectbox("选择疾病", list(diseases.keys()), key="dleps_select_disease")

    use_mock = st.checkbox("使用模拟预测（不依赖模型）", value=False, key="dleps_use_mock")

    with st.expander("云端预测", expanded=False):
        up_payload = None
        down_payload = None
        if selected and selected in diseases:
            up_file, down_file = diseases[selected]
            try:
                if Path(up_file).exists():
                    up_payload = _cloud_encode_bytes(Path(up_file).read_bytes(), Path(up_file).name, "text/plain")
                if Path(down_file).exists():
                    down_payload = _cloud_encode_bytes(Path(down_file).read_bytes(), Path(down_file).name, "text/plain")
            except Exception as e:
                st.warning(f"读取疾病签名文件失败：{e}")

        payload = {
            "smiles": list(smiles_list),
            "disease": str(selected),
            "use_mock": bool(use_mock),
        }
        if up_payload and down_payload:
            payload["disease_up"] = up_payload
            payload["disease_down"] = down_payload

        if not smiles_list:
            st.info("请先输入 SMILES 后再提交云端预测。")
        elif not selected:
            st.info("请选择疾病后再提交云端预测。")
        else:
            _cloud_submit_section(
                "dleps_predict",
                payload,
                button_label="提交云端预测",
                key="cloud_dleps_predict_btn",
                download_name="dleps_results_cloud.csv",
            )

    def _load_array_from_uploaded(uploaded, expected_exts):
        if uploaded is None:
            return None
        name = getattr(uploaded, "name", "")
        suffix = Path(name).suffix.lower()
        if suffix not in expected_exts:
            raise ValueError(f"文件类型不支持：{suffix}，支持：{', '.join(sorted(expected_exts))}")

        data = uploaded.getvalue()
        if suffix in {".npy"}:
            return np.load(io.BytesIO(data), allow_pickle=False)
        if suffix in {".npz"}:
            npz = np.load(io.BytesIO(data), allow_pickle=False)
            if len(npz.files) != 1:
                raise ValueError(f".npz 里需要且只能有 1 个数组，当前包含：{npz.files}")
            return npz[npz.files[0]]

        tmp_path = None
        try:
            import tempfile
            import h5py

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(data)
                tmp.flush()
                tmp_path = tmp.name
            with h5py.File(tmp_path, "r") as hf:
                if "data" not in hf:
                    raise ValueError("HDF5 文件缺少 key='data' 的数据集")
                return np.array(hf["data"])
        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    # Train UI (optional)
    st.subheader("训练（可选）")
    with st.expander("展开以上传训练数据并训练/微调模型"):
        st.write(
            "这里训练的是 DLEPS 的 Dense 网络（潜向量→978 表达）。"
            "训练依赖 data/ 下的基础文件（vae.hdf5 / denseweight.h5 / benchmark.csv / gene_info.txt）。"
        )
        train_smiles_file = st.file_uploader(
            "上传 SMILES 训练数据（one-hot 数组，.h5/.hdf5/.npy/.npz；HDF5 需 key='data'）",
            type=["h5", "hdf5", "npy", "npz"],
            key="dleps_train_smiles",
        )
        train_rna_file = st.file_uploader(
            "上传 RNA 训练标签（978维数组，.h5/.hdf5/.npy/.npz；HDF5 需 key='data'）",
            type=["h5", "hdf5", "npy", "npz"],
            key="dleps_train_rna",
        )
        val_split = st.slider("验证集比例（从训练集中划分）", 0.05, 0.5, 0.2, 0.05, key="dleps_val_split")
        epochs = st.number_input("epochs", min_value=1, max_value=50000, value=100, step=1, key="dleps_epochs")
        batch_size = st.number_input("batch_size", min_value=1, max_value=4096, value=64, step=1, key="dleps_batch")
        shuffle = st.checkbox("shuffle", value=True, key="dleps_shuffle")
        out_weights_name = st.text_input(
            "保存权重文件名（保存在 data/ 下；若想预测自动加载可命名为 DLEPS_30000_tune_gvae10000.h5）",
            value="DLEPS_custom.h5",
            key="dleps_save_name",
        )

        with st.expander("云端训练", expanded=False):
            if train_smiles_file is None or train_rna_file is None:
                st.info("请先上传训练数据后再提交云端训练。")
            else:
                smiles_payload = _cloud_encode_bytes(
                    train_smiles_file.getvalue(),
                    getattr(train_smiles_file, "name", "smiles"),
                    "application/octet-stream",
                )
                rna_payload = _cloud_encode_bytes(
                    train_rna_file.getvalue(),
                    getattr(train_rna_file, "name", "rna"),
                    "application/octet-stream",
                )
                payload = {
                    "smiles": smiles_payload,
                    "rna": rna_payload,
                    "val_split": float(val_split),
                    "epochs": int(epochs),
                    "batch_size": int(batch_size),
                    "shuffle": bool(shuffle),
                    "out_weights_name": str(out_weights_name),
                }
                _cloud_submit_section(
                    "dleps_train",
                    payload,
                    button_label="提交云端训练",
                    key="cloud_dleps_train_btn",
                )

        if st.button("开始训练并保存权重", key="dleps_train_btn"):
            required = [
                dleps_data / "vae.hdf5",
                dleps_data / "denseweight.h5",
                dleps_data / "benchmark.csv",
                dleps_data / "gene_info.txt",
            ]
            missing = [str(p) for p in required if not p.exists()]
            if missing:
                st.error("缺少基础文件，无法训练：\n" + "\n".join(missing))
                st.stop()

            if train_smiles_file is None or train_rna_file is None:
                st.error("请先上传训练数据（SMILES 与 RNA）。")
                st.stop()

            try:
                x = _load_array_from_uploaded(train_smiles_file, {".h5", ".hdf5", ".npy", ".npz"})
                y = _load_array_from_uploaded(train_rna_file, {".h5", ".hdf5", ".npy", ".npz"})
            except Exception as e:
                st.error(f"读取训练数据失败：{e}")
                st.stop()

            if x is None or y is None:
                st.error("训练数据为空。")
                st.stop()

            if x.shape[0] != y.shape[0]:
                st.error(f"样本数不一致：SMILES={x.shape[0]}，RNA={y.shape[0]}")
                st.stop()
            if y.ndim != 2 or y.shape[1] != 978:
                st.error(f"RNA 标签形状应为 (N, 978)，当前为 {y.shape}")
                st.stop()

            n = x.shape[0]
            val_n = max(1, int(n * float(val_split)))
            if n - val_n < 1:
                st.error("训练集太小，无法划分验证集。")
                st.stop()

            rng = np.random.default_rng(42)
            idx = rng.permutation(n)
            val_idx = idx[:val_n]
            train_idx = idx[val_n:]
            x_train, y_train = x[train_idx], y[train_idx]
            x_val, y_val = x[val_idx], y[val_idx]

            DLEPS = _get_dleps_class()
            if DLEPS is None:
                st.stop()

            try:
                dleps_p = DLEPS()
                with st.spinner("正在训练中（可能需要较长时间）..."):
                    history = dleps_p.train(
                        x_train,
                        y_train,
                        (x_val, y_val),
                        epochs=int(epochs),
                        batch_size=int(batch_size),
                        shuffle=bool(shuffle),
                    )

                out_path = dleps_data / out_weights_name
                dleps_p.model[0].save_weights(str(out_path))

                last_loss = None
                last_val_loss = None
                try:
                    last_loss = history.history.get("loss", [None])[-1]
                    last_val_loss = history.history.get("val_loss", [None])[-1]
                except Exception:
                    pass

                st.success(f"训练完成，权重已保存：{out_path}")
                if last_loss is not None:
                    st.write(f"最后一轮 loss: {last_loss}")
                if last_val_loss is not None:
                    st.write(f"最后一轮 val_loss: {last_val_loss}")
            except Exception as e:
                st.error(f"训练失败：{e}")
                st.stop()

    # Predict
    if st.button("预测", key="dleps_predict_btn"):
        if not smiles_list:
            st.error("请先输入有效的 SMILES。")
        elif not selected:
            st.error("请选择疾病。")
        else:
            up_file, down_file = diseases[selected]
            if use_mock:
                rng = np.random.default_rng(2026)
                scores = rng.uniform(-1, 1, size=len(smiles_list)).tolist()
            else:
                DLEPS = _get_dleps_class()
                if DLEPS is None:
                    st.stop()

                dleps = DLEPS(up_name=up_file, down_name=down_file)
                with st.spinner("正在预测中..."):
                    scores = dleps.predict(smiles_list)

            results_df = pd.DataFrame({
                "SMILES": smiles_list,
                "富集评分 (Enrichment Score)": scores,
            })
            st.subheader("预测结果")
            st.dataframe(results_df)

            csv = results_df.to_csv(index=False)
            st.download_button("下载结果 CSV", csv, "dleps_results.csv", "text/csv")

    st.write("---")
    st.write("**注意**：")
    st.write("- 评分范围通常为 -1 到 1。正值表示药物可能逆转疾病。")
    st.write("- 如果输入 SMILES 过多，预测可能需要时间。")
    st.write("- 数据基于项目中的基因签名文件。")



def _load_table_from_bytes(data: bytes, name: str) -> pd.DataFrame:
    ext = Path(name).suffix.lower().lstrip(".")
    buf = io.BytesIO(data)
    if ext in {"csv", "txt"}:
        text = buf.read().decode("utf-8", errors="ignore")
        cleaned, _ = _strip_leading_comments(text)
        return pd.read_csv(io.StringIO(cleaned), sep=None, engine="python")
    if ext == "tsv":
        text = buf.read().decode("utf-8", errors="ignore")
        cleaned, _ = _strip_leading_comments(text)
        return pd.read_csv(io.StringIO(cleaned), sep="\t")
    if ext in {"xlsx", "xls"}:
        return pd.read_excel(buf)
    if ext in {"json"}:
        return pd.read_json(buf)
    if ext in {"jsonl", "ndjson"}:
        return pd.read_json(buf, lines=True)
    if ext in {"parquet", "pq"}:
        return pd.read_parquet(buf)
    if ext in {"feather", "arrow"}:
        return pd.read_feather(buf)
    raise ValueError(f"不支持的文件类型: .{ext}")


@st.cache_data(show_spinner=False, max_entries=50)
def _cached_load_table_from_bytes(data: bytes, name: str) -> pd.DataFrame:
    return _load_table_from_bytes(data, name)


@st.cache_data(show_spinner=False, max_entries=20)
def _cached_concat_tables(files: List[tuple[str, bytes]], add_source_col: str) -> pd.DataFrame:
    frames = []
    for name, data in files:
        df = _cached_load_table_from_bytes(data, name)
        if add_source_col:
            df[add_source_col] = name
        frames.append(df)
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, axis=0, ignore_index=True)


def _load_tables_from_uploads(
    uploads: List[Any], *, add_source_col: str = "_source"
) -> pd.DataFrame:
    if not uploads:
        raise ValueError("未选择文件")
    files = []
    for up in uploads:
        name = getattr(up, "name", "uploaded")
        files.append((str(name), up.getvalue()))
    return _cached_concat_tables(files, str(add_source_col))


def _guess_numeric_columns(df: pd.DataFrame, *, max_cols: int = 20) -> List[str]:
    cols = []
    for c in df.columns:
        ser = pd.to_numeric(df[c], errors="coerce")
        ratio = float(ser.notna().mean()) if len(ser) else 0.0
        if ratio >= 0.8:
            cols.append(str(c))
        if len(cols) >= max_cols:
            break
    return cols


def _guess_categorical_columns(df: pd.DataFrame, *, exclude: List[str], max_cols: int = 20) -> List[str]:
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        ser = df[c]
        n_unique = int(ser.nunique(dropna=True)) if len(ser) else 0
        if 1 < n_unique <= 30:
            cols.append(str(c))
        if len(cols) >= max_cols:
            break
    return cols


def _apply_preprocess(
    df: pd.DataFrame,
    *,
    trim_cols: bool,
    lower_cols: bool,
    strip_cells: bool,
    drop_empty_rows: bool,
    drop_empty_cols: bool,
    drop_duplicates: bool,
    numeric_cols: List[str],
    categorical_cols: List[str],
    fill_numeric: str,
    fill_categorical: str,
) -> pd.DataFrame:
    out = df.copy()

    if trim_cols:
        out.columns = [str(c).strip() for c in out.columns]
    if lower_cols:
        out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]

    if strip_cells:
        for c in out.columns:
            if out[c].dtype == object:
                out[c] = out[c].astype("string").str.strip()

    if drop_empty_rows:
        out = out.dropna(axis=0, how="all")
    if drop_empty_cols:
        out = out.dropna(axis=1, how="all")
    if drop_duplicates:
        out = out.drop_duplicates()

    # Coerce numeric columns
    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if fill_numeric != "不填充":
        for c in numeric_cols:
            if c not in out.columns:
                continue
            if fill_numeric == "中位数":
                out[c] = out[c].fillna(out[c].median())
            elif fill_numeric == "均值":
                out[c] = out[c].fillna(out[c].mean())
            elif fill_numeric == "0":
                out[c] = out[c].fillna(0)

    if fill_categorical != "不填充":
        for c in categorical_cols:
            if c not in out.columns:
                continue
            if fill_categorical == "众数":
                mode = out[c].mode(dropna=True)
                if len(mode) > 0:
                    out[c] = out[c].fillna(mode.iloc[0])
            elif fill_categorical == "unknown":
                out[c] = out[c].fillna("unknown")

    return out


def _render_preprocess_panel(df: pd.DataFrame, key_prefix: str) -> Dict[str, Any]:
    with st.expander("导入预处理", expanded=False):
        trim_cols = st.checkbox("列名去空格", value=True, key=f"{key_prefix}_trim_cols")
        lower_cols = st.checkbox("列名小写并用下划线", value=False, key=f"{key_prefix}_lower_cols")
        strip_cells = st.checkbox("字符串去空格", value=True, key=f"{key_prefix}_strip_cells")
        drop_empty_rows = st.checkbox("删除全空行", value=True, key=f"{key_prefix}_drop_empty_rows")
        drop_empty_cols = st.checkbox("删除全空列", value=True, key=f"{key_prefix}_drop_empty_cols")
        drop_duplicates = st.checkbox("删除重复行", value=False, key=f"{key_prefix}_drop_duplicates")

        auto_numeric = _guess_numeric_columns(df)
        numeric_cols = st.multiselect(
            "数值列（用于类型转换/缺失填充）",
            options=list(df.columns),
            default=[c for c in auto_numeric if c in df.columns],
            key=f"{key_prefix}_numeric_cols",
        )
        auto_cat = _guess_categorical_columns(df, exclude=list(numeric_cols))
        categorical_cols = st.multiselect(
            "分类型列（用于缺失填充）",
            options=[c for c in df.columns if c not in set(numeric_cols)],
            default=[c for c in auto_cat if c in df.columns],
            key=f"{key_prefix}_categorical_cols",
        )
        fill_numeric = st.selectbox(
            "数值缺失填充",
            options=["不填充", "中位数", "均值", "0"],
            index=1,
            key=f"{key_prefix}_fill_numeric",
        )
        fill_categorical = st.selectbox(
            "分类型缺失填充",
            options=["不填充", "众数", "unknown"],
            index=0,
            key=f"{key_prefix}_fill_categorical",
        )

    return {
        "trim_cols": trim_cols,
        "lower_cols": lower_cols,
        "strip_cells": strip_cells,
        "drop_empty_rows": drop_empty_rows,
        "drop_empty_cols": drop_empty_cols,
        "drop_duplicates": drop_duplicates,
        "numeric_cols": list(numeric_cols),
        "categorical_cols": list(categorical_cols),
        "fill_numeric": fill_numeric,
        "fill_categorical": fill_categorical,
    }

def _parse_kv_lines(text: str) -> Dict[str, float]:
    params: Dict[str, float] = {}
    if not text:
        return params
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
        elif ":" in line:
            k, v = line.split(":", 1)
        else:
            raise ValueError(f"参数行格式错误: '{line}'，应为 key=value 或 key: value")
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"参数行 key 为空: '{line}'")
        try:
            params[k] = float(v)
        except Exception:
            raise ValueError(f"参数值需要是数值: '{line}'")
    return params


def _list_local_models() -> List[str]:
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    items = [str(p) for p in models_dir.glob("*.joblib")]
    pretrained_dir = models_dir / "pretrained"
    if pretrained_dir.exists():
        items.extend([str(p) for p in pretrained_dir.glob("*.joblib")])
    return sorted(items)


def _get_file_mtime(path: str) -> float:
    try:
        return float(Path(path).stat().st_mtime)
    except Exception:
        return 0.0


@st.cache_resource(show_spinner=False, max_entries=20)
def _cached_joblib_from_bytes(data: bytes):
    return joblib.load(io.BytesIO(data))


@st.cache_resource(show_spinner=False, max_entries=20)
def _cached_joblib_from_path(path: str, mtime: float):
    _ = mtime
    return joblib.load(path)


@st.cache_resource(show_spinner=False, max_entries=10)
def _cached_torch_bundle_from_bytes(data: bytes):
    from src.drug.torch_predictor import load_torch_bundle_from_bytes  # type: ignore

    return load_torch_bundle_from_bytes(data)


@st.cache_resource(show_spinner=False, max_entries=10)
def _cached_torch_bundle_from_path(path: str, mtime: float):
    _ = mtime
    from src.drug.torch_predictor import load_torch_bundle  # type: ignore

    return load_torch_bundle(path)


@st.cache_resource(show_spinner=False, max_entries=10)
def _cached_docking_bundle_from_bytes(data: bytes):
    from src.drug.docking_cross_attention import load_docking_bundle_from_bytes

    return load_docking_bundle_from_bytes(data)


@st.cache_resource(show_spinner=False, max_entries=10)
def _cached_docking_bundle_from_path(path: str, mtime: float):
    _ = mtime
    from src.drug.docking_cross_attention import load_docking_bundle

    return load_docking_bundle(path)


@st.cache_resource(show_spinner=False, max_entries=10)
def _cached_transformer_bundle_from_bytes(data: bytes):
    from src.drug.transformer_predictor import load_transformer_bundle_from_bytes  # type: ignore

    return load_transformer_bundle_from_bytes(data)


@st.cache_resource(show_spinner=False, max_entries=10)
def _cached_transformer_bundle_from_path(path: str, mtime: float):
    _ = mtime
    from src.drug.transformer_predictor import load_transformer_bundle  # type: ignore

    return load_transformer_bundle(path)


@st.cache_resource(show_spinner=False, max_entries=10)
def _cached_sequence_featurizer(version: int):
    return SequenceFeatures(version=int(version))


@st.cache_resource(show_spinner=False, max_entries=10)
def _cached_molecule_featurizer(version: int, radius: int, n_bits: int):
    from src.drug.featurizer import MoleculeFeatures  # type: ignore

    return MoleculeFeatures(version=int(version), radius=int(radius), n_bits=int(n_bits))


def _load_bundle(uploaded_file, local_path: Optional[str]) -> EpitopeModelBundle:
    if uploaded_file is not None:
        return _cached_joblib_from_bytes(uploaded_file.getvalue())
    if local_path:
        return _cached_joblib_from_path(local_path, _get_file_mtime(local_path))
    raise ValueError("请上传模型文件或选择本地模型")


def _load_drug_bundle(uploaded_file, local_path: Optional[str]):
    if uploaded_file is not None:
        return _cached_joblib_from_bytes(uploaded_file.getvalue())
    if local_path:
        return _cached_joblib_from_path(local_path, _get_file_mtime(local_path))
    raise ValueError("请上传模型文件或选择本地模型")


def _load_torch_bundle(uploaded_file, local_path: Optional[str]):
    if uploaded_file is not None:
        return _cached_torch_bundle_from_bytes(uploaded_file.getvalue())
    if local_path:
        return _cached_torch_bundle_from_path(local_path, _get_file_mtime(local_path))
    raise ValueError("请上传模型文件或选择本地模型")


def _render_plot_images(out_dir: Path, prefix: str) -> None:
    p1 = out_dir / f"{prefix}_scatter.png"
    p2 = out_dir / f"{prefix}_residual_hist.png"
    p3 = out_dir / f"{prefix}_residual_vs_pred.png"

    cols = st.columns(3)
    if p1.exists():
        cols[0].image(str(p1), caption=p1.name, use_container_width=True)
    if p2.exists():
        cols[1].image(str(p2), caption=p2.name, use_container_width=True)
    if p3.exists():
        cols[2].image(str(p3), caption=p3.name, use_container_width=True)


def _bundle_input_vector(bundle: EpitopeModelBundle, sequence: str, env: Dict[str, float]) -> np.ndarray:
    feat_v = int(getattr(bundle, "featurizer_version", 1) or 1)
    featurizer = _cached_sequence_featurizer(feat_v)
    seq_x = featurizer.transform_one(sequence).astype(np.float32)

    env_vec = []
    for c in bundle.env_cols:
        if c in env:
            env_vec.append(float(env[c]))
        else:
            env_vec.append(float(bundle.env_medians.get(c, 0.0)))

    env_x = np.asarray(env_vec, dtype=np.float32)
    x = np.concatenate([seq_x, env_x], axis=0)
    return x.astype(np.float32)


def _make_x_only_epitope(
    df: pd.DataFrame,
    *,
    sequence_col: str,
    env_cols: List[str],
    env_medians: Dict[str, float],
    featurizer_version: int,
) -> np.ndarray:
    featurizer = SequenceFeatures(version=int(featurizer_version))
    seq_x = featurizer.transform_many(df[sequence_col].astype(str).tolist())

    if env_cols:
        env_df = df[env_cols].copy()
        for c in env_cols:
            env_df[c] = pd.to_numeric(env_df[c], errors="coerce").astype(float)
            env_df[c] = env_df[c].fillna(float(env_medians.get(c, float(env_df[c].median()))))
        env_x = env_df.to_numpy(dtype=np.float32)
    else:
        env_x = np.zeros((len(df), 0), dtype=np.float32)

    return np.concatenate([seq_x, env_x], axis=1).astype(np.float32)


def epitope_train_ui() -> None:
    st.subheader("训练（CSV → 模型）")

    uploads = st.file_uploader(
        "上传训练集（支持多文件）",
        type=_UPLOAD_TYPES,
        accept_multiple_files=True,
        key="train_csv",
    )
    if not uploads:
        st.info("需要一个包含 sequence + 实验参数数值列 + 目标列 的 CSV")
        return

    try:
        df_raw = _load_tables_from_uploads(uploads)
    except Exception as e:
        st.error(f"读取文件失败：{e}")
        return

    cfg = _render_preprocess_panel(df_raw, "epitope_train")
    df = _apply_preprocess(df_raw, **cfg)
    _preview_df(df, title="数据预览")

    sequence_col = st.selectbox("sequence 列名", options=list(df.columns), index=(list(df.columns).index("sequence") if "sequence" in df.columns else 0))
    target_col = st.selectbox("目标列（疗效）", options=[c for c in df.columns if c != sequence_col])

    drop_target_na = st.checkbox("过滤目标列为空", value=True, key="epitope_train_drop_target_na")
    drop_target_zero = st.checkbox("将目标列=0 视为无标注", value=False, key="epitope_train_drop_target_zero")

    auto_env_cols = infer_env_cols(df, sequence_col=sequence_col, target_col=target_col, env_cols=None)
    env_cols = st.multiselect("实验参数列（数值）", options=[c for c in df.columns if c not in (sequence_col, target_col)], default=auto_env_cols)

    epitope_algo_map = {
        "Confluencia 1.0 HGB": "hgb",
        "Confluencia 1.0 GBR": "gbr",
        "Confluencia 1.0 RF": "rf",
        "Confluencia 1.0 MLP": "mlp",
        "Confluencia 1.0 SGD": "sgd",
    }
    epitope_algo_label = st.selectbox("训练算法（参考 Confluencia 1.0）", options=list(epitope_algo_map.keys()), index=0)
    model_name = epitope_algo_map[epitope_algo_label]
    st.caption(
        "模型说明：hgb=大规模友好、支持早停；gbr=稳健但训练较慢；rf=抗噪、可解释性较好；"
        "mlp=非线性强但需更多数据；sgd=线性+弹性网，速度快、内存占用低，适合大规模数据。"
    )
    with st.expander("训练优化参数", expanded=False):
        mlp_alpha = st.number_input("MLP 正则强度 (alpha)", min_value=1e-8, max_value=1e-2, value=1e-4, format="%.6f")
        mlp_early = st.checkbox("MLP 早停", value=True)
        mlp_patience = st.number_input("MLP 早停耐心", min_value=1, max_value=200, value=10, step=1)
        sgd_alpha = st.number_input("SGD 正则强度 (alpha)", min_value=1e-8, max_value=1e-2, value=1e-4, format="%.6f")
        sgd_l1_ratio = st.number_input("SGD l1_ratio", min_value=0.0, max_value=1.0, value=0.15, step=0.05)
        sgd_early = st.checkbox("SGD 早停", value=True)
        hgb_l2 = st.number_input("HGB L2 正则", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    test_size = st.slider("验证集比例", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    seed = st.number_input("随机种子", min_value=0, max_value=10_000_000, value=42, step=1)

    model_out = st.text_input("保存路径", value="models/epitope_model.joblib")

    with st.expander("云端训练", expanded=False):
        payload = {
            "data": _cloud_encode_dataframe(df),
            "sequence_col": str(sequence_col),
            "target_col": str(target_col),
            "env_cols": list(env_cols),
            "model_name": str(model_name),
            "test_size": float(test_size),
            "random_state": int(seed),
            "mlp_alpha": float(mlp_alpha),
            "mlp_early_stopping": bool(mlp_early),
            "mlp_patience": int(mlp_patience),
            "sgd_alpha": float(sgd_alpha),
            "sgd_l1_ratio": float(sgd_l1_ratio),
            "sgd_early_stopping": bool(sgd_early),
            "hgb_l2": float(hgb_l2),
            "drop_target_na": bool(drop_target_na),
            "drop_target_zero": bool(drop_target_zero),
            "model_out": str(model_out),
        }
        _cloud_submit_section(
            "epitope_train",
            payload,
            button_label="提交云端训练",
            key="cloud_epitope_train_btn",
            model_id_state_key="cloud_epitope_model_id",
        )

    with st.expander("超参数调参（Grid / Randomized）", expanded=False):
        st.markdown("使用 `param_grids/param_grid_examples.json` 示例或上传自定义 JSON 文件进行调参。")
        strategy = st.selectbox("搜索策略", options=["grid", "random"], index=0)
        n_iter = st.number_input("随机搜索迭代次数 (仅 random 有效)", min_value=1, value=20, step=1)
        cv = st.number_input("CV 折数", min_value=2, value=5, step=1)
        use_example = st.checkbox("使用示例参数网格（param_grids/param_grid_examples.json）", value=True)
        uploaded_grid = st.file_uploader("或上传 JSON 参数网格", type=["json"], key="epitope_param_grid")
        tune_model_out = st.text_input("调参后模型保存路径", value="models/epitope_model_tuned.joblib")
        if st.button("开始调参", key="epitope_start_tune"):
            try:
                if use_example and not uploaded_grid:
                    grid_path = _PROJECT_ROOT / "param_grids" / "param_grid_examples.json"
                    with open(grid_path, "r", encoding="utf-8") as f:
                        grid_all = json.load(f)
                else:
                    if not uploaded_grid:
                        st.error("请选择示例或上传参数网格 JSON 文件")
                        raise RuntimeError("no param grid")
                    grid_all = json.load(uploaded_grid)

                grid = grid_all.get(model_name, {}) if isinstance(grid_all, dict) else {}

                x, y, _, _ = make_xy(
                    df,
                    sequence_col=sequence_col,
                    target_col=target_col,
                    env_cols=list(env_cols),
                    featurizer=SequenceFeatures(version=int(getattr(st.session_state, "epitope_train_featurizer_version", 2)) if True else 1),
                    env_medians=None,
                )

                base = build_model(model_name=model_name, random_state=int(seed))

                with st.spinner("调参中，请稍候..."):
                    best_est, best_params, cv_results = run_hyper_search(
                        base,
                        grid,
                        x,
                        y,
                        strategy=strategy,
                        n_iter=int(n_iter),
                        cv=int(cv),
                        scoring=None,
                        n_jobs=-1,
                        random_state=int(seed),
                    )

                out_path = Path(tune_model_out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(best_est, out_path)

                st.success("调参完成")
                st.json(best_params)
                st.write(f"已保存模型: {out_path}")
            except Exception as e:
                st.error(f"调参失败：{e}")

    with st.expander("独立评估（专用评估模块）", expanded=False):
        eval_uploads = st.file_uploader(
            "上传评估集（支持多文件）",
            type=_UPLOAD_TYPES,
            accept_multiple_files=True,
            key="epitope_eval_csv",
        )
        eval_model_path = st.text_input("评估模型路径", value=str(model_out), key="epitope_eval_model_path")
        if st.button("开始评估", key="epitope_eval_btn"):
            if not eval_uploads:
                st.error("请先上传评估数据")
            elif not Path(eval_model_path).exists():
                st.error(f"模型不存在: {eval_model_path}")
            else:
                import tempfile

                try:
                    eval_df_raw = _load_tables_from_uploads(eval_uploads)
                    eval_cfg = _render_preprocess_panel(eval_df_raw, "epitope_eval")
                    eval_df = _apply_preprocess(eval_df_raw, **eval_cfg)

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tf:
                        eval_tmp_path = tf.name
                    eval_df.to_csv(eval_tmp_path, index=False)

                    eval_result = evaluate_epitope_from_csv(
                        model_path=str(eval_model_path),
                        data_path=str(eval_tmp_path),
                        sequence_col=str(sequence_col),
                        target_col=str(target_col),
                        env_cols=list(env_cols),
                    )
                    st.success("评估完成")
                    st.json(eval_result.metrics)
                except Exception as e:
                    st.error(f"评估失败：{e}")
                finally:
                    try:
                        Path(eval_tmp_path).unlink(missing_ok=True)  # type: ignore[name-defined]
                    except Exception:
                        pass

    if st.button("开始训练", type="primary"):
        train_df = df.copy()
        if drop_target_na and target_col in train_df.columns:
            train_df = train_df[train_df[target_col].notna()].copy()
        if drop_target_zero and target_col in train_df.columns:
            train_df = train_df[pd.to_numeric(train_df[target_col], errors="coerce") != 0].copy()

        with st.spinner("训练中..."):
            import tempfile

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tf:
                    train_tmp_path = tf.name
                train_df.to_csv(train_tmp_path, index=False)

                train_result = train_epitope_from_csv(
                    data_path=str(train_tmp_path),
                    model_out=str(model_out),
                    sequence_col=str(sequence_col),
                    target_col=str(target_col),
                    env_cols=list(env_cols),
                    model_name=str(model_name),
                    test_size=float(test_size),
                    seed=int(seed),
                    featurizer_version=2,
                    mlp_alpha=float(mlp_alpha),
                    mlp_early_stopping=bool(mlp_early),
                    mlp_patience=int(mlp_patience),
                    sgd_alpha=float(sgd_alpha),
                    sgd_l1_ratio=float(sgd_l1_ratio),
                    sgd_early_stopping=bool(sgd_early),
                    hgb_l2=float(hgb_l2),
                )
                bundle = joblib.load(train_result.model_path)
                metrics = train_result.metrics
            finally:
                try:
                    Path(train_tmp_path).unlink(missing_ok=True)  # type: ignore[name-defined]
                except Exception:
                    pass

        st.success("训练完成")
        render_training_result_panel(metrics, getattr(bundle, "feature_names", None))

        with st.expander("显示原始指标 JSON"):
            st.json(metrics)

        st.write("env_cols:", bundle.env_cols)

        out_path = Path(model_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, out_path)
        st.write("已保存到:", str(out_path))

        buf = io.BytesIO()
        joblib.dump(bundle, buf)
        data = buf.getvalue()
        st.download_button(
            "下载模型文件",
            data=data,
            file_name=out_path.name,
            mime="application/octet-stream",
        )


def epitope_crawl_ui() -> None:
    st.subheader("爬虫数据汇聚（URL/本地表格合并）")
    st.caption("支持 CSV/TSV/Excel 或 FASTA 的 URL/本地路径；FASTA 也支持 uniprot:ACC 与 pdb:ID（如 uniprot:P12345,Q8N158 或 pdb:1AKE）。表格会合并并增加 _source 列。")

    sources_text = st.text_area(
        "数据源（每行一个 URL 或本地路径）",
        value="",
        height=120,
        key="epitope_crawl_sources",
    )

    data_mode = st.selectbox(
        "数据格式",
        options=["表格 (CSV/TSV/Excel)", "FASTA"],
        index=0,
        key="epitope_crawl_mode",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        cache_dir = st.text_input("cache_dir", value="data/cache/http", key="epitope_crawl_cache")
    with col2:
        timeout_s = st.number_input("timeout（秒）", min_value=1.0, value=30.0, step=1.0, key="epitope_crawl_timeout")
    with col3:
        sleep_s = st.number_input("sleep（秒/请求）", min_value=0.0, value=0.2, step=0.1, key="epitope_crawl_sleep")

    col4, col5, col6 = st.columns(3)
    with col4:
        sequence_col = st.text_input("sequence 列名（表格可选）", value="sequence", key="epitope_crawl_seqcol")
    with col5:
        min_len = st.number_input("最短长度", min_value=1, value=8, step=1, key="epitope_crawl_minlen")
    with col6:
        max_len = st.number_input("最长长度", min_value=1, value=25, step=1, key="epitope_crawl_maxlen")

    col7, col8 = st.columns(2)
    with col7:
        allow_x = st.checkbox("允许 X", value=False, key="epitope_crawl_allow_x")
    with col8:
        drop_duplicates = st.checkbox("去重（按 sequence）", value=True, key="epitope_crawl_dedup")

    out_path = st.text_input("保存 CSV 路径（可选）", value="data/epitope_crawled.csv", key="epitope_crawl_out")

    with st.expander("云端汇聚", expanded=False):
        sources_cloud = [s.strip() for s in sources_text.splitlines() if s.strip()]
        payload = {
            "sources": sources_cloud,
            "data_mode": str(data_mode),
            "cache_dir": str(cache_dir),
            "timeout": float(timeout_s),
            "sleep": float(sleep_s),
            "sequence_col": str(sequence_col),
            "min_len": int(min_len),
            "max_len": int(max_len),
            "allow_x": bool(allow_x),
            "drop_duplicates": bool(drop_duplicates),
            "out_path": str(out_path),
        }
        _cloud_submit_section(
            "epitope_crawl",
            payload,
            button_label="提交云端汇聚",
            key="cloud_epitope_crawl_btn",
            download_name="epitope_crawled_cloud.csv",
        )

    if st.button("开始汇聚", type="primary", key="epitope_crawl_btn"):
        sources = [s.strip() for s in sources_text.splitlines() if s.strip()]
        if not sources:
            st.error("请至少提供一个数据源")
            return

        # 温和提示：当用户设置的请求间隔过短或期望的请求频率过高时，提醒遵守礼貌爬虫规范
        try:
            sleep_val = float(sleep_s)
        except Exception:
            sleep_val = 0.0
        req_per_sec = float('inf') if sleep_val <= 0 else 1.0 / sleep_val
        # 阈值：每秒 >5 次 或 sleep < 0.2s 视为较高频率
        if req_per_sec > 5 or sleep_val < 0.2:
            st.warning("文明爬虫，从我做起 — 你当前设置的请求频率较高，建议增大 sleep（秒/请求）以降低对目标服务器压力。")

        try:
            from src.epitope.crawler import crawl_epitope_csv_datasets, crawl_epitope_fasta_sources
        except Exception as e:
            st.error(f"无法导入表位爬虫模块：{e}")
            return

        with st.spinner("抓取/读取中..."):
            if data_mode.startswith("FASTA"):
                df = crawl_epitope_fasta_sources(
                    sources,
                    cache_dir=str(cache_dir),
                    timeout=float(timeout_s),
                    sleep_seconds=float(sleep_s),
                    user_agent="epitope-crawler/1.0 (research; contact: local)",
                    min_len=int(min_len),
                    max_len=int(max_len),
                    allow_x=bool(allow_x),
                    drop_duplicates=bool(drop_duplicates),
                )
            else:
                df = crawl_epitope_csv_datasets(
                    sources,
                    cache_dir=str(cache_dir),
                    timeout=float(timeout_s),
                    sleep_seconds=float(sleep_s),
                    user_agent="epitope-urlcsv/1.0 (research; contact: local)",
                    sequence_col=str(sequence_col).strip() or None,
                    min_len=int(min_len),
                    max_len=int(max_len),
                    allow_x=bool(allow_x),
                    drop_duplicates=bool(drop_duplicates),
                )

        st.success(f"完成：{len(df)} 行")
        _preview_df(df, title="抓取结果预览", max_rows=50)

        if out_path:
            p = Path(out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(p, index=False)
            st.write("已保存到:", str(p))

        st.download_button(
            "下载合并 CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=Path(out_path).name if out_path else "epitope_crawled.csv",
            mime="text/csv",
        )


def epitope_self_train_ui() -> None:
    st.subheader("自训练（伪标签 + 不确定性筛选）")

    labeled_up = st.file_uploader(
        "上传有标注数据（支持多文件）",
        type=_UPLOAD_TYPES,
        accept_multiple_files=True,
        key="epitope_labeled",
    )
    unlabeled_up = st.file_uploader(
        "上传无标注数据（支持多文件）",
        type=_UPLOAD_TYPES,
        accept_multiple_files=True,
        key="epitope_unlabeled",
    )

    if not labeled_up or not unlabeled_up:
        st.info("请同时上传有标注与无标注数据")
        return

    try:
        labeled_raw = _load_tables_from_uploads(labeled_up)
        unlabeled_raw = _load_tables_from_uploads(unlabeled_up)
    except Exception as e:
        st.error(f"读取文件失败：{e}")
        return

    labeled_cfg = _render_preprocess_panel(labeled_raw, "epitope_st_labeled")
    unlabeled_cfg = _render_preprocess_panel(unlabeled_raw, "epitope_st_unlabeled")
    labeled = _apply_preprocess(labeled_raw, **labeled_cfg)
    unlabeled = _apply_preprocess(unlabeled_raw, **unlabeled_cfg)

    _preview_df(labeled, title="标注数据预览")
    _preview_df(unlabeled, title="未标注数据预览")

    sequence_col = st.selectbox("sequence 列名", options=list(labeled.columns), key="epitope_st_seq")
    target_col = st.selectbox("目标列（疗效）", options=[c for c in labeled.columns if c != sequence_col], key="epitope_st_target")

    drop_target_na = st.checkbox("过滤目标列为空", value=True, key="epitope_st_drop_target_na")
    drop_target_zero = st.checkbox("将目标列=0 视为无标注", value=False, key="epitope_st_drop_target_zero")

    auto_env_cols = infer_env_cols(labeled, sequence_col=sequence_col, target_col=target_col, env_cols=None)
    env_cols = st.multiselect(
        "实验参数列（数值）",
        options=[c for c in labeled.columns if c not in (sequence_col, target_col)],
        default=auto_env_cols,
        key="epitope_st_env",
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        featurizer_version = st.selectbox("特征版本", options=[1, 2], index=0, key="epitope_st_feat")
    with col2:
        model_name = st.selectbox("模型", options=["hgb", "gbr", "rf", "mlp", "sgd"], index=0, key="epitope_st_model")
        st.caption(
            "模型说明：hgb=大规模友好、支持早停；gbr=稳健但训练较慢；rf=抗噪、可解释性较好；"
            "mlp=非线性强但需更多数据；sgd=线性+弹性网，速度快、内存占用低，适合大规模数据。"
        )
        with st.expander("训练优化参数", expanded=False):
            mlp_alpha = st.number_input("MLP 正则强度 (alpha)", min_value=1e-8, max_value=1e-2, value=1e-4, format="%.6f", key="epitope_st_mlp_alpha")
            mlp_early = st.checkbox("MLP 早停", value=True, key="epitope_st_mlp_early")
            mlp_patience = st.number_input("MLP 早停耐心", min_value=1, max_value=200, value=10, step=1, key="epitope_st_mlp_pat")
            sgd_alpha = st.number_input("SGD 正则强度 (alpha)", min_value=1e-8, max_value=1e-2, value=1e-4, format="%.6f", key="epitope_st_sgd_alpha")
            sgd_l1_ratio = st.number_input("SGD l1_ratio", min_value=0.0, max_value=1.0, value=0.15, step=0.05, key="epitope_st_sgd_l1")
            sgd_early = st.checkbox("SGD 早停", value=True, key="epitope_st_sgd_early")
            hgb_l2 = st.number_input("HGB L2 正则", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="epitope_st_hgb_l2")
    with col3:
        n_models = st.number_input("集成模型数", min_value=1, value=5, step=1, key="epitope_st_n_models")
    with col4:
        keep_frac = st.slider("保留低不确定性比例", min_value=0.05, max_value=1.0, value=0.5, step=0.05, key="epitope_st_keep")

    # 伪标签权重设置：训练时伪样本的相对权重（标注样本权重固定为 1.0）
    col_pw1, col_pw2 = st.columns(2)
    with col_pw1:
        pseudo_weight = st.slider("伪样本权重", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="epitope_st_pseudo_w")
    with col_pw2:
        use_soft_labels = st.checkbox("使用软标签（伪标签为模型均值，默认开启）", value=True, key="epitope_st_soft")

    # 筛选与一致性策略
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        selection_mode = st.selectbox(
            "伪样本筛选模式",
            options=["固定比例 (keep_frac)", "自适应分位 (percentile)", "sigma 阈值"],
            index=0,
            key="epitope_st_selmode",
        )
    with col_s2:
        use_consistency = st.checkbox("启用一致性增强筛选（对环境扰动鲁棒）", value=False, key="epitope_st_consistency")

    if selection_mode.startswith("自适应"):
        adaptive_percentile = st.slider("sigma 分位 (%) (越低越严格)", min_value=1, max_value=99, value=50, step=1, key="epitope_st_percentile")
    elif selection_mode.startswith("sigma"):
        sigma_thresh = st.number_input("sigma 最大阈值", min_value=0.0, value=0.5, step=0.01, format="%.4f", key="epitope_st_sigma_thresh")

    if use_consistency:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            n_augs = st.number_input("一致性扰动次数", min_value=1, max_value=10, value=3, step=1, key="epitope_st_naugs")
        with col_c2:
            noise_scale = st.number_input("环境扰动相对尺度", min_value=0.0, max_value=1.0, value=0.05, step=0.01, key="epitope_st_noise")
        consistency_tol = st.number_input("一致性最大允许 std", min_value=0.0, value=0.1, step=0.01, format="%.4f", key="epitope_st_cons_tol")

    save_pseudo_log = st.checkbox("保存伪样本日志", value=True, key="epitope_st_save_pseudo")
    pseudo_log_path = st.text_input("伪样本日志路径", value="logs/epitope_pseudo_labels.csv", key="epitope_st_pseudo_log_path")

    # 参数级 EMA 教师（仅在使用 PyTorch 模型时适用）
    enable_param_ema = st.checkbox("启用参数级 EMA 教师（适用于 PyTorch 模型）", value=False, key="epitope_st_param_ema")
    if enable_param_ema:
        ema_decay = st.number_input("EMA 衰减系数 (decay)", min_value=0.0, max_value=0.9999, value=0.99, step=0.001, format="%.4f", key="epitope_st_ema_decay")
    else:
        ema_decay = 0.0

    # EMA 教师（基于预测值的指数移动平均，模型无关）
    use_ema = st.checkbox("启用 EMA 教师（预测级）", value=False, key="epitope_st_ema_use")
    if use_ema:
        ema_decay = st.number_input("EMA 衰减系数 (0-1, 接近 1 更慢更新)", min_value=0.0, max_value=0.9999, value=0.9, step=0.01, format="%.4f", key="epitope_st_ema_alpha")

    col5, col6, col7 = st.columns(3)
    with col5:
        test_size = st.slider("验证集比例", min_value=0.1, max_value=0.5, value=0.2, step=0.05, key="epitope_st_test")
    with col6:
        seed = st.number_input("随机种子", min_value=0, max_value=10_000_000, value=42, step=1, key="epitope_st_seed")
    with col7:
        min_labeled = st.number_input("最少标注样本", min_value=5, value=20, step=1, key="epitope_st_min")

    model_out = st.text_input("模型保存路径", value="models/epitope_self_trained.joblib", key="epitope_st_out")
    data_out = st.text_input("合并数据保存路径（可选）", value="data/epitope_self_train.csv", key="epitope_st_data_out")

    with st.expander("云端自训练", expanded=False):
        payload = {
            "labeled": _cloud_encode_dataframe(labeled),
            "unlabeled": _cloud_encode_dataframe(unlabeled),
            "sequence_col": str(sequence_col),
            "target_col": str(target_col),
            "env_cols": list(env_cols),
            "featurizer_version": int(featurizer_version),
            "model_name": str(model_name),
            "mlp_alpha": float(mlp_alpha),
            "mlp_early_stopping": bool(mlp_early),
            "mlp_patience": int(mlp_patience),
            "sgd_alpha": float(sgd_alpha),
            "sgd_l1_ratio": float(sgd_l1_ratio),
            "sgd_early_stopping": bool(sgd_early),
            "hgb_l2": float(hgb_l2),
            "n_models": int(n_models),
            "keep_frac": float(keep_frac),
            "test_size": float(test_size),
            "random_state": int(seed),
            "min_labeled": int(min_labeled),
            "drop_target_na": bool(drop_target_na),
            "drop_target_zero": bool(drop_target_zero),
            "model_out": str(model_out),
            "data_out": str(data_out),
        }
        _cloud_submit_section(
            "epitope_self_train",
            payload,
            button_label="提交云端自训练",
            key="cloud_epitope_self_train_btn",
            model_id_state_key="cloud_epitope_self_model_id",
        )

    if st.button("开始自训练", type="primary", key="epitope_st_btn"):
        labeled2 = labeled.copy()
        if drop_target_na:
            labeled2 = labeled2[labeled2[target_col].notna()].copy()
        if drop_target_zero:
            labeled2 = labeled2[pd.to_numeric(labeled2[target_col], errors="coerce") != 0].copy()
        if len(labeled2) < int(min_labeled):
            st.error(f"标注样本太少：{len(labeled2)} < min_labeled={int(min_labeled)}")
            return

        env_cols2 = infer_env_cols(labeled2, sequence_col=sequence_col, target_col=target_col, env_cols=list(env_cols))
        x_l, y_l, env_medians, _feature_names = make_xy(
            labeled2,
            sequence_col=sequence_col,
            target_col=target_col,
            env_cols=list(env_cols2),
            featurizer=SequenceFeatures(version=int(featurizer_version)),
            env_medians=None,
        )

        x_u = _make_x_only_epitope(
            unlabeled,
            sequence_col=sequence_col,
            env_cols=list(env_cols2),
            env_medians=env_medians,
            featurizer_version=int(featurizer_version),
        )

        rng = np.random.default_rng(int(seed))
        preds = []
        models_list = []
        teacher_preds = None
        for i in range(int(n_models)):
            idx = rng.integers(low=0, high=len(x_l), size=len(x_l), endpoint=False)
            model = build_model(
                model_name=str(model_name),
                random_state=int(seed) + i,  # type: ignore[arg-type]
                mlp_alpha=float(mlp_alpha),
                mlp_early_stopping=bool(mlp_early),
                mlp_patience=int(mlp_patience),
                sgd_alpha=float(sgd_alpha),
                sgd_l1_ratio=float(sgd_l1_ratio),
                sgd_early_stopping=bool(sgd_early),
                hgb_l2=float(hgb_l2),
            )
            model.fit(x_l[idx], y_l[idx])
            models_list.append(model)
            pred_i = np.asarray(model.predict(x_u), dtype=np.float32).reshape(-1)
            preds.append(pred_i)
            # 更新预测级 EMA 教师（如果启用）——对所有模型通用（无需访问参数）
            if bool(use_ema):
                if teacher_preds is None:
                    teacher_preds = pred_i.copy()
                else:
                    decay = float(ema_decay)
                    teacher_preds = decay * teacher_preds + (1.0 - decay) * pred_i

        pred_mat = np.stack(preds, axis=0)
        # 最终伪标签均值：若启用 EMA 教师则使用 EMA 结果，否则使用简单均值
        if bool(use_ema) and teacher_preds is not None:
            mu = teacher_preds
        else:
            mu = pred_mat.mean(axis=0)
        sigma = pred_mat.std(axis=0)

        # 选择候选索引：支持固定比例、自适应分位或绝对 sigma 阈值
        if selection_mode.startswith("固定"):
            k = max(1, int(round(len(unlabeled) * float(keep_frac))))
            keep_idx = np.argsort(sigma)[:k]
        elif selection_mode.startswith("自适应"):
            thresh = float(np.percentile(sigma, float(adaptive_percentile)))
            keep_idx = np.where(sigma <= thresh)[0]
            if len(keep_idx) == 0:
                keep_idx = np.argsort(sigma)[:1]
        else:
            # sigma 阈值模式
            keep_idx = np.where(sigma <= float(sigma_thresh))[0]
            if len(keep_idx) == 0:
                keep_idx = np.argsort(sigma)[:1]

        # 一致性增强：在环境参数上做小扰动，要求伪标签对扰动稳定
        if use_consistency and len(env_cols2) > 0:
            n_unl = len(unlabeled)
            env_meds_arr = np.array([env_medians.get(c, 0.0) for c in env_cols2], dtype=float)
            aug_means = []
            for a in range(int(n_augs)):
                x_u_aug = x_u.copy()
                noise = rng.normal(loc=0.0, scale=float(noise_scale) * (np.abs(env_meds_arr) + 1.0), size=(n_unl, len(env_cols2)))
                if x_u_aug.shape[1] >= len(env_cols2):
                    x_u_aug[:, -len(env_cols2) :] += noise
                # per-augmentation ensemble mean across stored models
                try:
                    aug_preds = np.stack([m.predict(x_u_aug) for m in models_list], axis=0)
                    aug_means.append(aug_preds.mean(axis=0))
                except Exception:
                    # if prediction fails for some models, skip this augmentation
                    pass
            if aug_means:
                aug_stack = np.stack(aug_means, axis=0)
                consistency_std = aug_stack.std(axis=0)
                # filter keep_idx by consistency tolerance
                cons_ok = np.where(consistency_std <= float(consistency_tol))[0]
                # intersection
                keep_idx = np.array([i for i in keep_idx if i in cons_ok], dtype=int)
                if len(keep_idx) == 0:
                    # fallback: choose top by sigma
                    keep_idx = np.argsort(sigma)[: max(1, int(round(len(unlabeled) * float(keep_frac))))]
        pseudo = unlabeled.iloc[keep_idx].copy()
        # soft vs hard labels: for regression we use the ensemble mean as the pseudo target
        if bool(use_soft_labels):
            pseudo[target_col] = mu[keep_idx]
        else:
            # hard label fallback: round to nearest (keeps as float though)
            pseudo[target_col] = mu[keep_idx]
        pseudo["pseudo_uncertainty_std"] = sigma[keep_idx]
        pseudo["pseudo_labeled"] = True
        # assign sample weights: labeled samples weight=1.0, pseudo samples weight=pseudo_weight
        labeled2["pseudo_labeled"] = False
        labeled2["sample_weight"] = 1.0
        pseudo["sample_weight"] = float(pseudo_weight)

        combined = pd.concat([labeled2, pseudo], axis=0, ignore_index=True)

        # 保存伪标签日志（可选）
        if bool(save_pseudo_log):
            try:
                plog = pseudo.copy()
                plog["_mu"] = mu[keep_idx]
                plog["_sigma"] = sigma[keep_idx]
                plog["_sample_weight"] = plog.get("sample_weight", float(pseudo_weight))
                ppath = Path(pseudo_log_path)
                ppath.parent.mkdir(parents=True, exist_ok=True)
                if ppath.exists():
                    plog.to_csv(ppath, mode="a", header=False, index=False)
                else:
                    plog.to_csv(ppath, index=False)
                st.write("已保存伪样本日志:", str(ppath))
            except Exception as e:
                st.warning(f"保存伪样本日志失败: {e}")

        with st.spinner("训练中..."):
            bundle, metrics = train_bundle(
                combined,
                sequence_col=sequence_col,
                target_col=target_col,
                env_cols=list(env_cols2),
                model_name=str(model_name),  # type: ignore[arg-type]
                test_size=float(test_size),
                random_state=int(seed),
                featurizer_version=int(featurizer_version),
                sample_weight_col="sample_weight",
                mlp_alpha=float(mlp_alpha),
                mlp_early_stopping=bool(mlp_early),
                mlp_patience=int(mlp_patience),
                sgd_alpha=float(sgd_alpha),
                sgd_l1_ratio=float(sgd_l1_ratio),
                sgd_early_stopping=bool(sgd_early),
                hgb_l2=float(hgb_l2),
            )

        out_path = Path(model_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, out_path)

        if data_out:
            dp = Path(data_out)
            dp.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(dp, index=False)
            st.write("已保存合并数据:", str(dp))

        st.success("自训练完成")
        render_training_result_panel(metrics, feature_names=metrics.get("feature_names", None))
        st.write(f"伪标签样本数: {len(pseudo)}")

        buf = io.BytesIO()
        joblib.dump(bundle, buf)
        st.download_button(
            "下载模型文件",
            data=buf.getvalue(),
            file_name=out_path.name,
            mime="application/octet-stream",
        )


def epitope_plot_ui() -> None:
    st.subheader("绘图（回归诊断）")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_model = st.file_uploader("上传模型 .joblib", type=["joblib"], key="epitope_plot_model")
    with col2:
        local_models = _list_local_models()
        local_path = st.selectbox("或选择本地 models/*.joblib", options=[""] + local_models, key="epitope_plot_local")

    data_up = st.file_uploader(
        "上传评估数据（支持多文件）",
        type=_UPLOAD_TYPES,
        accept_multiple_files=True,
        key="epitope_plot_data",
    )
    if not data_up:
        return

    if uploaded_model is None and not local_path:
        st.error("请上传模型或选择本地模型")
        return

    bundle: EpitopeModelBundle = _load_bundle(uploaded_model, local_path if local_path else None)  # type: ignore[assignment]
    try:
        df_raw = _load_tables_from_uploads(data_up)
    except Exception as e:
        st.error(f"读取文件失败：{e}")
        return

    plot_cfg = _render_preprocess_panel(df_raw, "epitope_plot")
    df = _apply_preprocess(df_raw, **plot_cfg)

    if bundle.sequence_col not in df.columns:
        st.error(f"缺少 sequence 列: {bundle.sequence_col}")
        return
    if bundle.target_col not in df.columns:
        st.error(f"缺少 target 列: {bundle.target_col}")
        return

    out_dir = Path(st.text_input("输出目录", value="plots/epitope", key="epitope_plot_out_dir"))
    prefix = st.text_input("文件前缀", value="epitope", key="epitope_plot_prefix")
    title = st.text_input("标题（可选）", value="epitope regression", key="epitope_plot_title")

    with st.expander("云端绘图", expanded=False):
        model_payload = _resolve_model_payload(uploaded_model, local_path if local_path else None, content_type="application/octet-stream")
        if model_payload is None:
            st.info("请上传模型或选择本地模型以提交云端绘图。")
        else:
            payload = {
                "model": model_payload,
                "data": _cloud_encode_dataframe(df),
                "out_dir": str(out_dir),
                "prefix": str(prefix),
                "title": str(title),
            }
            _cloud_submit_section(
                "epitope_plot",
                payload,
                button_label="提交云端绘图",
                key="cloud_epitope_plot_btn",
            )

    if st.button("生成诊断图", type="primary", key="epitope_plot_btn"):
        x, y, _, _ = make_xy(
            df,
            sequence_col=bundle.sequence_col,
            target_col=bundle.target_col,
            env_cols=list(bundle.env_cols),
            featurizer=SequenceFeatures(version=int(getattr(bundle, "featurizer_version", 1) or 1)),
            env_medians=dict(bundle.env_medians),
        )
        y_pred = np.asarray(bundle.model.predict(x), dtype=float).reshape(-1)

        save_regression_diagnostic_plots(
            y_true=y,
            y_pred=y_pred,
            out_dir=out_dir,
            prefix=str(prefix),
            title=str(title),
        )

        st.success("已生成诊断图")
        _render_plot_images(out_dir=out_dir, prefix=str(prefix))


def epitope_predict_ui() -> None:
    st.subheader("单条预测（模型 + 序列 + 实验条件）")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_model = st.file_uploader("上传模型 .joblib", type=["joblib"], key="pred_model")
    with col2:
        local_models = _list_local_models()
        local_path = st.selectbox("或选择本地 models/*.joblib", options=[""] + local_models)

    sequence = st.text_input("表位序列", value="SIINFEKL")
    kv_text = st.text_area(
        "实验参数（每行 key=value，可留空用训练集中位数填充）",
        value="concentration=10\ncell_density=1000000\n",
        height=120,
    )

    with st.expander("云端预测", expanded=False):
        try:
            env_cloud = _parse_kv_lines(kv_text)
        except ValueError as e:
            st.error(str(e))
            env_cloud = {}

        model_payload = _resolve_model_payload(uploaded_model, local_path if local_path else None, content_type="application/octet-stream")
        if model_payload is None:
            st.info("请上传模型或选择本地模型以提交云端预测。")
        else:
            payload = {
                "model": model_payload,
                "sequence": str(sequence),
                "env_params": {k: float(v) for k, v in env_cloud.items()},
            }
            _cloud_submit_section(
                "epitope_predict",
                payload,
                button_label="提交云端预测",
                key="cloud_epitope_pred_btn",
            )

    if st.button("预测", type="primary"):
        bundle = _load_bundle(uploaded_model, local_path if local_path else None)
        try:
            env = _parse_kv_lines(kv_text)
        except ValueError as e:
            st.error(str(e))
            return

        y = predict_one(bundle, sequence=sequence, env_params=env)
        st.metric("预测值", value=f"{y:.6g}")

        if bundle.env_cols:
            resolved = {c: float(env.get(c, bundle.env_medians.get(c, 0.0))) for c in bundle.env_cols}
            st.write("实际使用的 env:", resolved)

        # 差分进化建议环境
        if bundle.env_cols:
            with st.expander("差分进化建议环境（优化数值型 env）", expanded=False):
                st.markdown("为每个环境变量指定搜索区间，点击下方按钮开始差分进化搜索。")
                env_bounds = []
                cols = bundle.env_cols
                cols_vals = {}
                for c in cols:
                    med = float(bundle.env_medians.get(c, 0.0))
                    if med == 0.0:
                        lo_def = -1.0
                        hi_def = 1.0
                    else:
                        lo_def = med * 0.5
                        hi_def = med * 1.5
                    col1, col2 = st.columns(2)
                    with col1:
                        lo = st.number_input(f"{c} 下界", value=float(lo_def), key=f"de_{c}_lo")
                    with col2:
                        hi = st.number_input(f"{c} 上界", value=float(hi_def), key=f"de_{c}_hi")
                    env_bounds.append((float(lo), float(hi)))
                    cols_vals[c] = (float(lo), float(hi))

                run_de = st.button("运行差分进化建议环境", key="run_de_epitope")
                if run_de:
                    try:
                        from src.epitope.predictor import suggest_env_by_de_epitope

                        best_env, best_val = suggest_env_by_de_epitope(bundle, sequence=sequence, env_bounds=env_bounds)
                        mapped = {c: float(v) for c, v in zip(cols, best_env.tolist())}
                        st.success(f"优化完成，预测值={best_val:.6g}")
                        st.write("建议环境:", mapped)
                    except Exception as e:
                        st.error(f"差分进化失败：{e}")

        # --- Multiscale integration: optional small-molecule analysis ---
        st.markdown("---")
        st.subheader("关联小分子多尺度分析（可选）")
        smiles_input = st.text_input("小分子 SMILES（可选）", value="", key="epitope_ms_smiles")
        ms_steps = st.number_input("GNN 步数", min_value=1, max_value=8, value=3, key="epitope_ms_steps")
        ms_hidden = st.number_input("GNN 隐藏维度", min_value=8, max_value=256, value=64, step=8, key="epitope_ms_hidden")
        ms_dropout = st.slider("GNN Dropout", min_value=0.0, max_value=0.5, value=0.1, step=0.05, key="epitope_ms_dropout")
        run_ms = st.button("运行多尺度分析（与该表位关联）")
        if run_ms:
            if not smiles_input:
                st.error("请先填写小分子 SMILES")
            else:
                cloud_cfg = _get_cloud_cfg()
                if cloud_cfg.get("enabled"):
                    payload = {"smiles": smiles_input, "steps": int(ms_steps), "hidden": int(ms_hidden), "context": {"sequence": sequence, "pred": float(y)}}
                    with st.spinner("提交云端多尺度任务..."):
                        res = _submit_cloud_job("multiscale", payload, cloud_cfg)
                    if not res.get("ok"):
                        st.error(f"云端计算失败: {res.get('text')} (HTTP {res.get('status')})")
                    else:
                        j = res.get("json") or {}
                        scores = j.get("scores")
                        pinn_losses = j.get("pinn_losses")
                        svg_b64 = j.get("svg_b64") or j.get("png_b64")
                        if scores:
                            try:
                                import pandas as _pd

                                df = _pd.DataFrame([{"atom_index": k, "sensitivity": float(v)} for k, v in scores.items()])
                                st.dataframe(df.sort_values("sensitivity", ascending=False))
                            except Exception:
                                st.write(scores)
                        if pinn_losses is not None:
                            st.write("PINN 训练损失（云端）:", pinn_losses)
                        if svg_b64:
                            import base64 as _b64

                            try:
                                st.image(_b64.b64decode(svg_b64))
                            except Exception:
                                st.write("收到云端图片，但无法显示")
                else:
                    # local execution
                    try:
                        X, A, mol = mol_to_graph(smiles_input)
                    except Exception as e:
                        st.error(f"SMILES 解析失败: {e}")
                        return
                    in_dim = X.shape[1]
                    gnn = SimpleGNN(in_dim, hidden_dim=ms_hidden, steps=ms_steps, dropout=float(ms_dropout))
                    model_fn, probe_module = example_model_fn_factory(ms_hidden)
                    try:
                        scores = sensitivity_masking(smiles_input, gnn, model_fn)
                    except Exception as e:
                        st.error(f"敏感性计算失败: {e}")
                        return
                    try:
                        import pandas as _pd

                        df = _pd.DataFrame([{"atom_index": k, "sensitivity": float(v)} for k, v in scores.items()])
                        st.dataframe(df.sort_values("sensitivity", ascending=False))
                    except Exception:
                        st.write(scores)
                    # build small demo PINN and run short loop
                    msm = MultiScaleModel(gnn)
                    mol_emb = msm.encode_molecule(smiles_input).detach()
                    msm.build_pinn(spatial_dim=1, mol_emb_dim=mol_emb.shape[0], hidden=64, dropout=0.1)
                    optimizer = torch.optim.Adam(msm.pinn.parameters(), lr=1e-3)
                    collocation = torch.rand((64, 2))
                    losses = []
                    for epoch in range(8):
                        loss = msm.pinn_step(optimizer, collocation, mol_emb, 0.1, 0.5, 0.1)
                        losses.append(loss)
                    st.write("PINN 训练损失（本地示例）:", losses)


def epitope_screen_ui() -> None:
    st.subheader("批量虚拟筛选（候选 CSV → 预测 CSV）")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_model = st.file_uploader("上传模型 .joblib", type=["joblib"], key="screen_model")
    with col2:
        local_models = _list_local_models()
        local_path = st.selectbox("或选择本地 models/*.joblib", options=[""] + local_models, key="screen_local")

    uploaded_candidates = st.file_uploader(
        "上传候选（支持多文件）",
        type=_UPLOAD_TYPES,
        accept_multiple_files=True,
        key="candidates_csv",
    )
    if not uploaded_candidates:
        return

    try:
        candidates_raw = _load_tables_from_uploads(uploaded_candidates)
    except Exception as e:
        st.error(f"读取文件失败：{e}")
        return

    cand_cfg = _render_preprocess_panel(candidates_raw, "epitope_screen")
    candidates = _apply_preprocess(candidates_raw, **cand_cfg)
    _preview_df(candidates, title="候选预览")
    sequence_col = st.selectbox("sequence 列名", options=list(candidates.columns), index=(list(candidates.columns).index("sequence") if "sequence" in candidates.columns else 0))
    out_col = st.text_input("输出预测列名", value="pred")

    # Optional: per-candidate multiscale analysis settings
    smiles_col_options = [""] + list(candidates.columns)
    smiles_col = st.selectbox("（可选）候选表中小分子 SMILES 列名", options=smiles_col_options, index=0)
    multiscale_opt = st.checkbox("对每个候选同时运行多尺度分析（耗时）", value=False)
    if multiscale_opt:
        ms_exec_mode = st.radio("执行位置", options=["local", "cloud"], index=0)
        ms_steps = st.number_input("GNN 步数", min_value=1, max_value=8, value=3, key="batch_ms_steps")
        ms_hidden = st.number_input("GNN 隐藏维度", min_value=8, max_value=256, value=64, step=8, key="batch_ms_hidden")
        ms_dropout = st.slider("GNN Dropout", min_value=0.0, max_value=0.5, value=0.1, step=0.05, key="batch_ms_dropout")
        st.caption("提示：本地模式会在当前机器逐条计算；若候选数较多请使用云端或仅选择子集（建议 <=50）。")

    with st.expander("云端筛选", expanded=False):
        model_payload = _resolve_model_payload(uploaded_model, local_path if local_path else None, content_type="application/octet-stream")
        if model_payload is None:
            st.info("请上传模型或选择本地模型以提交云端筛选。")
        else:
            payload = {
                "model": model_payload,
                "candidates": _cloud_encode_dataframe(candidates),
                "sequence_col": str(sequence_col),
                "out_col": str(out_col),
            }
            _cloud_submit_section(
                "epitope_screen",
                payload,
                button_label="提交云端筛选",
                key="cloud_epitope_screen_btn",
                download_name="epitope_predictions_cloud.csv",
            )

    if st.button("开始筛选", type="primary"):
        bundle = _load_bundle(uploaded_model, local_path if local_path else None)

        df = candidates.copy()
        if sequence_col not in df.columns:
            st.error(f"候选 CSV 缺少列: {sequence_col}")
            return

        for c in bundle.env_cols:
            if c not in df.columns:
                df[c] = np.nan
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(bundle.env_medians.get(c, 0.0))

        with st.spinner("预测中..."):
            seqs = df[sequence_col].astype(str).tolist()
            featurizer = _cached_sequence_featurizer(int(getattr(bundle, "featurizer_version", 1)))
            seq_x = featurizer.transform_many(seqs)
            env_x = df[bundle.env_cols].to_numpy(dtype=np.float32) if bundle.env_cols else np.zeros((len(df), 0), dtype=np.float32)
            x = np.concatenate([seq_x, env_x], axis=1)

            preds = np.empty((x.shape[0],), dtype=np.float32)
            chunk = 10000
            for start in range(0, x.shape[0], chunk):
                end = min(x.shape[0], start + chunk)
                preds[start:end] = np.asarray(bundle.model.predict(x[start:end]), dtype=np.float32).reshape(-1)

        df[out_col] = np.array(preds, dtype=float)
        st.success("筛选完成")
        _preview_df(df, title="结果预览", max_rows=50)

        # 如果启用了多尺度批量分析选项，执行额外处理（依赖 smiles_col）
        if multiscale_opt:
            if not smiles_col:
                st.error("若启用多尺度分析，请在上方选择包含 SMILES 的列名或在候选表中提供 SMILES 列。")
            else:
                n = len(df)
                if ms_exec_mode == "local" and n > 200:
                    st.warning("候选数较多，局部逐条计算可能非常耗时（>200）。请考虑云端模式或筛选子集。")

                progress = st.progress(0)
                ms_top_atoms = []
                ms_top_vals = []
                ms_pinn_losses = []

                gnn_proto = None
                model_fn = None

                for idx, row in df.iterrows():
                    smiles_val = str(row.get(smiles_col, "") or "")
                    if not smiles_val:
                        ms_top_atoms.append(None)
                        ms_top_vals.append(None)
                        ms_pinn_losses.append(None)
                        progress.progress(int((idx + 1) / n * 100))
                        continue

                    if ms_exec_mode == "cloud":
                        try:
                            cloud_cfg = _get_cloud_cfg()
                            payload = {"smiles": smiles_val, "steps": int(ms_steps), "hidden": int(ms_hidden), "context": {"sequence": str(row[sequence_col])}}
                            res = _submit_cloud_job("multiscale", payload, cloud_cfg)
                            if res.get("ok"):
                                j = res.get("json") or {}
                                scores = j.get("scores") or {}
                                pinn_losses = j.get("pinn_losses")
                                if scores:
                                    top_atom = max(scores.items(), key=lambda kv: kv[1])[0]
                                    top_val = float(scores[top_atom])
                                else:
                                    top_atom = None
                                    top_val = None
                                ms_top_atoms.append(top_atom)
                                ms_top_vals.append(top_val)
                                ms_pinn_losses.append(pinn_losses)
                            else:
                                ms_top_atoms.append(None)
                                ms_top_vals.append(None)
                                ms_pinn_losses.append(None)
                        except Exception:
                            ms_top_atoms.append(None)
                            ms_top_vals.append(None)
                            ms_pinn_losses.append(None)
                        progress.progress(int((idx + 1) / n * 100))
                        continue

                    # local execution path
                    try:
                        X, A, mol = mol_to_graph(smiles_val)
                    except Exception:
                        ms_top_atoms.append(None)
                        ms_top_vals.append(None)
                        ms_pinn_losses.append(None)
                        progress.progress(int((idx + 1) / n * 100))
                        continue

                    if gnn_proto is None:
                        in_dim = X.shape[1]
                        gnn_proto = SimpleGNN(in_dim, hidden_dim=ms_hidden, steps=ms_steps, dropout=float(ms_dropout))
                        model_fn, _ = example_model_fn_factory(ms_hidden)

                    try:
                        scores = sensitivity_masking(smiles_val, gnn_proto, model_fn)
                        if scores:
                            top_atom = max(scores.items(), key=lambda kv: kv[1])[0]
                            top_val = float(scores[top_atom])
                        else:
                            top_atom = None
                            top_val = None

                        msm = MultiScaleModel(gnn_proto)
                        mol_emb = msm.encode_molecule(smiles_val).detach()
                        msm.build_pinn(spatial_dim=1, mol_emb_dim=mol_emb.shape[0], hidden=64, dropout=0.1)
                        collocation = torch.rand((32, 2))
                        try:
                            pinn_l = pinn_loss(msm.pinn, collocation, mol_emb, 0.1, 0.5, 0.1).item()
                        except Exception:
                            pinn_l = None

                        ms_top_atoms.append(top_atom)
                        ms_top_vals.append(top_val)
                        ms_pinn_losses.append(pinn_l)
                    except Exception:
                        ms_top_atoms.append(None)
                        ms_top_vals.append(None)
                        ms_pinn_losses.append(None)

                    progress.progress(int((idx + 1) / n * 100))

                df["ms_top_atom"] = ms_top_atoms
                df["ms_top_sensitivity"] = ms_top_vals
                df["ms_pinn_loss"] = ms_pinn_losses

                st.success("多尺度（批量）分析完成")
                _preview_df(df, title="带多尺度结果的预测预览", max_rows=50)

        out_csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("下载预测结果 CSV", data=out_csv, file_name="predictions.csv", mime="text/csv")


def epitope_sensitivity_ui() -> None:
    st.subheader("参数敏感性分析（局部梯度 / 数值梯度）")
    st.caption("对单个输入点估计 d(pred)/d(feature)，用于解释“当前这个表位+条件附近”哪些特征最敏感。")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_model = st.file_uploader("上传模型 .joblib", type=["joblib"], key="sens_model")
    with col2:
        local_models = _list_local_models()
        local_path = st.selectbox("或选择本地 models/*.joblib", options=[""] + local_models, key="sens_local")

    sequence = st.text_input("表位序列", value="SIINFEKL", key="sens_seq")
    kv_text = st.text_area(
        "实验参数（每行 key=value，可留空用训练集中位数填充）",
        value="concentration=10\ncell_density=1000000\n",
        height=120,
        key="sens_kv",
    )

    eps = st.number_input("数值梯度步长 eps", min_value=1e-6, max_value=1e-1, value=1e-3, step=1e-3, format="%.6f")
    topk = st.slider("显示 Top-K 特征", min_value=5, max_value=50, value=15, step=5)
    importance_mode = st.selectbox("重要性定义", options=["abs_grad", "grad_x"], index=0)

    with st.expander("云端敏感性分析", expanded=False):
        try:
            env_cloud = _parse_kv_lines(kv_text)
        except ValueError as e:
            st.error(str(e))
            env_cloud = {}

        model_payload = _resolve_model_payload(uploaded_model, local_path if local_path else None, content_type="application/octet-stream")
        if model_payload is None:
            st.info("请上传模型或选择本地模型以提交云端敏感性分析。")
        else:
            payload = {
                "model": model_payload,
                "sequence": str(sequence),
                "env_params": {k: float(v) for k, v in env_cloud.items()},
                "eps": float(eps),
                "top_k": int(topk),
                "importance": str(importance_mode),
            }
            _cloud_submit_section(
                "epitope_sensitivity",
                payload,
                button_label="提交云端敏感性分析",
                key="cloud_epitope_sensitivity_btn",
            )

    if st.button("开始分析", type="primary"):
        bundle = _load_bundle(uploaded_model, local_path if local_path else None)
        try:
            env = _parse_kv_lines(kv_text)
        except ValueError as e:
            st.error(str(e))
            return

        # Build input vector matching training.
        x = _bundle_input_vector(bundle, sequence=sequence, env=env)
        if hasattr(bundle, "feature_names") and bundle.feature_names and len(bundle.feature_names) != len(x):
            st.error(
                f"模型特征维度不匹配：bundle.feature_names={len(bundle.feature_names)}，但当前输入 x={len(x)}。\n"
                "常见原因：模型是旧版本/来自不同代码版本。建议用当前代码重新训练并导出模型。"
            )
            return

        # Prediction
        y = predict_one(bundle, sequence=sequence, env_params=env)
        st.metric("预测值", value=f"{y:.6g}")

        # Suggestion options UI
        with st.expander("建议参数（可选）", expanded=False):
            opt_mut = st.checkbox("包含突变/替换提示 (mutation_hints)", value=True)
            opt_exp = st.checkbox("包含实验设计建议 (experimental_design)", value=True)
            opt_val = st.checkbox("包含验证/统计建议 (validation)", value=True)
            opt_model = st.checkbox("包含模型稳健性检查 (model_checks)", value=True)
            opt_assay = st.checkbox("包含测定/assay 建议 (assay_suggestions)", value=True)

        suggestion_options = {
            "mutation_hints": bool(opt_mut),
            "experimental_design": bool(opt_exp),
            "validation": bool(opt_val),
            "model_checks": bool(opt_model),
            "assay_suggestions": bool(opt_assay),
        }

        with st.spinner("计算敏感性中..."):
            result = sensitivity_from_bundle(bundle, x=x, eps=float(eps), importance=str(importance_mode))

        # Structured report and human-readable summary
        report = sensitivity_report(result, top_k=int(topk), suggestion_options=suggestion_options)

        st.write("### 格式化结论与建议（摘要）")
        st.code(format_sensitivity_report(report, top_k=int(topk)), language="text")

        st.write("### 结构化报告（JSON）")
        try:
            st.json(report)
        except Exception:
            st.write(report)

        st.write("### Top 特征（表格）")
        rows = []
        for name, imp, grad in top_features(result, k=int(topk)):
            rows.append({"feature": name, "importance": imp, "grad": grad})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        groups = group_importance(result)
        st.write("### 分组重要性（粗粒度解释）")
        gdf = pd.DataFrame(
            [{"group": k, "importance": float(v)} for k, v in sorted(groups.items(), key=lambda kv: -kv[1])]
        )
        st.dataframe(gdf, use_container_width=True)
        st.bar_chart(gdf.set_index("group")["importance"], height=220)

        st.write("### 给 wet 组的结论")
        # wetlab_takeaway accepts either groups or the full SensitivityResult
        msg = wetlab_takeaway(result)
        st.success(msg) if str(msg).startswith("模型显示") else st.warning(msg)

        st.write("### WET 组智能体建议（实验）")
        with st.expander("智能体 API（实验）", expanded=False):
            st.caption("实验性功能：将当前分析数据发送给智能体 API，生成下一步实验建议。")

            mode = st.selectbox("API 类型", options=["OpenAI 兼容", "Raw JSON"], index=0, key="wet_agent_mode")
            endpoint = st.text_input(
                "API 地址",
                value="https://api.openai.com/v1/chat/completions",
                key="wet_agent_endpoint",
            )
            api_key = st.text_input("API Key", value="", type="password", key="wet_agent_key")
            timeout_s = st.number_input("timeout（秒）", min_value=1.0, value=60.0, step=1.0, key="wet_agent_timeout")

            note = st.text_area("附加说明（可选）", value="", height=80, key="wet_agent_note")

            agent_data = {
                "sequence": sequence,
                "env": env,
                "prediction": float(y),
                "sensitivity_report": report,
                "top_features": rows,
                "group_importance": groups,
                "wetlab_takeaway": msg,
                "note": str(note),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }

            with st.expander("发送数据预览", expanded=False):
                st.json(agent_data)

            if mode == "OpenAI 兼容":
                model = st.text_input("model", value="gpt-4o-mini", key="wet_agent_model")
                temperature = st.slider("temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
                max_tokens = st.number_input("max_tokens", min_value=64, max_value=4096, value=512, step=64)
                system_prompt = st.text_area(
                    "system prompt",
                    value=(
                        "你是实验设计助手。请基于输入数据，给出下一步湿实验建议：\n"
                        "1) 明确优先级最高的3个实验\n"
                        "2) 给出每个实验的关键变量与对照\n"
                        "3) 指出潜在风险与需要补充的数据\n"
                        "输出中文，分点列出。"
                    ),
                    height=120,
                    key="wet_agent_sys",
                )

                user_prompt = json.dumps(agent_data, ensure_ascii=False, indent=2)

                if st.button("调用智能体生成建议", type="primary", key="wet_agent_run_openai"):
                    with st.spinner("智能体生成中..."):
                        result_ai = call_openai_chat(
                            endpoint=str(endpoint),
                            api_key=str(api_key) if api_key else None,
                            model=str(model),
                            system_prompt=str(system_prompt),
                            user_prompt=user_prompt,
                            temperature=float(temperature),
                            max_tokens=int(max_tokens),
                            timeout=float(timeout_s),
                        )

                    if result_ai.ok and result_ai.content:
                        st.success("智能体建议已生成")
                        st.write(result_ai.content)
                        with st.expander("原始响应", expanded=False):
                            st.json(result_ai.raw)
                    else:
                        st.error(f"调用失败：{result_ai.error}")
                        if result_ai.raw:
                            st.write(result_ai.raw)

            else:
                extra_headers_text = st.text_area(
                    "额外 Headers（JSON，可选）",
                    value="{}",
                    height=80,
                    key="wet_agent_headers",
                )
                extra_payload_text = st.text_area(
                    "额外 Payload（JSON，可选，将与 data 合并）",
                    value="{}",
                    height=100,
                    key="wet_agent_payload",
                )

                extra_headers = safe_parse_json(extra_headers_text) or {}
                extra_payload = safe_parse_json(extra_payload_text) or {}
                payload = {"data": agent_data}
                payload.update(extra_payload)

                if st.button("调用智能体生成建议", type="primary", key="wet_agent_run_raw"):
                    with st.spinner("智能体生成中..."):
                        result_ai = call_raw_json(
                            endpoint=str(endpoint),
                            api_key=str(api_key) if api_key else None,
                            payload=payload,
                            timeout=float(timeout_s),
                            headers=extra_headers,
                        )

                    if result_ai.ok:
                        st.success("智能体建议已生成")
                        if result_ai.content:
                            st.write(result_ai.content)
                        with st.expander("原始响应", expanded=False):
                            st.json(result_ai.raw)
                    else:
                        st.error(f"调用失败：{result_ai.error}")
                        if result_ai.raw:
                            st.write(result_ai.raw)

        feat_v = int(getattr(bundle, "featurizer_version", 1) or 1)
        if feat_v < 2:
            st.info("提示：当前模型 featurizer_version=1，不含‘中部/两端’特征；建议用新版特征重新训练模型。")


def _list_local_models_drug() -> List[str]:
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    items = [str(p) for p in models_dir.glob("drug_*.joblib")]
    pretrained_dir = models_dir / "pretrained"
    if pretrained_dir.exists():
        items.extend([str(p) for p in pretrained_dir.glob("drug_*.joblib")])
        items.extend([str(p) for p in pretrained_dir.glob("drug_pretrained.joblib")])
    return sorted(items)


def _list_local_models_drug_torch() -> List[str]:
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    items = [str(p) for p in models_dir.glob("drug_torch*.pt")]
    pretrained_dir = models_dir / "pretrained"
    if pretrained_dir.exists():
        # Include any .pt in pretrained (not only names starting with 'drug_torch')
        items.extend([str(p) for p in pretrained_dir.glob("*.pt")])
    return sorted(items)


def _list_local_models_drug_transformer() -> List[str]:
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    items = [str(p) for p in models_dir.glob("drug_transformer*.pt")]
    pretrained_dir = models_dir / "pretrained"
    if pretrained_dir.exists():
        items.extend([str(p) for p in pretrained_dir.glob("*transformer*.pt")])
    return sorted(items)


def _list_local_models_docking() -> List[str]:
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    items = [str(p) for p in models_dir.glob("docking_*.pt")]
    items.extend([str(p) for p in models_dir.glob("drug_docking*.pt")])
    pretrained_dir = models_dir / "pretrained"
    if pretrained_dir.exists():
        items.extend([str(p) for p in pretrained_dir.glob("*docking*.pt")])
    return sorted(set(items))


def _parse_hidden_sizes(text: str) -> List[int]:
    raw = [s.strip() for s in str(text).split(",") if s.strip()]
    if not raw:
        return [512, 256]
    sizes: List[int] = []
    for s in raw:
        v = int(s)
        if v <= 0:
            raise ValueError("hidden_sizes 必须是正整数")
        sizes.append(v)
    return sizes


def render_training_result_panel(metrics: Dict[str, Any], feature_names: Optional[List[str]] = None) -> None:
    st.markdown("### 训练指标详解")
    cols = st.columns(5)
    cols[0].metric("R2", f"{metrics.get('r2', 0):.4f}")
    cols[1].metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
    cols[2].metric("MAE", f"{metrics.get('mae', 0):.4f}")
    if "explained_variance" in metrics:
        cols[3].metric("Exp. Var", f"{metrics.get('explained_variance', 0):.4f}")
    if "max_error" in metrics:
        cols[4].metric("Max Error", f"{metrics.get('max_error', 0):.4f}")

    col_cnt, col_feat = st.columns(2)
    with col_cnt:
        st.write(f"训练集: {metrics.get('n_train', 0)} | 验证集: {metrics.get('n_val', 0)}")
    with col_feat:
        st.write(f"特征数: {metrics.get('n_features', 0)}")

    tabs = st.tabs(["训练曲线", "残差分析", "特征重要性", "建议"])
    
    with tabs[0]:
        hist = metrics.get("history")
        if hist and isinstance(hist, dict) and (hist.get("train_loss") or hist.get("val_loss")):
            loss_df = pd.DataFrame({
                "epoch": range(1, len(hist.get("train_loss", [])) + 1),
                "train_loss": hist.get("train_loss", []),
            })
            if hist.get("val_loss"):
                loss_df["val_loss"] = hist["val_loss"]
            st.line_chart(loss_df.set_index("epoch"))
        else:
            st.info("当前模型未提供训练过程曲线（非迭代模型或未记录）。")

    with tabs[1]:
        y_val = metrics.get("y_val") or metrics.get("y_true")
        y_pred = metrics.get("y_pred")
        if y_val and y_pred and len(y_val) == len(y_pred):
            import matplotlib.pyplot as plt
            if len(y_val) > 2000:
                st.caption("验证集较大，仅采样前 2000 个点绘图。")
                y_val = y_val[:2000]
                y_pred = y_pred[:2000]

            y_v = np.array(y_val)
            y_p = np.array(y_pred)
            resid = y_p - y_v

            c1, c2 = st.columns(2)
            with c1:
                fig1, ax1 = plt.subplots(figsize=(5, 4))
                ax1.scatter(y_v, y_p, alpha=0.5, s=10)
                mmin = min(y_v.min(), y_p.min())
                mmax = max(y_v.max(), y_p.max())
                ax1.plot([mmin, mmax], [mmin, mmax], "r--")
                ax1.set_xlabel("True")
                ax1.set_ylabel("Pred")
                ax1.set_title("True vs Pred")
                st.pyplot(fig1)

            with c2:
                fig2, ax2 = plt.subplots(figsize=(5, 4))
                ax2.hist(resid, bins=30, alpha=0.7)
                ax2.set_xlabel("Residual (Pred - True)")
                ax2.set_title("Residual Distribution")
                st.pyplot(fig2)
        else:
            st.info("未提供验证集详细预测数据，无法绘制残差。")

    with tabs[2]:
        imps = metrics.get("feature_importances")
        if imps and feature_names and len(imps) == len(feature_names):
            idf = pd.DataFrame({"feature": feature_names, "importance": imps})
            idf = idf.sort_values("importance", ascending=False).head(20)
            st.bar_chart(idf.set_index("feature"))
        elif imps:
            # If feature names mismatch or not provided
            st.bar_chart(imps[:50])
            st.caption("显示前 50 个特征的重要性（无名称）")
        else:
            st.info("当前模型不支持特征重要性提取。")

    with tabs[3]:
        if metrics.get("suggestions"):
            st.write(metrics.get("suggestions"))
        else:
            st.write("暂无建议。")


def drug_train_ui() -> None:
    st.subheader("训练（CSV → 药物疗效模型）")
    st.caption("训练集需要至少包含 smiles 列 + 1 个目标列（例如 efficacy）。可选包含剂量/频次等数值列作为条件输入。")

    try:
        import rdkit  # type: ignore  # noqa: F401
        from src.drug.predictor import infer_env_cols, train_bundle  # type: ignore
        from src.drug.training_eval import evaluate_drug_from_csv, train_drug_from_csv  # type: ignore
    except Exception as e:
        st.warning(f"药物模块不可用（可能未安装 rdkit）：{e}")
        return

    uploaded = st.file_uploader(
        "上传训练集（支持多文件）",
        type=_UPLOAD_TYPES,
        accept_multiple_files=True,
        key="drug_train_csv",
    )
    if not uploaded:
        st.info("请上传包含 smiles 和目标列的 CSV")
        return

    try:
        df_raw = _load_tables_from_uploads(uploaded)
    except Exception as e:
        st.error(f"读取文件失败：{e}")
        return

    cfg = _render_preprocess_panel(df_raw, "drug_train")
    df = _apply_preprocess(df_raw, **cfg)
    _preview_df(df, title="数据预览")

    smiles_col = st.selectbox(
        "smiles 列名",
        options=list(df.columns),
        index=(list(df.columns).index("smiles") if "smiles" in df.columns else 0),
        key="drug_smiles_col",
    )
    target_candidates = [c for c in df.columns if c != smiles_col]
    target_col = st.selectbox("目标列（疗效）", options=target_candidates, index=(target_candidates.index("efficacy") if "efficacy" in target_candidates else 0))

    drop_target_na = st.checkbox("过滤目标列为空", value=True, key="drug_train_drop_target_na")
    drop_target_zero = st.checkbox("将目标列=0 视为无标注", value=False, key="drug_train_drop_target_zero")

    auto_env_cols = infer_env_cols(df, smiles_col=smiles_col, target_col=target_col, env_cols=None)
    env_cols = st.multiselect(
        "条件列（数值，可选）",
        options=[c for c in df.columns if c not in (smiles_col, target_col)],
        default=auto_env_cols,
    )

    drug_algo_map = {
        "Confluencia 1.0 HGB": "hgb",
        "Confluencia 1.0 GBR": "gbr",
        "Confluencia 1.0 RF": "rf",
        "Confluencia 1.0 Ridge": "ridge",
        "Confluencia 1.0 MLP": "mlp",
    }
    drug_algo_label = st.selectbox("训练算法（参考 Confluencia 1.0）", options=list(drug_algo_map.keys()), index=0)
    model_name = drug_algo_map[drug_algo_label]
    featurizer_version = st.selectbox("特征版本", options=[1, 2], index=1)
    with st.expander("训练优化参数", expanded=False):
        mlp_alpha = st.number_input("MLP 正则强度 (alpha)", min_value=1e-8, max_value=1e-2, value=1e-4, format="%.6f", key="drug_train_mlp_alpha")
        mlp_early = st.checkbox("MLP 早停", value=True, key="drug_train_mlp_early")
        mlp_patience = st.number_input("MLP 早停耐心", min_value=1, max_value=200, value=10, step=1, key="drug_train_mlp_pat")
        ridge_alpha = st.number_input("Ridge 正则 (alpha)", min_value=1e-6, max_value=100.0, value=1.0, step=0.5, key="drug_train_ridge_alpha")
        hgb_l2 = st.number_input("HGB L2 正则", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="drug_train_hgb_l2")
    test_size = st.slider("验证集比例", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    seed = st.number_input("随机种子", min_value=0, max_value=10_000_000, value=42, step=1)

    model_out = st.text_input("保存路径", value="models/drug_model.joblib")

    with st.expander("云端训练", expanded=False):
        payload = {
            "data": _cloud_encode_dataframe(df),
            "smiles_col": str(smiles_col),
            "target_col": str(target_col),
            "env_cols": list(env_cols),
            "model_name": str(model_name),
            "featurizer_version": int(featurizer_version),
            "test_size": float(test_size),
            "random_state": int(seed),
            "mlp_alpha": float(mlp_alpha),
            "mlp_early_stopping": bool(mlp_early),
            "mlp_patience": int(mlp_patience),
            "ridge_alpha": float(ridge_alpha),
            "hgb_l2": float(hgb_l2),
            "drop_target_na": bool(drop_target_na),
            "drop_target_zero": bool(drop_target_zero),
            "model_out": str(model_out),
        }
        _cloud_submit_section(
            "drug_train",
            payload,
            button_label="提交云端训练",
            key="cloud_drug_train_btn",
            model_id_state_key="cloud_drug_model_id",
        )

    with st.expander("超参数调参（Grid / Randomized）", expanded=False):
        st.markdown("使用 `param_grids/param_grid_examples.json` 示例或上传自定义 JSON 文件进行调参。")
        strategy = st.selectbox("搜索策略", options=["grid", "random"], index=0, key="drug_tune_strategy")
        n_iter = st.number_input("随机搜索迭代次数 (仅 random 有效)", min_value=1, value=20, step=1, key="drug_tune_niter")
        cv = st.number_input("CV 折数", min_value=2, value=5, step=1, key="drug_tune_cv")
        use_example = st.checkbox("使用示例参数网格（param_grids/param_grid_examples.json）", value=True, key="drug_tune_use_example")
        uploaded_grid = st.file_uploader("或上传 JSON 参数网格", type=["json"], key="drug_param_grid")
        tune_model_out = st.text_input("调参后模型保存路径", value="models/drug_model_tuned.joblib", key="drug_tune_out")
        if st.button("开始调参", key="drug_start_tune"):
            try:
                if use_example and not uploaded_grid:
                    grid_path = _PROJECT_ROOT / "param_grids" / "param_grid_examples.json"
                    with open(grid_path, "r", encoding="utf-8") as f:
                        grid_all = json.load(f)
                else:
                    if not uploaded_grid:
                        st.error("请选择示例或上传参数网格 JSON 文件")
                        raise RuntimeError("no param grid")
                    grid_all = json.load(uploaded_grid)

                grid = grid_all.get(model_name, {}) if isinstance(grid_all, dict) else {}

                x, y, _, _ = make_xy(
                    df,
                    smiles_col=smiles_col,
                    target_col=target_col,
                    env_cols=list(env_cols),
                    featurizer=MoleculeFeatures(version=int(getattr(st.session_state, "drug_train_featurizer_version", 2)), radius=int(radius), n_bits=int(n_bits)),
                    env_medians=None,
                )

                base = build_model(model_name=model_name, random_state=int(seed))

                with st.spinner("调参中，请稍候..."):
                    best_est, best_params, cv_results = run_hyper_search(
                        base,
                        grid,
                        x,
                        y,
                        strategy=strategy,
                        n_iter=int(n_iter),
                        cv=int(cv),
                        scoring=None,
                        n_jobs=-1,
                        random_state=int(seed),
                    )

                out_path = Path(tune_model_out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(best_est, out_path)

                st.success("调参完成")
                st.json(best_params)
                st.write(f"已保存模型: {out_path}")
            except Exception as e:
                st.error(f"调参失败：{e}")

    with st.expander("独立评估（专用评估模块）", expanded=False):
        eval_uploads = st.file_uploader(
            "上传评估集（支持多文件）",
            type=_UPLOAD_TYPES,
            accept_multiple_files=True,
            key="drug_eval_csv",
        )
        eval_model_path = st.text_input("评估模型路径", value=str(model_out), key="drug_eval_model_path")
        eval_keep_invalid = st.checkbox("评估时保留无效 SMILES", value=False, key="drug_eval_keep_invalid")
        if st.button("开始评估", key="drug_eval_btn"):
            if not eval_uploads:
                st.error("请先上传评估数据")
            elif not Path(eval_model_path).exists():
                st.error(f"模型不存在: {eval_model_path}")
            else:
                import tempfile

                try:
                    eval_df_raw = _load_tables_from_uploads(eval_uploads)
                    eval_cfg = _render_preprocess_panel(eval_df_raw, "drug_eval")
                    eval_df = _apply_preprocess(eval_df_raw, **eval_cfg)

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tf:
                        eval_tmp_path = tf.name
                    eval_df.to_csv(eval_tmp_path, index=False)

                    eval_result = evaluate_drug_from_csv(
                        model_path=str(eval_model_path),
                        data_path=str(eval_tmp_path),
                        smiles_col=str(smiles_col),
                        target_col=str(target_col),
                        env_cols=list(env_cols),
                        drop_invalid_smiles=(not bool(eval_keep_invalid)),
                    )
                    st.success("评估完成")
                    st.json(eval_result.metrics)
                except Exception as e:
                    st.error(f"评估失败：{e}")
                finally:
                    try:
                        Path(eval_tmp_path).unlink(missing_ok=True)  # type: ignore[name-defined]
                    except Exception:
                        pass

    if st.button("开始训练", type="primary", key="drug_train_btn"):
        train_df = df.copy()
        if drop_target_na and target_col in train_df.columns:
            train_df = train_df[train_df[target_col].notna()].copy()
        if drop_target_zero and target_col in train_df.columns:
            train_df = train_df[pd.to_numeric(train_df[target_col], errors="coerce") != 0].copy()

        if len(train_df) == 0:
            st.error("训练数据为空：请检查筛选条件、目标列是否为空或全为0。")
            return

        with st.spinner("训练中..."):
            try:
                import tempfile

                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tf:
                    train_tmp_path = tf.name
                train_df.to_csv(train_tmp_path, index=False)

                train_result = train_drug_from_csv(
                    data_path=str(train_tmp_path),
                    model_out=str(model_out),
                    smiles_col=str(smiles_col),
                    target_col=str(target_col),
                    env_cols=list(env_cols),
                    model_name=str(model_name),
                    test_size=float(test_size),
                    seed=int(seed),
                    featurizer_version=int(featurizer_version),
                    mlp_alpha=float(mlp_alpha),
                    mlp_early_stopping=bool(mlp_early),
                    mlp_patience=int(mlp_patience),
                    ridge_alpha=float(ridge_alpha),
                    hgb_l2=float(hgb_l2),
                )
                bundle = joblib.load(train_result.model_path)
                metrics = train_result.metrics
            except Exception as e:
                st.error(f"训练失败：{e}")
                return
            finally:
                try:
                    Path(train_tmp_path).unlink(missing_ok=True)  # type: ignore[name-defined]
                except Exception:
                    pass

        st.success("训练完成")
        render_training_result_panel(metrics, getattr(bundle, "feature_names", None))
        
        # st.json(metrics) # Only show raw json if expanded
        with st.expander("显示原始指标 JSON"):
            st.json(metrics)

        st.write("env_cols:", bundle.env_cols)

        out_path = Path(model_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, out_path)
        st.write("已保存到:", str(out_path))

        buf = io.BytesIO()
        joblib.dump(bundle, buf)
        st.download_button(
            "下载模型文件",
            data=buf.getvalue(),
            file_name=out_path.name,
            mime="application/octet-stream",
        )


def drug_train_torch_ui() -> None:
    st.subheader("训练（CSV → 药效 Torch 模型）")
    st.caption("Torch 版本支持自定义网络结构与训练轮次，适合更大数据或进一步微调。")

    try:
        import rdkit  # type: ignore  # noqa: F401
        import torch  # type: ignore  # noqa: F401
        from src.drug.torch_predictor import dump_torch_bundle, load_torch_bundle_from_bytes, train_torch_bundle  # type: ignore
    except Exception as e:
        st.warning(f"Torch 药物模块不可用（可能未安装 torch/rdkit）：{e}")
        return

    uploaded = st.file_uploader(
        "上传训练集（支持多文件）",
        type=_UPLOAD_TYPES,
        accept_multiple_files=True,
        key="drug_train_torch_csv",
    )
    if not uploaded:
        st.info("请上传包含 smiles 和目标列的 CSV")
        return

    try:
        df_raw = _load_tables_from_uploads(uploaded)
    except Exception as e:
        st.error(f"读取文件失败：{e}")
        return

    cfg = _render_preprocess_panel(df_raw, "drug_train_torch")
    df = _apply_preprocess(df_raw, **cfg)
    _preview_df(df, title="数据预览")

    smiles_col = st.selectbox(
        "smiles 列名",
        options=list(df.columns),
        index=(list(df.columns).index("smiles") if "smiles" in df.columns else 0),
        key="drug_torch_smiles_col",
    )
    target_candidates = [c for c in df.columns if c != smiles_col]
    target_col = st.selectbox(
        "目标列（疗效）",
        options=target_candidates,
        index=(target_candidates.index("efficacy") if "efficacy" in target_candidates else 0),
        key="drug_torch_target_col",
    )

    drop_target_na = st.checkbox("过滤目标列为空", value=True, key="drug_torch_drop_target_na")
    drop_target_zero = st.checkbox("将目标列=0 视为无标注", value=False, key="drug_torch_drop_target_zero")

    env_cols = st.multiselect(
        "条件列（数值，可选）",
        options=[c for c in df.columns if c not in (smiles_col, target_col)],
        default=[c for c in df.columns if c not in (smiles_col, target_col) and is_numeric_dtype(df[c])],
    )

    featurizer_version = st.selectbox("特征版本", options=[1, 2], index=1, key="drug_torch_feat")
    hidden_sizes_text = st.text_input("隐藏层（逗号分隔）", value="512,256", key="drug_torch_hidden")
    dropout = st.slider("Dropout", min_value=0.0, max_value=0.5, value=0.1, step=0.05, key="drug_torch_dropout")
    lr = st.number_input("学习率", min_value=1e-5, max_value=1e-2, value=1e-3, format="%.5f", key="drug_torch_lr")
    weight_decay = st.number_input("权重衰减（L2）", min_value=0.0, max_value=1e-2, value=1e-4, format="%.6f", key="drug_torch_wd")
    lr_schedule = st.selectbox("学习率调度", options=["cosine", "step", "none"], index=0, key="drug_torch_sched")
    step_size = st.number_input("阶梯步长", min_value=1, max_value=500, value=20, step=1, key="drug_torch_step")
    gamma = st.number_input("阶梯衰减系数", min_value=0.1, max_value=0.99, value=0.5, step=0.05, key="drug_torch_gamma")
    min_lr = st.number_input("最小学习率", min_value=1e-8, max_value=1e-3, value=1e-6, format="%.8f", key="drug_torch_minlr")
    early_patience = st.number_input("早停耐心", min_value=1, max_value=200, value=10, step=1, key="drug_torch_pat")
    max_grad_norm = st.number_input("梯度裁剪上限", min_value=0.0, max_value=100.0, value=5.0, step=0.5, key="drug_torch_clip")
    epochs = st.number_input("训练轮次", min_value=10, max_value=2000, value=200, step=10, key="drug_torch_epochs")
    batch_size = st.number_input("Batch 大小", min_value=16, max_value=2048, value=128, step=16, key="drug_torch_batch")
    test_size = st.slider("验证集比例", min_value=0.1, max_value=0.5, value=0.2, step=0.05, key="drug_torch_test")
    seed = st.number_input("随机种子", min_value=0, max_value=10_000_000, value=42, step=1, key="drug_torch_seed")

    use_cuda = st.checkbox("使用 CUDA（若可用）", value=False, key="drug_torch_cuda")
    model_out = st.text_input("保存路径", value="models/drug_torch_model.pt", key="drug_torch_out")

    with st.expander("知识蒸馏（可选）", expanded=False):
        teacher_up = st.file_uploader("上传教师模型 .pt", type=["pt"], key="drug_torch_teacher")
        distill_weight = st.slider("蒸馏权重", min_value=0.0, max_value=1.0, value=0.2, step=0.05, key="drug_torch_distill")

    with st.expander("云端训练", expanded=False):
        payload = {
            "data": _cloud_encode_dataframe(df),
            "smiles_col": str(smiles_col),
            "target_col": str(target_col),
            "env_cols": list(env_cols),
            "hidden_sizes": str(hidden_sizes_text),
            "dropout": float(dropout),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "lr_schedule": str(lr_schedule),
            "step_size": int(step_size),
            "gamma": float(gamma),
            "min_lr": float(min_lr),
            "early_stopping_patience": int(early_patience),
            "max_grad_norm": float(max_grad_norm),
            "distill_weight": float(distill_weight),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "test_size": float(test_size),
            "random_state": int(seed),
            "featurizer_version": int(featurizer_version),
            "use_cuda": bool(use_cuda),
            "drop_target_na": bool(drop_target_na),
            "drop_target_zero": bool(drop_target_zero),
            "model_out": str(model_out),
        }
        _cloud_submit_section(
            "drug_train_torch",
            payload,
            button_label="提交云端 Torch 训练",
            key="cloud_drug_torch_train_btn",
            model_id_state_key="cloud_drug_torch_model_id",
        )

    if st.button("开始 Torch 训练", type="primary", key="drug_torch_train_btn"):
        try:
            hidden_sizes = _parse_hidden_sizes(hidden_sizes_text)
        except Exception as e:
            st.error(f"隐藏层格式错误：{e}")
            return

        train_df = df.copy()
        if drop_target_na and target_col in train_df.columns:
            train_df = train_df[train_df[target_col].notna()].copy()
        if drop_target_zero and target_col in train_df.columns:
            train_df = train_df[pd.to_numeric(train_df[target_col], errors="coerce") != 0].copy()

        with st.spinner("Torch 训练中..."):
            bundle, metrics = train_torch_bundle(
                train_df,
                smiles_col=str(smiles_col),
                target_col=str(target_col),
                env_cols=list(env_cols),
                hidden_sizes=hidden_sizes,
                dropout=float(dropout),
                lr=float(lr),
                weight_decay=float(weight_decay),
                lr_schedule=str(lr_schedule),
                step_size=int(step_size),
                gamma=float(gamma),
                min_lr=float(min_lr),
                max_grad_norm=float(max_grad_norm),
                early_stopping_patience=int(early_patience),
                batch_size=int(batch_size),
                epochs=int(epochs),
                test_size=float(test_size),
                random_state=int(seed),
                featurizer_version=int(featurizer_version),
                use_cuda=bool(use_cuda),
                teacher_bundle=load_torch_bundle_from_bytes(teacher_up.getvalue()) if teacher_up is not None else None,
                distill_weight=float(distill_weight),
            )

        st.success("Torch 训练完成")
        st.json(metrics)
        hist = metrics.get("history") if isinstance(metrics, dict) else None
        if isinstance(hist, dict) and hist.get("train_loss"):
            st.write("训练曲线")
            st.line_chart({"train_loss": hist.get("train_loss", []), "val_loss": hist.get("val_loss", [])})
            df_hist = pd.DataFrame({
                "epoch": list(range(1, len(hist.get("train_loss", [])) + 1)),
                "train_loss": hist.get("train_loss", []),
                "val_loss": hist.get("val_loss", []),
            })
            st.dataframe(df_hist, use_container_width=True)
            df_hist = pd.DataFrame({
                "epoch": list(range(1, len(hist.get("train_loss", [])) + 1)),
                "train_loss": hist.get("train_loss", []),
                "val_loss": hist.get("val_loss", []),
            })
            st.dataframe(df_hist, use_container_width=True)
            df_hist = pd.DataFrame({
                "epoch": list(range(1, len(hist.get("train_loss", [])) + 1)),
                "train_loss": hist.get("train_loss", []),
                "val_loss": hist.get("val_loss", []),
            })
            st.dataframe(df_hist, use_container_width=True)
        if isinstance(metrics, dict) and metrics.get("suggestions"):
            st.write("训练建议:")
            st.write(metrics.get("suggestions"))
        st.write("env_cols:", bundle.env_cols)

        out_path = Path(model_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dump_torch_bundle(bundle, out_path)
        st.write("已保存到:", str(out_path))

        buf = io.BytesIO()
        dump_torch_bundle(bundle, buf)
        st.download_button(
            "下载 Torch 模型",
            data=buf.getvalue(),
            file_name=out_path.name,
            mime="application/octet-stream",
        )


def drug_transformer_train_ui() -> None:
    st.subheader("训练（CSV → 药效 Transformer 模型）")
    st.caption("基于字符级 SMILES 的 Transformer 回归模型，可选拼接实验条件数值特征。")

    try:
        import torch  # type: ignore  # noqa: F401
        from src.drug.transformer_predictor import dump_transformer_bundle, load_transformer_bundle_from_bytes, train_transformer_bundle  # type: ignore
    except Exception as e:
        st.warning(f"Transformer 药物模块不可用（可能未安装 torch）：{e}")
        return

    uploaded = st.file_uploader(
        "上传训练集（支持多文件）",
        type=_UPLOAD_TYPES,
        accept_multiple_files=True,
        key="drug_train_transformer_csv",
    )
    if not uploaded:
        st.info("请上传包含 smiles 和目标列的 CSV")
        return

    try:
        df_raw = _load_tables_from_uploads(uploaded)
    except Exception as e:
        st.error(f"读取文件失败：{e}")
        return

    cfg = _render_preprocess_panel(df_raw, "drug_train_transformer")
    df = _apply_preprocess(df_raw, **cfg)
    _preview_df(df, title="数据预览")

    smiles_col = st.selectbox(
        "smiles 列名",
        options=list(df.columns),
        index=(list(df.columns).index("smiles") if "smiles" in df.columns else 0),
        key="drug_transformer_smiles_col",
    )
    target_candidates = [c for c in df.columns if c != smiles_col]
    target_col = st.selectbox(
        "目标列（疗效）",
        options=target_candidates,
        index=(target_candidates.index("efficacy") if "efficacy" in target_candidates else 0),
        key="drug_transformer_target_col",
    )

    drop_target_na = st.checkbox("过滤目标列为空", value=True, key="drug_transformer_drop_target_na")
    drop_target_zero = st.checkbox("将目标列=0 视为无标注", value=False, key="drug_transformer_drop_target_zero")

    env_cols = st.multiselect(
        "条件列（数值，可选）",
        options=[c for c in df.columns if c not in (smiles_col, target_col)],
        default=[c for c in df.columns if c not in (smiles_col, target_col) and is_numeric_dtype(df[c])],
    )

    max_len = st.number_input("最大序列长度", min_value=32, max_value=512, value=128, step=16, key="drug_transformer_max_len")
    min_char_freq = st.number_input("最小字符频次（低于则视为UNK）", min_value=1, max_value=10, value=1, step=1, key="drug_transformer_min_char")
    emb_dim = st.number_input("Embedding 维度", min_value=32, max_value=512, value=128, step=32, key="drug_transformer_emb")
    n_heads = st.number_input("注意力头数", min_value=1, max_value=16, value=4, step=1, key="drug_transformer_heads")
    n_layers = st.number_input("Transformer 层数", min_value=1, max_value=12, value=2, step=1, key="drug_transformer_layers")
    ff_dim = st.number_input("前馈层维度", min_value=64, max_value=2048, value=256, step=64, key="drug_transformer_ff")
    dropout = st.slider("Dropout", min_value=0.0, max_value=0.5, value=0.1, step=0.05, key="drug_transformer_dropout")
    use_lstm = st.checkbox("加入 LSTM", value=False, key="drug_transformer_lstm")
    lstm_hidden = st.number_input("LSTM 隐藏维度", min_value=32, max_value=512, value=128, step=32, key="drug_transformer_lstm_hidden")
    lstm_layers = st.number_input("LSTM 层数", min_value=1, max_value=4, value=1, step=1, key="drug_transformer_lstm_layers")
    lstm_bi = st.checkbox("双向 LSTM", value=True, key="drug_transformer_lstm_bi")
    lr = st.number_input("学习率", min_value=1e-5, max_value=1e-2, value=2e-4, format="%.5f", key="drug_transformer_lr")
    weight_decay = st.number_input("权重衰减（L2）", min_value=0.0, max_value=1e-2, value=1e-4, format="%.6f", key="drug_transformer_wd")
    lr_schedule = st.selectbox("学习率调度", options=["cosine", "step", "none"], index=0, key="drug_transformer_sched")
    step_size = st.number_input("阶梯步长", min_value=1, max_value=500, value=20, step=1, key="drug_transformer_step")
    gamma = st.number_input("阶梯衰减系数", min_value=0.1, max_value=0.99, value=0.5, step=0.05, key="drug_transformer_gamma")
    min_lr = st.number_input("最小学习率", min_value=1e-8, max_value=1e-3, value=1e-6, format="%.8f", key="drug_transformer_minlr")
    early_patience = st.number_input("早停耐心", min_value=1, max_value=200, value=8, step=1, key="drug_transformer_pat")
    max_grad_norm = st.number_input("梯度裁剪上限", min_value=0.0, max_value=100.0, value=5.0, step=0.5, key="drug_transformer_clip")
    epochs = st.number_input("训练轮次", min_value=10, max_value=2000, value=200, step=10, key="drug_transformer_epochs")
    batch_size = st.number_input("Batch 大小", min_value=8, max_value=2048, value=64, step=8, key="drug_transformer_batch")
    test_size = st.slider("验证集比例", min_value=0.1, max_value=0.5, value=0.2, step=0.05, key="drug_transformer_test")
    seed = st.number_input("随机种子", min_value=0, max_value=10_000_000, value=42, step=1, key="drug_transformer_seed")
    use_cuda = st.checkbox("使用 CUDA（若可用）", value=False, key="drug_transformer_cuda")
    model_out = st.text_input("保存路径", value="models/drug_transformer_model.pt", key="drug_transformer_out")

    with st.expander("知识蒸馏（可选）", expanded=False):
        teacher_up = st.file_uploader("上传教师模型 .pt", type=["pt"], key="drug_transformer_teacher")
        distill_weight = st.slider("蒸馏权重", min_value=0.0, max_value=1.0, value=0.2, step=0.05, key="drug_transformer_distill")
        tr_enable_ema = st.checkbox("启用参数级 EMA 教师（训练期间更新）", value=False, key="drug_transformer_enable_ema")
        if tr_enable_ema:
            tr_ema_decay = st.number_input("EMA 衰减系数", min_value=0.0, max_value=0.9999, value=0.99, step=0.001, format="%.4f", key="drug_transformer_ema_decay")
        else:
            tr_ema_decay = 0.99

    with st.expander("云端训练", expanded=False):
        payload = {
            "data": _cloud_encode_dataframe(df),
            "smiles_col": str(smiles_col),
            "target_col": str(target_col),
            "env_cols": list(env_cols),
            "max_len": int(max_len),
            "min_char_freq": int(min_char_freq),
            "emb_dim": int(emb_dim),
            "n_heads": int(n_heads),
            "n_layers": int(n_layers),
            "ff_dim": int(ff_dim),
            "dropout": float(dropout),
            "use_lstm": bool(use_lstm),
            "lstm_hidden": int(lstm_hidden),
            "lstm_layers": int(lstm_layers),
            "lstm_bidirectional": bool(lstm_bi),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "lr_schedule": str(lr_schedule),
            "step_size": int(step_size),
            "gamma": float(gamma),
            "min_lr": float(min_lr),
            "max_grad_norm": float(max_grad_norm),
            "early_stopping_patience": int(early_patience),
            "distill_weight": float(distill_weight),
            "use_ema": bool(tr_enable_ema),
            "ema_decay": float(tr_ema_decay),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "test_size": float(test_size),
            "random_state": int(seed),
            "use_cuda": bool(use_cuda),
            "drop_target_na": bool(drop_target_na),
            "drop_target_zero": bool(drop_target_zero),
            "model_out": str(model_out),
        }
        _cloud_submit_section(
            "drug_transformer_train",
            payload,
            button_label="提交云端 Transformer 训练",
            key="cloud_drug_transformer_train_btn",
            model_id_state_key="cloud_drug_transformer_model_id",
        )

    if st.button("开始 Transformer 训练", type="primary", key="drug_transformer_train_btn"):
        train_df = df.copy()
        if drop_target_na and target_col in train_df.columns:
            train_df = train_df[train_df[target_col].notna()].copy()
        if drop_target_zero and target_col in train_df.columns:
            train_df = train_df[pd.to_numeric(train_df[target_col], errors="coerce") != 0].copy()

        with st.spinner("Transformer 训练中..."):
            bundle, metrics = train_transformer_bundle(
                train_df,
                smiles_col=str(smiles_col),
                target_col=str(target_col),
                env_cols=list(env_cols),
                max_len=int(max_len),
                min_char_freq=int(min_char_freq),
                emb_dim=int(emb_dim),
                n_heads=int(n_heads),
                n_layers=int(n_layers),
                ff_dim=int(ff_dim),
                dropout=float(dropout),
                use_lstm=bool(use_lstm),
                lstm_hidden=int(lstm_hidden),
                lstm_layers=int(lstm_layers),
                lstm_bidirectional=bool(lstm_bi),
                lr=float(lr),
                weight_decay=float(weight_decay),
                lr_schedule=str(lr_schedule),
                step_size=int(step_size),
                gamma=float(gamma),
                min_lr=float(min_lr),
                max_grad_norm=float(max_grad_norm),
                batch_size=int(batch_size),
                epochs=int(epochs),
                test_size=float(test_size),
                random_state=int(seed),
                use_cuda=bool(use_cuda),
                early_stopping_patience=int(early_patience),
                teacher_bundle=load_transformer_bundle_from_bytes(teacher_up.getvalue()) if teacher_up is not None else None,
                distill_weight=float(distill_weight),
                use_ema=bool(tr_enable_ema),
                ema_decay=float(tr_ema_decay),
            )

        st.success("Transformer 训练完成")
        st.json(metrics)
        hist = metrics.get("history") if isinstance(metrics, dict) else None
        if isinstance(hist, dict) and hist.get("train_loss"):
            st.write("训练曲线")
            st.line_chart({"train_loss": hist.get("train_loss", []), "val_loss": hist.get("val_loss", [])})
        if isinstance(metrics, dict) and metrics.get("suggestions"):
            st.write("训练建议:")
            st.write(metrics.get("suggestions"))
        st.write("env_cols:", bundle.env_cols)

        # 展示训练历史或评估指标
        hist = metrics.get("history") if isinstance(metrics, dict) else None
        if hist:
            try:
                import pandas as _pd

                df_hist = _pd.DataFrame(hist)
                st.subheader("训练曲线")
                st.line_chart(df_hist)

                # 提供历史记录下载
                csv_buf = df_hist.to_csv(index=False).encode("utf-8")
                st.download_button("下载训练历史 CSV", data=csv_buf, file_name="transformer_history.csv", mime="text/csv")
            except Exception as _e:
                st.warning(f"无法绘制训练历史：{_e}")
        else:
            # 直接显示常用指标
            metrics_map = {}
            for k in ("mae", "rmse", "r2", "n_train", "n_val", "vocab_size"):
                if isinstance(metrics, dict) and k in metrics:
                    metrics_map[k] = metrics[k]

            if metrics_map:
                cols = st.columns(len(metrics_map)) if len(metrics_map) else []
                for i, (k, v) in enumerate(metrics_map.items()):
                    try:
                        cols[i].metric(label=k.upper(), value=f"{v:.6g}" if isinstance(v, float) else str(v))
                    except Exception:
                        try:
                            cols[i].metric(label=k.upper(), value=str(v))
                        except Exception:
                            pass

        # 如果有验证集，绘制预测 vs 实际 和 残差直方图
        try:
            val_seqs = metrics.get('val_seqs') if isinstance(metrics, dict) else None
            val_targets = metrics.get('val_targets') if isinstance(metrics, dict) else None
            if val_seqs and val_targets and len(val_seqs) == len(val_targets):
                from src.drug.transformer_predictor import predict_transformer_batch
                import matplotlib.pyplot as plt
                import numpy as _np

                y_true = _np.array(val_targets, dtype=float)
                y_pred = _np.array(
                    predict_transformer_batch(bundle, smiles_list=val_seqs, env_params_list=[{} for _ in range(len(val_seqs))], batch_size=512),
                    dtype=float,
                )

                # scatter plot
                fig1, ax1 = plt.subplots()
                ax1.scatter(y_true, y_pred, alpha=0.7)
                mmin = min(y_true.min(), y_pred.min())
                mmax = max(y_true.max(), y_pred.max())
                ax1.plot([mmin, mmax], [mmin, mmax], color='r', linestyle='--')
                ax1.set_xlabel('实际值')
                ax1.set_ylabel('预测值')
                ax1.set_title('预测 vs 实际')
                st.pyplot(fig1)

                # residuals
                resid = y_pred - y_true
                fig2, ax2 = plt.subplots()
                ax2.hist(resid, bins=30)
                ax2.set_xlabel('残差 (pred - true)')
                ax2.set_title('残差分布')
                st.pyplot(fig2)

                # small summary
                st.write('验证样本数:', len(y_true))
                st.write('残差均值', float(_np.mean(resid)), '残差标准差', float(_np.std(resid)))
        except Exception as e:
            st.warning(f'绘制验证图失败: {e}')

        out_path = Path(model_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dump_transformer_bundle(bundle, out_path)
        st.write("已保存到:", str(out_path))

        buf = io.BytesIO()
        dump_transformer_bundle(bundle, buf)
        st.download_button(
            "下载 Transformer 模型",
            data=buf.getvalue(),
            file_name=out_path.name,
            mime="application/octet-stream",
        )


def drug_transformer_predict_ui() -> None:
    st.subheader("单条预测（Transformer 模型 + SMILES + 条件）")

    try:
        from src.drug.transformer_predictor import predict_transformer_one  # type: ignore
    except Exception as e:
        st.warning(f"Transformer 预测不可用：{e}")
        return

    col1, col2 = st.columns(2)
    with col1:
        uploaded_model = st.file_uploader("上传模型 .pt", type=["pt"], key="drug_transformer_pred_model")
    with col2:
        local_models = _list_local_models_drug_transformer()
        local_path = st.selectbox("或选择本地 models/drug_transformer*.pt", options=[""] + local_models, key="drug_transformer_pred_local")

    smiles = st.text_input("SMILES", value="CCO", key="drug_transformer_pred_smiles")
    kv_text = st.text_area(
        "条件参数（每行 key=value；缺失会用训练集中位数填充）",
        value="dose=10\nfreq=2\n",
        height=120,
        key="drug_transformer_pred_kv",
    )

    with st.expander("云端预测", expanded=False):
        try:
            env_cloud = _parse_kv_lines(kv_text)
        except ValueError as e:
            st.error(str(e))
            env_cloud = {}

        model_payload = _resolve_model_payload(uploaded_model, local_path if local_path else None, content_type="application/octet-stream")
        if model_payload is None:
            st.info("请上传模型或选择本地模型以提交云端预测。")
        else:
            payload = {
                "model": model_payload,
                "smiles": str(smiles),
                "env_params": {k: float(v) for k, v in env_cloud.items()},
            }
            _cloud_submit_section(
                "drug_transformer_predict",
                payload,
                button_label="提交云端预测",
                key="cloud_drug_transformer_pred_btn",
            )

    if st.button("预测", type="primary", key="drug_transformer_pred_btn"):
        if uploaded_model is None and not local_path:
            st.error("请上传模型或选择本地模型")
            return

        try:
            env = _parse_kv_lines(kv_text)
        except ValueError as e:
            st.error(str(e))
            return

        if uploaded_model is not None:
            bundle = _cached_transformer_bundle_from_bytes(uploaded_model.getvalue())
        else:
            bundle = _cached_transformer_bundle_from_path(local_path, _get_file_mtime(local_path))

        y = predict_transformer_one(bundle, smiles=smiles, env_params=env)
        st.metric("预测值", value=f"{y:.6g}")

        update_feedback_context({
            "event": "predict_transformer",
            "smiles": str(smiles),
            "env_params": {k: float(v) for k, v in env.items()},
            "pred": float(y),
        })


def docking_train_ui() -> None:
    st.subheader("训练（交叉注意力：SMILES × 蛋白序列 → 对接效果）")
    st.caption("输入包含 smiles 与 protein（或 receptor）序列列，以及对接评分/打分列。")

    try:
        import torch  # type: ignore  # noqa: F401
        from src.drug.docking_cross_attention import dump_docking_bundle, load_docking_bundle_from_bytes, train_docking_bundle
    except Exception as e:
        st.warning(f"对接模型不可用（可能未安装 torch）：{e}")
        return

    uploaded = st.file_uploader(
        "上传训练集（支持多文件）",
        type=_UPLOAD_TYPES,
        accept_multiple_files=True,
        key="docking_train_csv",
    )
    use_crawled = False
    crawled_df = st.session_state.get("docking_crawl_df")
    if (not uploaded) and isinstance(crawled_df, pd.DataFrame) and len(crawled_df) > 0:
        use_crawled = st.checkbox("使用爬虫抓取的数据", value=True, key="docking_use_crawl")

    if not uploaded and not use_crawled:
        st.info("请上传包含 smiles、protein、目标列 的 CSV，或使用爬虫数据")
        return

    try:
        df_raw = _load_tables_from_uploads(uploaded) if uploaded else crawled_df.copy()
    except Exception as e:
        st.error(f"读取文件失败：{e}")
        return

    cfg = _render_preprocess_panel(df_raw, "docking_train")
    df = _apply_preprocess(df_raw, **cfg)
    _preview_df(df, title="数据预览")

    smiles_col = st.selectbox(
        "smiles 列名",
        options=list(df.columns),
        index=(list(df.columns).index("smiles") if "smiles" in df.columns else 0),
        key="docking_smiles_col",
    )
    protein_candidates = [c for c in df.columns if c != smiles_col]
    if not protein_candidates:
        st.error("缺少蛋白/受体序列列")
        return
    protein_col = st.selectbox(
        "蛋白/受体序列列名",
        options=protein_candidates,
        index=(protein_candidates.index("protein") if "protein" in protein_candidates else 0),
        key="docking_protein_col",
    )
    target_candidates = [c for c in df.columns if c not in (smiles_col, protein_col)]
    if not target_candidates:
        st.error("缺少目标列（对接评分）")
        return
    target_col = st.selectbox(
        "目标列（对接效果/打分）",
        options=target_candidates,
        index=(target_candidates.index("docking_score") if "docking_score" in target_candidates else 0),
        key="docking_target_col",
    )

    st.markdown("#### 模型参数")
    lig_max_len = st.number_input("SMILES 最大长度", min_value=16, max_value=512, value=128, step=16, key="docking_lig_max")
    prot_max_len = st.number_input("蛋白序列最大长度", min_value=64, max_value=4096, value=512, step=64, key="docking_prot_max")
    min_char_freq = st.number_input("最小字符频次", min_value=1, max_value=10, value=1, step=1, key="docking_min_char")
    emb_dim = st.number_input("Embedding 维度", min_value=32, max_value=512, value=128, step=32, key="docking_emb")
    n_heads = st.number_input("注意力头数", min_value=1, max_value=16, value=4, step=1, key="docking_heads")
    n_layers = st.number_input("编码层数", min_value=1, max_value=8, value=2, step=1, key="docking_layers")
    ff_dim = st.number_input("前馈层维度", min_value=64, max_value=2048, value=256, step=64, key="docking_ff")
    dropout = st.slider("Dropout", min_value=0.0, max_value=0.5, value=0.1, step=0.05, key="docking_dropout")
    use_lstm = st.checkbox("加入 LSTM", value=False, key="docking_lstm")
    lstm_hidden = st.number_input("LSTM 隐藏维度", min_value=32, max_value=512, value=128, step=32, key="docking_lstm_hidden")
    lstm_layers = st.number_input("LSTM 层数", min_value=1, max_value=4, value=1, step=1, key="docking_lstm_layers")
    lstm_bi = st.checkbox("双向 LSTM", value=True, key="docking_lstm_bi")

    st.markdown("#### 训练参数")
    lr = st.number_input("学习率", min_value=1e-5, max_value=1e-2, value=2e-4, format="%.5f", key="docking_lr")
    weight_decay = st.number_input("权重衰减（L2）", min_value=0.0, max_value=1e-2, value=1e-4, format="%.6f", key="docking_wd")
    lr_schedule = st.selectbox("学习率调度", options=["cosine", "step", "none"], index=0, key="docking_sched")
    step_size = st.number_input("阶梯步长", min_value=1, max_value=500, value=20, step=1, key="docking_step")
    gamma = st.number_input("阶梯衰减系数", min_value=0.1, max_value=0.99, value=0.5, step=0.05, key="docking_gamma")
    min_lr = st.number_input("最小学习率", min_value=1e-8, max_value=1e-3, value=1e-6, format="%.8f", key="docking_minlr")
    early_patience = st.number_input("早停耐心", min_value=1, max_value=200, value=8, step=1, key="docking_pat")
    max_grad_norm = st.number_input("梯度裁剪上限", min_value=0.0, max_value=100.0, value=5.0, step=0.5, key="docking_clip")
    batch_size = st.number_input("Batch size", min_value=4, max_value=512, value=32, step=4, key="docking_bs")
    epochs = st.number_input("训练轮数", min_value=1, max_value=500, value=30, step=1, key="docking_epochs")
    test_size = st.slider("验证集比例", min_value=0.1, max_value=0.5, value=0.2, step=0.05, key="docking_test")
    seed = st.number_input("随机种子", min_value=0, max_value=10_000_000, value=42, step=1, key="docking_seed")
    use_cuda = st.checkbox("使用 CUDA（若可用）", value=False, key="docking_cuda")
    model_out = st.text_input("保存路径", value="models/docking_crossattn.pt", key="docking_out")

    with st.expander("知识蒸馏（可选）", expanded=False):
        teacher_up = st.file_uploader("上传教师模型 .pt", type=["pt"], key="docking_teacher")
        distill_weight = st.slider("蒸馏权重", min_value=0.0, max_value=1.0, value=0.2, step=0.05, key="docking_distill")

    with st.expander("云端训练", expanded=False):
        payload = {
            "data": _cloud_encode_dataframe(df),
            "ligand_col": str(smiles_col),
            "protein_col": str(protein_col),
            "target_col": str(target_col),
            "params": {
                "lig_max_len": int(lig_max_len),
                "prot_max_len": int(prot_max_len),
                "min_char_freq": int(min_char_freq),
                "emb_dim": int(emb_dim),
                "n_heads": int(n_heads),
                "n_layers": int(n_layers),
                "ff_dim": int(ff_dim),
                "dropout": float(dropout),
                "use_lstm": bool(use_lstm),
                "lstm_hidden": int(lstm_hidden),
                "lstm_layers": int(lstm_layers),
                "lstm_bidirectional": bool(lstm_bi),
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "lr_schedule": str(lr_schedule),
                "step_size": int(step_size),
                "gamma": float(gamma),
                "min_lr": float(min_lr),
                "max_grad_norm": float(max_grad_norm),
                "batch_size": int(batch_size),
                "epochs": int(epochs),
                "test_size": float(test_size),
                "random_state": int(seed),
                "use_cuda": bool(use_cuda),
                "early_stopping_patience": int(early_patience),
                "distill_weight": float(distill_weight),
            },
        }
        _cloud_submit_section(
            "docking_train",
            payload,
            button_label="提交云端训练",
            key="cloud_docking_train_btn",
            model_id_state_key="cloud_docking_model_id",
        )

    if st.button("开始训练", type="primary", key="docking_train_btn"):
        with st.spinner("交叉注意力模型训练中..."):
            bundle, metrics = train_docking_bundle(
                df,
                ligand_col=str(smiles_col),
                protein_col=str(protein_col),
                target_col=str(target_col),
                lig_max_len=int(lig_max_len),
                prot_max_len=int(prot_max_len),
                min_char_freq=int(min_char_freq),
                emb_dim=int(emb_dim),
                n_heads=int(n_heads),
                n_layers=int(n_layers),
                ff_dim=int(ff_dim),
                dropout=float(dropout),
                use_lstm=bool(use_lstm),
                lstm_hidden=int(lstm_hidden),
                lstm_layers=int(lstm_layers),
                lstm_bidirectional=bool(lstm_bi),
                lr=float(lr),
                weight_decay=float(weight_decay),
                lr_schedule=str(lr_schedule),
                step_size=int(step_size),
                gamma=float(gamma),
                min_lr=float(min_lr),
                max_grad_norm=float(max_grad_norm),
                batch_size=int(batch_size),
                epochs=int(epochs),
                test_size=float(test_size),
                random_state=int(seed),
                use_cuda=bool(use_cuda),
                early_stopping_patience=int(early_patience),
                teacher_bundle=load_docking_bundle_from_bytes(teacher_up.getvalue()) if teacher_up is not None else None,
                distill_weight=float(distill_weight),
            )

        st.success("训练完成")
        st.json(metrics)
        hist = metrics.get("history") if isinstance(metrics, dict) else None
        if isinstance(hist, dict) and hist.get("train_loss"):
            st.write("训练曲线")
            st.line_chart({"train_loss": hist.get("train_loss", []), "val_loss": hist.get("val_loss", [])})
        if isinstance(metrics, dict) and metrics.get("suggestions"):
            st.write("训练建议:")
            st.write(metrics.get("suggestions"))

        if model_out:
            Path(model_out).parent.mkdir(parents=True, exist_ok=True)
            from src.drug.docking_cross_attention import save_docking_bundle

            save_docking_bundle(bundle, str(model_out))
            st.write("已保存模型：", str(model_out))

        buf = io.BytesIO()
        dump_docking_bundle(bundle, buf)
        st.download_button(
            "下载模型 .pt",
            data=buf.getvalue(),
            file_name=Path(model_out).name if model_out else "docking_crossattn.pt",
            mime="application/octet-stream",
        )


def docking_predict_ui() -> None:
    st.subheader("单条预测（交叉注意力：SMILES × 蛋白序列）")

    try:
        from src.drug.docking_cross_attention import predict_docking_one
    except Exception as e:
        st.warning(f"对接预测不可用（可能未安装 torch）：{e}")
        return

    col1, col2 = st.columns(2)
    with col1:
        uploaded_model = st.file_uploader("上传模型 .pt", type=["pt"], key="docking_pred_model")
    with col2:
        local_models = _list_local_models_docking()
        local_path = st.selectbox("或选择本地 models/docking_*.pt", options=[""] + local_models, key="docking_pred_local")

    smiles = st.text_input("SMILES", value="CCO", key="docking_pred_smiles")
    protein = st.text_area(
        "蛋白/受体序列",
        value="MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAG",
        height=120,
        key="docking_pred_protein",
    )
    use_cuda = st.checkbox("使用 CUDA（若可用）", value=False, key="docking_pred_cuda")

    with st.expander("云端预测", expanded=False):
        model_payload = _resolve_model_payload(uploaded_model, local_path if local_path else None, content_type="application/octet-stream")
        if model_payload is None:
            st.info("请上传模型或选择本地模型以提交云端预测。")
        else:
            payload = {
                "model": model_payload,
                "smiles": str(smiles),
                "protein": str(protein),
                "use_cuda": bool(use_cuda),
            }
            _cloud_submit_section(
                "docking_predict",
                payload,
                button_label="提交云端预测",
                key="cloud_docking_pred_btn",
            )

    if st.button("预测", type="primary", key="docking_pred_btn"):
        if uploaded_model is None and not local_path:
            st.error("请上传模型或选择本地模型")
            return

        if uploaded_model is not None:
            bundle = _cached_docking_bundle_from_bytes(uploaded_model.getvalue())
        else:
            bundle = _cached_docking_bundle_from_path(local_path, _get_file_mtime(local_path))

        y = predict_docking_one(bundle, smiles=str(smiles), protein=str(protein), use_cuda=bool(use_cuda))
        st.metric("预测对接效果", value=f"{y:.6g}")

        update_feedback_context({
            "event": "docking_predict",
            "smiles": str(smiles),
            "protein_len": int(len(str(protein))),
            "pred": float(y),
        })


def docking_screen_ui() -> None:
    st.subheader("批量预测（CSV → 对接效果）")

    try:
        from src.drug.docking_cross_attention import predict_docking_batch
    except Exception as e:
        st.warning(f"对接预测不可用（可能未安装 torch）：{e}")
        return

    col1, col2 = st.columns(2)
    with col1:
        uploaded_model = st.file_uploader("上传模型 .pt", type=["pt"], key="docking_screen_model")
    with col2:
        local_models = _list_local_models_docking()
        local_path = st.selectbox("或选择本地 models/docking_*.pt", options=[""] + local_models, key="docking_screen_local")

    uploaded = st.file_uploader(
        "上传候选 CSV（需包含 smiles 与 protein 列）",
        type=_UPLOAD_TYPES,
        accept_multiple_files=True,
        key="docking_screen_csv",
    )
    if not uploaded:
        st.info("请上传候选数据")
        return

    try:
        candidates_raw = _load_tables_from_uploads(uploaded)
    except Exception as e:
        st.error(f"读取文件失败：{e}")
        return

    cand_cfg = _render_preprocess_panel(candidates_raw, "docking_screen")
    candidates = _apply_preprocess(candidates_raw, **cand_cfg)
    _preview_df(candidates, title="候选预览")

    smiles_col = st.selectbox(
        "smiles 列名",
        options=list(candidates.columns),
        index=(list(candidates.columns).index("smiles") if "smiles" in candidates.columns else 0),
        key="docking_screen_smiles",
    )
    protein_candidates = [c for c in candidates.columns if c != smiles_col]
    if not protein_candidates:
        st.error("缺少蛋白/受体序列列")
        return
    protein_col = st.selectbox(
        "蛋白/受体序列列名",
        options=protein_candidates,
        index=(protein_candidates.index("protein") if "protein" in protein_candidates else 0),
        key="docking_screen_protein",
    )
    out_col = st.text_input("输出预测列名", value="dock_pred", key="docking_screen_out")
    use_cuda = st.checkbox("使用 CUDA（若可用）", value=False, key="docking_screen_cuda")

    with st.expander("云端批量预测", expanded=False):
        model_payload = _resolve_model_payload(uploaded_model, local_path if local_path else None, content_type="application/octet-stream")
        if model_payload is None:
            st.info("请上传模型或选择本地模型以提交云端批量预测。")
        else:
            payload = {
                "model": model_payload,
                "candidates": _cloud_encode_dataframe(candidates),
                "smiles_col": str(smiles_col),
                "protein_col": str(protein_col),
                "out_col": str(out_col),
                "use_cuda": bool(use_cuda),
            }
            _cloud_submit_section(
                "docking_screen",
                payload,
                button_label="提交云端批量预测",
                key="cloud_docking_screen_btn",
                download_name="docking_predictions_cloud.csv",
            )

    if st.button("开始预测", type="primary", key="docking_screen_btn"):
        if uploaded_model is None and not local_path:
            st.error("请上传模型或选择本地模型")
            return

        if uploaded_model is not None:
            bundle = _cached_docking_bundle_from_bytes(uploaded_model.getvalue())
        else:
            bundle = _cached_docking_bundle_from_path(local_path, _get_file_mtime(local_path))

        ligands = candidates[smiles_col].astype(str).tolist()
        proteins = candidates[protein_col].astype(str).tolist()
        preds = predict_docking_batch(bundle, ligands, proteins, batch_size=64, use_cuda=bool(use_cuda))

        out_df = candidates.copy()
        out_df[out_col] = preds
        _preview_df(out_df, title="结果预览", max_rows=50)
        st.download_button(
            "下载预测结果 CSV",
            data=out_df.to_csv(index=False).encode("utf-8"),
            file_name="docking_predictions.csv",
            mime="text/csv",
        )

        update_feedback_context({
            "event": "docking_screen",
            "n": int(len(out_df)),
        })


def drug_predict_ui() -> None:
    st.subheader("单条预测（模型 + SMILES + 条件）")

    try:
        import rdkit  # type: ignore  # noqa: F401
        from src.drug.predictor import DrugModelBundle, predict_one  # type: ignore
    except Exception as e:
        st.warning(f"药物模块不可用（可能未安装 rdkit）：{e}")
        return

    model_type = st.selectbox("模型类型", options=["sklearn(joblib)", "torch(pt)"], index=0, key="drug_pred_model_type")
    use_cuda_pred = False
    if model_type == "torch(pt)":
        use_cuda_pred = st.checkbox("使用 CUDA（若可用）", value=False, key="drug_pred_cuda")

    col1, col2 = st.columns(2)
    if model_type == "torch(pt)":
        with col1:
            uploaded_model = st.file_uploader("上传模型 .pt", type=["pt"], key="drug_pred_model_torch")
        with col2:
            local_models = _list_local_models_drug_torch()
            local_path = st.selectbox("或选择本地 models/drug_torch*.pt", options=[""] + local_models, key="drug_pred_local_torch")
    else:
        with col1:
            uploaded_model = st.file_uploader("上传模型 .joblib", type=["joblib"], key="drug_pred_model")
        with col2:
            local_models = _list_local_models_drug()
            local_path = st.selectbox("或选择本地 models/drug_*.joblib", options=[""] + local_models, key="drug_pred_local")

    smiles = st.text_input("SMILES", value="CCO", key="drug_pred_smiles")
    kv_text = st.text_area(
        "条件参数（每行 key=value；缺失会用训练集中位数填充）",
        value="dose=10\nfreq=2\n",
        height=120,
        key="drug_pred_kv",
    )

    with st.expander("云端预测", expanded=False):
        try:
            env_cloud = _parse_kv_lines(kv_text)
        except ValueError as e:
            st.error(str(e))
            env_cloud = {}

        model_payload = _resolve_model_payload(uploaded_model, local_path if local_path else None, content_type="application/octet-stream")
        if model_payload is None:
            st.info("请上传模型或选择本地模型以提交云端预测。")
        else:
            payload = {
                "model": model_payload,
                "model_type": str(model_type),
                "smiles": str(smiles),
                "env_params": {k: float(v) for k, v in env_cloud.items()},
            }
            _cloud_submit_section(
                "drug_predict",
                payload,
                button_label="提交云端预测",
                key="cloud_drug_pred_btn",
            )

    if st.button("预测", type="primary", key="drug_pred_btn"):
        if uploaded_model is None and not local_path:
            st.error("请上传模型或选择本地模型")
            return

        try:
            env = _parse_kv_lines(kv_text)
        except ValueError as e:
            st.error(str(e))
            return

        if model_type == "torch(pt)":
            try:
                from src.drug.torch_predictor import predict_torch_one  # type: ignore
            except Exception as e:
                st.error(f"Torch 预测不可用：{e}")
                return

            bundle = _load_torch_bundle(uploaded_model, local_path if local_path else None)
            y = predict_torch_one(cast(Any, bundle), smiles=smiles, env_params=env, use_cuda=bool(use_cuda_pred))
        else:
            bundle = _load_drug_bundle(uploaded_model, local_path if local_path else None)
            y = predict_one(cast(Any, bundle), smiles=smiles, env_params=env)

        st.metric("预测值", value=f"{y:.6g}")

        update_feedback_context({
            "event": "predict",
            "model_type": str(model_type),
            "smiles": str(smiles),
            "env_params": {k: float(v) for k, v in env.items()},
            "pred": float(y),
        })

        if getattr(bundle, "env_cols", None):
            resolved = {c: float(env.get(c, bundle.env_medians.get(c, 0.0))) for c in bundle.env_cols}
            st.write("实际使用的条件:", resolved)

        # 差分进化建议环境（药物预测）
        if getattr(bundle, "env_cols", None):
            with st.expander("差分进化建议环境（优化数值型 env）", expanded=False):
                st.markdown("为每个环境变量指定搜索区间，点击开始差分进化搜索（适用于当前模型类型）。")
                env_bounds = []
                cols = bundle.env_cols
                for c in cols:
                    med = float(bundle.env_medians.get(c, 0.0))
                    if med == 0.0:
                        lo_def = -1.0
                        hi_def = 1.0
                    else:
                        lo_def = med * 0.5
                        hi_def = med * 1.5
                    col1, col2 = st.columns(2)
                    with col1:
                        lo = st.number_input(f"{c} 下界", value=float(lo_def), key=f"de_drug_{c}_lo")
                    with col2:
                        hi = st.number_input(f"{c} 上界", value=float(hi_def), key=f"de_drug_{c}_hi")
                    env_bounds.append((float(lo), float(hi)))

                run_de_drug = st.button("运行差分进化建议环境（药物）", key="run_de_drug")
                if run_de_drug:
                    try:
                        if model_type == "torch(pt)":
                            from src.drug.torch_predictor import suggest_env_by_de_torch

                            best_env, best_val = suggest_env_by_de_torch(bundle, smiles=smiles, env_bounds=env_bounds, use_cuda=bool(use_cuda_pred))
                        elif model_type == "transformer(pt)":
                            from src.drug.transformer_predictor import suggest_env_by_de

                            best_env, best_val = suggest_env_by_de(bundle, smiles=smiles, env_bounds=env_bounds)
                        else:
                            from src.drug.predictor import suggest_env_by_de_drug

                            best_env, best_val = suggest_env_by_de_drug(bundle, smiles=smiles, env_bounds=env_bounds)

                        mapped = {c: float(v) for c, v in zip(cols, best_env.tolist())}
                        st.success(f"优化完成，预测值={best_val:.6g}")
                        st.write("建议环境:", mapped)
                    except Exception as e:
                        st.error(f"差分进化失败：{e}")


def drug_screen_ui() -> None:
    st.subheader("批量虚拟筛选（候选 CSV → 预测 CSV）")

    try:
        import rdkit  # type: ignore  # noqa: F401
        from src.drug.predictor import DrugModelBundle, predict_one  # type: ignore
    except Exception as e:
        st.warning(f"药物模块不可用（可能未安装 rdkit）：{e}")
        return

    model_type = st.selectbox(
        "模型类型",
        options=["sklearn(joblib)", "torch(pt)", "transformer(pt)"],
        index=0,
        key="drug_screen_model_type",
    )
    use_cuda_screen = False
    if model_type in {"torch(pt)", "transformer(pt)"}:
        use_cuda_screen = st.checkbox("使用 CUDA（若可用）", value=False, key="drug_screen_cuda")

    col1, col2 = st.columns(2)
    if model_type == "torch(pt)":
        with col1:
            uploaded_model = st.file_uploader("上传模型 .pt", type=["pt"], key="drug_screen_model_torch")
        with col2:
            local_models = _list_local_models_drug_torch()
            local_path = st.selectbox("或选择本地 models/drug_torch*.pt", options=[""] + local_models, key="drug_screen_local_torch")
    elif model_type == "transformer(pt)":
        with col1:
            uploaded_model = st.file_uploader("上传模型 .pt", type=["pt"], key="drug_screen_model_transformer")
        with col2:
            local_models = _list_local_models_drug_transformer()
            local_path = st.selectbox("或选择本地 models/drug_transformer*.pt", options=[""] + local_models, key="drug_screen_local_transformer")
    else:
        with col1:
            uploaded_model = st.file_uploader("上传模型 .joblib", type=["joblib"], key="drug_screen_model")
        with col2:
            local_models = _list_local_models_drug()
            local_path = st.selectbox("或选择本地 models/drug_*.joblib", options=[""] + local_models, key="drug_screen_local")

    uploaded_candidates = st.file_uploader(
        "上传候选（支持多文件）",
        type=_UPLOAD_TYPES,
        accept_multiple_files=True,
        key="drug_candidates_csv",
    )

    generated_df = st.session_state.get("drug_generated_df")
    use_generated = False
    if isinstance(generated_df, pd.DataFrame) and not generated_df.empty:
        use_generated = st.checkbox("使用最近生成结果作为候选", value=False, key="drug_candidates_use_generated")

    if not uploaded_candidates and not use_generated:
        return

    try:
        if use_generated:
            candidates_raw = generated_df.copy()
        else:
            candidates_raw = _load_tables_from_uploads(uploaded_candidates)
    except Exception as e:
        st.error(f"读取文件失败：{e}")
        return

    cand_cfg = _render_preprocess_panel(candidates_raw, "drug_screen")
    candidates = _apply_preprocess(candidates_raw, **cand_cfg)
    _preview_df(candidates, title="候选预览")

    smiles_col = st.selectbox(
        "smiles 列名",
        options=list(candidates.columns),
        index=(list(candidates.columns).index("smiles") if "smiles" in candidates.columns else 0),
        key="drug_candidates_smiles",
    )
    out_col = st.text_input("输出预测列名", value="pred", key="drug_out_col")

    with st.expander("云端筛选", expanded=False):
        model_payload = _resolve_model_payload(uploaded_model, local_path if local_path else None, content_type="application/octet-stream")
        if model_payload is None:
            st.info("请上传模型或选择本地模型以提交云端筛选。")
        else:
            payload = {
                "model": model_payload,
                "model_type": str(model_type),
                "candidates": _cloud_encode_dataframe(candidates),
                "smiles_col": str(smiles_col),
                "out_col": str(out_col),
            }
            _cloud_submit_section(
                "drug_screen",
                payload,
                button_label="提交云端筛选",
                key="cloud_drug_screen_btn",
                download_name="drug_predictions_cloud.csv",
            )

    if st.button("开始筛选", type="primary", key="drug_screen_btn"):
        if uploaded_model is None and not local_path:
            st.error("请上传模型或选择本地模型")
            return

        if model_type == "torch(pt)":
            try:
                from src.drug.torch_predictor import predict_torch_batch  # type: ignore
            except Exception as e:
                st.error(f"Torch 预测不可用：{e}")
                return
            bundle = _load_torch_bundle(uploaded_model, local_path if local_path else None)
        elif model_type == "transformer(pt)":
            try:
                from src.drug.transformer_predictor import predict_transformer_batch  # type: ignore
            except Exception as e:
                st.error(f"Transformer 预测不可用：{e}")
                return
            if uploaded_model is not None:
                bundle = _cached_transformer_bundle_from_bytes(uploaded_model.getvalue())
            else:
                bundle = _cached_transformer_bundle_from_path(local_path, _get_file_mtime(local_path))
        else:
            bundle = _load_drug_bundle(uploaded_model, local_path if local_path else None)

        df = candidates.copy()
        if smiles_col not in df.columns:
            st.error(f"候选 CSV 缺少列: {smiles_col}")
            return

        # Fill missing env columns by medians
        for c in bundle.env_cols:
            if c not in df.columns:
                df[c] = np.nan
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(bundle.env_medians.get(c, 0.0))

        with st.spinner("预测中..."):
            if model_type == "torch(pt)":
                preds = predict_torch_batch(
                    cast(Any, bundle),
                    smiles_list=df[smiles_col].astype(str).tolist(),
                    env_matrix=(df[bundle.env_cols].to_numpy(dtype=np.float32) if bundle.env_cols else None),
                    batch_size=1024,
                    use_cuda=bool(use_cuda_screen),
                )
            elif model_type == "transformer(pt)":
                preds = predict_transformer_batch(
                    cast(Any, bundle),
                    smiles_list=df[smiles_col].astype(str).tolist(),
                    env_matrix=(df[bundle.env_cols].to_numpy(dtype=np.float32) if bundle.env_cols else None),
                    batch_size=512,
                    use_cuda=bool(use_cuda_screen),
                )
            else:
                featurizer = _cached_molecule_featurizer(
                    version=int(getattr(bundle, "featurizer_version", 2)),
                    radius=int(getattr(bundle, "radius", 2)),
                    n_bits=int(getattr(bundle, "n_bits", 2048)),
                )
                mol_x, valids = featurizer.transform_many(df[smiles_col].astype(str).tolist())
                env_x = df[bundle.env_cols].to_numpy(dtype=np.float32) if bundle.env_cols else np.zeros((len(df), 0), dtype=np.float32)
                x = np.concatenate([mol_x, env_x], axis=1)

                preds = np.empty((x.shape[0],), dtype=np.float32)
                chunk = 10000
                for start in range(0, x.shape[0], chunk):
                    end = min(x.shape[0], start + chunk)
                    preds[start:end] = np.asarray(bundle.model.predict(x[start:end]), dtype=np.float32).reshape(-1)
                if valids is not None:
                    invalid_mask = ~valids.astype(bool)
                    preds = preds.astype(float)
                    preds[invalid_mask] = float("nan")

        df[out_col] = np.array(preds, dtype=float)
        st.success("筛选完成")
        _preview_df(df, title="结果预览", max_rows=50)

        update_feedback_context({
            "event": "screen",
            "model_type": str(model_type),
            "n_rows": int(len(df)),
            "out_col": str(out_col),
            "pred_summary": {
                "min": float(np.nanmin(df[out_col].astype(float).to_numpy())) if len(df) else None,
                "max": float(np.nanmax(df[out_col].astype(float).to_numpy())) if len(df) else None,
                "mean": float(np.nanmean(df[out_col].astype(float).to_numpy())) if len(df) else None,
            },
        })

        out_csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("下载预测结果 CSV", data=out_csv, file_name="drug_predictions.csv", mime="text/csv")


def drug_export_embeddings_ui() -> None:
    """Generate and download embeddings from SMILES using SequenceVectorizer."""
    st.subheader("导出向量（SMILES → Embeddings）")

    try:
        from src.representations.sequence_vectorizer import SequenceVectorizer
    except Exception as e:
        st.warning(f"向量化器不可用：{e}")
        return

    uploaded = st.file_uploader(
        "上传候选 CSV（包含 smiles 列）",
        type=_UPLOAD_TYPES,
        accept_multiple_files=True,
        key="drug_export_embs_csv",
    )
    if not uploaded:
        st.info("请上传包含 smiles 列的 CSV 文件")
        return

    try:
        df_raw = _load_tables_from_uploads(uploaded)
    except Exception as e:
        st.error(f"读取文件失败：{e}")
        return

    cfg = _render_preprocess_panel(df_raw, "drug_export_emb")
    df = _apply_preprocess(df_raw, **cfg)
    _preview_df(df, title="数据预览")

    if len(df.columns) == 0:
        st.error("数据中未检测到任何列")
        return

    smiles_col = st.selectbox(
        "smiles 列名",
        options=list(df.columns),
        index=(list(df.columns).index("smiles") if "smiles" in df.columns else 0),
        key="drug_export_smiles_col",
    )

    max_len = st.number_input("最大序列长度", min_value=16, max_value=1024, value=128, step=16, key="drug_export_max_len")
    emb_dim = st.number_input("Embedding 维度", min_value=8, max_value=1024, value=128, step=8, key="drug_export_emb_dim")
    seed = st.number_input("随机种子", min_value=0, max_value=10_000_000, value=42, step=1, key="drug_export_seed")

    if st.button("生成并下载向量", type="primary", key="drug_export_emb_btn"):
        seqs = df[smiles_col].astype(str).tolist()
        sv = SequenceVectorizer(max_len=int(max_len), emb_dim=int(emb_dim), seed=int(seed))
        sv.fit(seqs)
        emb = sv.embed_random(seqs)

        # .npy download
        import io
        buf = io.BytesIO()
        np.save(buf, emb)
        buf.seek(0)
        st.download_button("下载 embeddings (.npy)", data=buf.getvalue(), file_name="embeddings.npy", mime="application/octet-stream")

        # csv with embedding columns
        out_df = df.reset_index(drop=True).copy()
        for i in range(emb.shape[1]):
            out_df[f"emb_{i}"] = emb[:, i]
        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("下载带向量的 CSV", data=csv_bytes, file_name="candidates_with_embeddings.csv", mime="text/csv")


def drug_self_train_ui() -> None:
    st.subheader("自训练（伪标签 + 不确定性筛选）")

    try:
        import rdkit  # type: ignore  # noqa: F401
        from src.drug.featurizer import MoleculeFeatures  # type: ignore
        from src.drug.predictor import DrugModelBundle, build_model, infer_env_cols as drug_infer_env_cols, make_xy, train_bundle  # type: ignore
    except Exception as e:
        st.warning(f"药物模块不可用（可能未安装 rdkit）：{e}")
        return

    labeled_up = st.file_uploader(
        "上传有标注数据（支持多文件）",
        type=_UPLOAD_TYPES,
        accept_multiple_files=True,
        key="drug_st_labeled",
    )
    unlabeled_up = st.file_uploader(
        "上传无标注数据（支持多文件）",
        type=_UPLOAD_TYPES,
        accept_multiple_files=True,
        key="drug_st_unlabeled",
    )

    if not labeled_up or not unlabeled_up:
        st.info("请同时上传有标注与无标注数据")
        return

    try:
        labeled_raw = _load_tables_from_uploads(labeled_up)
        unlabeled_raw = _load_tables_from_uploads(unlabeled_up)
    except Exception as e:
        st.error(f"读取文件失败：{e}")
        return

    labeled_cfg = _render_preprocess_panel(labeled_raw, "drug_st_labeled")
    unlabeled_cfg = _render_preprocess_panel(unlabeled_raw, "drug_st_unlabeled")
    labeled = _apply_preprocess(labeled_raw, **labeled_cfg)
    unlabeled = _apply_preprocess(unlabeled_raw, **unlabeled_cfg)

    _preview_df(labeled, title="标注数据预览")
    _preview_df(unlabeled, title="未标注数据预览")

    smiles_col = st.selectbox("smiles 列名", options=list(labeled.columns), key="drug_st_smiles")
    target_col = st.selectbox("目标列（疗效）", options=[c for c in labeled.columns if c != smiles_col], key="drug_st_target")

    drop_target_na = st.checkbox("过滤目标列为空", value=True, key="drug_st_drop_target_na")
    drop_target_zero = st.checkbox("将目标列=0 视为无标注", value=False, key="drug_st_drop_target_zero")

    auto_env_cols = drug_infer_env_cols(labeled, smiles_col=smiles_col, target_col=target_col, env_cols=None)
    env_cols = st.multiselect(
        "条件列（数值）",
        options=[c for c in labeled.columns if c not in (smiles_col, target_col)],
        default=auto_env_cols,
        key="drug_st_env",
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        featurizer_version = st.selectbox("特征版本", options=[1, 2], index=1, key="drug_st_feat")
    with col2:
        model_name = st.selectbox("模型", options=["hgb", "gbr", "rf", "ridge", "mlp"], index=0, key="drug_st_model")
    with col3:
        n_models = st.number_input("集成模型数", min_value=1, value=5, step=1, key="drug_st_n_models")
    with col4:
        keep_frac = st.slider("保留低不确定性比例", min_value=0.05, max_value=1.0, value=0.5, step=0.05, key="drug_st_keep")

    col5, col6, col7 = st.columns(3)
    with col5:
        test_size = st.slider("验证集比例", min_value=0.1, max_value=0.5, value=0.2, step=0.05, key="drug_st_test")
    with col6:
        seed = st.number_input("随机种子", min_value=0, max_value=10_000_000, value=42, step=1, key="drug_st_seed")
    with col7:
        min_labeled = st.number_input("最少标注样本", min_value=5, value=20, step=1, key="drug_st_min")

    with st.expander("训练优化参数", expanded=False):
        mlp_alpha = st.number_input("MLP 正则强度 (alpha)", min_value=1e-8, max_value=1e-2, value=1e-4, format="%.6f", key="drug_st_mlp_alpha")
        mlp_early = st.checkbox("MLP 早停", value=True, key="drug_st_mlp_early")
        mlp_patience = st.number_input("MLP 早停耐心", min_value=1, max_value=200, value=10, step=1, key="drug_st_mlp_pat")
        ridge_alpha = st.number_input("Ridge 正则 (alpha)", min_value=1e-6, max_value=100.0, value=1.0, step=0.5, key="drug_st_ridge_alpha")
        hgb_l2 = st.number_input("HGB L2 正则", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="drug_st_hgb_l2")

    radius = st.number_input("指纹半径", min_value=1, max_value=6, value=2, step=1, key="drug_st_radius")
    n_bits = st.number_input("指纹位数", min_value=256, max_value=8192, value=2048, step=256, key="drug_st_bits")
    keep_invalid_smiles = st.checkbox("保留无效 SMILES（不建议）", value=False, key="drug_st_keep_invalid")

    model_out = st.text_input("模型保存路径", value="models/drug_self_trained.joblib", key="drug_st_out")
    data_out = st.text_input("合并数据保存路径（可选）", value="data/drug_self_train.csv", key="drug_st_data_out")

    with st.expander("云端自训练", expanded=False):
        payload = {
            "labeled": _cloud_encode_dataframe(labeled),
            "unlabeled": _cloud_encode_dataframe(unlabeled),
            "smiles_col": str(smiles_col),
            "target_col": str(target_col),
            "env_cols": list(env_cols),
            "featurizer_version": int(featurizer_version),
            "model_name": str(model_name),
            "mlp_alpha": float(mlp_alpha),
            "mlp_early_stopping": bool(mlp_early),
            "mlp_patience": int(mlp_patience),
            "ridge_alpha": float(ridge_alpha),
            "hgb_l2": float(hgb_l2),
            "n_models": int(n_models),
            "keep_frac": float(keep_frac),
            "test_size": float(test_size),
            "random_state": int(seed),
            "min_labeled": int(min_labeled),
            "radius": int(radius),
            "n_bits": int(n_bits),
            "keep_invalid_smiles": bool(keep_invalid_smiles),
            "drop_target_na": bool(drop_target_na),
            "drop_target_zero": bool(drop_target_zero),
            "model_out": str(model_out),
            "data_out": str(data_out),
        }
        _cloud_submit_section(
            "drug_self_train",
            payload,
            button_label="提交云端自训练",
            key="cloud_drug_self_train_btn",
            model_id_state_key="cloud_drug_self_model_id",
        )

    if st.button("开始自训练", type="primary", key="drug_st_btn"):
        labeled2 = labeled.copy()
        if drop_target_na:
            labeled2 = labeled2[labeled2[target_col].notna()].copy()
        if drop_target_zero:
            labeled2 = labeled2[pd.to_numeric(labeled2[target_col], errors="coerce") != 0].copy()
        if len(labeled2) < int(min_labeled):
            st.error(f"标注样本太少：{len(labeled2)} < min_labeled={int(min_labeled)}")
            return

        env_cols2 = drug_infer_env_cols(labeled2, smiles_col=smiles_col, target_col=target_col, env_cols=list(env_cols))

        x_l, y_l, env_medians, _feature_names, valids_l = make_xy(
            labeled2,
            smiles_col=smiles_col,
            target_col=target_col,
            env_cols=list(env_cols2),
            featurizer=MoleculeFeatures(version=int(featurizer_version), radius=int(radius), n_bits=int(n_bits)),
            env_medians=None,
        )

        keep_l = valids_l.astype(bool)
        x_l = x_l[keep_l]
        y_l = y_l[keep_l]

        def _make_x_only_drug(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
            featurizer = MoleculeFeatures(version=int(featurizer_version), radius=int(radius), n_bits=int(n_bits))
            mol_x, valids = featurizer.transform_many(df[smiles_col].astype(str).tolist())
            if env_cols2:
                env_df = df[list(env_cols2)].copy()
                for c in env_cols2:
                    env_df[c] = pd.to_numeric(env_df[c], errors="coerce").astype(float)
                    env_df[c] = env_df[c].fillna(float(env_medians.get(c, float(env_df[c].median()))))
                env_x = env_df.to_numpy(dtype=np.float32)
            else:
                env_x = np.zeros((len(df), 0), dtype=np.float32)
            x = np.concatenate([mol_x, env_x], axis=1).astype(np.float32)
            return x, valids

        x_u, valids_u = _make_x_only_drug(unlabeled)

        rng = np.random.default_rng(int(seed))
        preds = []
        for i in range(int(n_models)):
            idx = rng.integers(low=0, high=len(x_l), size=len(x_l), endpoint=False)
            model = build_model(
                model_name=str(model_name),
                random_state=int(seed) + i,  # type: ignore[arg-type]
                mlp_alpha=float(mlp_alpha),
                mlp_early_stopping=bool(mlp_early),
                mlp_patience=int(mlp_patience),
                ridge_alpha=float(ridge_alpha),
                hgb_l2=float(hgb_l2),
            )
            model.fit(x_l[idx], y_l[idx])
            preds.append(np.asarray(model.predict(x_u), dtype=np.float32).reshape(-1))

        pred_mat = np.stack(preds, axis=0)
        mu = pred_mat.mean(axis=0)
        sigma = pred_mat.std(axis=0)

        pool_idx = np.where(valids_u.astype(bool))[0] if not bool(keep_invalid_smiles) else np.arange(len(unlabeled))
        if len(pool_idx) == 0:
            st.error("未标注数据中没有可用 SMILES")
            return

        k = max(1, int(round(len(pool_idx) * float(keep_frac))))
        best = pool_idx[np.argsort(sigma[pool_idx])[:k]]
        pseudo = unlabeled.iloc[best].copy()
        pseudo[target_col] = mu[best]
        pseudo["pseudo_uncertainty_std"] = sigma[best]
        pseudo["pseudo_labeled"] = True

        labeled2["pseudo_labeled"] = False
        combined = pd.concat([labeled2, pseudo], axis=0, ignore_index=True)

        with st.spinner("训练中..."):
            bundle, metrics = train_bundle(
                combined,
                smiles_col=smiles_col,
                target_col=target_col,
                env_cols=list(env_cols2),
                model_name=str(model_name),  # type: ignore[arg-type]
                test_size=float(test_size),
                random_state=int(seed),
                featurizer_version=int(featurizer_version),
                radius=int(radius),
                n_bits=int(n_bits),
                drop_invalid_smiles=(not bool(keep_invalid_smiles)),
                mlp_alpha=float(mlp_alpha),
                mlp_early_stopping=bool(mlp_early),
                mlp_patience=int(mlp_patience),
                ridge_alpha=float(ridge_alpha),
                hgb_l2=float(hgb_l2),
            )

        out_path = Path(model_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, out_path)

        if data_out:
            dp = Path(data_out)
            dp.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(dp, index=False)
            st.write("已保存合并数据:", str(dp))

        st.success("自训练完成")
        render_training_result_panel(metrics, feature_names=metrics.get("feature_names", None))
        st.write(f"伪标签样本数: {len(pseudo)}")

        buf = io.BytesIO()
        joblib.dump(bundle, buf)
        st.download_button(
            "下载模型文件",
            data=buf.getvalue(),
            file_name=out_path.name,
            mime="application/octet-stream",
        )


def drug_plot_ui() -> None:
    st.subheader("绘图（回归诊断）")

    try:
        import rdkit  # type: ignore  # noqa: F401
        from src.drug.featurizer import MoleculeFeatures  # type: ignore
        from src.drug.predictor import DrugModelBundle, make_xy as drug_make_xy  # type: ignore
    except Exception as e:
        st.warning(f"药物模块不可用（可能未安装 rdkit）：{e}")
        return

    col1, col2 = st.columns(2)
    with col1:
        uploaded_model = st.file_uploader("上传模型 .joblib", type=["joblib"], key="drug_plot_model")
    with col2:
        local_models = _list_local_models_drug()
        local_path = st.selectbox("或选择本地 models/drug_*.joblib", options=[""] + local_models, key="drug_plot_local")

    data_up = st.file_uploader(
        "上传评估数据（支持多文件）",
        type=_UPLOAD_TYPES,
        accept_multiple_files=True,
        key="drug_plot_data",
    )
    if not data_up:
        return

    if uploaded_model is None and not local_path:
        st.error("请上传模型或选择本地模型")
        return

    if uploaded_model is not None:
        buf = io.BytesIO(uploaded_model.getvalue())
        bundle: DrugModelBundle = joblib.load(buf)
    else:
        bundle = joblib.load(local_path)
    try:
        df_raw = _load_tables_from_uploads(data_up)
    except Exception as e:
        st.error(f"读取文件失败：{e}")
        return

    plot_cfg = _render_preprocess_panel(df_raw, "drug_plot")
    df = _apply_preprocess(df_raw, **plot_cfg)

    if bundle.smiles_col not in df.columns:
        st.error(f"缺少 smiles 列: {bundle.smiles_col}")
        return
    if bundle.target_col not in df.columns:
        st.error(f"缺少 target 列: {bundle.target_col}")
        return

    out_dir = Path(st.text_input("输出目录", value="plots/drug", key="drug_plot_out_dir"))
    prefix = st.text_input("文件前缀", value="drug", key="drug_plot_prefix")
    title = st.text_input("标题（可选）", value="drug regression", key="drug_plot_title")
    keep_invalid_smiles = st.checkbox("保留无效 SMILES（不建议）", value=False, key="drug_plot_keep_invalid")

    with st.expander("云端绘图", expanded=False):
        model_payload = _resolve_model_payload(uploaded_model, local_path if local_path else None, content_type="application/octet-stream")
        if model_payload is None:
            st.info("请上传模型或选择本地模型以提交云端绘图。")
        else:
            payload = {
                "model": model_payload,
                "data": _cloud_encode_dataframe(df),
                "out_dir": str(out_dir),
                "prefix": str(prefix),
                "title": str(title),
                "keep_invalid_smiles": bool(keep_invalid_smiles),
            }
            _cloud_submit_section(
                "drug_plot",
                payload,
                button_label="提交云端绘图",
                key="cloud_drug_plot_btn",
            )

    if st.button("生成诊断图", type="primary", key="drug_plot_btn"):
        x, y, _, _, valids = drug_make_xy(
            df,
            smiles_col=bundle.smiles_col,
            target_col=bundle.target_col,
            env_cols=list(bundle.env_cols),
            featurizer=MoleculeFeatures(
                version=int(getattr(bundle, "featurizer_version", 2) or 2),
                radius=int(getattr(bundle, "radius", 2) or 2),
                n_bits=int(getattr(bundle, "n_bits", 2048) or 2048),
            ),
            env_medians=dict(bundle.env_medians),
        )

        if not bool(keep_invalid_smiles):
            keep = valids.astype(bool)
            x = x[keep]
            y = y[keep]

        y_pred = np.asarray(bundle.model.predict(x), dtype=float).reshape(-1)
        save_regression_diagnostic_plots(
            y_true=y,
            y_pred=y_pred,
            out_dir=out_dir,
            prefix=str(prefix),
            title=str(title),
        )

        st.success("已生成诊断图")
        _render_plot_images(out_dir=out_dir, prefix=str(prefix))


def drug_pubchem_crawl_train_ui() -> None:
    st.subheader("爬虫自主训练（PubChem BioAssay proxy 标签）")
    st.caption(
        "说明：这里的标签是基于 PubChem BioAssay 汇总的代理标签（activity_score），"
        "不是临床真实疗效。适合快速生成一个可跑通的基线模型。"
    )

    # Crawl dependencies only
    try:
        from src.drug.crawler import (
            crawl_pubchem_activity_dataset,
            crawl_docking_training_dataset,
            crawl_multiscale_training_dataset,
            PUBCHEM_PROPERTY_FIELDS,
        )
        from src.common.dataset_fetch import concat_tables
    except Exception as e:
        st.error(f"无法导入爬虫模块：{e}")
        return

    def _parse_cid_text(text: str) -> List[int]:
        if not text:
            return []
        tokens = re.split(r"[\s,;]+", str(text).strip())
        out: List[int] = []
        for t in tokens:
            if not t:
                continue
            try:
                v = int(float(t))
                if v > 0:
                    out.append(v)
            except Exception:
                continue
        return out

    crawl_mode = st.selectbox(
        "抓取模式",
        options=["PubChem CID 区间", "PubChem CID 列表", "URL/本地表格合并", "分子对接数据（表格合并）", "多尺度训练数据（表格合并）"],
        index=0,
        key="drug_crawl_mode",
    )

    # defaults for shared params
    start_cid = 1
    n = 300
    min_total = 1
    sleep_s = 0.2
    timeout_s = 20.0
    cache_dir = "data/cache/pubchem"
    indication_label = "不限"
    min_active = 0
    treat_zero_unlabeled = False
    max_workers = 4
    rate_limit = 5.0
    weighted = False
    include_outcomes = False
    normalize_smiles = False
    include_properties = False
    include_synonyms = False
    property_fields: List[str] = []
    cid_text = ""
    sources_text = ""
    docking_smiles_col = ""
    docking_protein_seq_col = ""
    docking_protein_pdb_col = ""
    docking_score_col = ""
    docking_pocket_col = ""
    multiscale_smiles_col = ""
    multiscale_target_col = ""
    multiscale_d_col = ""
    multiscale_vmax_col = ""
    multiscale_km_col = ""
    multiscale_default_D = 0.1
    multiscale_default_Vmax = 0.5
    multiscale_default_Km = 0.1

    if crawl_mode.startswith("PubChem"):
        col1, col2, col3 = st.columns(3)
        with col1:
            if crawl_mode == "PubChem CID 区间":
                start_cid = st.number_input("start_cid", min_value=1, value=1, step=1)
                n = st.number_input("n（CID 数量）", min_value=10, value=300, step=50)
            else:
                cid_text = st.text_area(
                    "CID 列表（逗号/空格/换行分隔）",
                    value="",
                    height=120,
                    key="drug_crawl_cid_text",
                )
        with col2:
            min_total = st.number_input("min_total（Active+Inactive）", min_value=1, value=1, step=1)
            sleep_s = st.number_input("sleep（秒/请求）", min_value=0.0, value=0.2, step=0.1, format="%.2f")
        with col3:
            timeout_s = st.number_input("timeout（秒）", min_value=1.0, value=20.0, step=1.0, format="%.1f")
            cache_dir = st.text_input("cache_dir", value="data/cache/pubchem")

        st.markdown("**有效病症与标注处理**")
        indication_options = ["不限", "肿瘤", "感染", "炎症", "免疫", "神经", "心血管", "代谢", "其他（自定义）"]
        indication_choice = st.selectbox("有效病症（用于元数据标记）", options=indication_options, index=0)
        indication_custom = ""
        if indication_choice.startswith("其他"):
            indication_custom = st.text_input("自定义病症", value="")
        indication_label = indication_custom.strip() if indication_choice.startswith("其他") else indication_choice

        min_active = st.number_input("min_active（最低 Active 数）", min_value=0, value=0, step=1)
        treat_zero_unlabeled = st.checkbox("将 activity_score=0 视为无标注", value=False)

        st.markdown("**高级选项**")
        col4, col5 = st.columns(2)
        with col4:
            max_workers = st.number_input("并发线程数", min_value=1, value=4, step=1)
            rate_limit = st.number_input("全局速率（requests/sec）", min_value=0.1, value=5.0, step=0.1, format="%.1f")
        with col5:
            weighted = st.checkbox("按 assay 权重加权得分", value=False)
            include_outcomes = st.checkbox("保留 outcome 细分（输出列）", value=False)
        normalize_smiles = st.checkbox("启用 SMILES 规范化（RDKit，可选）", value=False)

        st.markdown("**字段扩展**")
        col6a, col6b = st.columns(2)
        with col6a:
            include_properties = st.checkbox("扩展 PubChem 属性字段", value=False)
        with col6b:
            include_synonyms = st.checkbox("扩展 PubChem 同义名", value=False)
        if include_properties:
            default_props = [
                "MolecularFormula",
                "MolecularWeight",
                "XLogP",
                "TopologicalPolarSurfaceArea",
                "HBondDonorCount",
                "HBondAcceptorCount",
                "RotatableBondCount",
                "IUPACName",
                "InChIKey",
            ]
            property_fields = st.multiselect(
                "属性字段（可多选）",
                options=list(PUBCHEM_PROPERTY_FIELDS),
                default=[p for p in default_props if p in PUBCHEM_PROPERTY_FIELDS],
                key="drug_pubchem_property_fields",
            )
    else:
        sources_text = st.text_area(
            "数据源（每行一个 URL 或本地路径）",
            value="",
            height=120,
            key="drug_crawl_sources",
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            cache_dir = st.text_input("cache_dir", value="data/cache/http")
        with col2:
            timeout_s = st.number_input("timeout（秒）", min_value=1.0, value=30.0, step=1.0, format="%.1f")
        with col3:
            sleep_s = st.number_input("sleep（秒/请求）", min_value=0.0, value=0.2, step=0.1, format="%.2f")

        if crawl_mode.startswith("分子对接"):
            st.markdown("**列名映射（可选）**")
            c1, c2 = st.columns(2)
            with c1:
                docking_smiles_col = st.text_input("ligand_smiles 列名", value="", key="docking_smiles_col")
                docking_protein_seq_col = st.text_input("protein_sequence 列名", value="", key="docking_protein_seq_col")
                docking_protein_pdb_col = st.text_input("protein_pdb 列名", value="", key="docking_protein_pdb_col")
            with c2:
                docking_score_col = st.text_input("binding_score 列名", value="", key="docking_score_col")
                docking_pocket_col = st.text_input("pocket_path 列名", value="", key="docking_pocket_col")
            normalize_smiles = st.checkbox("启用 SMILES 规范化（RDKit，可选）", value=False, key="docking_norm_smiles")

        if crawl_mode.startswith("多尺度"):
            st.markdown("**列名映射（可选）**")
            c1, c2 = st.columns(2)
            with c1:
                multiscale_smiles_col = st.text_input("smiles 列名", value="", key="ms_crawl_smiles_col")
                multiscale_target_col = st.text_input("target 列名", value="", key="ms_crawl_target_col")
                multiscale_d_col = st.text_input("D 列名", value="", key="ms_crawl_d_col")
            with c2:
                multiscale_vmax_col = st.text_input("Vmax 列名", value="", key="ms_crawl_vmax_col")
                multiscale_km_col = st.text_input("Km 列名", value="", key="ms_crawl_km_col")
            st.markdown("**默认物理参数（缺失时填充）**")
            d1, d2, d3 = st.columns(3)
            with d1:
                multiscale_default_D = st.number_input("默认 D", min_value=0.0, value=0.1, step=0.01, key="ms_crawl_default_D")
            with d2:
                multiscale_default_Vmax = st.number_input("默认 Vmax", min_value=0.0, value=0.5, step=0.05, key="ms_crawl_default_Vmax")
            with d3:
                multiscale_default_Km = st.number_input("默认 Km", min_value=0.0, value=0.1, step=0.01, key="ms_crawl_default_Km")
            normalize_smiles = st.checkbox("启用 SMILES 规范化（RDKit，可选）", value=False, key="ms_crawl_norm_smiles")

    st.markdown("**合并与去重**")
    col6, col7 = st.columns(2)
    with col6:
        dedupe_smiles = st.checkbox("按 smiles 去重", value=True)
        dedupe_cid = st.checkbox("按 cid 去重", value=False)
    with col7:
        merge_existing = st.checkbox("与已有 CSV 合并", value=False)
        merge_path = st.text_input("已有 CSV 路径", value="", disabled=(not merge_existing))

    st.markdown("**自动化训练**")
    col8, col9 = st.columns(2)
    with col8:
        auto_train = st.checkbox("抓取后自动训练", value=False)
    with col9:
        auto_min_samples = st.number_input("最少标注样本（自动训练）", min_value=5, value=10, step=1)

    st.divider()
    st.write("### 1) 抓取数据集")
    data_out_default = "data/pubchem_activity.csv"
    if crawl_mode.startswith("分子对接"):
        data_out_default = "data/docking_crawl_out.csv"
    elif crawl_mode.startswith("多尺度"):
        data_out_default = "data/multiscale_crawl_out.csv"
    data_out = st.text_input("保存 CSV 路径（可选）", value=data_out_default, key="drug_crawl_out")

    with st.expander("云端抓取", expanded=False):
        if crawl_mode != "PubChem CID 区间":
            st.info("云端抓取当前仅支持 PubChem CID 区间模式。")
        else:
            payload = {
                "start_cid": int(start_cid),
                "n": int(n),
                "min_total": int(min_total),
                "sleep": float(sleep_s),
                "timeout": float(timeout_s),
                "cache_dir": str(cache_dir),
                "indication": str(indication_label),
                "min_active": int(min_active),
                "treat_zero_unlabeled": bool(treat_zero_unlabeled),
                "max_workers": int(max_workers),
                "rate_limit": float(rate_limit),
                "weighted": bool(weighted),
                "include_outcomes": bool(include_outcomes),
                "normalize_smiles": bool(normalize_smiles),
                "include_properties": bool(include_properties),
                "property_fields": list(property_fields),
                "include_synonyms": bool(include_synonyms),
                "data_out": str(data_out),
            }
            _cloud_submit_section(
                "drug_pubchem_crawl",
                payload,
                button_label="提交云端抓取",
                key="cloud_drug_pubchem_crawl_btn",
                download_name="pubchem_activity_cloud.csv",
            )

    if st.button("开始抓取", type="primary", key="drug_pubchem_crawl"):
        prog = st.progress(0)
        status = st.empty()

        def _progress_cb(done: int, tot: int) -> None:
            try:
                prog.progress(int(done / tot * 100))
                status.write(f"已完成 {done}/{tot}")
            except Exception:
                pass

        df = pd.DataFrame()
        if crawl_mode == "URL/本地表格合并":
            sources = [s.strip() for s in sources_text.splitlines() if s.strip()]
            if not sources:
                st.error("请至少提供一个数据源")
                return
            with st.spinner("抓取/读取中..."):
                df = concat_tables(
                    sources,
                    cache_dir=str(cache_dir),
                    timeout=float(timeout_s),
                    sleep_seconds=float(sleep_s),
                    headers={"User-Agent": "drug-urlcsv/1.0 (research; contact: local)"},
                )
        elif crawl_mode.startswith("分子对接"):
            sources = [s.strip() for s in sources_text.splitlines() if s.strip()]
            if not sources:
                st.error("请至少提供一个数据源")
                return
            with st.spinner("抓取/读取中..."):
                df = crawl_docking_training_dataset(
                    sources=sources,
                    cache_dir=str(cache_dir),
                    timeout=float(timeout_s),
                    sleep_seconds=float(sleep_s),
                    ligand_smiles_col=str(docking_smiles_col) if docking_smiles_col else None,
                    protein_seq_col=str(docking_protein_seq_col) if docking_protein_seq_col else None,
                    protein_pdb_col=str(docking_protein_pdb_col) if docking_protein_pdb_col else None,
                    binding_score_col=str(docking_score_col) if docking_score_col else None,
                    pocket_path_col=str(docking_pocket_col) if docking_pocket_col else None,
                    normalize_smiles=bool(normalize_smiles),
                    drop_invalid=True,
                )
        elif crawl_mode.startswith("多尺度"):
            sources = [s.strip() for s in sources_text.splitlines() if s.strip()]
            if not sources:
                st.error("请至少提供一个数据源")
                return
            with st.spinner("抓取/读取中..."):
                df = crawl_multiscale_training_dataset(
                    sources=sources,
                    cache_dir=str(cache_dir),
                    timeout=float(timeout_s),
                    sleep_seconds=float(sleep_s),
                    smiles_col=str(multiscale_smiles_col) if multiscale_smiles_col else None,
                    target_col=str(multiscale_target_col) if multiscale_target_col else None,
                    d_col=str(multiscale_d_col) if multiscale_d_col else None,
                    vmax_col=str(multiscale_vmax_col) if multiscale_vmax_col else None,
                    km_col=str(multiscale_km_col) if multiscale_km_col else None,
                    default_D=float(multiscale_default_D),
                    default_Vmax=float(multiscale_default_Vmax),
                    default_Km=float(multiscale_default_Km),
                    normalize_smiles=bool(normalize_smiles),
                    drop_invalid=True,
                )
        else:
            indication_filter = None if (not indication_label or indication_label == "不限") else indication_label
            cid_list = _parse_cid_text(cid_text) if crawl_mode == "PubChem CID 列表" else None
            if crawl_mode == "PubChem CID 列表" and not cid_list:
                st.error("CID 列表为空，请输入至少一个 CID")
                return
            total = len(cid_list) if cid_list is not None else int(n)
            with st.spinner("抓取中..."):
                df = crawl_pubchem_activity_dataset(
                    start_cid=int(start_cid),
                    n=int(total),
                    cids=cid_list,
                    sleep_seconds=float(sleep_s),
                    min_total_outcomes=int(min_total),
                    min_active=int(min_active),
                    treat_zero_unlabeled=bool(treat_zero_unlabeled),
                    drop_invalid=True,
                    cache_dir=str(cache_dir),
                    timeout=float(timeout_s),
                    user_agent="drug-efficacy-crawler/1.0 (frontend)",
                    max_workers=int(max_workers),
                    rate_limit=float(rate_limit),
                    retries=3,
                    backoff_factor=0.5,
                    indication=indication_filter,
                    weighted=bool(weighted),
                    include_outcome_breakdown=bool(include_outcomes),
                    normalize_smiles=bool(normalize_smiles),
                    include_properties=bool(include_properties),
                    property_fields=list(property_fields) if property_fields else None,
                    include_synonyms=bool(include_synonyms),
                    progress_callback=_progress_cb,
                )

        if merge_existing and merge_path:
            try:
                exist_df = pd.read_csv(merge_path)
                df = pd.concat([exist_df, df], axis=0, ignore_index=True)
            except Exception as e:
                st.warning(f"合并已有 CSV 失败：{e}")

        if dedupe_smiles:
            if "smiles" in df.columns:
                df = df.drop_duplicates(subset=["smiles"]).copy()
            elif "ligand_smiles" in df.columns:
                df = df.drop_duplicates(subset=["ligand_smiles"]).copy()
        if dedupe_cid and "cid" in df.columns:
            df = df.drop_duplicates(subset=["cid"]).copy()

        # normalize output for UI
        if "activity_score" in df.columns:
            df["has_label"] = df["activity_score"].notna()
        if crawl_mode.startswith("PubChem"):
            df["indication"] = str(indication_label)

        if crawl_mode.startswith("分子对接"):
            st.session_state["docking_crawl_df"] = df
        elif crawl_mode.startswith("多尺度"):
            st.session_state["multiscale_crawl_df"] = df
        else:
            st.session_state["drug_pubchem_df"] = df

        usable = int(df["activity_score"].notna().sum()) if "activity_score" in df.columns else 0
        st.success(f"抓取完成：共 {len(df)} 行，可用标注 {usable} 行")
        _preview_df(df, title="抓取结果预览", max_rows=50)

        if data_out:
            out_path = Path(data_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            st.write("已保存到:", str(out_path))

        st.download_button(
            "下载 CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=Path(data_out).name if data_out else "pubchem_activity.csv",
            mime="text/csv",
        )

        if auto_train:
            st.info("已启用抓取后自动训练，将使用下方训练配置。")

    st.divider()
    st.write("### 2) 用抓取的数据训练模型")

    featurizer_version = st.selectbox("特征版本", options=[1, 2], index=1, key="drug_pubchem_feat_v")
    model_name = st.selectbox("模型", options=["hgb", "gbr", "rf", "ridge", "mlp"], index=0, key="drug_pubchem_model")
    with st.expander("训练优化参数", expanded=False):
        mlp_alpha = st.number_input("MLP 正则强度 (alpha)", min_value=1e-8, max_value=1e-2, value=1e-4, format="%.6f", key="drug_pubchem_mlp_alpha")
        mlp_early = st.checkbox("MLP 早停", value=True, key="drug_pubchem_mlp_early")
        mlp_patience = st.number_input("MLP 早停耐心", min_value=1, max_value=200, value=10, step=1, key="drug_pubchem_mlp_pat")
        ridge_alpha = st.number_input("Ridge 正则 (alpha)", min_value=1e-6, max_value=100.0, value=1.0, step=0.5, key="drug_pubchem_ridge_alpha")
        hgb_l2 = st.number_input("HGB L2 正则", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="drug_pubchem_hgb_l2")
    test_size = st.slider("验证集比例", min_value=0.1, max_value=0.5, value=0.2, step=0.05, key="drug_pubchem_test")
    seed = st.number_input("随机种子", min_value=0, max_value=10_000_000, value=42, step=1, key="drug_pubchem_seed")
    model_out = st.text_input("模型保存路径", value="models/drug_pubchem_activity.joblib", key="drug_pubchem_out")

    with st.expander("云端训练", expanded=False):
        df_cloud = st.session_state.get("drug_pubchem_df")
        if df_cloud is None or not isinstance(df_cloud, pd.DataFrame):
            st.info("请先抓取数据集后再提交云端训练。")
        else:
            payload = {
                "data": _cloud_encode_dataframe(df_cloud),
                "model_name": str(model_name),
                "mlp_alpha": float(mlp_alpha),
                "mlp_early_stopping": bool(mlp_early),
                "mlp_patience": int(mlp_patience),
                "ridge_alpha": float(ridge_alpha),
                "hgb_l2": float(hgb_l2),
                "test_size": float(test_size),
                "random_state": int(seed),
                "featurizer_version": int(featurizer_version),
                "model_out": str(model_out),
            }
            _cloud_submit_section(
                "drug_pubchem_train",
                payload,
                button_label="提交云端训练",
                key="cloud_drug_pubchem_train_btn",
                model_id_state_key="cloud_drug_pubchem_model_id",
            )

    if st.button("训练并导出模型", type="primary", key="drug_pubchem_train"):
        df = st.session_state.get("drug_pubchem_df")
        if df is None or not isinstance(df, pd.DataFrame):
            st.error("请先抓取数据集（或先在本页运行一次抓取）")
            return

        train_df = df[df["activity_score"].notna()].copy()
        train_df = train_df[train_df["n_total"].astype(float) >= float(min_total)]
        train_df = train_df[train_df["smiles"].notna()].copy()

        st.write(f"训练用样本数：{len(train_df)}")
        if len(train_df) < 10:
            st.error("可用标注样本太少。建议增大 n 或降低 min_total。")
            return

        # Training needs RDKit
        try:
            import rdkit  # type: ignore  # noqa: F401
            from src.drug.predictor import train_bundle  # type: ignore
        except Exception as e:
            st.error(f"无法训练（可能未安装 rdkit）：{e}")
            return

        with st.spinner("训练中..."):
            bundle, metrics = train_bundle(
                train_df,
                smiles_col="smiles",
                target_col="activity_score",
                env_cols=[],
                model_name=model_name,  # type: ignore[arg-type]
                test_size=float(test_size),
                random_state=int(seed),
                featurizer_version=int(featurizer_version),
                mlp_alpha=float(mlp_alpha),
                mlp_early_stopping=bool(mlp_early),
                mlp_patience=int(mlp_patience),
                ridge_alpha=float(ridge_alpha),
                hgb_l2=float(hgb_l2),
            )

        st.success("训练完成")
        st.json(metrics)

        out_path = Path(model_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, out_path)
        st.write("已保存到:", str(out_path))

        buf = io.BytesIO()
        joblib.dump(bundle, buf)
        st.download_button(
            "下载模型文件",
            data=buf.getvalue(),
            file_name=out_path.name,
            mime="application/octet-stream",
        )



def drug_advanced_crawl_ui() -> None:
    st.subheader("高级数据对接（爬虫 → 训练数据）")
    st.caption("统一入口：为多尺度分析、分子对接、分子生成提供可直接训练/使用的爬虫数据，并写入对应页面的会话状态。")

    try:
        from src.drug.crawler import (
            crawl_docking_training_dataset,
            crawl_multiscale_training_dataset,
            crawl_pubchem_activity_dataset,
            PUBCHEM_PROPERTY_FIELDS,
        )
        from src.common.dataset_fetch import concat_tables
    except Exception as e:
        st.error(f"无法导入爬虫模块：{e}")
        return

    task = st.radio(
        "目标模块",
        ["多尺度分析训练集", "分子对接训练集", "分子生成种子/打分"],
        index=0,
        horizontal=True,
        key="adv_crawl_task",
    )

    col_base1, col_base2, col_base3 = st.columns(3)
    with col_base1:
        cache_dir = st.text_input("cache_dir", value="data/cache/http", key="adv_crawl_cache")
    with col_base2:
        timeout_s = st.number_input("timeout（秒）", min_value=1.0, value=30.0, step=1.0, key="adv_crawl_timeout")
    with col_base3:
        sleep_s = st.number_input("sleep（秒/请求）", min_value=0.0, value=0.2, step=0.1, format="%.2f", key="adv_crawl_sleep")

    dedupe_smiles = st.checkbox("按 smiles 去重", value=True, key="adv_crawl_dedupe")
    normalize_smiles = st.checkbox("启用 SMILES 规范化（RDKit，可选）", value=False, key="adv_crawl_norm")

    if task == "多尺度分析训练集":
        st.markdown("#### 数据源（表格合并）")
        sources_text = st.text_area("URL 或本地路径（每行一个）", value="", height=100, key="adv_ms_sources")

        st.markdown("#### 列名映射与默认值")
        c1, c2 = st.columns(2)
        with c1:
            ms_smiles_col = st.text_input("smiles 列名", value="smiles", key="adv_ms_smiles_col")
            ms_target_col = st.text_input("target 列名（可选）", value="target", key="adv_ms_target_col")
            ms_d_col = st.text_input("D 列名（可选）", value="D", key="adv_ms_d_col")
        with c2:
            ms_vmax_col = st.text_input("Vmax 列名（可选）", value="Vmax", key="adv_ms_vmax_col")
            ms_km_col = st.text_input("Km 列名（可选）", value="Km", key="adv_ms_km_col")
            ms_default_d = st.number_input("缺省 D", min_value=0.0, value=0.1, step=0.01, key="adv_ms_default_d")
            ms_default_vmax = st.number_input("缺省 Vmax", min_value=0.0, value=0.5, step=0.05, key="adv_ms_default_vmax")
            ms_default_km = st.number_input("缺省 Km", min_value=0.0, value=0.1, step=0.01, key="adv_ms_default_km")

        data_out = st.text_input("保存 CSV 路径", value="data/multiscale_crawl_out.csv", key="adv_ms_out")

        if st.button("抓取并推送到多尺度模块", type="primary", key="adv_ms_run"):
            sources = [s.strip() for s in sources_text.splitlines() if s.strip()]
            if not sources:
                st.error("请至少提供一个数据源")
                return
            with st.spinner("抓取/合并中..."):
                df = crawl_multiscale_training_dataset(
                    sources=sources,
                    cache_dir=str(cache_dir),
                    timeout=float(timeout_s),
                    sleep_seconds=float(sleep_s),
                    smiles_col=str(ms_smiles_col) if ms_smiles_col else None,
                    target_col=str(ms_target_col) if ms_target_col else None,
                    d_col=str(ms_d_col) if ms_d_col else None,
                    vmax_col=str(ms_vmax_col) if ms_vmax_col else None,
                    km_col=str(ms_km_col) if ms_km_col else None,
                    default_D=float(ms_default_d),
                    default_Vmax=float(ms_default_vmax),
                    default_Km=float(ms_default_km),
                    normalize_smiles=bool(normalize_smiles),
                    drop_invalid=True,
                )

            if dedupe_smiles and "smiles" in df.columns:
                df = df.drop_duplicates(subset=["smiles"]).copy()

            st.session_state["multiscale_crawl_df"] = df
            st.success(f"完成：{len(df)} 行，已写入多尺度页面的会话缓存，可直接在多尺度分析页选择“来自爬虫”数据。")
            _preview_df(df, title="多尺度训练数据预览", max_rows=50)

            if data_out:
                p = Path(data_out)
                p.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(p, index=False)
                st.write("已保存到:", str(p))

            st.download_button(
                "下载 CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=Path(data_out).name if data_out else "multiscale_crawl_out.csv",
                mime="text/csv",
                key="adv_ms_download",
            )

    elif task == "分子对接训练集":
        st.markdown("#### 数据源（表格合并）")
        sources_text = st.text_area("URL 或本地路径（每行一个）", value="", height=100, key="adv_dock_sources")

        st.markdown("#### 列名映射（可选）")
        c1, c2 = st.columns(2)
        with c1:
            dock_smiles_col = st.text_input("ligand_smiles 列名", value="ligand_smiles", key="adv_dock_smiles_col")
            dock_seq_col = st.text_input("protein_sequence 列名", value="protein_sequence", key="adv_dock_seq_col")
            dock_pdb_col = st.text_input("protein_pdb 列名", value="protein_pdb", key="adv_dock_pdb_col")
        with c2:
            dock_score_col = st.text_input("binding_score 列名", value="binding_score", key="adv_dock_score_col")
            dock_pocket_col = st.text_input("pocket_path 列名", value="pocket_path", key="adv_dock_pocket_col")

        data_out = st.text_input("保存 CSV 路径", value="data/docking_crawl_out.csv", key="adv_dock_out")

        if st.button("抓取并推送到对接模块", type="primary", key="adv_dock_run"):
            sources = [s.strip() for s in sources_text.splitlines() if s.strip()]
            if not sources:
                st.error("请至少提供一个数据源")
                return
            with st.spinner("抓取/合并中..."):
                df = crawl_docking_training_dataset(
                    sources=sources,
                    cache_dir=str(cache_dir),
                    timeout=float(timeout_s),
                    sleep_seconds=float(sleep_s),
                    ligand_smiles_col=str(dock_smiles_col) if dock_smiles_col else None,
                    protein_seq_col=str(dock_seq_col) if dock_seq_col else None,
                    protein_pdb_col=str(dock_pdb_col) if dock_pdb_col else None,
                    binding_score_col=str(dock_score_col) if dock_score_col else None,
                    pocket_path_col=str(dock_pocket_col) if dock_pocket_col else None,
                    normalize_smiles=bool(normalize_smiles),
                    drop_invalid=True,
                )

            if dedupe_smiles:
                if "ligand_smiles" in df.columns:
                    df = df.drop_duplicates(subset=["ligand_smiles"]).copy()
                elif "smiles" in df.columns:
                    df = df.drop_duplicates(subset=["smiles"]).copy()

            st.session_state["docking_crawl_df"] = df
            st.success(f"完成：{len(df)} 行，已写入分子对接页面的会话缓存，可在对接训练/筛选页直接使用。")
            _preview_df(df, title="分子对接训练数据预览", max_rows=50)

            if data_out:
                p = Path(data_out)
                p.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(p, index=False)
                st.write("已保存到:", str(p))

            st.download_button(
                "下载 CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=Path(data_out).name if data_out else "docking_crawl_out.csv",
                mime="text/csv",
                key="adv_dock_download",
            )

    else:
        st.markdown("#### 采集方式")
        seed_mode = st.selectbox(
            "来源",
            options=["PubChem CID 区间", "URL/本地表格合并"],
            index=0,
            key="adv_seed_mode",
        )

        data_out = st.text_input("保存 CSV 路径", value="data/gen_seed_crawl.csv", key="adv_seed_out")

        if seed_mode.startswith("PubChem"):
            c1, c2, c3 = st.columns(3)
            with c1:
                start_cid = st.number_input("start_cid", min_value=1, value=1, step=1, key="adv_seed_start")
            with c2:
                n_seed = st.number_input("抓取数量", min_value=10, value=200, step=10, key="adv_seed_n")
            with c3:
                min_total = st.number_input("min_total（Active+Inactive）", min_value=0, value=1, step=1, key="adv_seed_min_total")

            st.markdown("**属性字段（生成/打分可用）**")
            default_props = [
                "MolecularWeight",
                "XLogP",
                "TopologicalPolarSurfaceArea",
                "HBondDonorCount",
                "HBondAcceptorCount",
                "RotatableBondCount",
            ]
            prop_fields = st.multiselect(
                "选择属性字段",
                options=list(PUBCHEM_PROPERTY_FIELDS),
                default=[p for p in default_props if p in PUBCHEM_PROPERTY_FIELDS],
                key="adv_seed_props",
            )

            if st.button("抓取并推送到分子生成", type="primary", key="adv_seed_run_pubchem"):
                with st.spinner("爬取中..."):
                    df = crawl_pubchem_activity_dataset(
                        start_cid=int(start_cid),
                        n=int(n_seed),
                        min_total_outcomes=int(min_total),
                        cache_dir=str(cache_dir),
                        timeout=float(timeout_s),
                        sleep_seconds=float(sleep_s),
                        include_properties=True,
                        property_fields=list(prop_fields),
                        normalize_smiles=bool(normalize_smiles),
                        drop_invalid=True,
                    )

                if dedupe_smiles and "smiles" in df.columns:
                    df = df.drop_duplicates(subset=["smiles"]).copy()

                st.session_state["drug_gen_seed_df"] = df
                st.success("已写入分子生成页会话缓存，可在“分子生成”页选择“使用爬虫种子”直接加载。")
                _preview_df(df, title="分子生成种子/打分数据预览", max_rows=50)

                if data_out:
                    p = Path(data_out)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(p, index=False)
                    st.write("已保存到:", str(p))

                st.download_button(
                    "下载 CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=Path(data_out).name if data_out else "gen_seed_crawl.csv",
                    mime="text/csv",
                    key="adv_seed_download_pubchem",
                )

        else:
            sources_text = st.text_area("URL 或本地路径（每行一个）", value="", height=100, key="adv_seed_sources")
            seed_col = st.text_input("smiles 列名", value="smiles", key="adv_seed_col")

            if st.button("合并并推送到分子生成", type="primary", key="adv_seed_run_url"):
                sources = [s.strip() for s in sources_text.splitlines() if s.strip()]
                if not sources:
                    st.error("请至少提供一个数据源")
                    return
                with st.spinner("读取/合并中..."):
                    df = concat_tables(
                        sources,
                        cache_dir=str(cache_dir),
                        timeout=float(timeout_s),
                        sleep_seconds=float(sleep_s),
                        headers={"User-Agent": "adv-seed-crawler/1.0 (research; contact: local)"},
                    )
                if seed_col and seed_col in df.columns:
                    df = df.rename(columns={seed_col: "smiles"})
                if "smiles" in df.columns:
                    df["smiles"] = df["smiles"].astype(str)
                if dedupe_smiles and "smiles" in df.columns:
                    df = df.drop_duplicates(subset=["smiles"]).copy()

                st.session_state["drug_gen_seed_df"] = df
                st.success("已写入分子生成页会话缓存，可在“分子生成”页选择“使用爬虫种子”直接加载。")
                _preview_df(df, title="分子生成种子数据预览", max_rows=50)

                if data_out:
                    p = Path(data_out)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(p, index=False)
                    st.write("已保存到:", str(p))

                st.download_button(
                    "下载 CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=Path(data_out).name if data_out else "gen_seed_crawl.csv",
                    mime="text/csv",
                    key="adv_seed_download_url",
                )


def literature_autolearn_ui() -> None:
    st.subheader("文献自主学习（训练数据建议）")
    st.caption("输入关键词与领域，自动检索公开文献并输出可用数据源建议。")

    domain_label = st.selectbox(
        "领域",
        options=["药物/小分子", "表位", "分子对接", "多尺度/动力学", "自定义"],
        index=0,
        key="lit_domain",
    )
    domain_custom = ""
    if domain_label == "自定义":
        domain_custom = st.text_input("自定义领域关键词", value="", key="lit_domain_custom")

    query = st.text_input("核心关键词", value="", key="lit_query")
    extra_terms = st.text_area("补充关键词（分号/换行分隔）", value="", key="lit_extra_terms")

    source_options_by_domain = {
        "drug": ["PubChem", "ChEMBL", "BindingDB", "DrugBank", "ZINC"],
        "epitope": ["IEDB", "UniProt", "PDB", "SAbDab"],
        "docking": ["PDBbind", "BindingDB", "DUD-E", "PDB"],
        "multiscale": ["BRENDA", "SABIO-RK", "BioModels", "KEGG"],
        "custom": ["Zenodo", "Figshare", "Dryad", "Dataverse", "GitHub"],
    }

    col1, col2, col3 = st.columns(3)
    with col1:
        year_from = st.number_input("起始年份", min_value=1900, max_value=2100, value=2015, step=1, key="lit_year_from")
    with col2:
        year_to = st.number_input("截止年份", min_value=1900, max_value=2100, value=int(time.strftime("%Y")), step=1, key="lit_year_to")
    with col3:
        max_results = st.number_input("最大结果数", min_value=5, max_value=200, value=30, step=5, key="lit_max_results")

    col4, col5, col6 = st.columns(3)
    with col4:
        include_preprints = st.checkbox("包含预印本", value=False, key="lit_preprints")
    with col5:
        timeout_s = st.number_input("timeout（秒）", min_value=5.0, value=20.0, step=1.0, key="lit_timeout")
    with col6:
        cache_dir = st.text_input("cache_dir", value="data/cache/literature", key="lit_cache_dir")

    domain_key = "custom"
    if domain_label == "药物/小分子":
        domain_key = "drug"
    elif domain_label == "表位":
        domain_key = "epitope"
    elif domain_label == "分子对接":
        domain_key = "docking"
    elif domain_label == "多尺度/动力学":
        domain_key = "multiscale"
    elif domain_label == "自定义":
        domain_key = "custom"

    default_sources = source_options_by_domain.get(domain_key, source_options_by_domain["custom"])
    sources_filter = st.multiselect(
        "可选查找源（用于过滤/强化检索）",
        options=sorted({*default_sources, *source_options_by_domain["custom"]}),
        default=list(default_sources),
        key="lit_sources_filter",
    )

    def _split_terms(text: str) -> List[str]:
        tokens = [t.strip() for t in re.split(r"[;\n]+", str(text)) if t.strip()]
        return tokens

    keyword_list = _split_terms(extra_terms)
    if domain_custom:
        keyword_list.insert(0, domain_custom)

    with st.expander("云端检索", expanded=False):
        payload = {
            "query": str(query),
            "domain": str(domain_key),
            "keywords": list(keyword_list),
            "sources": list(sources_filter),
            "year_from": int(year_from) if year_from else None,
            "year_to": int(year_to) if year_to else None,
            "include_preprints": bool(include_preprints),
            "max_results": int(max_results),
            "timeout": float(timeout_s),
            "cache_dir": str(cache_dir),
            "include_csv": True,
        }
        _cloud_submit_section(
            "literature_autolearn",
            payload,
            button_label="提交云端检索",
            key="cloud_lit_autolearn_btn",
            download_name="literature_autolearn.csv",
        )

    if st.button("开始检索", type="primary", key="lit_autolearn_btn"):
        try:
            from src.common.literature_autolearn import literature_autolearn
        except Exception as e:
            st.error(f"无法导入文献检索模块：{e}")
            return

        with st.spinner("检索中..."):
            res = literature_autolearn(
                query=str(query),
                domain=str(domain_key),
                keywords=keyword_list,
                sources=list(sources_filter),
                year_from=int(year_from) if year_from else None,
                year_to=int(year_to) if year_to else None,
                include_preprints=bool(include_preprints),
                max_results=int(max_results),
                timeout=float(timeout_s),
                cache_dir=str(cache_dir) if cache_dir else None,
                include_csv=False,
            )

        st.session_state["literature_autolearn"] = res
        st.write("检索式:", res.get("query", ""))

        suggestions = res.get("suggestions", {}) if isinstance(res, dict) else {}
        sources = suggestions.get("suggested_sources", []) if isinstance(suggestions, dict) else []
        fields = suggestions.get("field_hints", []) if isinstance(suggestions, dict) else []
        actions = suggestions.get("actions", []) if isinstance(suggestions, dict) else []

        if sources:
            st.markdown("**建议数据源**：" + ", ".join([str(s) for s in sources]))
        if fields:
            st.markdown("**建议字段**：" + ", ".join([str(s) for s in fields]))
        if actions:
            st.markdown("**建议步骤**：" + "；".join([str(s) for s in actions]))

        items = res.get("items", []) if isinstance(res, dict) else []
        if not items:
            st.info("未找到结果，请尝试调整关键词或时间范围。")
            return

        df = pd.DataFrame(items)

        def _row_id(row: pd.Series) -> str:
            for key in ["doi", "pmcid", "pmid", "title"]:
                v = str(row.get(key, "")).strip()
                if v:
                    return v
            return f"row_{hash(str(row.to_dict())) & 0xFFFFFFFF}"

        df["__row_id"] = df.apply(_row_id, axis=1)

        stored = st.session_state.get("literature_editor_last")
        if isinstance(stored, list) and stored:
            try:
                stored_df = pd.DataFrame(stored)
                if "__row_id" in stored_df.columns:
                    stored_map = {str(r["__row_id"]): r for r in stored_df.to_dict(orient="records")}
                    for idx, row in df.iterrows():
                        rid = str(row.get("__row_id", ""))
                        srow = stored_map.get(rid)
                        if srow:
                            for col in ["dataset_selected", "notes"]:
                                if col in df.columns and col in srow:
                                    df.at[idx, col] = srow[col]
            except Exception:
                pass
        if "dataset_selected" not in df.columns:
            # allow users to manually mark which dataset is directly usable
            def _default_sel(v: Any) -> str:
                s = str(v or "").strip()
                return s.split("|")[0].strip() if s else ""

            df.insert(1, "dataset_selected", df.get("dataset_hints", "").map(_default_sel) if "dataset_hints" in df.columns else "")
        if "notes" not in df.columns:
            df.insert(2, "notes", "")

        show_cols = [
            c
            for c in [
                "dataset_selected",
                "notes",
                "year",
                "title",
                "journal",
                "authors",
                "dataset_hints",
                "paper_url",
                "fulltext_urls",
                "doi",
                "pmid",
                "pmcid",
            ]
            if c in df.columns
        ]

        st.markdown('<a id="lit_anchor"></a>', unsafe_allow_html=True)
        df_for_edit = df.set_index("__row_id")
        df_for_edit.index.name = "__row_id"

        edited = st.data_editor(
            df_for_edit.loc[:, show_cols],
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "dataset_selected": st.column_config.TextColumn("可用数据集（手填/标注）", help="写数据集名称/链接提示，例如：ChEMBL bioactivity / IEDB export / PDBbind v2020"),
                "notes": st.column_config.TextColumn("备注", help="记录列名、许可、下载方式等"),
                "paper_url": st.column_config.LinkColumn("论文链接"),
                "fulltext_urls": st.column_config.TextColumn("全文/资源链接"),
            },
            key="lit_preview_editor",
        )

        if isinstance(edited, pd.DataFrame):
            edited_reset = edited.reset_index().rename(columns={"__row_id": "__row_id"})
            st.session_state["literature_editor_last"] = edited_reset.to_dict(orient="records")

        latest = st.session_state.get("literature_editor_last")
        if isinstance(latest, list) and latest:
            edited_df = pd.DataFrame(latest)
        elif isinstance(edited, pd.DataFrame):
            edited_df = edited.reset_index().rename(columns={"__row_id": "__row_id"})
        else:
            edited_df = df.reset_index().rename(columns={"__row_id": "__row_id"})

        # 已移除“勾选”和相关的自动下载功能以简化界面
        st.info("已禁用勾选与自动下载功能；检索结果将在下方显示，若需导出请在后续模块中处理。")


def drug_legacy_demo_ui() -> None:
    st.subheader("药物疗效预测（torch）")
    st.caption("抓取 PubChem 并使用 torch 训练（较慢）。建议优先用上方‘训练/预测’模块加载你自己的标注数据。")

    drug = None
    try:
        import importlib.util as _importlib_util

        if not _importlib_util.find_spec("torch"):
            import sys

            is_packaged = bool(getattr(sys, "_MEIPASS", None))
            st.warning(
                "torch 模块需要 torch，但当前运行环境无法导入 torch。\n"
                f"当前解释器：{sys.executable}\n"
                "\n"
                "解决方式：\n"
                "- 如果你运行的是打包版 exe：请用 full 版本重新打包（minimal/denoise 不包含 torch）\n"
                "  powershell build_windows.ps1 -BuildProfile full -InstallDeps\n"
                "- 如果你运行的是源码版：请用工作区 .venv 启动并安装依赖\n"
                "  D:/IGEM集成方案/.venv/Scripts/python.exe -m pip install -r requirements-drug-demo.txt\n"
                "  D:/IGEM集成方案/.venv/Scripts/python.exe -m streamlit run src/frontend.py\n"
                + ("" if not is_packaged else "\n提示：当前检测为打包环境（PyInstaller）。")
            )
            return

        # 注意：这里用 importlib 动态导入，避免 PyInstaller 静态分析时强制把 torch 等大依赖打进包里。
        # 如果用户确实需要，可在源码环境中安装 torch 后运行即可。
        import importlib
        import importlib.util as _importlib_util
        import sys

        # 这里刻意不用“可被常量折叠”的拼接表达式，避免 PyInstaller 启发式地把 src.main 作为 hidden-import 打进包。
        # （例如 "src." + "main" 在编译期会被折叠成常量）
        module_name = "src."
        module_name += "main"
        try:
            drug = importlib.import_module(module_name)  # type: ignore
        except Exception:
            # 尝试从打包路径载入（_MEIPASS/src/main.py）以启用旧版 torch 功能
            meipass = getattr(sys, "_MEIPASS", None)
            if meipass:
                module_path = Path(meipass) / "src" / "main.py"
                if module_path.exists():
                    spec = _importlib_util.spec_from_file_location("src.main", module_path)
                    if spec and spec.loader:
                        mod = _importlib_util.module_from_spec(spec)
                        spec.loader.exec_module(mod)  # type: ignore[call-arg]
                        drug = mod
                    else:
                        raise
                else:
                    raise
            else:
                raise

        required = ["scrape_drug_data", "prepare_data", "pretrain_feature_extractor", "MultiTaskModel", "finetune_multi_task", "smiles_to_fingerprint"]
        for name in required:
            if not hasattr(drug, name):
                raise RuntimeError(f"src.main 缺少函数/类: {name}")

    except Exception as e:
        st.warning(f"旧版模块不可用：{e}")
        return

    assert drug is not None

    if "drug_model" not in st.session_state:
        st.session_state["drug_model"] = None

    num_drugs = st.slider("抓取样本数（CID 1..N）", min_value=10, max_value=200, value=30, step=10)
    pretrain_epochs = st.slider("预训练 epochs（越小越快）", min_value=1, max_value=20, value=3, step=1)
    finetune_epochs = st.slider("微调 epochs（越小越快）", min_value=1, max_value=20, value=3, step=1)

    if st.button("抓取并训练", type="primary", key="drug_legacy_train"):
        with st.spinner("抓取数据中..."):
            df = drug.scrape_drug_data(num_drugs=int(num_drugs))
        if len(df) == 0:
            st.error("没有抓取到数据（可能网络受限）")
            return
        _preview_df(df, title="抓取到的样本预览", max_rows=10)

        with st.spinner("特征提取中..."):
            x, y = drug.prepare_data(df)

        with st.spinner("预训练特征提取器中..."):
            fe = drug.pretrain_feature_extractor(x, epochs=int(pretrain_epochs))

        with st.spinner("微调多任务模型中..."):
            mt = drug.MultiTaskModel(fe)
            drug.finetune_multi_task(mt, x, y, epochs=int(finetune_epochs))

        st.session_state["drug_model"] = mt
        st.success("训练完成（会话内缓存）")

    st.divider()
    st.write("### 预测（SMILES）")
    smiles = st.text_input("输入 SMILES", value="CCO", key="drug_legacy_smiles")

    if st.button("预测疗效/毒性", key="drug_legacy_pred"):
        mt = st.session_state.get("drug_model")
        if mt is None:
            st.error("请先训练一个模型")
            return

        fp = drug.smiles_to_fingerprint(smiles)
        x = np.asarray(fp, dtype=np.float32).reshape(1, -1)

        try:
            import importlib

            # 同上：避免 PyInstaller 把 torch 直接打进包。
            torch_name = "tor"
            torch_name += "ch"
            torch = importlib.import_module(torch_name)  # type: ignore
        except Exception as e:
            st.error(f"当前环境无法导入 torch：{e}")
            return

        with torch.no_grad():
            outputs = mt(torch.tensor(x, dtype=torch.float32))

        st.write({
            "efficacy_pred": float(outputs["efficacy"].reshape(-1)[0].item()),
            "toxicity_pred": float(outputs["toxicity"].reshape(-1)[0].item()),
        })

    st.divider()
    st.write("### 批量预测（CSV）")
    batch_up = st.file_uploader("上传包含 smiles 列的表格", type=_UPLOAD_TYPES, key="drug_legacy_batch")
    if batch_up is not None:
        mt = st.session_state.get("drug_model")
        if mt is None:
            st.error("请先训练一个模型")
            return
        try:
            df = _load_table_from_bytes(batch_up.getvalue(), getattr(batch_up, "name", "batch.csv"))
        except Exception as e:
            st.error(f"读取文件失败：{e}")
            return
        if "smiles" not in df.columns:
            st.error("缺少 smiles 列")
            return

        try:
            import importlib

            torch_name = "tor"
            torch_name += "ch"
            torch = importlib.import_module(torch_name)  # type: ignore
        except Exception as e:
            st.error(f"当前环境无法导入 torch：{e}")
            return

        preds = []
        with torch.no_grad():
            for s in df["smiles"].astype(str).tolist():
                fp = drug.smiles_to_fingerprint(s)
                x = np.asarray(fp, dtype=np.float32).reshape(1, -1)
                efficacy_pred, toxicity_pred = mt(torch.tensor(x, dtype=torch.float32))
                preds.append(
                    {
                        "smiles": s,
                        "efficacy_pred": float(efficacy_pred.reshape(-1)[0].item()),
                        "toxicity_pred": float(toxicity_pred.reshape(-1)[0].item()),
                    }
                )

        out_df = pd.DataFrame(preds)
        _preview_df(out_df, title="结果预览", max_rows=50)
        st.download_button(
            "下载预测结果 CSV",
            data=out_df.to_csv(index=False).encode("utf-8"),
            file_name="legacy_torch_predictions.csv",
            mime="text/csv",
        )


def drug_generate_ui() -> None:
    st.subheader("分子生成（GAN + 进化算法）")
    st.caption("使用分子指纹 GAN 扩增种群，并通过遗传算法生成候选 SMILES，可选模型评分与性质过滤。")

    try:
        import rdkit  # type: ignore  # noqa: F401
        from src.drug.generative import (
            GanConfig,
            PropertyFilters,
            calc_props,
            default_score,
            generate_molecules,
            passes_filters,
            select_diverse_ranked,
        )  # type: ignore
        from src.drug.predictor import DrugModelBundle, predict_one  # type: ignore
    except Exception as e:
        st.warning(f"分子生成不可用（可能未安装 rdkit）：{e}")
        return

    uploaded = st.file_uploader(
        "上传种子数据（可选）",
        type=_UPLOAD_TYPES,
        accept_multiple_files=True,
        key="drug_gen_seed_csv",
    )

    seed_smiles: List[str] = []
    df_seeds: Optional[pd.DataFrame] = None
    seed_frames: List[pd.DataFrame] = []

    session_seed_df = st.session_state.get("drug_gen_seed_df")
    use_session_seed = False
    if isinstance(session_seed_df, pd.DataFrame) and len(session_seed_df) > 0:
        st.info("检测到来自“高级数据对接”页的爬虫种子，可直接用于生成。")
        use_session_seed = st.checkbox(
            "使用爬虫抓取的种子数据",
            value=(not uploaded),
            key="drug_gen_use_session_seed",
        )
        if use_session_seed:
            seed_frames.append(session_seed_df.copy())

    if uploaded:
        try:
            df_raw = _load_tables_from_uploads(uploaded)
            seed_frames.append(df_raw)
        except Exception as e:
            st.error(f"读取文件失败：{e}")
            return

    if seed_frames:
        df_seeds = pd.concat(seed_frames, axis=0, ignore_index=True)
        if "smiles" in df_seeds.columns:
            df_seeds = df_seeds.drop_duplicates(subset=["smiles"]).copy()
        _preview_df(df_seeds, title="种子数据预览")

    smiles_col = None
    if df_seeds is not None:
        smiles_col = st.selectbox(
            "smiles 列名",
            options=list(df_seeds.columns),
            index=(list(df_seeds.columns).index("smiles") if "smiles" in df_seeds.columns else 0),
            key="drug_gen_smiles_col",
        )

    seed_text = st.text_area(
        "补充种子 SMILES（每行一个，可选）",
        value="",
        height=120,
        key="drug_gen_seed_text",
    )

    st.markdown("#### 评分方式")
    score_mode = st.selectbox("评分模式", options=["auto", "qed", "model", "combined"], index=0, key="drug_gen_score_mode")

    model_bundle: Optional[DrugModelBundle] = None
    env_params: Dict[str, float] = {}
    uploaded_model = None
    local_path = ""
    if score_mode in ("model", "combined", "auto"):
        col1, col2 = st.columns(2)
        with col1:
            uploaded_model = st.file_uploader("上传模型 .joblib（可选）", type=["joblib"], key="drug_gen_model_upload")
        with col2:
            local_models = _list_local_models_drug()
            local_path = st.selectbox("或选择本地 models/drug_*.joblib", options=[""] + local_models, key="drug_gen_model_local")

        if uploaded_model is not None:
            model_bundle = joblib.load(io.BytesIO(uploaded_model.getvalue()))
        elif local_path:
            model_bundle = joblib.load(local_path)

        kv_text = st.text_area(
            "条件参数（每行 key=value；缺失会用训练集中位数填充）",
            value="",
            height=100,
            key="drug_gen_kv",
        )
        if kv_text.strip():
            try:
                env_params = _parse_kv_lines(kv_text)
            except ValueError as e:
                st.error(str(e))
                return
        if model_bundle is not None and model_bundle.env_cols:
            for c in model_bundle.env_cols:
                env_params.setdefault(c, float(model_bundle.env_medians.get(c, 0.0)))

        st.markdown("#### 训练打分模型（可选）")
        score_csv = st.file_uploader(
            "上传带有 'smiles' 和 'score' 列的 CSV 用于训练打分模型（可选）",
            type=["csv"],
            key="drug_gen_score_csv",
        )
        train_epochs = st.number_input("训练轮数", min_value=1, max_value=10000, value=100, key="drug_gen_train_epochs")
        train_batch = st.number_input("训练 Batch", min_value=1, max_value=4096, value=128, key="drug_gen_train_batch")
        train_lr = st.number_input("学习率", min_value=1e-6, max_value=1.0, value=1e-3, format="%.6g", key="drug_gen_train_lr")
        train_hidden = st.text_input("隐藏层 sizes", value="512,256", key="drug_gen_train_hidden")
        train_dropout = st.number_input("dropout", min_value=0.0, max_value=1.0, value=0.1, step=0.01, key="drug_gen_train_dropout")
        train_test_size = st.slider("验证集比例", min_value=0.0, max_value=0.5, value=0.2, step=0.05, key="drug_gen_train_test_size")
        use_cuda_train = st.checkbox("训练时使用 CUDA（若可用）", value=False, key="drug_gen_train_cuda")

        if score_csv is not None:
            if st.button("训练并保存评分模型", key="drug_gen_train_score_btn"):
                try:
                    import time
                    from src.drug.scoring import train_scoring_model_from_df, save_torch_bundle  # type: ignore

                    df_score = pd.read_csv(score_csv)
                    hidden_sizes = [int(x) for x in train_hidden.split(",") if x.strip()]
                    with st.spinner("训练中... 可能需要几分钟，请耐心等待"):
                        bundle, metrics = train_scoring_model_from_df(
                            df_score,
                            smiles_col="smiles",
                            score_col="score",
                            hidden_sizes=hidden_sizes,
                            dropout=float(train_dropout),
                            lr=float(train_lr),
                            batch_size=int(train_batch),
                            epochs=int(train_epochs),
                            test_size=float(train_test_size),
                            random_state=42,
                            use_cuda=bool(use_cuda_train),
                        )

                    out_path = _PROJECT_ROOT / "data" / f"scoring_model_{int(time.time())}.pt"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    save_torch_bundle(bundle, str(out_path))
                    st.success(f"训练完成，模型已保存：{out_path}")
                    st.write(metrics)
                except Exception as e:
                    st.error(f"训练失败：{e}")

    weight_model = st.number_input("模型权重（combined）", min_value=0.0, max_value=10.0, value=1.0, step=0.1, key="drug_gen_w_model")
    weight_qed = st.number_input("QED 权重（combined）", min_value=0.0, max_value=10.0, value=1.0, step=0.1, key="drug_gen_w_qed")

    st.markdown("#### GAN 参数（可选）")
    use_gan = st.checkbox("启用 GAN 扩增种群", value=False, key="drug_gen_use_gan")
    gan_samples = st.number_input("GAN 采样数量", min_value=16, max_value=4096, value=256, step=16, key="drug_gen_gan_samples")
    gan_latent_dim = st.number_input("GAN 潜在维度", min_value=8, max_value=512, value=64, step=8, key="drug_gen_gan_latent")
    gan_hidden_dim = st.number_input("GAN 隐藏维度", min_value=64, max_value=1024, value=256, step=32, key="drug_gen_gan_hidden")
    gan_epochs = st.number_input("GAN 训练轮数", min_value=10, max_value=5000, value=200, step=10, key="drug_gen_gan_epochs")
    gan_batch = st.number_input("GAN Batch", min_value=16, max_value=2048, value=128, step=16, key="drug_gen_gan_batch")
    gan_lr = st.number_input("GAN 学习率", min_value=1e-5, max_value=1e-2, value=2e-4, format="%.5f", key="drug_gen_gan_lr")
    gan_device = st.selectbox("GAN 设备", options=["cpu", "cuda"], index=0, key="drug_gen_gan_device")

    st.markdown("#### 进化参数")
    population = st.number_input("种群规模", min_value=20, max_value=2000, value=200, step=10, key="drug_gen_pop")
    generations = st.number_input("迭代代数", min_value=1, max_value=200, value=20, step=1, key="drug_gen_gen")
    elite_frac = st.slider("精英比例", min_value=0.05, max_value=0.8, value=0.2, step=0.05, key="drug_gen_elite")
    mutation_rate = st.slider("变异概率", min_value=0.0, max_value=1.0, value=0.4, step=0.05, key="drug_gen_mut")
    crossover_rate = st.slider("交叉概率", min_value=0.0, max_value=1.0, value=0.6, step=0.05, key="drug_gen_cross")

    st.markdown("#### 指纹与多样性")
    radius = st.number_input("指纹半径", min_value=1, max_value=6, value=2, step=1, key="drug_gen_radius")
    n_bits = st.number_input("指纹位数", min_value=256, max_value=8192, value=2048, step=256, key="drug_gen_bits")
    diversity_max_sim = st.slider("输出多样性（最大相似度）", min_value=0.1, max_value=1.0, value=0.9, step=0.05, key="drug_gen_div")

    st.markdown("#### 性质过滤（可选）")
    min_qed = st.number_input("最小 QED", min_value=0.0, max_value=1.0, value=0.0, step=0.05, key="drug_gen_min_qed")
    min_mw = st.number_input("最小分子量", min_value=0.0, max_value=2000.0, value=0.0, step=10.0, key="drug_gen_min_mw")
    max_mw = st.number_input("最大分子量", min_value=0.0, max_value=2000.0, value=800.0, step=10.0, key="drug_gen_max_mw")
    min_logp = st.number_input("最小 LogP", min_value=-5.0, max_value=10.0, value=-5.0, step=0.5, key="drug_gen_min_logp")
    max_logp = st.number_input("最大 LogP", min_value=-5.0, max_value=10.0, value=6.0, step=0.5, key="drug_gen_max_logp")
    max_hbd = st.number_input("最大 HBD", min_value=0, max_value=20, value=8, step=1, key="drug_gen_max_hbd")
    max_hba = st.number_input("最大 HBA", min_value=0, max_value=30, value=12, step=1, key="drug_gen_max_hba")
    max_tpsa = st.number_input("最大 TPSA", min_value=0.0, max_value=300.0, value=140.0, step=5.0, key="drug_gen_max_tpsa")

    include_props = st.checkbox("输出附带性质列", value=True, key="drug_gen_with_props")
    top_k = st.number_input("导出 Top K", min_value=10, max_value=1000, value=50, step=10, key="drug_gen_topk")
    seed = st.number_input("随机种子", min_value=0, max_value=10_000_000, value=42, step=1, key="drug_gen_seed")

    with st.expander("云端生成", expanded=False):
        seed_smiles_cloud: List[str] = []
        if df_seeds is not None and smiles_col:
            seed_smiles_cloud.extend(df_seeds[smiles_col].astype(str).tolist())
        if seed_text.strip():
            seed_smiles_cloud.extend([s.strip() for s in seed_text.splitlines() if s.strip()])

        model_payload = None
        if score_mode in ("model", "combined", "auto"):
            model_payload = _resolve_model_payload(uploaded_model, local_path if local_path else None, content_type="application/octet-stream")

        payload = {
            "seed_smiles": seed_smiles_cloud,
            "score_mode": str(score_mode),
            "env_params": {k: float(v) for k, v in env_params.items()},
            "weight_model": float(weight_model),
            "weight_qed": float(weight_qed),
            "use_gan": bool(use_gan),
            "gan_params": {
                "samples": int(gan_samples),
                "latent_dim": int(gan_latent_dim),
                "hidden_dim": int(gan_hidden_dim),
                "epochs": int(gan_epochs),
                "batch_size": int(gan_batch),
                "lr": float(gan_lr),
                "device": str(gan_device),
            },
            "evolution_params": {
                "population": int(population),
                "generations": int(generations),
                "elite_frac": float(elite_frac),
                "mutation_rate": float(mutation_rate),
                "crossover_rate": float(crossover_rate),
            },
            "fingerprint": {
                "radius": int(radius),
                "n_bits": int(n_bits),
            },
            "diversity_max_sim": float(diversity_max_sim),
            "filters": {
                "min_qed": float(min_qed),
                "min_mw": float(min_mw),
                "max_mw": float(max_mw),
                "min_logp": float(min_logp),
                "max_logp": float(max_logp),
                "max_hbd": int(max_hbd),
                "max_hba": int(max_hba),
                "max_tpsa": float(max_tpsa),
            },
            "include_props": bool(include_props),
            "top_k": int(top_k),
            "seed": int(seed),
        }
        if model_payload is not None:
            payload["model"] = model_payload

        if not seed_smiles_cloud:
            st.info("请先提供种子 SMILES 后再提交云端生成。")
        else:
            _cloud_submit_section(
                "drug_generate",
                payload,
                button_label="提交云端生成",
                key="cloud_drug_generate_btn",
                download_name="drug_generated_cloud.csv",
            )

    if st.button("开始生成", type="primary", key="drug_gen_btn"):
        if df_seeds is not None and smiles_col:
            seed_smiles.extend(df_seeds[smiles_col].astype(str).tolist())
        if seed_text.strip():
            seed_smiles.extend([s.strip() for s in seed_text.splitlines() if s.strip()])
        if not seed_smiles:
            st.error("请至少提供一个种子 SMILES（上传文件或文本输入）")
            return

        if use_gan:
            try:
                import torch  # type: ignore  # noqa: F401
            except Exception as e:
                st.error(f"启用 GAN 需要 torch：{e}")
                return

        filters = PropertyFilters(
            min_qed=min_qed if min_qed > 0 else None,
            min_mw=min_mw if min_mw > 0 else None,
            max_mw=max_mw if max_mw > 0 else None,
            min_logp=min_logp if min_logp > -5 else None,
            max_logp=max_logp if max_logp < 10 else None,
            max_hbd=max_hbd if max_hbd > 0 else None,
            max_hba=max_hba if max_hba > 0 else None,
            max_tpsa=max_tpsa if max_tpsa > 0 else None,
        )

        resolved_mode = score_mode
        if score_mode == "auto":
            resolved_mode = "model" if model_bundle is not None else "qed"

        def score_fn(smiles: str) -> float:
            if not passes_filters(smiles, filters):
                return float("-inf")
            if resolved_mode == "model" and model_bundle is not None:
                return float(predict_one(model_bundle, smiles=smiles, env_params=env_params))
            if resolved_mode == "combined" and model_bundle is not None:
                return float(weight_model * predict_one(model_bundle, smiles=smiles, env_params=env_params) + weight_qed * default_score(smiles))
            return float(default_score(smiles))

        gan_cfg = GanConfig(
            latent_dim=int(gan_latent_dim),
            hidden_dim=int(gan_hidden_dim),
            epochs=int(gan_epochs),
            batch_size=int(gan_batch),
            lr=float(gan_lr),
            device=str(gan_device),
        )

        with st.spinner("生成中..."):
            ranked = generate_molecules(
                seed_smiles=seed_smiles,
                use_gan=bool(use_gan),
                gan_cfg=gan_cfg,
                radius=int(radius),
                n_bits=int(n_bits),
                gan_samples=int(gan_samples),
                score_fn=score_fn,
                population_size=int(population),
                generations=int(generations),
                elite_frac=float(elite_frac),
                mutation_rate=float(mutation_rate),
                crossover_rate=float(crossover_rate),
                rng=random.Random(int(seed)),
            )

        ranked = select_diverse_ranked(
            ranked,
            radius=int(radius),
            n_bits=int(n_bits),
            max_sim=float(diversity_max_sim),
        )

        rows = []
        for i, (smi, score) in enumerate(ranked[: int(top_k)], start=1):
            row = {"rank": i, "smiles": smi, "score": float(score)}
            if include_props:
                props = calc_props(smi)
                if props:
                    row.update(props)
            rows.append(row)

        if not rows:
            st.warning("未生成满足条件的分子，请放宽过滤条件或提高种群/代数。")
            return

        out_df = pd.DataFrame(rows)
        st.session_state["drug_generated_df"] = out_df
        _preview_df(out_df, title="结果预览", max_rows=50)
        st.download_button(
            "下载生成结果 CSV",
            data=out_df.to_csv(index=False).encode("utf-8"),
            file_name="drug_generated.csv",
            mime="text/csv",
        )

        update_feedback_context({
            "event": "drug_generate",
            "n_seeds": len(seed_smiles),
            "n_output": len(out_df),
            "score_mode": resolved_mode,
            "use_gan": bool(use_gan),
        })


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="🧬")
    _inject_global_styles()
    st.title(f"🧬 {APP_TITLE}")

    render_user_guidance_sidebar()
    render_acknowledgement_sidebar()
    render_performance_sidebar(_PROJECT_ROOT)
    render_feedback_sidebar()
    _cloud_ui()

    with st.sidebar.expander("运行内存管理", expanded=False):
        st.caption("长时间会话后可手动释放缓存与显存，降低内存占用峰值。")
        clear_st_cache = st.checkbox("同时清理 Streamlit 缓存", value=False, key="runtime_clear_st_cache")
        if st.button("释放运行内存", key="runtime_release_memory_btn"):
            release_runtime_memory(clear_streamlit_cache=bool(clear_st_cache))
            st.success("已执行内存释放。")

    (
        tab_drug,
        tab_epitope,
        tab_docking,
        tab_data,
        tab_train,
        tab_whitebox,
        tab_ms,
    ) = st.tabs(
        ["药效预测", "表位预测", "对接预测", "数据处理与获取", "模型训练", "结果解释", "多尺度分析"]
    )

    with tab_drug:
        t_pred, t_trans_pred, t_screen, t_plot, t_dleps, t_legacy = st.tabs(
            ["单条预测", "Transformer预测", "批量筛选", "绘图", "DLEPS 富集", "torch预测"]
        )
        with t_pred:
            set_feedback_context("drug", "predict")
            drug_predict_ui()
        with t_trans_pred:
            set_feedback_context("drug", "predict_transformer")
            drug_transformer_predict_ui()
        with t_screen:
            set_feedback_context("drug", "screen")
            drug_screen_ui()
        with t_plot:
            set_feedback_context("drug", "plot")
            drug_plot_ui()
        with t_dleps:
            set_feedback_context("dleps", "app")
            st.markdown("### DLEPS 药物富集预测")
            st.caption("若此处仍无内容，请刷新页面或检查控制台错误日志。")
            st.info("DLEPS 标签已渲染（调试提示）。")
            try:
                dleps_ui()
            except Exception as e:
                st.error(f"加载 DLEPS 前端失败：{e}")
                try:
                    st.exception(e)
                except Exception:
                    pass
        with t_legacy:
            set_feedback_context("drug", "legacy_demo")
            drug_legacy_demo_ui()

    with tab_epitope:
        t_pred, t_screen, t_sens, t_plot = st.tabs(["单条预测", "批量筛选", "敏感性分析", "绘图"])
        with t_pred:
            set_feedback_context("epitope", "predict")
            epitope_predict_ui()
        with t_screen:
            set_feedback_context("epitope", "screen")
            epitope_screen_ui()
        with t_sens:
            set_feedback_context("epitope", "sensitivity")
            epitope_sensitivity_ui()
        with t_plot:
            set_feedback_context("epitope", "plot")
            epitope_plot_ui()

    with tab_docking:
        d_pred, d_screen = st.tabs(["单条预测", "批量筛选"])
        with d_pred:
            set_feedback_context("docking", "predict")
            docking_predict_ui()
        with d_screen:
            set_feedback_context("docking", "screen")
            docking_screen_ui()

    with tab_data:
        t_denoise, t_gen, t_adv_crawl, t_epi_crawl, t_drug_crawl, t_lit = st.tabs([
            "数据增强与去噪",
            "分子生成",
            "高级数据对接",
            "表位爬虫",
            "药物爬虫",
            "文献自主学习",
        ])
        with t_denoise:
            set_feedback_context("denoise", "ui")
            data_aug_denoise_ui()
        with t_gen:
            set_feedback_context("drug", "generate")
            drug_generate_ui()
        with t_adv_crawl:
            set_feedback_context("data", "advanced_crawl")
            drug_advanced_crawl_ui()
        with t_epi_crawl:
            set_feedback_context("epitope", "crawl")
            epitope_crawl_ui()
        with t_drug_crawl:
            set_feedback_context("drug", "crawl_train")
            drug_pubchem_crawl_train_ui()
        with t_lit:
            set_feedback_context("literature", "autolearn")
            literature_autolearn_ui()

    with tab_train:
        (
            t_epi_train,
            t_drug_train,
            t_torch_train,
            t_trans_train,
            t_dock_train,
            t_epi_self,
            t_drug_self,
        ) = st.tabs(["表位训练", "药物训练", "Torch训练", "Transformer训练", "对接训练", "表位自训练", "药物自训练"])
        with t_epi_train:
            set_feedback_context("epitope", "train")
            epitope_train_ui()
        with t_drug_train:
            set_feedback_context("drug", "train")
            drug_train_ui()
        with t_torch_train:
            set_feedback_context("drug", "train_torch")
            drug_train_torch_ui()
        with t_trans_train:
            set_feedback_context("drug", "train_transformer")
            drug_transformer_train_ui()
        with t_dock_train:
            set_feedback_context("docking", "train")
            docking_train_ui()
        with t_epi_self:
            set_feedback_context("epitope", "self_train")
            epitope_self_train_ui()
        with t_drug_self:
            set_feedback_context("drug", "self_train")
            drug_self_train_ui()

    with tab_whitebox:
        set_feedback_context("whitebox", "ui")
        render_whitebox_panel(_PROJECT_ROOT)

    with tab_ms:
        set_feedback_context("multiscale", "ui")
        try:
            multiscale_ui()
        except Exception as e:
            st.error(f"加载多尺度界面失败：{e}")
            try:
                st.exception(e)
            except Exception:
                pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        msg = (
            "看起来您遇到了一些麻烦，请联系开发者18806370529@163.com，我们会尽快解决问题。\n"
            "如是用户输入或环境导致，请先检查：CSV 列名/格式、模型是否匹配、依赖是否安装。"
        )
        try:
            st.error(msg)
            st.exception(e)
        except Exception:
            print(msg)
