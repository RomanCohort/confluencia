"""src.ui.feedback -- 用户反馈系统。

从 frontend.py 提取，包含反馈上下文管理、日志记录和侧边栏渲染。
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

from src.ui.constants import _PROJECT_ROOT, IGEM_FBH_URL


def _feedback_log_path() -> Path:
    """返回反馈日志文件路径。"""
    log_dir = _PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "feedback.csv"


def set_feedback_context(module: str, page: str, extra: Optional[Dict[str, Any]] = None) -> None:
    """设置反馈上下文，记录当前模块和页面。"""
    ctx: Dict[str, Any] = {
        "module": str(module),
        "page": str(page),
    }
    if extra:
        ctx.update(extra)
    st.session_state["_feedback_context"] = ctx


def update_feedback_context(extra: Dict[str, Any]) -> None:
    """更新当前反馈上下文。"""
    ctx = st.session_state.get("_feedback_context")
    if not isinstance(ctx, dict):
        ctx = {}
    ctx.update(extra)
    st.session_state["_feedback_context"] = ctx


def _append_feedback_row(row: Dict[str, Any]) -> None:
    """追加一行反馈记录到 CSV 日志。"""
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


def render_feedback_sidebar() -> None:
    """渲染用户反馈侧边栏。"""
    with st.sidebar.expander("用户反馈", expanded=False):
        st.markdown("如需反馈，请发送邮件至：**18806370529@163.com**")
        ctx = (
            st.session_state.get("_feedback_context")
            if isinstance(st.session_state.get("_feedback_context"), dict)
            else {}
        )
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
                "context_json": json.dumps(ctx, ensure_ascii=False)
                if include_ctx and isinstance(ctx, dict)
                else "",
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
    """渲染用户指导侧边栏。"""
    with st.sidebar.expander("用户指导", expanded=False):
        st.markdown(
            f"""
**简介**
本前端集成四大模块：**表位虚拟筛选**、**药物疗效预测**、**分子对接预测（交叉注意力）**、**数据增强与去噪**。支持训练、预测、批量筛选、爬虫数据汇聚、自训练（伪标签）、诊断绘图与异常识别。

**基本用法**
1) 选择模块标签（表位/药物/对接/去噪）。
2) **训练**：上传带标注 CSV → 设置列名/模型参数 → 训练并下载模型。
3) **预测**：上传模型 + 输入序列/SMILES + 条件参数 → 得到预测值。
4) **批量筛选**：上传候选 CSV → 生成预测 CSV。
5) **对接预测**：上传对接模型 + SMILES + 蛋白序列 → 输出对接效果。
6) **爬虫数据**：填写 URL/本地表格路径 → 合并成训练数据。
7) **自训练**：上传有标注与无标注数据 → 自动伪标签扩充训练集。
8) **绘图**：上传模型 + 评估数据 → 输出回归诊断图（散点/残差）。

**常见问题自查**
1) **CSV 列名与格式**：确认含必需列（表位：`sequence`；药物：`smiles`；目标列与数值列）。
2) **数值列类型**：条件列需为数值；缺失值建议用空/NaN。
3) **模型匹配**：表位模型不要用于药物，药物模型不要用于表位。
4) **表格注释**：若文件前几行是注释（`#`/`//`/`;`/`%` 开头），系统会自动剔除。
5) **依赖环境**：药物模块需要 `rdkit`；对接预测需要 `torch`；去噪需要 `tensorflow/keras`。

**提示**：若仍无法解决，请在"用户反馈"中提交错误描述与输入样例。
**致谢**：感谢 DLEPS 团队的算法与模型支持、感谢IGEM 团队指导老师与成员的支持与反馈，感谢 GitHub、Kaggle 等开源社区的贡献。
**IGEM-FBH 官方网页**：[{IGEM_FBH_URL}]({IGEM_FBH_URL})
感谢各位用户使用！
"""
        )

        st.markdown("""
    **新增功能提示（最近更新）**
    - 爬虫新增：分子对接数据、多尺度训练数据抓取与标准化列映射。
    - 多尺度分析：支持从爬虫数据选择 SMILES 并自动填充 $D,V_{max},K_m$。
    - 云端 PINN：支持批量提交并显示结果汇总与 CSV 下载。
    - 对接训练：支持教师模型蒸馏（本地/云端）。
    - 反馈模块：支持表单提交、上下文附带与日志下载。

    **多尺度 GNN-PINN 使用指南（新增）**
    - 在"多尺度 GNN-PINN 分析"侧栏，可以选择 `SimpleGNN` 或 `EnhancedGNN`。
    - 启用 `EnhancedGNN` 时，系统会使用多头注意力 (GAT) 与残差连接以更好地表征邻域相互作用。
    - 若启用 `coeff_net`，模型将从分子嵌入预测 PDE 系数（例如扩散系数/反应速率），用于 PINN 的自适应残差项。
    - `训练并演示 PINN（本地短训）` 会在本地做短时间训练并允许你下载 checkpoint，用于后续加载或云端对比。
    - 对大量候选进行多尺度分析时建议使用云端执行（速度与资源更充足）。
    """, unsafe_allow_html=True)