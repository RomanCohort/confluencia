"""src.ui.epitope_ui -- 表位预测 UI 模块。从 frontend.py 提取。"""
from __future__ import annotations

import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.ui.constants import UPLOAD_TYPES as _UPLOAD_TYPES
from src.ui.data_utils import (
    _load_tables_from_uploads,
    _render_preprocess_panel,
    _apply_preprocess,
    _parse_kv_lines,
)
from src.ui.model_utils import (
    _list_local_models,
    _load_bundle,
    _cached_sequence_featurizer,
    _make_x_only_epitope,
    _render_plot_images,
    _bundle_input_vector,
)
from src.ui.common import (
    preview_df as _preview_df,
    _cloud_encode_dataframe,
    _cloud_encode_bytes,
    _cloud_request,
    _cloud_submit_section,
    _cloud_payload_bytes,
    _cloud_try_download_csv,
)

from src.epitope.featurizer import SequenceFeatures
from src.epitope.predictor import (
    EpitopeModelBundle,
    build_model,
    infer_env_cols,
    make_xy,
    predict_one,
    train_bundle,
)
from src.epitope.sensitivity import (
    group_importance,
    sensitivity_from_bundle,
    top_features,
    wetlab_takeaway,
    sensitivity_report,
    format_sensitivity_report,
)
from src.common.plotting import save_regression_diagnostic_plots
from src.common.agent_api import call_openai_chat, call_raw_json, safe_parse_json

from src.gnn import mol_to_graph, SimpleGNN
from src.gnn_sensitivity import sensitivity_masking, example_model_fn_factory
from src.multiscale import MultiScaleModel
from src.pinn import pinn_loss


# ----------------------------------------------------------------------
# 云端辅助函数
# ----------------------------------------------------------------------
def _get_cloud_cfg() -> Dict[str, Any]:
    """从 session_state 获取云端配置。"""
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
    """提交云端任务（带重试）。"""
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
            headers=dict(cloud_cfg.get("headers", {})),
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


def _resolve_model_payload(uploaded: Any, local_path: Optional[str], *, content_type: str) -> Optional[Dict[str, Any]]:
    """将上传文件或本地路径编码为云端传输格式。"""
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


# ----------------------------------------------------------------------
# 表位训练 UI
# ----------------------------------------------------------------------

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

    model_name = st.selectbox("模型", options=["hgb", "gbr", "rf", "mlp", "sgd"], index=0)
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

    if st.button("开始训练", type="primary"):
        train_df = df.copy()
        if drop_target_na and target_col in train_df.columns:
            train_df = train_df[train_df[target_col].notna()].copy()
        if drop_target_zero and target_col in train_df.columns:
            train_df = train_df[pd.to_numeric(train_df[target_col], errors="coerce") != 0].copy()

        with st.spinner("训练中..."):
            bundle, metrics = train_bundle(
                train_df,
                sequence_col=sequence_col,
                target_col=target_col,
                env_cols=list(env_cols),
                model_name=model_name,  # type: ignore[arg-type]
                test_size=float(test_size),
                random_state=int(seed),
                mlp_alpha=float(mlp_alpha),
                mlp_early_stopping=bool(mlp_early),
                mlp_patience=int(mlp_patience),
                sgd_alpha=float(sgd_alpha),
                sgd_l1_ratio=float(sgd_l1_ratio),
                sgd_early_stopping=bool(sgd_early),
                hgb_l2=float(hgb_l2),
            )

        st.success("训练完成")
        st.json(metrics)
        if isinstance(metrics, dict) and metrics.get("suggestions"):
            st.write("训练建议:")
            st.write(metrics.get("suggestions"))
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
        if isinstance(metrics, dict) and metrics.get("suggestions"):
            st.write("训练建议:")
            st.write(metrics.get("suggestions"))
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
            preds.append(np.asarray(model.predict(x_u), dtype=np.float32).reshape(-1))

        pred_mat = np.stack(preds, axis=0)
        mu = pred_mat.mean(axis=0)
        sigma = pred_mat.std(axis=0)

        k = max(1, int(round(len(unlabeled) * float(keep_frac))))
        keep_idx = np.argsort(sigma)[:k]
        pseudo = unlabeled.iloc[keep_idx].copy()
        pseudo[target_col] = mu[keep_idx]
        pseudo["pseudo_uncertainty_std"] = sigma[keep_idx]
        pseudo["pseudo_labeled"] = True

        labeled2["pseudo_labeled"] = False
        combined = pd.concat([labeled2, pseudo], axis=0, ignore_index=True)

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
        if isinstance(metrics, dict) and metrics.get("suggestions"):
            st.write("训练建议:")
            st.write(metrics.get("suggestions"))
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


