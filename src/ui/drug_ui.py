"""src.ui.drug_ui -- Drug prediction UI module.
"""
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
    _load_table_from_bytes,
    _render_preprocess_panel,
    _apply_preprocess,
    _parse_kv_lines,
)
from src.ui.model_utils import (
    _list_local_models_drug,
    _list_local_models_drug_torch,
    _list_local_models_drug_transformer,
    _list_local_models_docking,
    _parse_hidden_sizes,
    _get_file_mtime,
    _load_bundle,
    _load_drug_bundle,
    _load_torch_bundle,
    _cached_joblib_from_bytes,
    _cached_joblib_from_path,
    _cached_torch_bundle_from_bytes,
    _cached_torch_bundle_from_path,
    _cached_docking_bundle_from_bytes,
    _cached_docking_bundle_from_path,
    _cached_transformer_bundle_from_bytes,
    _cached_transformer_bundle_from_path,
    _cached_molecule_featurizer,
    _render_plot_images,
    _bundle_input_vector,
    _make_x_only_epitope,
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

try:
    from src.drug.predictor import infer_env_cols, train_bundle
except Exception:
    pass
try:
    from src.drug.featurizer import MoleculeFeatures
except Exception:
    pass
try:
    from src.drug.torch_predictor import load_torch_bundle, load_torch_bundle_from_bytes
except Exception:
    pass
try:
    from src.drug.transformer_predictor import load_transformer_bundle, load_transformer_bundle_from_bytes
except Exception:
    pass
try:
    from src.drug.docking_cross_attention import load_docking_bundle, load_docking_bundle_from_bytes
except Exception:
    pass
try:
    from src.common.plotting import save_regression_diagnostic_plots
except Exception:
    pass
try:
    from src.common.agent_api import call_openai_chat, call_raw_json, safe_parse_json
except Exception:
    pass
try:
    from src.common.training import EarlyStopping, build_scheduler
except Exception:
    pass
try:
    from src.rl_sampling import AtomPolicyNet, sample_atoms, reinforce_update
except Exception:
    pass


def _get_cloud_cfg() -> Dict[str, Any]:
    cfg = st.session_state.get("_cloud_cfg")
    if isinstance(cfg, dict):
        return cfg
    return {"enabled": False, "base_url": "", "headers": {"Content-Type": "application/json"},
            "timeout": 60, "retry": 1, "retry_backoff": 0.8, "show_resp": False, "job_path": "/jobs"}


def _submit_cloud_job(task, payload, cloud_cfg):
    if not cloud_cfg.get("enabled"):
        st.error("cloud not enabled")
        return {"ok": False, "status": 0, "text": "cloud disabled", "json": None}
    base_url = str(cloud_cfg.get("base_url", "")).strip()
    if not base_url:
        st.error("no cloud url")
        return {"ok": False, "status": 0, "text": "missing base_url", "json": None}
    url = base_url.rstrip("/") + str(cloud_cfg.get("job_path", "/jobs"))
    req_payload = {"task": task, "payload": payload}
    retries = int(cloud_cfg.get("retry", 0))
    backoff = float(cloud_cfg.get("retry_backoff", 0.8))
    last = None
    for i in range(retries + 1):
        last = _cloud_request("POST", url, headers=dict(cloud_cfg.get("headers", {})),
                              payload=req_payload, timeout=int(cloud_cfg.get("timeout", 60)))
        if last.get("ok"):
            return last
        status = int(last.get("status", 0) or 0)
        if status and status < 500:
            break
        if i < retries:
            time.sleep(backoff)
    return last or {"ok": False, "status": 0, "text": "unknown error", "json": None}


def _resolve_model_payload(uploaded, local_path, *, content_type):
    if uploaded is not None:
        name = getattr(uploaded, "name", "model")
        return _cloud_encode_bytes(uploaded.getvalue(), name, content_type)
    if local_path:
        try:
            data = Path(local_path).read_bytes()
            return _cloud_encode_bytes(data, Path(local_path).name, content_type)
        except Exception as e:
            st.error(f"read local model failed: {e}")
            return None
    return None


def _parse_cid_text(text):
    cids = []
    if not text:
        return cids
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").replace("	", " ").split()
        for p in parts:
            try:
                cids.append(int(p.strip()))
            except ValueError:
                pass
    return cids


def drug_train_ui() -> None:
    st.subheader("训练（CSV → 药物疗效模型）")
    st.caption("训练集需要至少包含 smiles 列 + 1 个目标列（例如 efficacy）。可选包含剂量/频次等数值列作为条件输入。")

    try:
        import rdkit  # type: ignore  # noqa: F401
        from src.drug.predictor import infer_env_cols, train_bundle  # type: ignore
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

    model_name = st.selectbox("模型", options=["hgb", "gbr", "rf", "ridge", "mlp"], index=0)
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
                bundle, metrics = train_bundle(
                    train_df,
                    smiles_col=str(smiles_col),
                    target_col=str(target_col),
                    env_cols=list(env_cols),
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
            except Exception as e:
                st.error(f"训练失败：{e}")
                return

        st.success("训练完成")
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
    if uploaded:
        try:
            df_raw = _load_tables_from_uploads(uploaded)
            df_seeds = df_raw
            _preview_df(df_raw, title="种子数据预览")
        except Exception as e:
            st.error(f"读取文件失败：{e}")
            return

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
