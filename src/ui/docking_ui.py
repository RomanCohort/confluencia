"""src.ui.docking_ui -- 分子对接 UI 模块。

从 frontend.py 提取的三个对接函数：
- docking_train_ui()
- docking_predict_ui()
- docking_screen_ui()
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import streamlit as st

from src.ui.constants import _PROJECT_ROOT
from src.ui.common import (
    _cloud_encode_bytes,
    _cloud_encode_dataframe,
    _cloud_submit_section,
    preview_df,
)
from src.ui.feedback import update_feedback_context
from src.ui.model_utils import (
    _cached_docking_bundle_from_bytes,
    _cached_docking_bundle_from_path,
    _get_file_mtime,
    _list_local_models_docking,
)


# ----------------------------------------------------------------------
# 常量与辅助（直接从 frontend.py 复制的内部函数）
# ----------------------------------------------------------------------
_UPLOAD_TYPES: List[str] = [
    "csv", "tsv", "txt", "xlsx", "xls",
    "json", "jsonl", "ndjson",
    "parquet", "pq",
    "feather", "arrow",
]


def _resolve_model_payload(
    uploaded: Any,
    local_path: Optional[str],
    *,
    content_type: str,
) -> Optional[Dict[str, Any]]:
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


# ----------------------------------------------------------------------
# 数据加载与预处理（直接从 frontend.py 复制）
# ----------------------------------------------------------------------
def _load_tables_from_uploads(
    uploads: List[Any],
    *,
    add_source_col: str = "_source",
) -> Any:
    """Load and concatenate uploaded table files."""
    from pandas.api.types import is_numeric_dtype

    frames: List[Any] = []
    for up in uploads:
        name = getattr(up, "name", "uploaded")
        data = up.getvalue()
        try:
            if name.endswith(".parquet") or name.endswith(".pq"):
                df = st.session_state.get("_parquet_cache", {}).get(name)
                if df is None:
                    import pandas as pd
                    import io as _io
                    df = pd.read_parquet(_io.BytesIO(data))
            elif name.endswith(".jsonl") or name.endswith(".ndjson"):
                import pandas as pd
                import io as _io
                df = pd.read_json(_io.BytesIO(data), lines=True, encoding="utf-8")
            elif name.endswith(".json"):
                import pandas as pd
                import io as _io
                df = pd.read_json(_io.BytesIO(data), encoding="utf-8")
            elif name.endswith(".feather") or name.endswith(".arrow"):
                import pandas as pd
                import io as _io
                df = pd.read_feather(_io.BytesIO(data))
            elif name.endswith((".xlsx", ".xls")):
                import pandas as pd
                import io as _io
                df = pd.read_excel(_io.BytesIO(data), engine="openpyxl" if name.endswith(".xlsx") else None)
            else:
                import pandas as pd
                import io as _io
                txt = data.decode("utf-8", errors="ignore")
                # strip leading comments
                markers = ("#", "//", ";", "%")
                lines = txt.splitlines()
                i = 0
                while i < len(lines):
                    s = lines[i].strip()
                    if not s:
                        i += 1
                        continue
                    if any(s.startswith(m) for m in markers):
                        i += 1
                        continue
                    break
                txt = "\n".join(lines[i:])
                sep = "\t" if name.endswith(".tsv") else ","
                df = pd.read_csv(_io.StringIO(txt), sep=sep, encoding="utf-8")
        except Exception:
            import pandas as pd
            import io as _io
            df = pd.read_csv(_io.StringIO(data.decode("utf-8", errors="ignore")), encoding="utf-8")

        if add_source_col:
            df[add_source_col] = str(name)
        frames.append(df)

    if not frames:
        return frames[0] if frames else None
    if len(frames) == 1:
        return frames[0]
    import pandas as pd
    return pd.concat(frames, axis=0, ignore_index=True)


def _guess_numeric_columns(df: Any, *, max_cols: int = 20) -> List[str]:
    cols = []
    for c in df.columns:
        ser = df[c].apply(lambda x: str(x) if not isinstance(x, (int, float)) else x)
        ser = pd.to_numeric(df[c], errors="coerce")
        ratio = float(ser.notna().mean()) if len(ser) else 0.0
        if ratio >= 0.8:
            cols.append(str(c))
        if len(cols) >= max_cols:
            break
    return cols


def _guess_categorical_columns(df: Any, *, exclude: List[str], max_cols: int = 20) -> List[str]:
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
    df: Any,
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
) -> Any:
    import pandas as pd
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


def _render_preprocess_panel(df: Any, key_prefix: str) -> Dict[str, Any]:
    import pandas as pd

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


# ----------------------------------------------------------------------
# 对接训练 UI
# ----------------------------------------------------------------------
def docking_train_ui() -> None:
    """分子对接训练 UI：交叉注意力模型训练（SMILES × 蛋白序列）。"""
    import pandas as pd
    st.subheader("训练（交叉注意力：SMILES × 蛋白序列 → 对接效果）")
    st.caption("输入包含 smiles 与 protein（或 receptor）序列列，以及对接评分/打分列。")

    try:
        import torch  # type: ignore
        from src.drug.docking_cross_attention import (
            dump_docking_bundle,
            load_docking_bundle_from_bytes,
            train_docking_bundle,
        )
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
    preview_df(df, title="数据预览")

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


# ----------------------------------------------------------------------
# 对接单条预测 UI
# ----------------------------------------------------------------------
def docking_predict_ui() -> None:
    """分子对接单条预测 UI：SMILES × 蛋白序列。"""
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


# ----------------------------------------------------------------------
# 对接批量筛选 UI
# ----------------------------------------------------------------------
def docking_screen_ui() -> None:
    """分子对接批量筛选 UI：CSV → 对接效果。"""
    import pandas as pd
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
    preview_df(candidates, title="候选预览")

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
        preview_df(out_df, title="结果预览", max_rows=50)
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
