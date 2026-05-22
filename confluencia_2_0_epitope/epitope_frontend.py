from __future__ import annotations

import importlib.util as importlib_util
import io
import platform

import numpy as np
import pandas as pd
import streamlit as st

from core.training import (
    EpitopeTrainingReport,
    build_artifacts_from_model,
    export_epitope_model_bytes,
    import_epitope_model_bytes,
    predict_epitope_model,
    train_epitope_model,
)
from core.pipeline import SensitivityArtifacts
from core.torch_mamba import TorchMambaConfig, real_mamba_available, torch_available
from core.cloud_client import CloudEpitopeClient
from core.cloud_config import CloudConfig, load_cloud_config, save_cloud_config

st.set_page_config(page_title="Confluencia 表位前端", layout="wide", page_icon="app.png")

st.title("Confluencia 2.0 表位专用前端")
st.caption("面向 circRNA 疫苗场景的表位训练与疗效预测")


def _ssm_unavailable_reason() -> str:
    if platform.system().lower() == "windows":
        return "当前是 Windows 环境，requirements 配置会跳过 mamba-ssm。"
    if not torch_available():
        return "PyTorch 不可用，无法启用 mamba-ssm。"
    if importlib_util.find_spec("mamba_ssm") is None:
        return "当前 Python 环境未安装 mamba-ssm。"
    return "mamba-ssm 导入失败，已自动切换到 fallback 模块。"


def _demo_data(n: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    aa = list("ACDEFGHIKLMNPQRSTVWY")

    def rand_seq(lo: int = 12, hi: int = 220) -> str:
        k = int(rng.integers(lo, hi + 1))
        return "".join(rng.choice(aa, size=k).tolist())

    df = pd.DataFrame(
        {
            "epitope_seq": [rand_seq() for _ in range(n)],
            "dose": rng.uniform(0.1, 8.0, size=n),
            "freq": rng.uniform(0.5, 3.0, size=n),
            "treatment_time": rng.uniform(0.0, 96.0, size=n),
            "circ_expr": rng.uniform(0.0, 2.0, size=n),
            "ifn_score": rng.uniform(0.0, 1.8, size=n),
        }
    )

    seq_len = df["epitope_seq"].str.len().to_numpy(dtype=np.float32)
    noise = rng.normal(0.0, 0.18, size=n)
    df["efficacy"] = (
        0.12 * seq_len
        + 0.48 * df["dose"].to_numpy(dtype=np.float32)
        + 0.35 * df["freq"].to_numpy(dtype=np.float32)
        + 0.26 * df["circ_expr"].to_numpy(dtype=np.float32)
        + 0.23 * df["ifn_score"].to_numpy(dtype=np.float32)
        + noise
    )
    return df


with st.sidebar:
    st.header("表位设置")

    # ---- 云服务器接口插槽 ----
    with st.expander("云服务器接口", expanded=False):
        st.caption("配置远程云服务器以实现云端训练和预测")
        cloud_server_url = st.text_input("服务器地址", value="", placeholder="https://your-server.example.com", key="ef_cloud_url")
        cloud_token = st.text_input("API Token", value="", type="password", key="ef_cloud_token")
        cloud_mode_sel = st.selectbox("运行模式", ["本地运行", "云端运行", "混合模式"], index=0, key="ef_cloud_mode")
        cloud_api_prefix = st.text_input("API 路径前缀", value="/api/v1", key="ef_cloud_prefix")
        cloud_timeout = st.number_input("请求超时(秒)", min_value=10, max_value=3600, value=300, key="ef_cloud_timeout")

        cloud_client: CloudEpitopeClient | None = None
        cloud_connected = False

        if cloud_server_url and cloud_token:
            try:
                _mode_map = {"本地运行": "local", "云端运行": "cloud", "混合模式": "hybrid"}
                cloud_client = CloudEpitopeClient.from_params(
                    server_url=cloud_server_url,
                    token=cloud_token,
                    api_prefix=cloud_api_prefix,
                    timeout=int(cloud_timeout),
                    mode=_mode_map.get(cloud_mode_sel, "local"),
                )
                health = cloud_client.health_check()
                if health.available:
                    cloud_connected = True
                    st.success(f"已连接 | 延迟 {health.latency_ms:.0f}ms | GPU: {'是' if health.gpu_available else '否'}")
                else:
                    st.warning(f"连接失败: {health.message}")
            except Exception as e:
                st.warning(f"云配置异常: {e}")
        else:
            try:
                _file_client = CloudEpitopeClient.from_config()
                if _file_client.is_available():
                    _health = _file_client.health_check()
                    if _health.available:
                        cloud_client = _file_client
                        cloud_connected = True
                        st.info(f"已从配置文件连接 | 延迟 {_health.latency_ms:.0f}ms")
            except Exception:
                pass

        if not cloud_connected:
            st.caption("未配置云服务器，将使用本地计算")

        if st.button("保存云配置", key="ef_save_cloud_cfg"):
            try:
                _mode_map2 = {"本地运行": "local", "云端运行": "cloud", "混合模式": "hybrid"}
                from core.cloud_config import CloudServerConfig, CloudAuthConfig, CloudModeConfig
                _save_cfg = CloudConfig(
                    enabled=bool(cloud_server_url and cloud_token),
                    server=CloudServerConfig(url=cloud_server_url or "", api_prefix=cloud_api_prefix, timeout=int(cloud_timeout)),
                    auth=CloudAuthConfig(token=cloud_token or ""),
                    mode=CloudModeConfig(default=_mode_map2.get(cloud_mode_sel, "local")),
                )
                saved_path = save_cloud_config(_save_cfg)
                st.success(f"配置已保存至 {saved_path}")
            except Exception as e:
                st.warning(f"保存失败: {e}")

    # ---- 原有设置 ----
    model_backend = st.selectbox("模型后端", ["torch-mamba", "sklearn-moe"], index=0)
    compute_mode = st.selectbox("计算档位", ["auto", "low", "medium", "high"], index=0)
    use_demo = st.checkbox("使用演示数据", value=True)
    sensitivity_idx = st.number_input("敏感性样本索引", min_value=0, value=0, step=1)

    st.markdown("---")
    st.caption(f"PyTorch 可用: {'是' if torch_available() else '否'}")
    st.caption(f"mamba-ssm 可用: {'是' if real_mamba_available() else '否'}")
    if not real_mamba_available():
        st.info(f"SSM 不可用原因: {_ssm_unavailable_reason()}")

    with st.expander("Torch 超参数", expanded=False):
        torch_epochs = st.slider("训练轮数", min_value=5, max_value=120, value=40, step=5)
        torch_batch_size = st.select_slider("批大小", options=[16, 32, 64, 96, 128], value=64)
        torch_lr = st.select_slider("学习率", options=[5e-4, 1e-3, 2e-3, 3e-3, 5e-3], value=2e-3)
        torch_d_model = st.select_slider("d_model", options=[64, 96, 128, 160], value=96)
        torch_layers = st.select_slider("Mamba 层数", options=[1, 2, 3, 4], value=2)
        torch_max_len = st.select_slider("最大序列长度", options=[256, 512, 768, 1024, 1536, 2048], value=1024)

    with st.expander("检查点与恢复", expanded=False):
        ckpt_enabled = st.checkbox("启用即时保存", value=False, help="训练过程中定期保存模型参数，防止意外中断导致进度丢失")
        ckpt_save_every = st.number_input("保存间隔（epoch）", min_value=1, max_value=50, value=5, step=1, disabled=not ckpt_enabled)
        ckpt_keep_last = st.number_input("保留最近 N 个检查点", min_value=1, max_value=10, value=3, step=1, disabled=not ckpt_enabled)

        # 检查是否有可恢复的检查点
        import glob
        from pathlib import Path
        _ckpt_dir = Path("./checkpoints/epitope_train")
        _existing_ckpts = sorted(_ckpt_dir.glob("mamba_epoch*_best.npz")) if _ckpt_dir.exists() else []
        if not _existing_ckpts:
            _existing_ckpts = sorted(_ckpt_dir.glob("mamba_epoch*.npz")) if _ckpt_dir.exists() else []

        if _existing_ckpts:
            _ckpt_options = [str(p) for p in _existing_ckpts]
            st.success(f"发现 {len(_ckpt_options)} 个检查点")
            resume_checkpoint = st.selectbox("恢复训练", ["不恢复"] + _ckpt_options, index=0, disabled=not ckpt_enabled)
        else:
            resume_checkpoint = "不恢复"
            st.caption("暂无已保存的检查点")

uploaded = st.file_uploader("上传表位 CSV", type=["csv"])
if use_demo:
    df = _demo_data(180)
elif uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    st.info("请上传 CSV 文件，或启用演示模式。")
    st.stop()

if "epitope_model_bundle" not in st.session_state:
    st.session_state["epitope_model_bundle"] = None
if "epitope_train_report" not in st.session_state:
    st.session_state["epitope_train_report"] = None
if "epitope_result_df" not in st.session_state:
    st.session_state["epitope_result_df"] = None
if "epitope_sens" not in st.session_state:
    st.session_state["epitope_sens"] = None
if "epitope_artifacts" not in st.session_state:
    st.session_state["epitope_artifacts"] = None
if "cloud_model_id" not in st.session_state:
    st.session_state["cloud_model_id"] = None

st.subheader("输入预览")
st.dataframe(df.head(15), use_container_width=True)

cfg = TorchMambaConfig(
    d_model=int(torch_d_model),
    n_layers=int(torch_layers),
    lr=float(torch_lr),
    epochs=int(torch_epochs),
    batch_size=int(torch_batch_size),
    max_len=int(torch_max_len),
)

train_col, predict_col = st.columns(2)
with train_col:
    run_train = st.button("仅训练模型", type="primary", use_container_width=True)
with predict_col:
    has_model = st.session_state.get("epitope_model_bundle") is not None or st.session_state.get("cloud_model_id") is not None
    run_predict = st.button("仅执行预测", disabled=not has_model, use_container_width=True)

with st.expander("导入已有模型", expanded=False):
    st.caption("默认禁止不安全模型导入；仅在内部可信来源时手动放开。")
    uploaded_model_zip = st.file_uploader("上传模型文件 (.zip)", type=["zip"], key="epitope_frontend_model_import_zip")
    allow_unsafe_import = st.checkbox("允许不安全导入（pickle，仅可信内部文件）", value=False)
    load_imported_model = st.button("加载导入模型", disabled=uploaded_model_zip is None)
    if load_imported_model and uploaded_model_zip is not None:
        try:
            imported_bundle = import_epitope_model_bytes(
                uploaded_model_zip.getvalue(),
                allow_unsafe=bool(allow_unsafe_import),
            )
            st.session_state["epitope_model_bundle"] = imported_bundle
            st.session_state["epitope_artifacts"] = build_artifacts_from_model(imported_bundle)
            st.session_state["epitope_train_report"] = EpitopeTrainingReport(
                sample_count=0,
                used_label=False,
                metrics={"mae": 0.0, "rmse": 0.0, "r2": 0.0},
                model_backend=imported_bundle.model_backend,
            )
            st.session_state["epitope_result_df"] = None
            st.session_state["epitope_sens"] = None
            st.success(f"模型导入成功：{imported_bundle.model_backend}")
        except Exception as e:
            st.error(f"模型导入失败: {e}")

if run_train:
    # 判断是否使用云端训练
    _use_cloud_train = cloud_connected and cloud_client is not None and cloud_client.config.should_use_cloud_train()

    if _use_cloud_train:
        with st.spinner("正在云端训练模型，请稍候..."):
            from core.pipeline import EpitopeArtifacts
            cloud_result = cloud_client.train_sync(
                df,
                model_backend=model_backend,
                compute_mode=compute_mode,
                torch_cfg={"d_model": int(torch_d_model), "n_layers": int(torch_layers), "lr": float(torch_lr), "epochs": int(torch_epochs), "batch_size": int(torch_batch_size), "max_len": int(torch_max_len)},
            )
        if cloud_result.status == "completed":
            st.session_state["cloud_model_id"] = cloud_result.model_id
            st.session_state["epitope_train_report"] = EpitopeTrainingReport(
                sample_count=cloud_result.sample_count,
                used_label=True,
                metrics=cloud_result.metrics,
                model_backend=cloud_result.model_backend,
            )
            st.session_state["epitope_artifacts"] = EpitopeArtifacts(
                compute_profile=compute_mode,
                moe_weights={},
                moe_metrics=cloud_result.metrics,
                used_proxy_label=False,
                feature_dim=0,
                env_cols=[],
                model_backend=cloud_result.model_backend,
                used_real_mamba=False,
            )
            st.session_state["epitope_model_bundle"] = None
            st.session_state["epitope_result_df"] = None
            st.session_state["epitope_sens"] = None
            st.success(f"云端训练完成。模型 ID: {cloud_result.model_id}")
        else:
            st.error(f"云端训练失败: {cloud_result.error_message}")
    else:
        # 检查点参数
        _ckpt_dir = "./checkpoints/epitope_train" if ckpt_enabled else None
        _resume = resume_checkpoint if resume_checkpoint != "不恢复" else None

        # 训练进度显示
        progress_bar = st.progress(0, text="训练准备中...")
        _train_status = st.empty()
        _epoch_losses: list = []

        def _on_epoch(epoch, tr_loss, va_loss, best_val):
            _epoch_losses.append((epoch, tr_loss, va_loss, best_val))
            progress = min((epoch + 1) / torch_epochs, 1.0)
            progress_bar.progress(
                progress,
                text=f"Epoch {epoch+1}/{torch_epochs} | train={tr_loss:.4f} val={va_loss:.4f} best={best_val:.4f}"
            )

        try:
            model_bundle, train_report = train_epitope_model(
                df,
                compute_mode=compute_mode,
                model_backend=model_backend,
                torch_cfg=cfg,
                checkpoint_dir=_ckpt_dir,
                checkpoint_save_every=int(ckpt_save_every),
                checkpoint_keep_last=int(ckpt_keep_last),
                resume_from=_resume,
                on_epoch_end=_on_epoch,
            )
            st.session_state["epitope_model_bundle"] = model_bundle
            st.session_state["epitope_train_report"] = train_report
            st.session_state["epitope_artifacts"] = build_artifacts_from_model(model_bundle)
            progress_bar.empty()
            _train_status.empty()

            if _ckpt_dir:
                st.success(f"训练完成（共 {_epoch_losses[-1][0]+1} 轮）。检查点已保存至 {_ckpt_dir}")
            else:
                st.success("训练完成。你现在可以在右侧栏位执行预测。")
        except Exception as _e:
            progress_bar.empty()
            _train_status.empty()
            if _ckpt_dir:
                st.warning(f"训练中断: {_e}。检查点已保存，可从此处恢复训练。")
            else:
                st.error(f"训练失败: {_e}")
            raise

if run_predict:
    model_bundle = st.session_state.get("epitope_model_bundle")
    cloud_model_id = st.session_state.get("cloud_model_id")

    # 判断是否使用云端预测
    _use_cloud_predict = (
        cloud_connected
        and cloud_client is not None
        and cloud_client.config.should_use_cloud_predict()
        and cloud_model_id is not None
    )

    if _use_cloud_predict:
        with st.spinner("正在云端执行预测..."):
            cloud_pred = cloud_client.predict_sync(
                cloud_model_id,
                df,
                sensitivity_sample_idx=int(sensitivity_idx),
            )
        if cloud_pred.status == "completed" and cloud_pred.predictions is not None:
            result_df = df.copy()
            result_df["efficacy_pred"] = cloud_pred.predictions
            result_df["pred_uncertainty"] = np.zeros_like(cloud_pred.predictions, dtype=np.float32)
            sens_data = cloud_pred.sensitivity or {}
            sens = SensitivityArtifacts(
                sample_index=int(sens_data.get("sample_index", 0)),
                prediction=float(sens_data.get("prediction", 0.0)),
                top_rows=pd.DataFrame(sens_data.get("top_rows", [])) if sens_data.get("top_rows") else pd.DataFrame(columns=["feature", "importance", "grad"]),
                neighborhood_rows=pd.DataFrame(sens_data.get("neighborhood_rows", [])) if sens_data.get("neighborhood_rows") else pd.DataFrame(columns=["group", "importance"]),
            )
            st.session_state["epitope_result_df"] = result_df
            st.session_state["epitope_sens"] = sens
            st.success("云端预测完成。")
        else:
            st.error(f"云端预测失败: {cloud_pred.error_message}")
    elif model_bundle is None:
        st.warning("请先在左侧栏位训练模型。")
    else:
        with st.spinner("正在执行预测..."):
            result_df, sens = predict_epitope_model(
                model_bundle,
                df,
                sensitivity_sample_idx=int(sensitivity_idx),
            )
        st.session_state["epitope_result_df"] = result_df
        st.session_state["epitope_sens"] = sens
        st.success("预测完成。")

result_df = st.session_state.get("epitope_result_df")
sens = st.session_state.get("epitope_sens")
artifacts = st.session_state.get("epitope_artifacts")
train_report = st.session_state.get("epitope_train_report")

if artifacts is None:
    st.info("尚未训练模型。请先点击“仅训练模型”。")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("样本数", str(len(df)))
m2.metric("后端", artifacts.model_backend)
m3.metric("特征维度", str(artifacts.feature_dim))
m4.metric("代理标签模式", "是" if artifacts.used_proxy_label else "否")

if artifacts.model_backend == "torch-mamba":
    st.info(f"真实 mamba-ssm: {'是' if artifacts.used_real_mamba else '否（回退模块）'}")

st.subheader("训练模块报告")
t1, t2, t3 = st.columns(3)
t1.metric("MAE", f"{(train_report.metrics.get('mae', 0.0) if train_report else 0.0):.4f}")
t2.metric("RMSE", f"{(train_report.metrics.get('rmse', 0.0) if train_report else 0.0):.4f}")
t3.metric("R2", f"{(train_report.metrics.get('r2', 0.0) if train_report else 0.0):.4f}")
if train_report is not None and not train_report.used_label:
    st.info("未检测到 efficacy 标签，当前训练指标为代理目标参考值。")


tab_overview, tab_model, tab_sensitivity, tab_export = st.tabs(
    ["总览", "模型", "敏感性", "导出"]
)

with tab_overview:
    st.subheader("当前输入预览")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("预测结果预览")
    if result_df is None:
        st.info("尚未执行预测。请点击右侧“仅执行预测”。")
    else:
        show_cols = [
            c
            for c in [
                "epitope_seq",
                "dose",
                "freq",
                "treatment_time",
                "circ_expr",
                "ifn_score",
                "efficacy",
                "efficacy_pred",
                "pred_uncertainty",
            ]
            if c in result_df.columns
        ]
        st.dataframe(result_df[show_cols].head(120), use_container_width=True)

with tab_model:
    st.subheader("模型诊断")
    metric_df = pd.DataFrame({"metric": list(artifacts.moe_metrics.keys()), "value": list(artifacts.moe_metrics.values())})
    st.dataframe(metric_df, use_container_width=True)

    st.subheader("MOE 权重（仅 sklearn 后端）")
    weights_df = pd.DataFrame({"expert": list(artifacts.moe_weights.keys()), "weight": list(artifacts.moe_weights.values())})
    if weights_df.empty:
        st.info("当前后端不提供 MOE 权重。")
    else:
        st.bar_chart(weights_df.set_index("expert"))

with tab_sensitivity:
    if sens is None:
        st.info("尚未执行预测，暂无敏感性结果。")
    else:
        st.subheader(f"样本 #{sens.sample_index} 的敏感性")

        if sens.neighborhood_rows.empty:
            st.info("暂无邻域敏感性数据")
        else:
            st.bar_chart(sens.neighborhood_rows.set_index("group"))

        st.subheader("局部敏感性 Top")
        st.dataframe(sens.top_rows.head(30), use_container_width=True)

with tab_export:
    st.subheader("导出预测结果")
    model_bundle = st.session_state.get("epitope_model_bundle")
    if model_bundle is not None:
        try:
            model_bytes = export_epitope_model_bytes(model_bundle)
            st.download_button(
                "下载模型文件 (.zip)",
                data=model_bytes,
                file_name=f"confluencia2_epitope_frontend_model_{artifacts.model_backend}.zip",
                mime="application/zip",
            )
        except Exception as e:
            st.warning(f"当前模型暂不支持安全导出: {e}")

    if result_df is None:
        st.info("尚未执行预测，暂无可导出结果。")
    else:
        buf = io.StringIO()
        result_df.to_csv(buf, index=False)
        st.download_button(
            "下载表位预测 CSV",
            data=buf.getvalue(),
            file_name="confluencia2_epitope_frontend_predictions.csv",
            mime="text/csv",
        )
