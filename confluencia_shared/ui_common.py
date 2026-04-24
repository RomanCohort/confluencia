from __future__ import annotations

import os
import math
import platform
import shutil
import base64
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, cast

import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype


def inject_global_styles() -> None:
    st.markdown(
        """
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
.stTabs [data-baseweb="tab-list"] {gap: 8px; flex-wrap: wrap;}
.stTabs [data-baseweb="tab"] {padding: 6px 12px; border-radius: 8px;}
.stButton>button, .stDownloadButton>button {border-radius: 8px;}
.stMetric {background: rgba(255,255,255,0.03); padding: 6px 10px; border-radius: 8px;}
</style>
""",
        unsafe_allow_html=True,
    )


def _log_slider(
    label: str,
    *,
    min_exp: float,
    max_exp: float,
    value: float,
    step: float,
    key: str,
    help_text: Optional[str] = None,
) -> float:
    safe_value = value if value > 0 else 10**min_exp
    init_exp = math.log10(safe_value)
    init_exp = max(min_exp, min(max_exp, init_exp))
    exp = st.slider(
        f"{label} (log10)",
        min_value=min_exp,
        max_value=max_exp,
        value=init_exp,
        step=step,
        key=key,
        help=help_text,
    )
    return float(10**exp)


def _cloud_request(
    method: str,
    url: str,
    *,
    headers: Dict[str, str],
    payload: Optional[Dict[str, Any]],
    timeout: int,
) -> Dict[str, Any]:
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
            return {"ok": True, "status": resp.status, "text": text, "json": j}
    except urllib.error.HTTPError as e:
        body = e.read() if hasattr(e, "read") else b""
        text = body.decode("utf-8", errors="ignore")
        return {"ok": False, "status": int(getattr(e, "code", 0)), "text": text, "json": None}
    except Exception as e:
        return {"ok": False, "status": 0, "text": str(e), "json": None}


def _cloud_cfg_from_session() -> Dict[str, Any]:
    base_url = str(st.session_state.get("cloud_base_url", ""))
    api_key = str(st.session_state.get("cloud_api_key", ""))
    timeout = int(st.session_state.get("cloud_timeout", 60))
    extra_headers_text = str(st.session_state.get("cloud_headers", "{}"))
    job_path = str(st.session_state.get("cloud_job_path", "/jobs"))

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        extra_headers = json.loads(extra_headers_text or "{}")
        if isinstance(extra_headers, dict):
            for k, v in extra_headers.items():
                headers[str(k)] = str(v)
    except Exception:
        pass

    return {
        "enabled": bool(st.session_state.get("cloud_enabled", False)),
        "base_url": base_url,
        "headers": headers,
        "timeout": timeout,
        "job_path": job_path,
    }


def _cloud_encode_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return {
        "filename": "data.csv",
        "content_b64": base64.b64encode(csv_bytes).decode("ascii"),
        "content_type": "text/csv",
        "bytes": int(len(csv_bytes)),
    }


def _cloud_submit_job(task: str, payload: Dict[str, Any]) -> None:
    cfg = _cloud_cfg_from_session()
    if not cfg.get("enabled"):
        st.error("请先启用云算力")
        return
    base_url = str(cfg.get("base_url", "")).strip()
    if not base_url:
        st.error("未配置云服务地址")
        return
    url = base_url.rstrip("/") + str(cfg.get("job_path", "/jobs"))
    req_payload = {"task": task, "payload": payload}
    res = _cloud_request(
        "POST",
        url,
        headers=cast(Dict[str, str], cfg.get("headers", {})),
        payload=req_payload,
        timeout=int(cfg.get("timeout", 60)),
    )
    if not res.get("ok"):
        st.error(f"云端请求失败 (HTTP {res.get('status')}): {res.get('text')}")
        return
    resp_json = res.get("json") or {}
    st.success("已提交云端任务")
    if isinstance(resp_json, dict) and resp_json:
        st.json(resp_json)


def render_performance_sidebar(project_root: Path) -> None:
    with st.sidebar.expander("性能与显示", expanded=False):
        st.number_input(
            "预览行数",
            min_value=5,
            max_value=200,
            value=int(st.session_state.get("preview_rows", 20)),
            step=5,
            key="preview_rows",
        )
        st.checkbox("显示数据预览", value=bool(st.session_state.get("show_preview", True)), key="show_preview")
        if st.button("清空缓存", key="clear_app_cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("缓存已清空")

        st.markdown("---")
        st.caption("当前电脑状况")
        st.caption(f"CPU 核心数: {os.cpu_count() or 0}")
        st.caption(f"Python 版本: {platform.python_version()}")
        st.caption(f"系统: {platform.system()} {platform.release()}")
        try:
            import psutil  # type: ignore

            mem = psutil.virtual_memory()
            st.caption(f"内存可用/总量: {mem.available / (1024**3):.1f} GB / {mem.total / (1024**3):.1f} GB")
        except Exception:
            try:
                if platform.system().lower().startswith("win"):
                    import ctypes

                    class _MemStatus(ctypes.Structure):
                        _fields_ = [
                            ("dwLength", ctypes.c_uint32),
                            ("dwMemoryLoad", ctypes.c_uint32),
                            ("ullTotalPhys", ctypes.c_uint64),
                            ("ullAvailPhys", ctypes.c_uint64),
                            ("ullTotalPageFile", ctypes.c_uint64),
                            ("ullAvailPageFile", ctypes.c_uint64),
                            ("ullTotalVirtual", ctypes.c_uint64),
                            ("ullAvailVirtual", ctypes.c_uint64),
                            ("ullAvailExtendedVirtual", ctypes.c_uint64),
                        ]

                    mem_status = _MemStatus()
                    mem_status.dwLength = ctypes.sizeof(_MemStatus)
                    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem_status))
                    st.caption(
                        f"内存可用/总量: {mem_status.ullAvailPhys / (1024**3):.1f} GB / {mem_status.ullTotalPhys / (1024**3):.1f} GB"
                    )
                else:
                    st.caption("内存信息: 未安装 psutil")
            except Exception:
                st.caption("内存信息: 未安装 psutil")

        try:
            total, used, free = shutil.disk_usage(project_root)
            st.caption(f"磁盘可用/总量: {free / (1024**3):.1f} GB / {total / (1024**3):.1f} GB")
        except Exception:
            pass
        try:
            import torch  # type: ignore

            cuda = "可用" if torch.cuda.is_available() else "不可用"
            st.caption(f"CUDA: {cuda}")
        except Exception:
            st.caption("CUDA: 未检测到 torch")

        st.markdown("---")
        st.caption("桌面快捷方式")
        shortcut_name = st.text_input("快捷方式名称", value="confluencia", key="shortcut_name")
        if st.button("生成桌面快捷方式", key="create_shortcut"):
            try:
                desktop = _get_desktop_dir()
                if desktop is None:
                    raise RuntimeError("未找到桌面路径")
                frontend_path = project_root / "src" / "frontend.py"
                target = Path(sys.executable)
                arguments = f"-m streamlit run \"{frontend_path}\""
                lnk_path = desktop / f"{shortcut_name}.lnk"
                _create_windows_shortcut(lnk_path, target, project_root, arguments=arguments)
                st.success(f"已生成桌面快捷方式: {lnk_path.name}")
            except Exception as e:
                st.error(f"生成快捷方式失败: {e}")


def _get_desktop_dir() -> Optional[Path]:
    candidates = []
    userprofile = os.environ.get("USERPROFILE")
    if userprofile:
        candidates.append(Path(userprofile) / "Desktop")
    onedrive = os.environ.get("OneDrive") or os.environ.get("OneDriveConsumer")
    if onedrive:
        candidates.append(Path(onedrive) / "Desktop")
    for p in candidates:
        if p.exists():
            return p
    return None


def _create_windows_shortcut(lnk_path: Path, target: Path, workdir: Path, *, arguments: str = "") -> None:
    if platform.system().lower() != "windows":
        raise RuntimeError("仅支持 Windows 生成快捷方式")
    icon_path = workdir / "assets" / "app.ico"
    icon_arg = ""
    if icon_path.exists():
        icon_arg = f"$Shortcut.IconLocation = '{icon_path}';"
    args_arg = ""
    if str(arguments).strip():
        args_arg = f"$Shortcut.Arguments = '{arguments}';"
    ps = (
        "$WScriptShell = New-Object -ComObject WScript.Shell;"
        f"$Shortcut = $WScriptShell.CreateShortcut('{lnk_path}');"
        f"$Shortcut.TargetPath = '{target}';"
        f"{args_arg}"
        f"$Shortcut.WorkingDirectory = '{workdir}';"
        f"{icon_arg}"
        "$Shortcut.Save();"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)


def render_whitebox_panel(project_root: Path) -> None:
    with st.expander("白箱化解释（逻辑回归）", expanded=False):
        st.caption("使用逻辑回归对数值型表格进行白箱化贡献分析（coef * scaled_value）")
        uploaded = st.file_uploader("上传 CSV（用于训练/解释）", type=["csv"], key="wb_csv_upload")
        if uploaded is not None:
            try:
                tmp_dir = Path("tmp")
                tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = tmp_dir / f"wb_input_{int(time.time())}.csv"
                tmp_path.write_bytes(uploaded.getvalue())
                st.session_state["wb_csv_path"] = str(tmp_path)
                st.success(f"已保存临时文件: {tmp_path.name}")
            except Exception as e:
                st.error(f"保存上传文件失败: {e}")

        csv_path = st.session_state.get("wb_csv_path", "")
        target_col = st.text_input("目标列名", value=st.session_state.get("wb_target", "label"), key="wb_target_input")
        st.session_state["wb_target"] = target_col
        model_out = st.text_input("模型保存路径", value="models/logistic_whitebox.joblib", key="wb_model_out")
        st.markdown("---")
        st.caption("训练超参数")
        # initialize defaults in session_state before creating widgets to avoid Streamlit errors
        st.session_state.setdefault("wb_C", 1.0)
        st.session_state.setdefault("wb_penalty", "l2")
        st.session_state.setdefault("wb_max_iter", 1000)
        st.session_state.setdefault("wb_class_weight", "None")
        st.session_state.setdefault("wb_solver", None)
        st.session_state.setdefault("wb_l1_ratio", 0.5)
        st.session_state.setdefault("wb_tol", 1e-4)
        st.session_state.setdefault("wb_fit_intercept", "True")
        st.session_state.setdefault("wb_warm_start", False)

        preset = st.selectbox("预设 (default/fast/accurate)", options=["default", "fast", "accurate"], index=0, key="wb_preset_select")
        if st.button("应用预设", key="wb_apply_preset"):
            if preset == "fast":
                st.session_state["wb_C"] = 0.1
                st.session_state["wb_penalty"] = "l2"
                st.session_state["wb_max_iter"] = 200
                st.session_state["wb_solver"] = "liblinear"
            elif preset == "accurate":
                st.session_state["wb_C"] = 10.0
                st.session_state["wb_penalty"] = "l2"
                st.session_state["wb_max_iter"] = 2000
                st.session_state["wb_solver"] = None
            else:
                st.session_state["wb_C"] = 1.0
                st.session_state["wb_penalty"] = "l2"
                st.session_state["wb_max_iter"] = 1000
                st.session_state["wb_solver"] = None
            st.session_state["wb_C_log"] = math.log10(float(st.session_state["wb_C"]))

        model_type = st.selectbox("替代模型类型", options=["logistic", "tree"], index=0, key="wb_model_type")

        tree_max_depth = st.slider("tree max_depth (0 表示不限制)", min_value=0, max_value=1024, value=0, step=1, key="wb_tree_max_depth")
        tree_min_leaf = st.slider("tree min_samples_leaf", min_value=1, max_value=100, value=1, step=1, key="wb_tree_min_leaf")

        C_val = _log_slider(
            "正则化强度（C, 值越小正则越强）",
            min_exp=-6.0,
            max_exp=6.0,
            value=float(st.session_state.get("wb_C", 1.0)),
            step=0.1,
            key="wb_C_log",
        )
        st.session_state["wb_C"] = C_val
        penalty = st.selectbox("惩罚项 (penalty)", options=["l2", "l1", "elasticnet", "none"], index=["l2", "l1", "elasticnet", "none"].index(st.session_state.get("wb_penalty", "l2")), key="wb_penalty")
        max_iter = st.slider("最大迭代次数 (max_iter)", min_value=10, max_value=100000, value=int(st.session_state.get("wb_max_iter", 1000)), step=10, key="wb_max_iter")
        class_weight = st.selectbox("class_weight", options=["None", "balanced"], index=0 if st.session_state.get("wb_class_weight", "None") == "None" else 1, key="wb_class_weight")
        solver = st.selectbox("solver (自动选择 None 时会根据 penalty 选择)", options=["None", "liblinear", "lbfgs", "saga", "newton-cg"], index=0, key="wb_solver")
        l1_ratio = st.slider("l1_ratio (elasticnet 时有效)", min_value=0.0, max_value=1.0, value=float(st.session_state.get("wb_l1_ratio", 0.5)), step=0.05, key="wb_l1_ratio")
        tol = _log_slider(
            "tol (优化停止阈值)",
            min_exp=-12.0,
            max_exp=0.0,
            value=float(st.session_state.get("wb_tol", 1e-4)),
            step=0.1,
            key="wb_tol_log",
        )
        st.session_state["wb_tol"] = tol
        fit_intercept = st.selectbox("fit_intercept", options=["True", "False"], index=0 if st.session_state.get("wb_fit_intercept", "True") == "True" else 1, key="wb_fit_intercept")
        warm_start = st.checkbox("warm_start", value=bool(st.session_state.get("wb_warm_start", False)), key="wb_warm_start")

        if st.button("训练逻辑回归", key="wb_train"):
            if not csv_path:
                st.error("请先上传 CSV 文件")
            elif not target_col:
                st.error("请填写目标列名")
            else:
                if _cloud_cfg_from_session().get("enabled"):
                    try:
                        df = pd.read_csv(csv_path)
                    except Exception as e:
                        st.error(f"读取 CSV 失败: {e}")
                    else:
                        payload = {
                            "data": _cloud_encode_dataframe(df),
                            "target": str(target_col),
                            "model_type": str(model_type),
                            "C": float(C_val),
                            "penalty": str(penalty),
                            "max_iter": int(max_iter),
                            "tol": float(tol),
                            "fit_intercept": str(st.session_state.get("wb_fit_intercept", "True")),
                            "solver": str(st.session_state.get("wb_solver", "")),
                            "l1_ratio": float(st.session_state.get("wb_l1_ratio", 0.0)),
                            "class_weight": str(class_weight),
                            "tree_max_depth": int(tree_max_depth),
                            "tree_min_samples_leaf": int(tree_min_leaf),
                        }
                        _cloud_submit_job("whitebox_train", payload)
                else:
                    import subprocess

                    with st.spinner("正在训练逻辑回归..."):
                        try:
                            cmd = [
                                sys.executable,
                                str(project_root / "scripts" / "logistic_whitebox.py"),
                                "train",
                                "--csv",
                                csv_path,
                                "--target",
                                target_col,
                                "--out",
                                model_out,
                                "--model-type",
                                str(model_type),
                                "--C",
                                str(C_val),
                                "--penalty",
                                str(penalty),
                                "--max-iter",
                                str(int(max_iter)),
                                "--tol",
                                str(float(tol)),
                                "--fit-intercept",
                                str(st.session_state.get("wb_fit_intercept", "True")),
                            ]
                            solver_val = st.session_state.get("wb_solver", None)
                            if solver_val:
                                cmd.extend(["--solver", str(solver_val)])
                            if st.session_state.get("wb_l1_ratio", None) is not None:
                                cmd.extend(["--l1-ratio", str(float(st.session_state.get("wb_l1_ratio")))])
                            if bool(st.session_state.get("wb_warm_start", False)):
                                cmd.append("--warm-start")
                            if class_weight != "None":
                                cmd.extend(["--class-weight", str(class_weight)])
                            if model_type == "tree":
                                if int(tree_max_depth) > 0:
                                    cmd.extend(["--max-depth", str(int(tree_max_depth))])
                                cmd.extend(["--min-samples-leaf", str(int(tree_min_leaf))])
                            subprocess.run(cmd, check=True)
                        except subprocess.CalledProcessError as e:
                            st.error(f"外部脚本执行失败: {e}")
                        except Exception as e:
                            st.error(f"训练失败: {e}")
                        else:
                            st.success("训练完成")
                            coeff_img = Path(model_out).with_name(Path(model_out).stem + "_coeffs.png")
                            roc_img = Path(model_out).with_name(Path(model_out).stem + "_roc.png")
                            if coeff_img.exists():
                                st.image(str(coeff_img), caption=coeff_img.name, use_container_width=True)
                            if roc_img.exists():
                                st.image(str(roc_img), caption=roc_img.name, use_container_width=True)

        explain_idx = st.text_input("要解释的行索引（逗号分隔）", value="0", key="wb_explain_idx")
        if st.button("生成解释", key="wb_explain"):
            csv_path = st.session_state.get("wb_csv_path", "")
            if not csv_path:
                st.error("请先上传 CSV 或先执行训练以生成临时 CSV")
            else:
                try:
                    idxs = [int(x.strip()) for x in explain_idx.split(",") if x.strip()]
                except Exception:
                    st.error("行索引格式错误，请输入逗号分隔的整数")
                    idxs = []

                if idxs:
                    if _cloud_cfg_from_session().get("enabled"):
                        try:
                            df = pd.read_csv(csv_path)
                        except Exception as e:
                            st.error(f"读取 CSV 失败: {e}")
                        else:
                            payload = {
                                "data": _cloud_encode_dataframe(df),
                                "model_path": str(model_out),
                                "model_type": str(model_type),
                                "indices": idxs,
                                "tree_max_depth": int(tree_max_depth),
                                "tree_min_samples_leaf": int(tree_min_leaf),
                            }
                            _cloud_submit_job("whitebox_explain", payload)
                    else:
                        import subprocess
                        try:
                            cmd = [
                                sys.executable,
                                str(project_root / "scripts" / "logistic_whitebox.py"),
                                "explain",
                                "--csv",
                                csv_path,
                                "--model",
                                model_out,
                                "--model-type",
                                str(model_type),
                                "--export-csv",
                                str(Path(model_out).with_name(Path(model_out).stem + "_contribs.csv")),
                            ]
                            for i in idxs:
                                cmd.append(str(i))
                            with st.spinner("正在生成解释..."):
                                if str(model_type) == "tree":
                                    if int(tree_max_depth) > 0:
                                        cmd.extend(["--max-depth", str(int(tree_max_depth))])
                                    cmd.extend(["--min-samples-leaf", str(int(tree_min_leaf))])
                                subprocess.run(cmd, check=True)
                        except subprocess.CalledProcessError as e:
                            st.error(f"外部脚本执行失败: {e}")
                        except Exception as e:
                            st.error(f"生成解释失败: {e}")
                        else:
                            st.success("解释已生成")
                            first_img = Path(model_out).with_name(Path(model_out).stem + f"_explain_{idxs[0]}.png")
                            export_csv = Path(model_out).with_name(Path(model_out).stem + "_contribs.csv")
                            if first_img.exists():
                                st.image(str(first_img), caption=first_img.name, use_container_width=True)
                            if export_csv.exists():
                                st.download_button("下载贡献 CSV", data=export_csv.read_bytes(), file_name=export_csv.name, mime="text/csv")

    with st.expander("模型参数可视化（Transformer/GNN/Torch）", expanded=False):
        st.caption("上传 .pt/.pth 模型文件以查看权重统计与分布。支持 Transformer/GNN 等通用 PyTorch state_dict。")
        uploaded = st.file_uploader("上传模型文件", type=["pt", "pth"], key="wb_torch_model")
        if uploaded is not None:
            try:
                import torch  # type: ignore

                obj = torch.load(uploaded, map_location="cpu")
                state = obj.get("model_state") if isinstance(obj, dict) and "model_state" in obj else obj
                if not isinstance(state, dict):
                    st.error("无法解析 state_dict")
                else:
                    rows = []
                    for name, tensor in state.items():
                        if not isinstance(tensor, torch.Tensor):
                            continue
                        if tensor.numel() == 0:
                            continue
                        t = tensor.float().view(-1)
                        rows.append(
                            {
                                "name": str(name),
                                "shape": tuple(tensor.shape),
                                "mean": float(t.mean().item()),
                                "std": float(t.std().item()),
                                "abs_mean": float(t.abs().mean().item()),
                                "abs_max": float(t.abs().max().item()),
                            }
                        )
                    if rows:
                        dfw = pd.DataFrame(rows).sort_values("abs_mean", ascending=False)
                        st.write("权重统计（按 abs_mean 排序）")
                        st.dataframe(dfw.head(50))
                        st.write("权重分布（abs_mean Top-20）")
                        st.bar_chart(dfw.head(20).set_index("name")["abs_mean"])
                    else:
                        st.warning("未发现可用权重张量")
            except Exception as e:
                st.error(f"加载模型失败: {e}")

    with st.expander("跨模块解释器（拟合线性替代模型）", expanded=False):
        st.caption("对其它模块输出（预测结果 CSV）拟合可解释的线性替代模型，再给出全局系数与单样本贡献。")
        mod_upload = st.file_uploader("上传模块输出 CSV（含特征列与预测列）", type=["csv"], key="mod_csv_upload")
        if mod_upload is not None:
            try:
                tmp_dir = Path("tmp")
                tmp_dir.mkdir(parents=True, exist_ok=True)
                mod_path = tmp_dir / f"mod_out_{int(time.time())}.csv"
                mod_path.write_bytes(mod_upload.getvalue())
                st.session_state["mod_csv_path"] = str(mod_path)
                st.success(f"已保存：{mod_path.name}")
            except Exception as e:
                st.error(f"保存失败：{e}")

        mod_csv = st.session_state.get("mod_csv_path", "")
        if mod_csv:
            try:
                df_mod = pd.read_csv(mod_csv)
            except Exception as e:
                st.error(f"读取 CSV 失败：{e}")
                df_mod = None
        else:
            df_mod = None

        if df_mod is not None:
            st.write("数据预览")
            st.dataframe(df_mod.head(5))
            numeric_cols = [c for c in df_mod.columns if is_numeric_dtype(df_mod[c])]
            st.caption("选择预测列（将作为替代模型的目标）")
            pred_col = st.selectbox("预测列", options=numeric_cols, index=(numeric_cols.index("prediction") if "prediction" in numeric_cols else 0), key="mod_pred_col")
            feat_choices = [c for c in numeric_cols if c != pred_col]
            st.caption("选择作为特征的列（数值型）")
            sel_feats = st.multiselect("特征列", options=feat_choices, default=feat_choices, key="mod_feat_cols")

            st.markdown("---")
            st.caption("拟合替代模型并展示：全局系数（条形图） + 单样本贡献（可选索引）")
            surrogate_out = st.text_input("替代模型保存路径", value="models/surrogate_module.joblib", key="sur_out")
            sur_C = _log_slider(
                "C (替代模型)",
                min_exp=-6.0,
                max_exp=6.0,
                value=float(st.session_state.get("sur_C", 1.0)),
                step=0.1,
                key="sur_C_log",
            )
            st.session_state["sur_C"] = sur_C
            sur_model_type = st.selectbox("替代模型类型", options=["logistic", "tree"], index=0, key="sur_model_type")
            sur_tree_max_depth = st.slider("tree max_depth (0 表示不限制)", min_value=0, max_value=1024, value=0, step=1, key="sur_tree_max_depth")
            sur_tree_min_leaf = st.slider("tree min_samples_leaf", min_value=1, max_value=100, value=1, step=1, key="sur_tree_min_leaf")
            run_sur = st.button("拟合并显示全局系数", key="run_surrogate")

            if run_sur:
                if not sel_feats:
                    st.error("请至少选择一个特征列")
                else:
                    if _cloud_cfg_from_session().get("enabled"):
                        try:
                            df_mod = pd.read_csv(mod_csv)
                        except Exception as e:
                            st.error(f"读取 CSV 失败：{e}")
                        else:
                            payload = {
                                "data": _cloud_encode_dataframe(df_mod),
                                "target": str(pred_col),
                                "features": list(sel_feats),
                                "model_type": str(sur_model_type),
                                "C": float(sur_C),
                                "tree_max_depth": int(sur_tree_max_depth),
                                "tree_min_samples_leaf": int(sur_tree_min_leaf),
                                "model_path": str(surrogate_out),
                            }
                            _cloud_submit_job("whitebox_surrogate_train", payload)
                    else:
                        import subprocess
                        csvp = mod_csv
                        try:
                            cmd = [
                                sys.executable,
                                str(project_root / "scripts" / "logistic_whitebox.py"),
                                "train",
                                "--csv",
                                csvp,
                                "--target",
                                pred_col,
                                "--out",
                                surrogate_out,
                                "--model-type",
                                str(sur_model_type),
                                "--C",
                                str(sur_C),
                            ]
                            for f in sel_feats:
                                cmd.extend(["--features", f])
                            if str(sur_model_type) == "tree":
                                if int(sur_tree_max_depth) > 0:
                                    cmd.extend(["--max-depth", str(int(sur_tree_max_depth))])
                                cmd.extend(["--min-samples-leaf", str(int(sur_tree_min_leaf))])
                            subprocess.run(cmd, check=True)
                        except subprocess.CalledProcessError as e:
                            st.error(f"替代模型训练失败: {e}")
                        except Exception as e:
                            st.error(f"训练错误: {e}")
                        else:
                            st.success("替代模型训练完成")
                            coeff_img = Path(surrogate_out).with_name(Path(surrogate_out).stem + "_coeffs.png")
                            if coeff_img.exists():
                                st.image(str(coeff_img), caption=coeff_img.name, use_container_width=True)

                explain_idx = st.text_input("要解释的行索引（逗号分隔）", value="0", key="sur_explain_idx")
                if st.button("生成并下载贡献 CSV", key="sur_explain_btn"):
                    try:
                        idxs = [int(x.strip()) for x in explain_idx.split(",") if x.strip()]
                    except Exception:
                        st.error("索引格式错误")
                        idxs = []
                    if idxs:
                        if _cloud_cfg_from_session().get("enabled"):
                            try:
                                df_mod = pd.read_csv(mod_csv)
                            except Exception as e:
                                st.error(f"读取 CSV 失败：{e}")
                            else:
                                payload = {
                                    "data": _cloud_encode_dataframe(df_mod),
                                    "model_path": str(surrogate_out),
                                    "model_type": str(sur_model_type),
                                    "indices": idxs,
                                    "tree_max_depth": int(sur_tree_max_depth),
                                    "tree_min_samples_leaf": int(sur_tree_min_leaf),
                                }
                                _cloud_submit_job("whitebox_surrogate_explain", payload)
                        else:
                            import subprocess
                            try:
                                cmd2 = [
                                    sys.executable,
                                    str(project_root / "scripts" / "logistic_whitebox.py"),
                                    "explain",
                                    "--csv",
                                    csvp,
                                    "--model",
                                    surrogate_out,
                                    "--model-type",
                                    str(sur_model_type),
                                    "--export-csv",
                                    str(Path(surrogate_out).with_name(Path(surrogate_out).stem + "_mod_contribs.csv")),
                                ]
                                for i in idxs:
                                    cmd2.append(str(i))
                                if str(sur_model_type) == "tree":
                                    if int(sur_tree_max_depth) > 0:
                                        cmd2.extend(["--max-depth", str(int(sur_tree_max_depth))])
                                    cmd2.extend(["--min-samples-leaf", str(int(sur_tree_min_leaf))])
                                subprocess.run(cmd2, check=True)
                            except Exception as e:
                                st.error(f"解释失败: {e}")
                            else:
                                export_csv = Path(surrogate_out).with_name(Path(surrogate_out).stem + "_mod_contribs.csv")
                                if export_csv.exists():
                                    st.success(f"导出完成: {export_csv}")
                                    st.download_button("下载贡献 CSV", data=export_csv.read_bytes(), file_name=export_csv.name, mime="text/csv")


def preview_rows(default: int = 20) -> int:
    val = st.session_state.get("preview_rows", default)
    try:
        n = int(val)
    except Exception:
        n = default
    return max(1, min(n, 200))


def preview_df(df: pd.DataFrame, *, title: Optional[str] = None, max_rows: Optional[int] = None) -> None:
    if not bool(st.session_state.get("show_preview", True)):
        return
    if title:
        st.write(title)
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.info("暂无可预览数据")
        return
    rows = int(max_rows) if max_rows is not None else preview_rows()
    st.dataframe(df.head(rows), use_container_width=True)
    st.caption(f"行数: {len(df):,}  列数: {len(df.columns):,}")
