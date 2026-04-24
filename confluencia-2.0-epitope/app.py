from __future__ import annotations

import io
import importlib.util as importlib_util
import json
import hashlib
import platform
import sys
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from core.reliability import credible_eval_epitope as core_credible_eval_epitope
from core.features import FeatureSpec as _EpitopeFeatureSpec
from core.training import (
    EpitopeTrainingReport,
    build_artifacts_from_model,
    export_epitope_model_bytes,
    import_epitope_model_bytes,
    predict_epitope_model,
    train_epitope_model,
)
from core.pipeline import SensitivityArtifacts
from core.torch_mamba import (
    TorchMambaConfig,
    real_mamba_available,
    torch_available,
)
from core.cloud_client import CloudEpitopeClient, CloudHealthStatus
from core.cloud_config import CloudConfig, load_cloud_config, save_cloud_config

st.set_page_config(page_title="Confluencia 2.0 表位模块", layout="wide")

st.title("Confluencia 2.0 表位预测与训练")
st.caption("面向 circRNA 的微观疗效预测，支持 PyTorch Mamba 训练与多邻域敏感性分析")


def _ssm_unavailable_reason() -> str:
    if platform.system().lower() == "windows":
        return "当前是 Windows 环境，requirements 配置会跳过 mamba-ssm。"
    if not torch_available():
        return "PyTorch 不可用，无法启用 mamba-ssm。"
    if importlib_util.find_spec("mamba_ssm") is None:
        return "当前 Python 环境未安装 mamba-ssm。"
    return "mamba-ssm 导入失败，已自动切换到 fallback 模块。"


def _input_template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "epitope_seq": ["SLYNTVATL", "GILGFVFTL"],
            "dose": [2.0, 1.0],
            "freq": [1.0, 0.8],
            "treatment_time": [24.0, 48.0],
            "circ_expr": [1.2, 0.6],
            "ifn_score": [0.7, 0.5],
            "efficacy": [1.8, 1.1],
        }
    )


def _read_uploaded_csv(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    last_err: Exception | None = None
    for enc in ["utf-8", "utf-8-sig", "gbk", "gb18030"]:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception as e:
            last_err = e
    if last_err is not None:
        raise last_err
    return pd.read_csv(io.BytesIO(raw))


def _normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "epitope_seq" in out.columns:
        out["epitope_seq"] = out["epitope_seq"].fillna("").astype(str).str.strip()
    for c in ["dose", "freq", "treatment_time", "circ_expr", "ifn_score", "efficacy"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _apply_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    alias_map = {
        "epitope_seq": ["epitope", "sequence", "peptide_seq"],
        "dose": ["dosage", "dose_mg"],
        "freq": ["frequency", "times", "dose_freq"],
        "treatment_time": ["time", "duration_h", "treatment_hours"],
        "circ_expr": ["circRNA_expr", "circ_expression", "circ_score"],
        "ifn_score": ["ifn", "ifn_gamma", "ifn_g_score"],
        "efficacy": ["label", "target", "y"],
    }
    for canonical, aliases in alias_map.items():
        if canonical in out.columns:
            continue
        for a in aliases:
            if a in out.columns:
                out = out.rename(columns={a: canonical})
                break
    return out


def _core_ready_ratio(df: pd.DataFrame, cols: list[str]) -> float:
    if not cols:
        return 1.0
    hit = sum(1 for c in cols if c in df.columns)
    return float(hit) / float(len(cols))


@st.cache_data(show_spinner=False)
def _demo_data(n: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    aa = list("ACDEFGHIKLMNPQRSTVWY")

    def rand_seq(lo: int = 12, hi: int = 180) -> str:
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
        0.15 * seq_len
        + 0.45 * df["dose"].to_numpy(dtype=np.float32)
        + 0.32 * df["freq"].to_numpy(dtype=np.float32)
        + 0.28 * df["circ_expr"].to_numpy(dtype=np.float32)
        + 0.21 * df["ifn_score"].to_numpy(dtype=np.float32)
        + noise
    )
    return df


@st.cache_data(show_spinner=False)
def _data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = max(len(df), 1)
    for c in df.columns:
        miss = float(df[c].isna().sum()) / float(n)
        rows.append({"列名": c, "缺失率": miss})
    if "epitope_seq" in df.columns:
        lengths = df["epitope_seq"].astype(str).str.len()
        rows.append({"列名": "seq_len_mean", "缺失率": float(lengths.mean())})
        rows.append({"列名": "seq_len_std", "缺失率": float(lengths.std(ddof=0))})
    return pd.DataFrame(rows)


def _append_experiment_log(module: str, config: dict, metrics: dict) -> None:
    base = Path(__file__).resolve().parents[1] / "logs"
    base.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "module": module,
        "config": json.dumps(config, ensure_ascii=False),
        "metrics": json.dumps(metrics, ensure_ascii=False),
    }
    csv_path = base / "experiments.csv"
    line_df = pd.DataFrame([row])
    if csv_path.exists():
        line_df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        line_df.to_csv(csv_path, index=False)


def _hash_dataframe(df: pd.DataFrame) -> str:
    csv_text = df.to_csv(index=False)
    return hashlib.sha256(csv_text.encode("utf-8")).hexdigest()


def _snapshot_env_deps() -> dict[str, str]:
    deps: dict[str, str] = {}
    for pkg in ["python", "numpy", "pandas", "scikit-learn", "streamlit", "torch"]:
        if pkg == "python":
            deps[pkg] = platform.python_version()
            continue
        try:
            deps[pkg] = importlib_metadata.version(pkg)
        except Exception:
            deps[pkg] = "not-installed"
    return deps


def _save_repro_bundle(module: str, data_df: pd.DataFrame, config: dict, metrics: dict) -> None:
    base = Path(__file__).resolve().parents[1] / "logs" / "reproduce"
    base.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_hash = _hash_dataframe(data_df)
    run_id = f"{module}_{ts}_{data_hash[:8]}"
    env_deps = _snapshot_env_deps()

    row = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "module": module,
        "rows": int(len(data_df)),
        "data_sha256": data_hash,
        "python_executable": sys.executable,
        "config": json.dumps(config, ensure_ascii=False),
        "metrics": json.dumps(metrics, ensure_ascii=False),
        "env_deps": json.dumps(env_deps, ensure_ascii=False),
    }

    csv_path = base / "runs.csv"
    row_df = pd.DataFrame([row])
    if csv_path.exists():
        row_df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        row_df.to_csv(csv_path, index=False)

    md_path = base / f"{run_id}.md"
    md_lines = [
        f"# Repro Report ({module})",
        f"- run_id: {run_id}",
        f"- timestamp: {row['timestamp']}",
        f"- rows: {row['rows']}",
        f"- data_sha256: {data_hash}",
        f"- python_executable: {sys.executable}",
        "",
        "## Config",
        "```json",
        json.dumps(config, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Metrics",
        "```json",
        json.dumps(metrics, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Environment Dependencies",
        "```json",
        json.dumps(env_deps, ensure_ascii=False, indent=2),
        "```",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")


with st.sidebar:
    st.header("运行设置")

    # ---- 云服务器接口插槽 ----
    with st.expander("云服务器接口", expanded=False):
        st.caption("配置远程云服务器以实现云端训练和预测")
        cloud_server_url = st.text_input("服务器地址", value="", placeholder="https://your-server.example.com", key="cloud_url")
        cloud_token = st.text_input("API Token", value="", type="password", key="cloud_token")
        cloud_mode = st.selectbox("运行模式", ["本地运行", "云端运行", "混合模式"], index=0, key="cloud_mode_select")
        cloud_api_prefix = st.text_input("API 路径前缀", value="/api/v1", key="cloud_api_prefix")
        cloud_timeout = st.number_input("请求超时(秒)", min_value=10, max_value=3600, value=300, key="cloud_timeout")

        # 初始化云客户端
        cloud_client: CloudEpitopeClient | None = None
        cloud_connected = False

        if cloud_server_url and cloud_token:
            try:
                _mode_map = {"本地运行": "local", "云端运行": "cloud", "混合模式": "hybrid"}
                _sel_mode = _mode_map.get(cloud_mode, "local")
                cloud_client = CloudEpitopeClient.from_params(
                    server_url=cloud_server_url,
                    token=cloud_token,
                    api_prefix=cloud_api_prefix,
                    timeout=int(cloud_timeout),
                    mode=_sel_mode,
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
            # 尝试从配置文件加载
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

        # 保存配置到文件
        if st.button("保存云配置", key="save_cloud_cfg"):
            try:
                _mode_map2 = {"本地运行": "local", "云端运行": "cloud", "混合模式": "hybrid"}
                _save_cfg = CloudConfig(
                    enabled=bool(cloud_server_url and cloud_token),
                    server=__import__("core.cloud_config", fromlist=["CloudServerConfig"]).CloudServerConfig(
                        url=cloud_server_url or "",
                        api_prefix=cloud_api_prefix,
                        timeout=int(cloud_timeout),
                    ),
                    auth=__import__("core.cloud_config", fromlist=["CloudAuthConfig"]).CloudAuthConfig(
                        token=cloud_token or "",
                    ),
                    mode=__import__("core.cloud_config", fromlist=["CloudModeConfig"]).CloudModeConfig(
                        default=_mode_map2.get(cloud_mode, "local"),
                    ),
                )
                saved_path = save_cloud_config(_save_cfg)
                st.success(f"配置已保存至 {saved_path}")
            except Exception as e:
                st.warning(f"保存失败: {e}")

    # ---- 原有设置 ----
    doc_mode = st.radio("说明模式", ["新手版", "专家版"], index=0, horizontal=True)
    preset = st.selectbox("参数预设", ["平衡", "快速", "高精度"], index=0)
    backend_map = {
        "Confluencia 2.0 Torch-Mamba": "torch-mamba",
        "Confluencia 2.0 sklearn-MOE": "sklearn-moe",
        "Confluencia 1.0 HGB": "hgb",
        "Confluencia 1.0 GBR": "gbr",
        "Confluencia 1.0 RF": "rf",
        "Confluencia 1.0 Ridge": "ridge",
        "Confluencia 1.0 MLP": "mlp",
    }
    backend_label = st.selectbox("训练算法后端", list(backend_map.keys()), index=2)
    model_backend = backend_map[backend_label]
    st.caption("默认使用 Confluencia 1.0 HGB，可切换到其他 1.0 算法。")
    compute_mode = st.selectbox("计算资源档位", ["auto", "low", "medium", "high"], index=0)
    run_mode = st.selectbox("运行模式", ["训练并预测", "仅查看最近结果"], index=0)
    do_benchmark = st.checkbox("启用后端对比", value=False)
    use_demo = st.checkbox("使用内置演示数据", value=True)
    sensitivity_idx = st.number_input("敏感性样本索引", min_value=0, value=0, step=1)

    if preset == "快速":
        compute_mode = "low"
    elif preset == "高精度":
        if compute_mode == "auto":
            compute_mode = "high"

    st.caption(f"PyTorch 可用: {'是' if torch_available() else '否'}")
    st.caption(f"mamba-ssm 可用: {'是' if real_mamba_available() else '否'}")
    if not real_mamba_available():
        st.info(f"SSM 不可用原因: {_ssm_unavailable_reason()}")

    with st.expander("训练超参数", expanded=False):
        torch_epochs = st.slider("训练轮数 (Epochs)", min_value=5, max_value=120, value=40, step=5)
        torch_batch_size = st.select_slider("批大小 (Batch Size)", options=[16, 32, 64, 96, 128], value=64)
        torch_lr = st.select_slider("学习率", options=[5e-4, 1e-3, 2e-3, 3e-3, 5e-3], value=2e-3)
        torch_d_model = st.select_slider("隐藏维度 (d_model)", options=[64, 96, 128, 160], value=96)
        torch_layers = st.select_slider("Mamba 层数", options=[1, 2, 3, 4], value=2)
        torch_max_len = st.select_slider("最大序列长度", options=[256, 512, 768, 1024, 1536, 2048], value=1024)

    with st.expander("预训练编码器配置", expanded=False):
        st.caption("启用 ESM-2 蛋白语言模型和 MHC 伪序列编码器以增强表位特征表示")
        use_esm2_epi = st.checkbox("启用 ESM-2 蛋白序列编码", value=False, help="facebook/esm2 系列，1280维(650M)，需网络下载")
        if use_esm2_epi:
            esm2_model_size = st.selectbox("ESM-2 模型大小", ["650M", "150M", "35M", "8M"], index=0, help="越大越准但越慢，650M=1280维, 8M=320维")
            esm2_pca_dim = st.selectbox("ESM-2 PCA 降维", [0, 32, 64, 128, 256], index=0, format_func=lambda x: "不降维" if x == 0 else f"降至 {x} 维", help="PCA 降维可大幅减少特征维度同时保留主要信息")
            esm2_cache_dir = st.text_input("ESM-2 缓存目录", value="D:/IGEM集成方案/data/cache/esm2", help="ESM-2 嵌入缓存存放路径")
        else:
            esm2_model_size = "650M"
            esm2_pca_dim = 0
            esm2_cache_dir = ""
        use_mhc_epi = st.checkbox("启用 MHC 伪序列编码", value=False, help="MHC Class I 等位基因伪序列特征，979维，AUC=0.917")
        if use_mhc_epi:
            mhc_allele_col = st.text_input("MHC 等位基因列名", value="mhc_allele", help="CSV 中 MHC 等位基因列的列名，如 HLA-A*02:01")
        else:
            mhc_allele_col = "mhc_allele"
        # 显示预计特征维度
        _est_dim = 317  # mamba + kmer + biochem + env baseline
        if use_esm2_epi:
            _raw_dim = {"650M": 1280, "150M": 640, "35M": 480, "8M": 320}[esm2_model_size]
            _est_dim += (esm2_pca_dim if esm2_pca_dim > 0 else _raw_dim)
        if use_mhc_epi:
            _est_dim += 979
        st.caption(f"预计特征维度: ~{_est_dim} 维（默认 317 维）")

    with st.expander("超参数调优", expanded=False):
        st.caption("启用后将对 sklearn 后端 (HGB/RF/Ridge/MLP) 执行交叉验证超参数搜索")
        tune_hyperparams = st.checkbox("启用超参数调优", value=False, help="在训练前对 HGB/RF/Ridge/MLP 参数执行 RandomizedSearchCV，耗时较长")
        if tune_hyperparams:
            tune_strategy = st.selectbox("调优策略", ["random", "grid"], index=0, help="random: 随机采样 n_iter 组配置; grid: 穷举所有组合")
            tune_n_iter = st.slider("随机搜索迭代次数", min_value=5, max_value=50, value=20, step=5, help="仅 random 策略生效")
            tune_cv = st.slider("调优交叉验证折数", min_value=2, max_value=5, value=3, step=1)
        else:
            tune_strategy = "random"
            tune_n_iter = 20
            tune_cv = 3

    with st.expander("参数介绍", expanded=False):
        if doc_mode == "新手版":
            st.markdown(
                """
                - `模型后端`：深度模型更强，轻量模型更快。
                - `训练算法后端`：支持 Confluencia 2.0 与 Confluencia 1.0 算法族切换。
                - `计算资源档位`：越高通常越准，但耗时更长。
                - `敏感性样本索引`：查看某个样本的解释结果。
                - `训练轮数`：训练次数；先从 20-40 轮开始。
                - `学习率`：太大不稳定，太小收敛慢，默认值通常可用。
                """
            )
        else:
            st.markdown(
                """
                                - `训练算法后端`：
                                    `torch-mamba/sklearn-moe` 为 Confluencia 2.0；
                                    `hgb/gbr/rf/ridge/mlp` 为 Confluencia 1.0 回归后端。
                - `计算资源档位`：影响专家规模/折数或计算预算，`auto` 按样本量自适配。
                - `敏感性样本索引`：指定解释对象，输出局部特征与邻域贡献。
                - `训练轮数`：优化迭代上限；高轮数需配合早停观察过拟合。
                - `批大小`：影响梯度噪声与吞吐，增大会提高稳定性但占用显存。
                - `学习率`：控制更新步长，建议与批大小联动调节。
                - `隐藏维度(d_model)`：主干容量参数，影响表征能力与计算量。
                - `Mamba 层数`：时序依赖建模深度，通常 2-4 层可覆盖大多数场景。
                - `最大序列长度`：输入截断上限，影响长序列保真与显存消耗。
                """
            )

with st.expander("机制解释（表位模块）", expanded=False):
    if doc_mode == "新手版":
        st.markdown(
            """
            - 系统会把表位序列和给药条件一起学习，输出疗效预测值。
            - 你可以看敏感性结果，知道模型主要依据了哪些信息。
            - 如果没有真实标签，会用代理目标跑通流程，用于前期筛选。
            """
        )
    else:
        st.markdown(
            """
            - 数据流：输入 `epitope_seq` 与环境变量（`dose/freq/treatment_time/circ_expr/ifn_score`），输出 `efficacy_pred`。
            - 训练机制：
              1. `torch-mamba`：序列编码器学习 token 依赖并融合环境变量做回归。
              2. `sklearn-moe`：多专家回归器加权融合，适合轻量与小样本场景。
                            3. Confluencia 1.0 后端：`hgb/gbr/rf/ridge/mlp`，用于经典基线与可复现实验。
            - 敏感性机制：计算局部特征/邻域贡献，量化预测驱动因素。
            - 标签机制：若缺少 `efficacy`，使用代理目标训练，仅用于流程验证和相对比较。
            """
        )

st.subheader("快速上手")
st.markdown("1. 下载模板并填数据。 2. 上传 CSV 或使用演示数据。 3. 左栏点击`仅训练模型`。 4. 右栏点击`仅执行预测`并在下方下载结果。")

tpl_buf = io.StringIO()
_input_template().to_csv(tpl_buf, index=False)
st.download_button(
    "下载输入模板 CSV",
    data=tpl_buf.getvalue(),
    file_name="confluencia2_epitope_input_template.csv",
    mime="text/csv",
)

uploaded = st.file_uploader(
    "上传 CSV 文件",
    type=["csv"],
    help="建议包含: epitope_seq, dose, freq, treatment_time, circ_expr, ifn_score。可选: efficacy",
)
df = pd.DataFrame()
if use_demo:
    df = _demo_data(160)
elif uploaded is not None:
    try:
        df = _read_uploaded_csv(uploaded)
    except Exception as e:
        st.error(f"CSV 读取失败，请检查编码或分隔符: {e}")
        st.stop()
else:
    st.info("请上传 CSV 文件，或启用演示数据。")
    st.stop()

df = _normalize_input(df)
df = _apply_column_aliases(df)
if df.empty:
    st.error("输入数据为空，请检查 CSV 内容。")
    st.stop()

missing_core = [c for c in ["epitope_seq", "dose", "freq", "treatment_time", "circ_expr", "ifn_score"] if c not in df.columns]
if missing_core:
    st.warning(f"检测到缺少核心列: {', '.join(missing_core)}。系统会自动补全缺失列为默认值，但建议完善后再训练。")

bad_numeric = 0
for c in ["dose", "freq", "treatment_time", "circ_expr", "ifn_score"]:
    if c in df.columns:
        bad_numeric += int(df[c].isna().sum())
if bad_numeric > 0:
    st.warning(f"检测到 {bad_numeric} 个数值单元无法解析，训练时将按 0 处理。建议修正后重试。")

core_cols = ["epitope_seq", "dose", "freq", "treatment_time", "circ_expr", "ifn_score"]
ready_ratio = _core_ready_ratio(df, core_cols)
cq1, cq2, cq3 = st.columns(3)
cq1.metric("数据行数", f"{len(df)}")
cq2.metric("核心列完备度", f"{ready_ratio * 100:.0f}%")
cq3.metric("数值异常单元", f"{bad_numeric}")
if ready_ratio < 1.0:
    st.info("已尝试自动识别常见别名列（如 `sequence`、`frequency`、`circRNA_expr`）。建议确认映射结果。")

st.subheader("输入预览")
st.dataframe(df.head(12), use_container_width=True)

with st.expander("数据质控报告", expanded=False):
    dq = _data_quality_report(df)
    st.dataframe(dq, use_container_width=True)

with st.expander("科研可信评估设置（建议开启）", expanded=False):
    st.caption("用于严格切分、5-fold CV 置信区间、基线对照与失败样本分析。")
    enable_credible_eval = st.checkbox("启用科研可信评估", value=True)
    ce1, ce2, ce3 = st.columns(3)
    credible_seed = ce1.number_input("随机种子", min_value=1, max_value=999999, value=42, step=1)
    credible_test_ratio = ce2.select_slider("测试集比例", options=[0.1, 0.15, 0.2, 0.25], value=0.2)
    credible_val_ratio = ce3.select_slider("验证集比例", options=[0.1, 0.15, 0.2], value=0.2)

    ce4, ce5 = st.columns(2)
    credible_cv_folds = ce4.select_slider("CV 折数", options=[3, 4, 5, 6, 8, 10], value=5)
    credible_top_fail_n = ce5.slider("失败样本导出 Top-N", min_value=5, max_value=100, value=20, step=5)

    uploaded_external = st.file_uploader(
        "上传外部独立测试集 CSV（可选，需含 efficacy）",
        type=["csv"],
        key="epitope_external_eval_csv",
    )
    external_eval_df: pd.DataFrame | None = None
    if uploaded_external is not None:
        try:
            external_eval_df = _normalize_input(_apply_column_aliases(_read_uploaded_csv(uploaded_external)))
            st.success(f"外部测试集已载入: {len(external_eval_df)} 行")
        except Exception as e:
            st.error(f"外部测试集读取失败: {e}")

# =============================================================================
# MHC 特征增强结果面板 (v2.1+)
# =============================================================================
with st.expander("MHC 特征增强与结合预测 (v2.1+)", expanded=False):
    st.markdown("""
    **MHC 等位基因特征工程 + HistGradientBoosting 分类器**

    在 IEDB 真实结合数据 (97,852 peptide-allele pairs) 上验证，AUC 达到 0.917。
    """)

    # 性能指标
    c1, c2, c3 = st.columns(3)
    c1.metric("外部验证 AUC", "0.917", delta="vs HGB baseline +0.186")
    c2.metric("NetMHCpan-4.1 AUC", "0.92-0.96", delta="差距 ~0.03-0.05")
    c3.metric("结合数据对数", "97,852", delta="26.3% binders")

    st.markdown("**关键发现**")
    st.markdown("""
    1. **功效标签 ≠ 结合标签**：原 288k 数据使用功效作为结合代理，两者生物信号不同
    2. **MHC 等位基因特征是主要区分因素**：单独贡献 +0.11 AUC
    3. **ESM-2 语言模型**：在小数据集 (N<2000) 上导致过拟合，应避免用于此任务
    """)

    # 结合预测对比表
    st.markdown("**模型对比 (IEDB held-out, N=2,166)**")
    model_compare = pd.DataFrame([
        {"模型": "NetMHCpan-4.1", "AUC": "0.92-0.96", "数据类型": "真实结合测量", "备注": "金标准"},
        {"模型": "HGB + MHC pseudo-seq (v2.1)", "AUC": "0.917", "数据类型": "真实结合标签", "备注": "153 alleles, 979 dims"},
        {"模型": "HGB (无 MHC)", "AUC": "0.731", "数据类型": "功效代理标签", "备注": "288k 数据"},
        {"模型": "Logistic Regression", "AUC": "0.663", "数据类型": "功效代理标签", "备注": "线性基线"},
        {"模型": "Random Forest", "AUC": "0.725", "数据类型": "功效代理标签", "备注": "集成基线"},
        {"模型": "MLP", "AUC": "0.644", "数据类型": "功效代理标签", "备注": "神经网络基线"},
    ])
    st.dataframe(model_compare, use_container_width=True, hide_index=True)

    # ESM-2 升级计划状态
    st.markdown("**ESM-2 650M 升级计划**")
    esm2_status = pd.DataFrame([
        {"组件": "ESM-2 编码器 (core/esm2_encoder.py)", "状态": "规划中", "输出维度": "1280 (650M)"},
        {"组件": "特征矩阵集成 (core/features.py)", "状态": "规划中", "总维度": "~1597 dims"},
        {"组件": "分类器训练 (core/train_binder_classifier.py)", "状态": "规划中", "目标 AUC": "0.92+"},
        {"组件": "NetMHCpan 对比评估 (benchmarks/netmhcpan_esm2_benchmark.py)", "状态": "规划中", "数据集": "61 行 held-out"},
    ])
    st.dataframe(esm2_status, use_container_width=True, hide_index=True)
    st.caption("ESM-2 升级目标：追平 NetMHCpan-4.1 (AUC 0.92-0.96)")

if "epitope_model_bundle" not in st.session_state:
    st.session_state["epitope_model_bundle"] = None
if "epitope_train_artifacts" not in st.session_state:
    st.session_state["epitope_train_artifacts"] = None
if "epitope_train_report" not in st.session_state:
    st.session_state["epitope_train_report"] = None
if "cloud_model_id" not in st.session_state:
    st.session_state["cloud_model_id"] = None

run_train = False
run_predict = False
if run_mode == "训练并预测":
    train_col, predict_col = st.columns(2)
    with train_col:
        run_train = st.button("仅训练模型", type="primary", use_container_width=True)
    with predict_col:
        has_model = st.session_state.get("epitope_model_bundle") is not None or st.session_state.get("cloud_model_id") is not None
        run_predict = st.button("仅执行预测", disabled=not has_model, use_container_width=True)

    with st.expander("导入已有模型", expanded=False):
        st.caption("默认禁止不安全模型导入；仅在内部可信来源时手动放开。")
        uploaded_model_zip = st.file_uploader("上传模型文件 (.zip)", type=["zip"], key="epitope_main_model_import_zip")
        allow_unsafe_import = st.checkbox("允许不安全导入（pickle，仅可信内部文件）", value=False, key="epitope_main_allow_unsafe_import")
        load_imported_model = st.button("加载导入模型", disabled=uploaded_model_zip is None)
        if load_imported_model and uploaded_model_zip is not None:
            try:
                imported_bundle = import_epitope_model_bytes(
                    uploaded_model_zip.getvalue(),
                    allow_unsafe=bool(allow_unsafe_import),
                )
                imported_artifacts = build_artifacts_from_model(imported_bundle)
                st.session_state["epitope_model_bundle"] = imported_bundle
                st.session_state["epitope_train_artifacts"] = imported_artifacts
                st.session_state["epitope_train_report"] = EpitopeTrainingReport(
                    sample_count=0,
                    used_label=False,
                    metrics={"mae": 0.0, "rmse": 0.0, "r2": 0.0},
                    model_backend=imported_bundle.model_backend,
                )
                st.session_state.pop("epitope_main_result", None)
                st.success(f"模型导入成功：{imported_bundle.model_backend}")
            except Exception as e:
                st.error(f"模型导入失败: {e}")
else:
    st.info("当前为《仅查看最近结果》模式，不会触发新的训练或预测。")

torch_cfg = TorchMambaConfig(
    d_model=int(torch_d_model),
    n_layers=int(torch_layers),
    lr=float(torch_lr),
    epochs=int(torch_epochs),
    batch_size=int(torch_batch_size),
    max_len=int(torch_max_len),
)

if run_train and run_mode == "训练并预测":
    # 判断是否使用云端训练
    _use_cloud_train = cloud_connected and cloud_client is not None and cloud_client.config.should_use_cloud_train()

    if _use_cloud_train:
        with st.spinner("正在云端训练模型，请稍候..."):
            cloud_result = cloud_client.train_sync(
                df,
                model_backend=model_backend,
                compute_mode=compute_mode,
                torch_cfg={"d_model": int(torch_d_model), "n_layers": int(torch_layers), "lr": float(torch_lr), "epochs": int(torch_epochs), "batch_size": int(torch_batch_size), "max_len": int(torch_max_len)},
            )
        if cloud_result.status == "completed":
            st.session_state["cloud_model_id"] = cloud_result.model_id
            train_report = EpitopeTrainingReport(
                sample_count=cloud_result.sample_count,
                used_label=True,
                metrics=cloud_result.metrics,
                model_backend=cloud_result.model_backend,
            )
            artifacts = EpitopeArtifacts(
                compute_profile=compute_mode,
                moe_weights={},
                moe_metrics=cloud_result.metrics,
                used_proxy_label=False,
                feature_dim=0,
                env_cols=[],
                model_backend=cloud_result.model_backend,
                used_real_mamba=False,
            )
            st.session_state["epitope_train_artifacts"] = artifacts
            st.session_state["epitope_train_report"] = train_report
            st.session_state["epitope_model_bundle"] = None  # Cloud mode, no local model
            st.session_state.pop("epitope_main_result", None)
            _append_experiment_log(
                module="epitope-main-cloud",
                config={"model_backend": model_backend, "compute_mode": compute_mode, "rows": int(len(df)), "cloud": True, "model_id": cloud_result.model_id},
                metrics=cloud_result.metrics,
            )
            st.success(f"云端训练完成。模型 ID: {cloud_result.model_id}")
        else:
            st.error(f"云端训练失败: {cloud_result.error_message}")
    else:
        with st.spinner("正在进行编码、训练与敏感性分析，请稍候..."):
            _epi_spec = _EpitopeFeatureSpec(
                use_esm2=use_esm2_epi,
                esm2_model_size=esm2_model_size,
                esm2_pca_dim=esm2_pca_dim,
                esm2_cache_dir=esm2_cache_dir,
                use_mhc=use_mhc_epi,
                mhc_allele_col=mhc_allele_col,
            )
            model_bundle, train_report = train_epitope_model(
                df,
                compute_mode=compute_mode,
                model_backend=model_backend,
                torch_cfg=torch_cfg,
                feature_spec=_epi_spec,
                tune_hyperparams=tune_hyperparams,
                tune_strategy=tune_strategy,
                tune_n_iter=tune_n_iter,
                tune_cv=tune_cv,
            )
            artifacts = build_artifacts_from_model(model_bundle)
        st.session_state["epitope_model_bundle"] = model_bundle
        st.session_state["epitope_train_artifacts"] = artifacts
        st.session_state["epitope_train_report"] = train_report
        st.session_state.pop("epitope_main_result", None)
    _append_experiment_log(
        module="epitope-main",
        config={"model_backend": model_backend, "compute_mode": compute_mode, "rows": int(len(df))},
        metrics={
            "mae": float(train_report.metrics.get("mae", 0.0)),
            "rmse": float(train_report.metrics.get("rmse", 0.0)),
            "r2": float(train_report.metrics.get("r2", 0.0)),
        },
    )
    _save_repro_bundle(
        module="epitope-main",
        data_df=df,
        config={"model_backend": model_backend, "compute_mode": compute_mode, "rows": int(len(df))},
        metrics={
            "mae": float(train_report.metrics.get("mae", 0.0)),
            "rmse": float(train_report.metrics.get("rmse", 0.0)),
            "r2": float(train_report.metrics.get("r2", 0.0)),
        },
    )

    if enable_credible_eval:
        _epi_cred_spec = _EpitopeFeatureSpec(
            use_esm2=use_esm2_epi,
            esm2_model_size=esm2_model_size,
            esm2_pca_dim=esm2_pca_dim,
            esm2_cache_dir=esm2_cache_dir,
            use_mhc=use_mhc_epi,
            mhc_allele_col=mhc_allele_col,
        )
        with st.spinner("正在执行严格切分 + CV + 基线对照评估..."):
            credible = core_credible_eval_epitope(
                df=df,
                backend=model_backend,
                compute_mode=compute_mode,
                seed=int(credible_seed),
                test_ratio=float(credible_test_ratio),
                val_ratio=float(credible_val_ratio),
                cv_folds=int(credible_cv_folds),
                top_n_failures=int(credible_top_fail_n),
                torch_cfg=torch_cfg,
                external_df=external_eval_df,
                feature_spec=_epi_cred_spec,
            )
        st.session_state["epitope_credible_eval"] = credible
        _append_experiment_log(
            module="epitope-credible-eval",
            config={
                "backend": model_backend,
                "seed": int(credible_seed),
                "test_ratio": float(credible_test_ratio),
                "val_ratio": float(credible_val_ratio),
                "cv_folds": int(credible_cv_folds),
            },
            metrics={
                "enabled": float(1.0 if credible.get("enabled", False) else 0.0),
                "test_rmse": float(credible.get("test_metrics", {}).get("rmse", 0.0)) if credible.get("enabled", False) else 0.0,
                "cv_rmse_mean": float(credible.get("cv_summary", {}).get("rmse_mean", 0.0)) if credible.get("enabled", False) else 0.0,
            },
        )
        sig = credible.get("significance", {}) if credible.get("enabled", False) else {}
        ood = credible.get("ood_eval", {}) if credible.get("enabled", False) else {}
        _save_repro_bundle(
            module="epitope-credible-eval",
            data_df=df,
            config={
                "backend": model_backend,
                "seed": int(credible_seed),
                "test_ratio": float(credible_test_ratio),
                "val_ratio": float(credible_val_ratio),
                "cv_folds": int(credible_cv_folds),
            },
            metrics={
                "enabled": float(1.0 if credible.get("enabled", False) else 0.0),
                "test_rmse": float(credible.get("test_metrics", {}).get("rmse", 0.0)) if credible.get("enabled", False) else 0.0,
                "cv_rmse_mean": float(credible.get("cv_summary", {}).get("rmse_mean", 0.0)) if credible.get("enabled", False) else 0.0,
                "cv_rmse_ci95": float(credible.get("cv_summary", {}).get("rmse_ci95", 0.0)) if credible.get("enabled", False) else 0.0,
                "sign_p_value": float(sig.get("p_value", 1.0)) if isinstance(sig, dict) else 1.0,
                "sign_effect_size_dz": float(sig.get("effect_size_dz", 0.0)) if isinstance(sig, dict) else 0.0,
                "ood_ratio": float(ood.get("ood_ratio", 0.0)) if isinstance(ood, dict) else 0.0,
            },
        )
    st.success("训练完成。请点击右侧《仅执行预测》。")

if run_predict and run_mode == "训练并预测":
    model_bundle = st.session_state.get("epitope_model_bundle")
    train_report = st.session_state.get("epitope_train_report")
    artifacts = st.session_state.get("epitope_train_artifacts")
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
            st.session_state["epitope_main_result"] = (result_df, artifacts, sens, train_report)
            st.success("云端预测完成")
        else:
            st.error(f"云端预测失败: {cloud_pred.error_message}")
    elif model_bundle is None or train_report is None or artifacts is None:
        st.warning("请先点击左侧《仅训练模型》。")
    else:
        with st.spinner("正在执行预测..."):
            result_df, sens = predict_epitope_model(
                model_bundle,
                df,
                sensitivity_sample_idx=int(sensitivity_idx),
            )
        st.session_state["epitope_main_result"] = (result_df, artifacts, sens, train_report)
        st.success("预测完成")

if run_train and do_benchmark and run_mode == "训练并预测":
    with st.spinner("正在对比后端表现..."):
        rows = []
        compare_backends = ["hgb", "gbr", "rf", "ridge", "mlp"]
        p = st.progress(0, text="后端对比进行中")
        total = max(len(compare_backends), 1)
        for i, b in enumerate(compare_backends):
            _bench_spec = _EpitopeFeatureSpec(
                use_esm2=use_esm2_epi,
                esm2_model_size=esm2_model_size,
                esm2_pca_dim=esm2_pca_dim,
                esm2_cache_dir=esm2_cache_dir,
                use_mhc=use_mhc_epi,
                mhc_allele_col=mhc_allele_col,
            )
            _, report_b = train_epitope_model(
                df,
                compute_mode=compute_mode,
                model_backend=b,
                torch_cfg=torch_cfg,
                feature_spec=_bench_spec,
            )
            rows.append(
                {
                    "backend": b,
                    "mae": float(report_b.metrics.get("mae", 0.0)),
                    "rmse": float(report_b.metrics.get("rmse", 0.0)),
                    "r2": float(report_b.metrics.get("r2", 0.0)),
                }
            )
            p.progress(int((i + 1) * 100 / total), text=f"已完成: {b}")
        p.empty()
        bench_df = pd.DataFrame(rows).sort_values(["rmse", "mae"], ascending=[True, True])
        st.session_state["epitope_benchmark"] = bench_df
        _append_experiment_log(
            module="epitope-benchmark",
            config={"backends": compare_backends, "rows": int(len(df))},
            metrics={"top_backend": str(bench_df.iloc[0]["backend"]) if not bench_df.empty else "none"},
        )
        _save_repro_bundle(
            module="epitope-benchmark",
            data_df=df,
            config={"backends": compare_backends, "rows": int(len(df))},
            metrics={"top_backend": str(bench_df.iloc[0]["backend"]) if not bench_df.empty else "none"},
        )

if "epitope_benchmark" in st.session_state:
    st.subheader("后端对比")
    bench_df = st.session_state["epitope_benchmark"]
    st.dataframe(bench_df, use_container_width=True)
    if not bench_df.empty:
        st.bar_chart(bench_df.set_index("backend")[["rmse", "mae"]])

if "epitope_main_result" not in st.session_state:
    if st.session_state.get("epitope_model_bundle") is None:
        st.info("请先点击《仅训练模型》，再点击《仅执行预测》。")
    else:
        st.info("模型已训练完成，等待执行预测。")
    st.stop()

if st.runtime.exists():
    result_df, artifacts, sens, train_report = st.session_state["epitope_main_result"]

    c1, c2, c3, c4 = st.columns(4)
c1.metric("样本数", f"{len(result_df)}")
c2.metric("计算档位", artifacts.compute_profile)
c3.metric("特征维度", f"{artifacts.feature_dim}")
c4.metric("代理标签模式", "是" if artifacts.used_proxy_label else "否")

c5, c6, c7 = st.columns(3)
c5.metric("敏感性样本", f"#{sens.sample_index}")
c6.metric("局部预测值", f"{sens.prediction:.4f}")
c7.metric("后端", artifacts.model_backend)

if artifacts.model_backend == "torch-mamba":
    st.info(f"真实 mamba-ssm 模块: {'是' if artifacts.used_real_mamba else '否（已切换回退模块）'}")

st.subheader("训练模块报告")
if train_report.used_label:
    t1, t2, t3 = st.columns(3)
    t1.metric("训练 MAE", f"{train_report.metrics.get('mae', 0.0):.4f}")
    t2.metric("训练 RMSE", f"{train_report.metrics.get('rmse', 0.0):.4f}")
    t3.metric("训练 R2", f"{train_report.metrics.get('r2', 0.0):.4f}")
else:
    st.warning("当前数据未提供真实 efficacy 标签。已隐藏 MAE/RMSE/R2，结果仅用于筛选/排序，不可作为最终结论。")

if "epitope_credible_eval" in st.session_state:
    st.subheader("科研可信评估（严格切分）")
    ce = st.session_state["epitope_credible_eval"]
    if not ce.get("enabled", False):
        reason = str(ce.get("reason", "unknown"))
        if reason == "missing_label":
            st.warning("未检测到真实 efficacy 标签，已跳过严格可信评估。")
        elif reason == "too_few_samples":
            st.warning("样本量不足，无法稳定执行 train/val/test + 5-fold 评估。")
        else:
            st.warning(f"可信评估未启用: {reason}")
    else:
        if not ce.get("backend_supported", True):
            st.info(f"当前后端在可信评估中暂不支持，已使用 {ce.get('backend_used', 'hgb')} 代理评估。")

        sp = ce.get("split_sizes", {})
        s1, s2, s3 = st.columns(3)
        s1.metric("Train", str(sp.get("train", 0)))
        s2.metric("Val", str(sp.get("val", 0)))
        s3.metric("Test", str(sp.get("test", 0)))

        test_m = ce.get("test_metrics", {})
        cv_s = ce.get("cv_summary", {})
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Test RMSE", f"{float(test_m.get('rmse', 0.0)):.4f}")
        cc2.metric("CV RMSE 均值", f"{float(cv_s.get('rmse_mean', 0.0)):.4f}")
        cc3.metric("CV RMSE 95%CI", f"+/- {float(cv_s.get('rmse_ci95', 0.0)):.4f}")

        cv_table = pd.DataFrame(
            [
                {
                    "metric": "MAE",
                    "mean": float(cv_s.get("mae_mean", 0.0)),
                    "std": float(cv_s.get("mae_std", 0.0)),
                    "ci95": float(cv_s.get("mae_ci95", 0.0)),
                },
                {
                    "metric": "RMSE",
                    "mean": float(cv_s.get("rmse_mean", 0.0)),
                    "std": float(cv_s.get("rmse_std", 0.0)),
                    "ci95": float(cv_s.get("rmse_ci95", 0.0)),
                },
                {
                    "metric": "R2",
                    "mean": float(cv_s.get("r2_mean", 0.0)),
                    "std": float(cv_s.get("r2_std", 0.0)),
                    "ci95": float(cv_s.get("r2_ci95", 0.0)),
                },
            ]
        )
        st.caption("5-fold 交叉验证汇总（均值 ± 标准差）")
        st.dataframe(cv_table, use_container_width=True)

        st.caption("固定基线对照（Linear / RF / HGB，测试集）")
        bdf = ce.get("baseline_df", pd.DataFrame())
        if isinstance(bdf, pd.DataFrame) and (not bdf.empty):
            st.dataframe(bdf, use_container_width=True)

        if bool(ce.get("pass_gate", False)):
            st.success("当前后端在测试集 RMSE 上已超过固定基线最佳模型。")
        else:
            st.error("当前后端未超过固定基线最佳模型，建议先优化再宣称改进有效。")

        ic_df = ce.get("interval_calibration_df", pd.DataFrame())
        if isinstance(ic_df, pd.DataFrame) and (not ic_df.empty):
            st.caption("预测区间校准（由验证集残差校准测试集覆盖率）")
            st.dataframe(ic_df, use_container_width=True)

        leak = ce.get("leakage_audit", None)
        if isinstance(leak, dict):
            l1, l2 = st.columns(2)
            l1.metric("Train/Test 精确重叠数", f"{int(float(leak.get('overlap_count', 0.0)))}")
            l2.metric("Train/Test 重叠比例", f"{float(leak.get('overlap_ratio', 0.0)) * 100:.2f}%")

        strat_df = ce.get("length_strat_df", pd.DataFrame())
        if isinstance(strat_df, pd.DataFrame) and (not strat_df.empty):
            st.caption("长度分层评估（短肽/中长肽）")
            st.dataframe(strat_df, use_container_width=True)

        sig = ce.get("significance", None)
        if isinstance(sig, dict):
            st.caption("关键模型对比的配对显著性检验（基于测试集绝对误差）")
            sg1, sg2, sg3 = st.columns(3)
            sg1.metric("p-value", f"{float(sig.get('p_value', 1.0)):.4g}")
            sg2.metric("effect size (dz)", f"{float(sig.get('effect_size_dz', 0.0)):.4f}")
            sg3.metric("non-zero pairs", f"{int(sig.get('n_nonzero', 0))}")
            if float(sig.get("p_value", 1.0)) < 0.05:
                st.success("检验达到统计显著性（p < 0.05）。")
            else:
                st.info("检验未达到统计显著性（p >= 0.05）。")

        ood = ce.get("ood_eval", None)
        if isinstance(ood, dict):
            st.caption("OOD 子集表现报告（按训练集 5%-95%分位阈值定义）")
            od1, od2, od3 = st.columns(3)
            od1.metric("OOD 占比", f"{float(ood.get('ood_ratio', 0.0)) * 100:.2f}%")
            od2.metric("OOD 样本数", f"{int(ood.get('ood_count', 0))}")
            od3.metric("ID 样本数", f"{int(ood.get('id_count', 0))}")

            ood_m = ood.get("ood_metrics", {}) if isinstance(ood.get("ood_metrics", {}), dict) else {}
            id_m = ood.get("id_metrics", {}) if isinstance(ood.get("id_metrics", {}), dict) else {}
            ood_table = pd.DataFrame(
                [
                    {"subset": "ID", "mae": float(id_m.get("mae", 0.0)), "rmse": float(id_m.get("rmse", 0.0)), "r2": float(id_m.get("r2", 0.0))},
                    {"subset": "OOD", "mae": float(ood_m.get("mae", 0.0)), "rmse": float(ood_m.get("rmse", 0.0)), "r2": float(ood_m.get("r2", 0.0))},
                ]
            )
            st.dataframe(ood_table, use_container_width=True)

            thr_df = ood.get("feature_threshold_df", pd.DataFrame())
            if isinstance(thr_df, pd.DataFrame) and (not thr_df.empty):
                st.caption("OOD 阈值与各特征触发比例")
                st.dataframe(thr_df, use_container_width=True)

        cal = ce.get("mamba_calibration", None)
        if isinstance(cal, dict):
            st.caption("Real-Mamba 与 Fallback 标定（测试集）")
            m1, m2, m3 = st.columns(3)
            m1.metric("Real RMSE", f"{float(cal.get('real_rmse', 0.0)):.4f}")
            m2.metric("Fallback RMSE", f"{float(cal.get('fallback_rmse', 0.0)):.4f}")
            m3.metric("Delta(Fallback-Real)", f"{float(cal.get('delta_rmse', 0.0)):.4f}")
            if not bool(cal.get("real_mamba_used", False)):
                st.info("当前环境不可用真实 mamba-ssm，Real 结果等价于回退路径。")

        aa_strat = ce.get("aa_composition_strat_df", pd.DataFrame())
        if isinstance(aa_strat, pd.DataFrame) and (not aa_strat.empty):
            st.caption("氨基酸组成分层评估（疏水/带电/芳香比例）")
            st.dataframe(aa_strat, use_container_width=True)

        ext_m = ce.get("external_metrics", None)
        if isinstance(ext_m, dict):
            st.caption("外部独立测试集指标")
            e1, e2, e3 = st.columns(3)
            e1.metric("External MAE", f"{float(ext_m.get('mae', 0.0)):.4f}")
            e2.metric("External RMSE", f"{float(ext_m.get('rmse', 0.0)):.4f}")
            e3.metric("External R2", f"{float(ext_m.get('r2', 0.0)):.4f}")

        fail_df = ce.get("failure_df", pd.DataFrame())
        if isinstance(fail_df, pd.DataFrame) and (not fail_df.empty):
            st.caption("失败样本 Top-N（按绝对误差）")
            show_fail_cols = [c for c in ["epitope_seq", "seq_len", "dose", "freq", "treatment_time", "circ_expr", "ifn_score", "y_true", "y_pred", "abs_error"] if c in fail_df.columns]
            st.dataframe(fail_df[show_fail_cols], use_container_width=True)
            fail_buf = io.StringIO()
            fail_df.to_csv(fail_buf, index=False)
            st.download_button(
                "下载失败样本 CSV",
                data=fail_buf.getvalue(),
                file_name="confluencia2_epitope_failure_samples.csv",
                mime="text/csv",
            )

st.subheader("MOE 可解释性")
col_a, col_b = st.columns(2)
with col_a:
    w_df = pd.DataFrame({"expert": list(artifacts.moe_weights.keys()), "weight": list(artifacts.moe_weights.values())})
    if not w_df.empty:
        st.bar_chart(w_df.set_index("expert"))
    else:
        st.info("当前后端下无可展示的 MOE 权重")
with col_b:
    m_df = pd.DataFrame({"metric": list(artifacts.moe_metrics.keys()), "value": list(artifacts.moe_metrics.values())})
    st.dataframe(m_df, use_container_width=True)

st.subheader("预测结果")
show_cols = [c for c in ["epitope_seq", "dose", "freq", "treatment_time", "circ_expr", "ifn_score", "efficacy", "efficacy_pred", "pred_uncertainty"] if c in result_df.columns]
st.dataframe(result_df[show_cols].head(100), use_container_width=True)

st.subheader("邻域敏感性")
if not sens.neighborhood_rows.empty:
    st.bar_chart(sens.neighborhood_rows.set_index("group"))
else:
    st.info("暂无邻域敏感性结果")

st.subheader("局部敏感性 Top")
st.dataframe(sens.top_rows.head(25), use_container_width=True)

st.subheader("下载结果")
model_bundle = st.session_state.get("epitope_model_bundle")
if model_bundle is not None:
    try:
        model_bytes = export_epitope_model_bytes(model_bundle)
        st.download_button(
            "下载模型文件 (.zip)",
            data=model_bytes,
            file_name=f"confluencia2_epitope_model_{artifacts.model_backend}.zip",
            mime="application/zip",
        )
    except Exception as e:
        st.warning(f"当前模型暂不支持安全导出: {e}")

buf = io.StringIO()
result_df.to_csv(buf, index=False)
st.download_button(
    "下载预测 CSV",
    data=buf.getvalue(),
    file_name="confluencia2_epitope_predictions.csv",
    mime="text/csv",
)
report_lines = [
    "# Confluencia Epitope Report",
    f"backend: {artifacts.model_backend}",
    f"samples: {len(result_df)}",
]
if train_report.used_label:
    report_lines.extend(
        [
            f"mae: {train_report.metrics.get('mae', 0.0):.6f}",
            f"rmse: {train_report.metrics.get('rmse', 0.0):.6f}",
            f"r2: {train_report.metrics.get('r2', 0.0):.6f}",
        ]
    )
else:
    report_lines.append("label_status: no_real_efficacy_label")
    report_lines.append("note: metrics hidden; ranking-only mode")

if "epitope_credible_eval" in st.session_state:
    ce = st.session_state["epitope_credible_eval"]
    if ce.get("enabled", False):
        tm = ce.get("test_metrics", {})
        cv = ce.get("cv_summary", {})
        report_lines.extend(
            [
                f"credible_backend_used: {ce.get('backend_used', artifacts.model_backend)}",
                f"credible_test_rmse: {float(tm.get('rmse', 0.0)):.6f}",
                f"credible_cv_rmse_mean: {float(cv.get('rmse_mean', 0.0)):.6f}",
                f"credible_cv_rmse_ci95: {float(cv.get('rmse_ci95', 0.0)):.6f}",
                f"credible_pass_baseline_gate: {bool(ce.get('pass_gate', False))}",
            ]
        )
        sig = ce.get("significance", None)
        if isinstance(sig, dict):
            report_lines.extend(
                [
                    f"significance_model_a: {sig.get('model_a', 'na')}",
                    f"significance_model_b: {sig.get('model_b', 'na')}",
                    f"significance_p_value: {float(sig.get('p_value', 1.0)):.8f}",
                    f"significance_effect_size_dz: {float(sig.get('effect_size_dz', 0.0)):.8f}",
                ]
            )
        ood = ce.get("ood_eval", None)
        if isinstance(ood, dict):
            ood_m = ood.get("ood_metrics", {}) if isinstance(ood.get("ood_metrics", {}), dict) else {}
            id_m = ood.get("id_metrics", {}) if isinstance(ood.get("id_metrics", {}), dict) else {}
            report_lines.extend(
                [
                    f"ood_ratio: {float(ood.get('ood_ratio', 0.0)):.8f}",
                    f"ood_count: {int(ood.get('ood_count', 0))}",
                    f"id_count: {int(ood.get('id_count', 0))}",
                    f"ood_rmse: {float(ood_m.get('rmse', 0.0)):.8f}",
                    f"id_rmse: {float(id_m.get('rmse', 0.0)):.8f}",
                ]
            )
        aa_strat = ce.get("aa_composition_strat_df", pd.DataFrame())
        if isinstance(aa_strat, pd.DataFrame) and (not aa_strat.empty):
            report_lines.append("aa_composition_stratification:")
            for _, row in aa_strat.iterrows():
                report_lines.append(f"  {row.get('property','?')}/{row.get('bin','?')}: n={int(row.get('n',0))}, rmse={float(row.get('rmse',0.0)):.6f}, mae={float(row.get('mae',0.0)):.6f}")

report_text = "\n".join(report_lines)
st.download_button(
    "下载运行报告 (Markdown)",
    data=report_text,
    file_name="confluencia2_epitope_report.md",
    mime="text/markdown",
)

with st.expander("常见问题（Epitope）", expanded=False):
    st.markdown(
        """
        - 上传失败: 优先使用 UTF-8；也支持 GBK/GB18030 自动识别。
        - 结果不稳定: 可固定参数后重复运行，或切换为 `sklearn-moe` 对照。
        - 训练指标为 0: 常见原因是未提供 `efficacy` 标签，系统进入代理目标模式。
        - 样本索引越界: `敏感性样本索引` 建议设置在 `0` 到 `样本数-1` 范围。
        """
    )
