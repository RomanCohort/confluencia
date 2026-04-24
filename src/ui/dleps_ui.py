"""src.ui.dleps_ui -- DLEPS 药物功效预测 UI 模块。

从 frontend.py 提取的 DLEPS 相关函数：
- dleps_ui()
- _resolve_dleps_data_dir()
- _load_array_from_uploaded() (内部辅助)
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.ui.constants import _PROJECT_ROOT
from src.ui.common import (
    _cloud_encode_bytes,
    _cloud_submit_section,
    preview_df,
)


# ----------------------------------------------------------------------
# 数据目录解析
# ----------------------------------------------------------------------
def _resolve_dleps_data_dir(project_root: Path) -> Path:
    """解析 DLEPS 数据目录，按优先级依次检查：
    1. 环境变量 DLEPS_DATA_DIR
    2. PyInstaller _MEIPASS/data
    3. DLEPS-main/DLEPS-main/data
    4. 当前目录 data/
    """
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


# ----------------------------------------------------------------------
# 内部辅助函数
# ----------------------------------------------------------------------
def _cloud_encode_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """将 DataFrame 编码为云端传输格式（Base64 CSV）。"""
    import base64
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return {
        "filename": "data.csv",
        "content_b64": base64.b64encode(csv_bytes).decode("ascii"),
        "content_type": "text/csv",
        "bytes": int(len(csv_bytes)),
    }


def _load_array_from_uploaded(uploaded: Any, expected_exts: set) -> Optional[np.ndarray]:
    """从上传文件加载 numpy 数组，支持 .npy, .npz, .h5/.hdf5。"""
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

    return None


# ----------------------------------------------------------------------
# DLEPS 主 UI
# ----------------------------------------------------------------------
def dleps_ui() -> None:
    """DLEPS 药物功效预测集成 UI。

    使用 DLEPS-main 目录下的 DLEPS 代码，输入 SMILES 字符串，
    选择疾病基因签名，预测富集评分。
    """
    st.subheader("DLEPS 药物功效预测前端")
    st.write("输入 SMILES 字符串，选择疾病基因签名，预测富集评分。")

    # ------------------------------------------------------------------
    # 内部函数：DLEPS 类获取
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 确保 DLEPS 代码路径可导入
    # ------------------------------------------------------------------
    dleps_code = _PROJECT_ROOT / "DLEPS-main" / "DLEPS-main" / "code"
    if not dleps_code.exists():
        st.error(f"未找到 DLEPS 代码目录：{dleps_code}")
        return

    if str(dleps_code) not in sys.path:
        sys.path.insert(0, str(dleps_code))

    # ------------------------------------------------------------------
    # 解析 DLEPS 数据目录
    # ------------------------------------------------------------------
    dleps_data = _resolve_dleps_data_dir(_PROJECT_ROOT)
    if not dleps_data.exists():
        st.error(
            "未找到 data 文件夹。请确认项目结构包含 data/，并从项目仓库运行该应用。\n"
            f"期望路径：{dleps_data}"
        )
        return

    # ------------------------------------------------------------------
    # 基础文件管理（可选）
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 疾病列表发现
    # ------------------------------------------------------------------
    diseases: Dict[str, tuple[str, str]] = {}
    try:
        for f in dleps_data.iterdir():
            if f.name.endswith("_up"):
                down = dleps_data / (f.name.replace("_up", "_down"))
                if down.exists():
                    diseases[f.stem.replace("_up", "")] = (str(f), str(down))
    except Exception:
        diseases = {}

    # ------------------------------------------------------------------
    # 测试工具（模拟数据）
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 添加新疾病
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 输入 SMILES
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 选择疾病
    # ------------------------------------------------------------------
    st.subheader("选择疾病")
    selected = ""
    if not diseases:
        st.warning("未找到疾病签名文件（*_up / *_down）。可以先用'测试工具'生成示例，或上传自定义签名。")
    else:
        selected = st.selectbox("选择疾病", list(diseases.keys()), key="dleps_select_disease")

    use_mock = st.checkbox("使用模拟预测（不依赖模型）", value=False, key="dleps_use_mock")

    # ------------------------------------------------------------------
    # 云端预测
    # ------------------------------------------------------------------
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

        payload: Dict[str, Any] = {
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

    # ------------------------------------------------------------------
    # 训练 UI（可选）
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 预测
    # ------------------------------------------------------------------
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