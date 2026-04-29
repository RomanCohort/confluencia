"""Confluencia 1.0 Streamlit 前端 - 主编排器。

模块化重构后的薄编排层，导入各功能模块并定义标签页结构。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 确保项目根目录在 sys.path
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
for _p in (_PROJECT_ROOT, _PROJECT_ROOT.parent):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

import streamlit as st

# ----------------------------------------------------------------------
# 共享模块导入
# ----------------------------------------------------------------------
from src.ui.constants import APP_TITLE, _PROJECT_ROOT
from src.ui.common import inject_global_styles, render_performance_sidebar, render_whitebox_panel
from src.ui.feedback import render_feedback_sidebar, render_user_guidance_sidebar, set_feedback_context

# ----------------------------------------------------------------------
# 标签页 UI 模块导入
# 尝试导入，若失败则使用占位符
# ----------------------------------------------------------------------
def _import_tab_modules():
    """延迟导入标签页模块，捕获导入错误。"""
    modules = {}

    # 表位预测相关
    try:
        from src.ui import epitope_ui
        modules["epitope_train_ui"] = epitope_ui.epitope_train_ui
        modules["epitope_predict_ui"] = epitope_ui.epitope_predict_ui
        modules["epitope_screen_ui"] = epitope_ui.epitope_screen_ui
        modules["epitope_sensitivity_ui"] = epitope_ui.epitope_sensitivity_ui
        modules["epitope_plot_ui"] = epitope_ui.epitope_plot_ui
        modules["epitope_crawl_ui"] = epitope_ui.epitope_crawl_ui
        modules["epitope_self_train_ui"] = epitope_ui.epitope_self_train_ui
    except ImportError:
        pass

    # 药物预测相关
    try:
        from src.ui import drug_ui
        modules["drug_predict_ui"] = drug_ui.drug_predict_ui
        modules["drug_train_ui"] = drug_ui.drug_train_ui
        modules["drug_train_torch_ui"] = drug_ui.drug_train_torch_ui
        modules["drug_transformer_train_ui"] = drug_ui.drug_transformer_train_ui
        modules["drug_transformer_predict_ui"] = drug_ui.drug_transformer_predict_ui
        modules["drug_screen_ui"] = drug_ui.drug_screen_ui
        modules["drug_plot_ui"] = drug_ui.drug_plot_ui
        modules["drug_pubchem_crawl_train_ui"] = drug_ui.drug_pubchem_crawl_train_ui
        modules["drug_generate_ui"] = drug_ui.drug_generate_ui
        modules["drug_legacy_demo_ui"] = drug_ui.drug_legacy_demo_ui
    except ImportError:
        pass

    # 对接预测
    try:
        from src.ui import docking_ui
        modules["docking_train_ui"] = docking_ui.docking_train_ui
        modules["docking_predict_ui"] = docking_ui.docking_predict_ui
        modules["docking_screen_ui"] = docking_ui.docking_screen_ui
    except ImportError:
        pass

    # DLEPS
    try:
        from src.ui import dleps_ui
        modules["dleps_ui"] = dleps_ui.dleps_ui
    except ImportError:
        pass

    # 多尺度分析
    try:
        from src.ui import multiscale_ui
        modules["multiscale_ui"] = multiscale_ui.multiscale_ui
    except ImportError:
        pass

    # 数据增强去噪
    try:
        from src.data_aug_denoise.ui import data_aug_denoise_ui
        modules["data_aug_denoise_ui"] = data_aug_denoise_ui
    except ImportError:
        pass

    # 四靶点基因签名
    try:
        from src.ui import four_target_ui
        modules["four_target_signature_ui"] = four_target_ui.four_target_signature_ui
        modules["four_target_combined_ui"] = four_target_ui.four_target_combined_ui
    except ImportError:
        pass

    return modules


def _placeholder_ui(name: str):
    """占位符 UI 函数，当模块导入失败时使用。"""
    def _inner():
        st.warning(f"模块 '{name}' 尚未加载，请检查是否已完成提取。")
        st.info("若需使用此功能，请运行完整版 frontend.py 或联系开发者。")
    return _inner


# ----------------------------------------------------------------------
# 云服务配置面板（内联，约 80 行）
# ----------------------------------------------------------------------
def _cloud_ui() -> dict:
    """渲染云服务配置侧边栏。"""
    import json
    import urllib.request
    import urllib.error

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

        headers = {"Content-Type": "application/json"}
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
                try:
                    req = urllib.request.Request(url, headers=headers, method="GET")
                    with urllib.request.urlopen(req, timeout=int(timeout)) as resp:
                        st.success(f"连接成功 (HTTP {resp.status})")
                        text = resp.read().decode("utf-8", errors="ignore")
                        if text:
                            st.code(text, language="json")
                except urllib.error.HTTPError as e:
                    st.error(f"连接失败 (HTTP {e.code}): {e.read().decode('utf-8', errors='ignore')}")
                except Exception as e:
                    st.error(f"连接失败: {e}")

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


# ----------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------
def main() -> None:
    """主入口：渲染页面配置、侧边栏和标签页结构。"""
    st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="🧬")
    inject_global_styles()
    st.title(f"🧬 {APP_TITLE}")

    # --- 全局侧边栏 ---
    render_user_guidance_sidebar()
    render_performance_sidebar(_PROJECT_ROOT)
    render_feedback_sidebar()
    _cloud_ui()

    # --- 获取 UI 模块 ---
    modules = _import_tab_modules()

    # 辅助函数：获取模块或占位符
    def get_ui(name: str):
        return modules.get(name, _placeholder_ui(name))

    # --- 主标签页 ---
    tab_drug, tab_epitope, tab_docking, tab_data, tab_train, tab_whitebox, tab_ms = st.tabs(
        ["药效预测", "表位预测", "对接预测", "数据处理与获取", "模型训练", "结果解释", "多尺度分析"]
    )

    # === tab_drug: 药效预测 ===
    with tab_drug:
        t_pred, t_trans_pred, t_screen, t_plot, t_dleps, t_four_target, t_legacy = st.tabs(
            ["单条预测", "Transformer预测", "批量筛选", "绘图", "DLEPS 富集", "四靶点签名", "torch预测"]
        )
        with t_pred:
            set_feedback_context("drug", "predict")
            get_ui("drug_predict_ui")()
        with t_trans_pred:
            set_feedback_context("drug", "predict_transformer")
            get_ui("drug_transformer_predict_ui")()
        with t_screen:
            set_feedback_context("drug", "screen")
            get_ui("drug_screen_ui")()
        with t_plot:
            set_feedback_context("drug", "plot")
            get_ui("drug_plot_ui")()
        with t_dleps:
            set_feedback_context("dleps", "app")
            st.markdown("### DLEPS 药物富集预测")
            st.caption("若此处仍无内容，请刷新页面或检查控制台错误日志。")
            try:
                get_ui("dleps_ui")()
            except Exception as e:
                st.error(f"加载 DLEPS 前端失败：{e}")
        with t_four_target:
            set_feedback_context("four_target", "signature")
            st.markdown("### 四靶点基因签名疗效预测")
            st.caption("基于 TROP2 / NECTIN4 / LIV-1 / B7-H4 的基因签名预测 circRNA 药物疗效。")
            try:
                get_ui("four_target_signature_ui")()
            except Exception as e:
                st.error(f"加载四靶点界面失败：{e}")
        with t_legacy:
            set_feedback_context("drug", "legacy_demo")
            get_ui("drug_legacy_demo_ui")()

    # === tab_epitope: 表位预测 ===
    with tab_epitope:
        t_pred, t_screen, t_sens, t_plot = st.tabs(["单条预测", "批量筛选", "敏感性分析", "绘图"])
        with t_pred:
            set_feedback_context("epitope", "predict")
            get_ui("epitope_predict_ui")()
        with t_screen:
            set_feedback_context("epitope", "screen")
            get_ui("epitope_screen_ui")()
        with t_sens:
            set_feedback_context("epitope", "sensitivity")
            get_ui("epitope_sensitivity_ui")()
        with t_plot:
            set_feedback_context("epitope", "plot")
            get_ui("epitope_plot_ui")()

    # === tab_docking: 对接预测 ===
    with tab_docking:
        d_pred, d_screen = st.tabs(["单条预测", "批量筛选"])
        with d_pred:
            set_feedback_context("docking", "predict")
            get_ui("docking_predict_ui")()
        with d_screen:
            set_feedback_context("docking", "screen")
            get_ui("docking_screen_ui")()

    # === tab_data: 数据处理与获取 ===
    with tab_data:
        t_denoise, t_gen, t_epi_crawl, t_drug_crawl = st.tabs(
            ["数据增强与去噪", "分子生成", "表位爬虫", "药物爬虫"]
        )
        with t_denoise:
            set_feedback_context("denoise", "ui")
            get_ui("data_aug_denoise_ui")()
        with t_gen:
            set_feedback_context("drug", "generate")
            get_ui("drug_generate_ui")()
        with t_epi_crawl:
            set_feedback_context("epitope", "crawl")
            get_ui("epitope_crawl_ui")()
        with t_drug_crawl:
            set_feedback_context("drug", "crawl_train")
            get_ui("drug_pubchem_crawl_train_ui")()

    # === tab_train: 模型训练 ===
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
            get_ui("epitope_train_ui")()
        with t_drug_train:
            set_feedback_context("drug", "train")
            get_ui("drug_train_ui")()
        with t_torch_train:
            set_feedback_context("drug", "train_torch")
            get_ui("drug_train_torch_ui")()
        with t_trans_train:
            set_feedback_context("drug", "train_transformer")
            get_ui("drug_transformer_train_ui")()
        with t_dock_train:
            set_feedback_context("docking", "train")
            get_ui("docking_train_ui")()
        with t_epi_self:
            set_feedback_context("epitope", "self_train")
            get_ui("epitope_self_train_ui")()
        with t_drug_self:
            set_feedback_context("drug", "self_train")
            get_ui("drug_self_train_ui")()

    # === tab_whitebox: 结果解释 ===
    with tab_whitebox:
        set_feedback_context("whitebox", "ui")
        render_whitebox_panel(_PROJECT_ROOT)

    # === tab_ms: 多尺度分析 ===
    with tab_ms:
        set_feedback_context("multiscale", "ui")
        try:
            get_ui("multiscale_ui")()
        except Exception as e:
            st.error(f"加载多尺度界面失败：{e}")


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