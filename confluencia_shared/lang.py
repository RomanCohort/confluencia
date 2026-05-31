"""Lightweight i18n support for Confluencia Streamlit UI.

Provides a simple translation dictionary and ``t(key)`` function.
Language is auto-detected from ``st.query_params`` or browser locale,
with a manual sidebar toggle.  Default: English.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Translation dictionaries  (key → {en, zh})
# ---------------------------------------------------------------------------
_TRANSLATIONS: dict[str, dict[str, str]] = {}

_LANGS = ("en", "zh")
_DEFAULT_LANG = "en"


def register(key: str, en: str, zh: str) -> None:
    """Register a translatable string."""
    _TRANSLATIONS[key] = {"en": en, "zh": zh}


def t(key: str, **kwargs) -> str:
    """Return the translated string for *key* in the current language.

    Optional ``kwargs`` are substituted via ``str.format()``.
    """
    lang = _current_lang()
    entry = _TRANSLATIONS.get(key, {})
    raw = entry.get(lang, entry.get(_DEFAULT_LANG, key))
    if kwargs:
        return raw.format(**kwargs)
    return raw


def _current_lang() -> str:
    """Return the active language code."""
    # 1) explicit sidebar toggle (persisted in session_state)
    try:
        if "lang" in st.session_state:
            return st.session_state.lang
    except Exception:
        pass
    # 2) auto-detect from query params
    try:
        qp = st.query_params.get("lang", "")
        if qp in _LANGS:
            st.session_state.lang = qp
            return qp
    except Exception:
        pass
    # 3) default
    try:
        st.session_state.lang = _DEFAULT_LANG
    except Exception:
        pass
    return _DEFAULT_LANG


def lang_toggle(label_en: str = "Language", label_zh: str = "语言") -> None:
    """Render a sidebar language radio toggle."""
    lang = _current_lang()
    labels = {"en": label_en, "zh": label_zh}
    options = {"English": "en", "中文": "zh"}
    chosen = st.sidebar.radio(
        labels[lang],
        list(options.keys()),
        index=list(options.values()).index(lang),
        horizontal=True,
        key="_lang_toggle",
    )
    st.session_state.lang = options[chosen]


# ---------------------------------------------------------------------------
# Bulk registration helper
# ---------------------------------------------------------------------------
def load_from_dict(data: dict[str, dict[str, str]]) -> None:
    """Load translations from a dict ``{key: {en: ..., zh: ...}}``."""
    for key, vals in data.items():
        register(key, vals.get("en", key), vals.get("zh", key))


# ---------------------------------------------------------------------------
# Common strings used across all modules
# ---------------------------------------------------------------------------
_COMMON_STRINGS = {
    "page_title_drug":     {"en": "Confluencia 2.0 Drug Module",          "zh": "Confluencia 2.0 药物模块"},
    "page_title_epitope":  {"en": "Confluencia 2.0 Epitope Module",       "zh": "Confluencia 2.0 表位模块"},
    "page_title_circrna":  {"en": "Confluencia circRNA Module",            "zh": "Confluencia circRNA 模块"},
    "title_drug":          {"en": "Confluencia 2.0: Drug Training & Micro-efficacy Prediction", "zh": "Confluencia 2.0：药物训练与微机制疗效预测"},
    "title_epitope":       {"en": "Confluencia 2.0 Epitope Prediction & Training", "zh": "Confluencia 2.0 表位预测与训练"},
    "caption_drug":        {"en": "MOE auto-modeling + CTM dynamic simulation + target/immune/inflammation multi-indicator prediction", "zh": "MOE 自动建模 + CTM 动态仿真 + 靶点/免疫/炎症多指标预测"},
    "caption_epitope":     {"en": "circRNA-oriented micro-efficacy prediction, supports PyTorch Mamba training and multi-neighborhood sensitivity analysis", "zh": "面向 circRNA 的微观疗效预测，支持 PyTorch Mamba 训练与多邻域敏感性分析"},
    "mode_beginner":       {"en": "Beginner",                              "zh": "新手版"},
    "mode_expert":         {"en": "Expert",                                "zh": "专家版"},
    "sidebar_settings":    {"en": "Settings",                              "zh": "设置"},
    "sidebar_run_mode":    {"en": "Run Mode",                              "zh": "运行模式"},
    "sidebar_features":    {"en": "Feature Configuration",                "zh": "特征配置"},
    "sidebar_dose":        {"en": "Dose",                                  "zh": "剂量"},
    "sidebar_freq":        {"en": "Frequency (doses/day)",                "zh": "频率（次/天）"},
    "sidebar_horizon":     {"en": "Simulation Horizon (h)",               "zh": "仿真时长 (h)"},
    "sidebar_modification":{"en": "RNA Modification",                     "zh": "RNA 修饰"},
    "sidebar_delivery":    {"en": "Delivery Vector",                      "zh": "递送载体"},
    "sidebar_route":       {"en": "Administration Route",                 "zh": "给药途径"},
    "btn_train":           {"en": "Train Model",                           "zh": "训练模型"},
    "btn_predict":         {"en": "Predict",                               "zh": "预测"},
    "btn_download":        {"en": "Download Results",                     "zh": "下载结果"},
    "btn_export_model":    {"en": "Export Model",                         "zh": "导出模型"},
    "btn_import_model":    {"en": "Import Model",                         "zh": "导入模型"},
    "btn_clear":           {"en": "Clear",                                "zh": "清除"},
    "btn_run_simulation":  {"en": "Run Simulation",                       "zh": "运行仿真"},
    "btn_upload_csv":      {"en": "Upload CSV",                           "zh": "上传 CSV"},
    "btn_use_demo":        {"en": "Use Demo Data",                        "zh": "使用示例数据"},
    "result_efficacy":     {"en": "Efficacy",                              "zh": "疗效"},
    "result_toxicity":     {"en": "Toxicity",                              "zh": "毒性"},
    "result_auc":          {"en": "AUC Efficacy",                         "zh": "AUC 疗效"},
    "result_peak":         {"en": "Peak Efficacy",                        "zh": "峰值疗效"},
    "result_half_life":    {"en": "RNA Half-life (h)",                    "zh": "RNA 半衰期 (h)"},
    "result_escape":       {"en": "Endosomal Escape (%)",                 "zh": "内体逃逸 (%)"},
    "result_window":       {"en": "Expression Window (h)",                "zh": "表达窗口 (h)"},
    "status_training":     {"en": "Training model...",                    "zh": "正在训练模型..."},
    "status_done":         {"en": "Training complete",                    "zh": "训练完成"},
    "status_error":        {"en": "Error occurred",                       "zh": "发生错误"},
    "status_no_data":      {"en": "No data provided",                     "zh": "未提供数据"},
    "cloud_title":         {"en": "Cloud Server",                         "zh": "云服务器"},
    "cloud_connect":       {"en": "Connect",                              "zh": "连接"},
    "cloud_disconnect":    {"en": "Disconnect",                           "zh": "断开"},
    "cloud_status_ok":     {"en": "Connected",                            "zh": "已连接"},
    "cloud_status_off":    {"en": "Offline",                              "zh": "离线"},
    "section_pk":          {"en": "Pharmacokinetic Simulation",           "zh": "药代动力学仿真"},
    "section_rnactm":      {"en": "RNACTM circRNA PK",                    "zh": "RNACTM circRNA 药代"},
    "section_admet":       {"en": "ADMET Screening",                      "zh": "ADMET 筛查"},
    "section_moe":         {"en": "MOE Ensemble",                         "zh": "MOE 集成"},
    "section_ctm_curve":   {"en": "CTM Concentration Curve",              "zh": "CTM 浓度曲线"},
    "section_rna_curve":   {"en": "RNACTM RNA/Protein Curve",             "zh": "RNACTM RNA/蛋白曲线"},
    "section_decision":    {"en": "5D Evaluation & Decision",             "zh": "5D 评估与决策"},
    "decision_go":         {"en": "Go",                                   "zh": "Go"},
    "decision_conditional":{"en": "Conditional",                          "zh": "Conditional"},
    "decision_nogo":       {"en": "No-Go",                                "zh": "No-Go"},
    "modification_none":   {"en": "Unmodified",                           "zh": "无修饰"},
    "modification_m6a":    {"en": "m6A",                                  "zh": "m6A"},
    "modification_psi":    {"en": "Psi",                                  "zh": "Ψ"},
    "modification_5mc":    {"en": "5mC",                                  "zh": "5mC"},
    "delivery_lnp_std":    {"en": "LNP Standard",                         "zh": "LNP 标准"},
    "delivery_lnp_liver":  {"en": "LNP Liver-targeted",                  "zh": "LNP 肝靶向"},
    "route_iv":            {"en": "IV",                                   "zh": "静脉注射"},
    "route_sc":            {"en": "SC",                                   "zh": "皮下注射"},
    "route_im":            {"en": "IM",                                   "zh": "肌肉注射"},
    "plot_time":           {"en": "Time (h)",                             "zh": "时间 (h)"},
    "plot_concentration":  {"en": "Concentration",                        "zh": "浓度"},
    "plot_protein":        {"en": "Translated Protein",                   "zh": "翻译蛋白"},
    "plot_rna_cyto":       {"en": "Cytoplasmic RNA",                      "zh": "胞质 RNA"},
    "tab_training":        {"en": "Training",                             "zh": "训练"},
    "tab_prediction":      {"en": "Prediction",                           "zh": "预测"},
    "tab_pk":              {"en": "PK Simulation",                        "zh": "PK 仿真"},
    "tab_rnactm":          {"en": "RNACTM",                               "zh": "RNACTM"},
    "tab_admet":           {"en": "ADMET",                                "zh": "ADMET"},
    "tab_decision":        {"en": "5D Decision",                          "zh": "5D 决策"},
    "tab_data":            {"en": "Data Input",                           "zh": "数据输入"},
    "tab_results":         {"en": "Results",                              "zh": "结果"},
    "epitope_seq_label":   {"en": "Epitope Sequence",                    "zh": "表位序列"},
    "mhc_allele_label":    {"en": "MHC Allele",                           "zh": "MHC 等位基因"},
    "binding_score_label": {"en": "Binding Score",                        "zh": "结合评分"},
    "immuno_score_label":  {"en": "Immunogenicity Score",                 "zh": "免疫原性评分"},
    "inflam_score_label":  {"en": "Inflammation Score",                   "zh": "炎症评分"},
    "training_epochs":     {"en": "Training Epochs",                      "zh": "训练轮数"},
    "training_batch":      {"en": "Batch Size",                           "zh": "批大小"},
    "training_lr":         {"en": "Learning Rate",                        "zh": "学习率"},
    "cloud_config_title":  {"en": "Cloud Server Configuration",           "zh": "云服务器配置"},
    "cloud_host":          {"en": "Host",                                 "zh": "主机"},
    "cloud_port":          {"en": "Port",                                 "zh": "端口"},
    "cloud_api_key":       {"en": "API Key",                              "zh": "API 密钥"},
    "cloud_save":          {"en": "Save Config",                          "zh": "保存配置"},
    "cloud_test":          {"en": "Test Connection",                      "zh": "测试连接"},
    "sensitivity_title":   {"en": "Sensitivity Analysis",                 "zh": "敏感性分析"},
    "sensitivity_n_iters": {"en": "Iterations",                           "zh": "迭代次数"},
    "sensitivity_n_range": {"en": "Parameter Range (±%)",                 "zh": "参数范围 (±%)"},
    "sensitivity_run":     {"en": "Run Sensitivity",                      "zh": "运行敏感性分析"},
    "upload_csv_hint":     {"en": "Upload a CSV file with columns: SMILES, epitope_seq, dose, freq, treatment_time", "zh": "上传 CSV 文件，包含列：SMILES, epitope_seq, dose, freq, treatment_time"},
    "demo_data_hint":      {"en": "Or use demo data to explore the platform", "zh": "或使用示例数据探索平台功能"},
    "col_smiles":          {"en": "SMILES",                               "zh": "SMILES"},
    "col_epitope":         {"en": "Epitope",                              "zh": "表位"},
    "col_dose":            {"en": "Dose",                                 "zh": "剂量"},
    "col_freq":            {"en": "Freq",                                 "zh": "频率"},
    "col_time":            {"en": "Time (h)",                             "zh": "时间 (h)"},
    "col_efficacy":        {"en": "Efficacy",                             "zh": "疗效"},
    "col_binding":         {"en": "Binding",                              "zh": "结合"},
    "col_immune":          {"en": "Immune",                               "zh": "免疫"},
    "col_inflammation":    {"en": "Inflammation",                         "zh": "炎症"},
    "tab_overview":        {"en": "Overview",                             "zh": "总览"},
    "ssm_unavailable_win": {"en": "Windows environment; mamba-ssm skipped in requirements.", "zh": "当前是 Windows 环境，requirements 配置会跳过 mamba-ssm。"},
    "ssm_unavailable_torch":{"en": "PyTorch unavailable; mamba-ssm cannot be enabled.", "zh": "PyTorch 不可用，无法启用 mamba-ssm。"},
    "ssm_unavailable_pkg": {"en": "mamba-ssm not installed in current environment.", "zh": "当前 Python 环境未安装 mamba-ssm。"},
    "ssm_unavailable_err": {"en": "mamba-ssm import failed; switched to fallback module.", "zh": "mamba-ssm 导入失败，已自动切换到 fallback 模块。"},
    "val_loss_rising":     {"en": "Validation loss rising; consider early stopping.", "zh": "验证损失回升，建议开启早停"},
    "r2_low_hint":         {"en": "R2 is low; consider increasing sample size.", "zh": "R2 偏低，建议增加样本量"},
    "data_preview":        {"en": "Data Preview",                         "zh": "数据预览"},
    "model_metadata":      {"en": "Model Metadata",                       "zh": "模型元数据"},
    "model_weights":       {"en": "Model Weights",                        "zh": "模型权重"},
    "feature_spec":        {"en": "Feature Specification",                "zh": "特征规格"},
    "output_columns":      {"en": "Output Columns",                       "zh": "输出列"},
    "use_mhc_ii":          {"en": "Enable MHC-II",                        "zh": "启用 MHC-II"},
    "mhc_auto_detect":     {"en": "Auto-detect MHC class",                "zh": "自动检测 MHC 类别"},
    "ires_score":          {"en": "IRES Score",                           "zh": "IRES 评分"},
    "gc_content":          {"en": "GC Content",                           "zh": "GC 含量"},
    "struct_stability":    {"en": "Structure Stability",                  "zh": "结构稳定性"},
    "innate_immune":       {"en": "Innate Immune Score",                  "zh": "先天免疫评分"},
}

load_from_dict(_COMMON_STRINGS)