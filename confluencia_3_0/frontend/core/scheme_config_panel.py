"""Scheme Configuration Panel - Parameter customization for all Schemes"""

import streamlit as st
import yaml
from pathlib import Path


class SchemeConfigPanel:
    """Parameter configuration interface for Schemes 0-7"""

    def __init__(self):
        self.default_configs = self._load_default_configs()

    def _load_default_configs(self):
        """Load default configurations from YAML"""
        config_path = Path('config_quality.yaml')
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {}

    def render_config_panel(self, scheme_id):
        """
        Render parameter configuration panel for given Scheme

        Args:
            scheme_id: Scheme number (0-7)

        Returns:
            config: Dictionary of configured parameters
        """
        st.subheader("⚙️ 参数配置")
        st.caption(f"Scheme {scheme_id} 自定义参数")

        config = {}

        # ═══════════════════════════════════════════════════════════
        # Common Parameters (All Schemes)
        # ═══════════════════════════════════════════════════════════

        with st.expander("🔧 基础参数", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                config['batch_size'] = st.slider(
                    "Batch Size",
                    min_value=1,
                    max_value=128,
                    value=32,
                    help="每次训练的样本数量"
                )

            with col2:
                config['learning_rate'] = st.number_input(
                    "Learning Rate",
                    min_value=1e-6,
                    max_value=1e-2,
                    value=1e-4,
                    format="%.6f",
                    help="优化器学习率"
                )

            with col3:
                config['epochs'] = st.number_input(
                    "Epochs",
                    min_value=1,
                    max_value=200,
                    value=50,
                    help="训练轮数"
                )

        # ═══════════════════════════════════════════════════════════
        # Scheme-Specific Parameters
        # ═══════════════════════════════════════════════════════════

        if scheme_id == 0:
            # CircFold Baseline (Pipeline)
            config.update(self._render_pipeline_config())

        elif scheme_id == 1:
            # EGNN + Physics
            config.update(self._render_egnn_config())

        elif scheme_id == 2:
            # Force Field
            config.update(self._render_forcefield_config())

        elif scheme_id == 3:
            # Dual-Engine Distillation
            config.update(self._render_distillation_config())

        elif scheme_id == 4:
            # Diffusion + EGNN
            config.update(self._render_diffusion_config())

        elif scheme_id == 6:
            # GNN Latent Diffusion
            config.update(self._render_gnn_config())

        elif scheme_id == 7:
            # Mamba + Transformer
            config.update(self._render_mamba_config())

        elif scheme_id == 8:
            # Sparse Pair-Guided
            config.update(self._render_sparse_config())

        # ═══════════════════════════════════════════════════════════
        # Config Import/Export
        # ═══════════════════════════════════════════════════════════

        col1, col2 = st.columns(2)

        with col1:
            if st.button("💾 导出配置", type="primary"):
                self._export_config(config, scheme_id)

        with col2:
            uploaded_file = st.file_uploader("📥 导入配置", type=['yaml', 'yml'])
            if uploaded_file:
                config = self._import_config(uploaded_file, scheme_id)

        return config

    def _render_pipeline_config(self):
        """Scheme 0: Pipeline parameters"""
        config = {}

        with st.expander("🔄 Pipeline参数", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                config['num_samples'] = st.slider(
                    "样本数 (num_samples)",
                    min_value=1,
                    max_value=20,
                    value=5,
                    help="每个序列生成的结构数量"
                )

                config['confidence_threshold'] = st.slider(
                    "置信度阈值",
                    min_value=0.5,
                    max_value=1.0,
                    value=0.70,
                    step=0.05,
                    help="质量过滤阈值"
                )

            with col2:
                config['bsj_distance_range'] = st.slider(
                    "BSJ距离范围 (Å)",
                    min_value=2.0,
                    max_value=6.0,
                    value=(2.8, 5.0),
                    help="磷酸二酯键合理距离范围"
                )

                config['energy_threshold'] = st.number_input(
                    "能量阈值 (kJ/mol)",
                    min_value=100,
                    max_value=1000,
                    value=800,
                    help="结构稳定性阈值"
                )

        return config

    def _render_egnn_config(self):
        """Scheme 1: EGNN parameters"""
        config = {}

        with st.expander("🧠 EGNN参数", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                config['num_layers'] = st.slider(
                    "EGNN层数",
                    min_value=2,
                    max_value=12,
                    value=6
                )

                config['hidden_dim'] = st.select_slider(
                    "隐藏层维度",
                    options=[64, 128, 256, 512],
                    value=128
                )

            with col2:
                config['num_heads'] = st.slider(
                    "注意力头数",
                    min_value=1,
                    max_value=16,
                    value=8
                )

                config['physics_weight'] = st.slider(
                    "物理约束权重",
                    min_value=0.0,
                    max_value=2.0,
                    value=1.0,
                    help="能量约束权重"
                )

        return config

    def _render_distillation_config(self):
        """Scheme 3: Distillation parameters"""
        config = {}

        with st.expander("📚 蒸馏参数", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                config['temperature'] = st.slider(
                    "蒸馏温度 (Temperature)",
                    min_value=1.0,
                    max_value=10.0,
                    value=2.0,
                    step=0.5,
                    help="控制软标签平滑度"
                )

                config['loss_weight_coords'] = st.slider(
                    "坐标损失权重",
                    min_value=0.1,
                    max_value=5.0,
                    value=1.0
                )

            with col2:
                config['loss_weight_confidence'] = st.slider(
                    "置信度损失权重",
                    min_value=0.1,
                    max_value=5.0,
                    value=0.5
                )

                config['loss_weight_bsj'] = st.slider(
                    "BSJ损失权重",
                    min_value=0.1,
                    max_value=5.0,
                    value=2.0
                )

        return config

    def _render_diffusion_config(self):
        """Scheme 4: Diffusion parameters"""
        config = {}

        with st.expander("🌊 扩散参数", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                config['num_diffusion_steps'] = st.slider(
                    "扩散步数",
                    min_value=100,
                    max_value=2000,
                    value=1000
                )

                config['noise_schedule'] = st.selectbox(
                    "噪声调度",
                    options=['linear', 'cosine', 'quadratic'],
                    index=0
                )

            with col2:
                config['guidance_scale'] = st.slider(
                    "引导强度",
                    min_value=1.0,
                    max_value=10.0,
                    value=3.0
                )

                config['egnn_refinement'] = st.checkbox(
                    "EGNN精修",
                    value=True
                )

        return config

    def _render_mamba_config(self):
        """Scheme 7: Mamba+Transformer parameters"""
        config = {}

        with st.expander("🐍 Mamba参数", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                config['n_mamba_layers'] = st.slider(
                    "Mamba层数",
                    min_value=2,
                    max_value=12,
                    value=4
                )

                config['d_state'] = st.slider(
                    "状态维度 (d_state)",
                    min_value=8,
                    max_value=32,
                    value=16
                )

            with col2:
                config['n_attn_layers'] = st.slider(
                    "Transformer层数",
                    min_value=1,
                    max_value=8,
                    value=2
                )

                config['attn_window'] = st.slider(
                    "注意力窗口",
                    min_value=10,
                    max_value=50,
                    value=20
                )

        return config

    def _render_forcefield_config(self):
        """Scheme 2: Force field parameters"""
        config = {}

        with st.expander("⚛️ 力场参数", expanded=True):
            config['forcefield'] = st.selectbox(
                "力场类型",
                options=['AMBER14', 'CHARMM36', 'OPLS-AA'],
                index=0
            )

            config['solvation'] = st.selectbox(
                "溶剂模型",
                options=['implicit', 'explicit'],
                index=0
            )

        return config

    def _render_gnn_config(self):
        """Scheme 6: GNN parameters"""
        config = {}

        with st.expander("🔗 GNN参数", expanded=True):
            config['latent_dim'] = st.select_slider(
                "潜空间维度",
                options=[32, 64, 128, 256],
                value=64
            )

            config['num_gnn_layers'] = st.slider(
                "GNN层数",
                min_value=2,
                max_value=8,
                value=4
            )

        return config

    def _render_sparse_config(self):
        """Scheme 8: Sparse pair parameters"""
        config = {}

        with st.expander("🎯 稀疏配对参数", expanded=True):
            config['sparse_threshold'] = st.slider(
                "稀疏阈值",
                min_value=0.1,
                max_value=0.9,
                value=0.5
            )

            config['pair_guidance'] = st.checkbox(
                "配对引导",
                value=True
            )

        return config

    def _export_config(self, config, scheme_id):
        """Export configuration to YAML file"""
        from datetime import datetime

        export_dir = Path('configs/custom')
        export_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'scheme{scheme_id}_config_{timestamp}.yaml'
        filepath = export_dir / filename

        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        st.success(f"✅ 配置已保存到: {filepath}")

        # Download button
        with open(filepath, 'rb') as f:
            st.download_button(
                label="📥 下载配置文件",
                data=f,
                file_name=filename,
                mime='text/yaml'
            )

    def _import_config(self, uploaded_file, scheme_id):
        """Import configuration from uploaded YAML file"""
        try:
            config = yaml.safe_load(uploaded_file)
            st.success(f"✅ 配置已加载: {uploaded_file.name}")
            return config
        except Exception as e:
            st.error(f"❌ 配置加载失败: {e}")
            return {}


# Convenience function for direct use
def render_scheme_config(scheme_id):
    """Render Scheme configuration panel"""
    panel = SchemeConfigPanel()
    return panel.render_config_panel(scheme_id)