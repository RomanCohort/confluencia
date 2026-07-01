"""Mutation Analyzer - Analyze mutation impact on circRNA structure"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


class MutationAnalyzer:
    """Mutation impact analysis interface"""

    def analyze_single_mutation(self, sequence, mutation_position, mutation_type):
        """
        Analyze impact of single mutation on structure

        Args:
            sequence: Original RNA sequence
            mutation_position: Position to mutate (0-indexed)
            mutation_type: New nucleotide (A/U/G/C)

        Returns:
            impact: Dictionary with mutation impact metrics
        """
        st.subheader("🧬 突变影响分析")

        # ═══════════════════════════════════════════════════════════
        # Mutation Input
        # ═══════════════════════════════════════════════════════════

        st.markdown(f"""
        **原始序列**: {sequence[:50]}...
        **突变位点**: 第 {mutation_position} 位
        **突变类型**: {sequence[mutation_position]} → {mutation_type}
        """)

        # Generate mutated sequence
        mutated_seq = self._apply_mutation(sequence, mutation_position, mutation_type)
        st.code(f"突变序列: {mutated_seq[:50]}...")

        # ═══════════════════════════════════════════════════════════
        # Impact Metrics
        # ═══════════════════════════════════════════════════════════

        # Mock prediction results
        original_result = {
            'confidence': 0.85,
            'bsj_distance': 3.5,
            'energy': 500,
            'rmsd': 2.0
        }

        mutated_result = {
            'confidence': 0.75,
            'bsj_distance': 4.2,
            'energy': 650,
            'rmsd': 3.5
        }

        # Calculate changes
        impact = self._calculate_impact(original_result, mutated_result)

        # Display metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "结构变化 (RMSD)",
                f"{impact['rmsd_change']:.2f} Å",
                delta=f"+{impact['rmsd_change']:.2f} Å",
                delta_color="inverse"
            )

        with col2:
            st.metric(
                "BSJ距离变化",
                f"{impact['bsj_distance_change']:.2f} Å",
                delta=f"+{impact['bsj_distance_change']:.2f} Å"
            )

        with col3:
            st.metric(
                "稳定性变化",
                f"{impact['energy_change']:.1f} kJ/mol",
                delta=f"+{impact['energy_change']:.1f} kJ/mol",
                delta_color="inverse"
            )

        # ═══════════════════════════════════════════════════════════
        # Impact Visualization
        # ═══════════════════════════════════════════════════════════

        self._render_impact_visualization(original_result, mutated_result)

        # ═══════════════════════════════════════════════════════════
        # Experimental Guidance
        # ═══════════════════════════════════════════════════════════

        self._render_mutation_guidance(impact)

        return impact

    def _apply_mutation(self, sequence, position, mutation_type):
        """Apply mutation to sequence"""
        return sequence[:position] + mutation_type + sequence[position+1:]

    def _calculate_impact(self, original, mutated):
        """Calculate mutation impact"""
        return {
            'rmsd_change': mutated['rmsd'] - original['rmsd'],
            'bsj_distance_change': mutated['bsj_distance'] - original['bsj_distance'],
            'energy_change': mutated['energy'] - original['energy'],
            'confidence_change': mutated['confidence'] - original['confidence']
        }

    def _render_impact_visualization(self, original, mutated):
        """Render impact comparison visualization"""

        st.subheader("📊 突变前后对比")

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("置信度", "BSJ距离", "能量", "RMSD")
        )

        metrics = ['confidence', 'bsj_distance', 'energy', 'rmsd']
        names = ['置信度', 'BSJ距离 (Å)', '能量 (kJ/mol)', 'RMSD (Å)']

        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

        for metric, name, pos in zip(metrics, names, positions):
            fig.add_trace(
                go.Bar(
                    x=['原始', '突变后'],
                    y=[original[metric], mutated[metric]],
                    name=name,
                    marker_color=['blue', 'red']
                ),
                row=pos[0], col=pos[1]
            )

        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig)

    def _render_mutation_guidance(self, impact):
        """Render experimental guidance based on impact"""

        st.markdown("---")
        st.subheader("💡 实验设计建议")

        bsj_change = impact['bsj_distance_change']

        if abs(bsj_change) > 1.0:
            st.error("""
            ⚠️ **突变显著影响BSJ连接**

            **建议：**
            1. 优先验证环化效率（RT-PCR检测）
            2. 检查突变位点是否影响剪接位点
            3. 考虑替代突变方案

            **预期影响：** 环化效率显著降低
            """)
        elif abs(bsj_change) > 0.5:
            st.warning("""
            ⚠️ **突变中度影响BSJ**

            **建议：**
            1. 进行环化验证实验
            2. 检查二级结构变化
            3. 监测表达量变化

            **预期影响：** 环化效率可能降低
            """)
        else:
            st.success("""
            ✅ **突变对BSJ影响较小**

            **建议：**
            1. 可直接进行功能验证
            2. 检查其他生物学特性
            3. 结构稳定性良好

            **预期影响：** 环化效率基本不变
            """)

    def batch_mutation_scan(self, sequence):
        """
        Scan all possible mutations across sequence

        Args:
            sequence: RNA sequence to scan

        Returns:
            mutations: List of mutation impacts
        """
        st.subheader("🔍 批量突变扫描")
        st.caption(f"扫描序列长度: {len(sequence)} nt")

        # ═══════════════════════════════════════════════════════════
        # Mutation Generation
        # ═══════════════════════════════════════════════════════════

        mutations = []
        nucleotides = ['A', 'U', 'G', 'C']

        for pos in range(len(sequence)):
            for mut_type in nucleotides:
                if sequence[pos] != mut_type:
                    # Mock impact calculation
                    impact = {
                        'position': pos,
                        'original': sequence[pos],
                        'mutation': mut_type,
                        'mutation_str': f"{sequence[pos]}{pos}{mut_type}",
                        'bsj_impact': np.random.uniform(-1.0, 1.0),
                        'energy_impact': np.random.uniform(-200, 200),
                        'confidence_impact': np.random.uniform(-0.2, 0.2)
                    }
                    mutations.append(impact)

        # ═══════════════════════════════════════════════════════════
        # Mutation Heatmap
        # ═══════════════════════════════════════════════════════════

        self._render_mutation_heatmap(mutations, sequence)

        # ═══════════════════════════════════════════════════════════
        # Top Impact Mutations
        # ═══════════════════════════════════════════════════════════

        self._render_top_mutations(mutations)

        return mutations

    def _render_mutation_heatmap(self, mutations, sequence):
        """Render mutation impact heatmap"""

        st.subheader("📈 突变影响热图")

        # Create heatmap matrix
        positions = list(range(len(sequence)))
        nucleotides = ['A', 'U', 'G', 'C']

        impact_matrix = np.zeros((4, len(sequence)))

        for mut in mutations:
            row = nucleotides.index(mut['mutation'])
            col = mut['position']
            impact_matrix[row, col] = abs(mut['bsj_impact'])

        fig = go.Figure(data=go.Heatmap(
            z=impact_matrix,
            x=positions,
            y=nucleotides,
            colorscale='RdYlGn_r',
            colorbar_title="影响程度"
        ))

        fig.update_layout(
            title="突变对BSJ的影响热图",
            xaxis_title="位置",
            yaxis_title="突变类型",
            height=400
        )

        st.plotly_chart(fig)

    def _render_top_mutations(self, mutations):
        """Render top impact mutations"""

        st.markdown("---")
        st.subheader("🎯 高影响突变位点")

        # Sort by BSJ impact
        sorted_mutations = sorted(
            mutations,
            key=lambda x: abs(x['bsj_impact']),
            reverse=True
        )[:10]

        top_mutations = []
        for mut in sorted_mutations:
            top_mutations.append({
                '位置': mut['position'],
                '突变': mut['mutation_str'],
                'BSJ影响': f"{mut['bsj_impact']:.2f} Å",
                '能量影响': f"{mut['energy_impact']:.1f} kJ/mol",
                '置信度影响': f"{mut['confidence_impact']:.2f}"
            })

        df = pd.DataFrame(top_mutations)
        st.table(df)

        st.info("""
        **提示：** 红色区域表示高影响突变位点，应优先避免或仔细验证
        """)


# Convenience function
def analyze_mutation(sequence, position=None, mutation_type=None):
    """Analyze mutation impact"""
    analyzer = MutationAnalyzer()

    if position is not None and mutation_type is not None:
        return analyzer.analyze_single_mutation(sequence, position, mutation_type)
    else:
        return analyzer.batch_mutation_scan(sequence)