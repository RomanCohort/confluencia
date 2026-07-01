"""Biological Interpreter - Translate computational results to biological meaning"""

import streamlit as st
import pandas as pd


class BiologicalInterpreter:
    """Biological significance interpretation layer"""

    # Terminology dictionary
    BIOLOGICAL_TERMS = {
        'EGNN': {
            '中文': '等变图神经网络',
            '生物学意义': '保持原子对称性的3D结构预测网络',
            '通俗解释': '一种能理解分子3D几何的AI模型'
        },
        'Mamba': {
            '中文': '状态空间模型',
            '生物学意义': '长序列全局依赖建模',
            '通俗解释': '能理解长RNA序列整体结构的AI模型'
        },
        'Distillation': {
            '中文': '知识蒸馏',
            '生物学意义': 'Teacher-Student知识传递',
            '通俗解释': '大模型教导小模型学习'
        },
        'BSJ': {
            '中文': '反向剪接连接位点',
            '生物学意义': 'circRNA环化关键位置',
            '通俗解释': 'circRNA成环的连接点'
        },
        'Confidence': {
            '中文': '置信度',
            '生物学意义': '结构预测可信度',
            '通俗解释': '预测结果的可靠程度'
        },
        'RMSD': {
            '中文': '均方根偏差',
            '生物学意义': '结构一致性',
            '通俗解释': '预测结构与真实结构的相似度'
        }
    }

    # Interpretation thresholds
    INTERPRETATION_THRESHOLDS = {
        'confidence': {
            'high': 0.80,
            'medium': 0.70,
            'low': 0.50,
            'guidance': {
                '≥0.80': '高度可信，推荐实验验证',
                '0.70-0.80': '中等可信，谨慎验证',
                '<0.70': '可信度较低，建议优化参数'
            }
        },
        'bsj_distance': {
            'ideal': 3.5,
            'acceptable_range': (2.8, 5.0),
            'guidance': {
                '3.5±0.5Å': '符合磷酸二酯键理想长度，环化效率高',
                '2.8-5.0Å': '在合理范围内，环化可能成功',
                '<2.8Å': '距离过短，可能结构冲突',
                '>5.0Å': '距离过长，环化效率低'
            }
        },
        'energy': {
            'stable': 500,
            'moderate': 800,
            'unstable': 1000,
            'guidance': {
                '<500kJ/mol': '结构稳定，热力学可靠',
                '500-800kJ/mol': '中等稳定，需验证',
                '>800kJ/mol': '结构不稳定，需优化'
            }
        },
        'rmsd': {
            'high_quality': 2.0,
            'acceptable': 5.0,
            'low_quality': 10.0,
            'guidance': {
                '<2Å': '高质量预测，与真实结构高度相似',
                '2-5Å': '中等质量，结构基本相似',
                '>5Å': '低质量，结构差异较大'
            }
        }
    }

    def interpret_prediction(self, prediction_result):
        """
        Interpret computational results in biological terms

        Args:
            prediction_result: Dictionary with prediction metrics

        Returns:
            interpretations: Dictionary with biological meanings
        """
        st.subheader("🧬 生物学意义解读")
        st.caption("将计算指标翻译为生物学语言")

        # ═══════════════════════════════════════════════════════════
        # Interpretation Table
        # ═══════════════════════════════════════════════════════════

        interpretations = self._build_interpretations(prediction_result)

        # Display as table
        df = pd.DataFrame(interpretations).T
        df.columns = ['数值', '生物学意义', '实验指导']

        st.table(df)

        # ═══════════════════════════════════════════════════════════
        # Overall Recommendation
        # ═══════════════════════════════════════════════════════════

        self._render_experiment_guidance(prediction_result)

        # ═══════════════════════════════════════════════════════════
        # Terminology Help
        # ═══════════════════════════════════════════════════════════

        self._render_terminology_help()

        return interpretations

    def _build_interpretations(self, result):
        """Build interpretation dictionary"""

        interpretations = {}

        # Confidence
        confidence = result.get('confidence', 0)
        interpretations['置信度'] = {
            '数值': f"{confidence:.2f}",
            '生物学意义': '结构预测可信度',
            '实验指导': self._get_confidence_guidance(confidence)
        }

        # BSJ distance
        bsj_dist = result.get('bsj_distance', 0)
        interpretations['BSJ距离'] = {
            '数值': f"{bsj_dist:.2f} Å",
            '生物学意义': '环化连接位点距离',
            '实验指导': self._get_bsj_guidance(bsj_dist)
        }

        # Energy
        energy = result.get('energy', 0)
        interpretations['能量'] = {
            '数值': f"{energy:.1f} kJ/mol",
            '生物学意义': '结构热力学稳定性',
            '实验指导': self._get_energy_guidance(energy)
        }

        # RMSD
        rmsd = result.get('rmsd', 0)
        interpretations['RMSD'] = {
            '数值': f"{rmsd:.2f} Å",
            '生物学意义': '结构一致性',
            '实验指导': self._get_rmsd_guidance(rmsd)
        }

        return interpretations

    def _get_confidence_guidance(self, confidence):
        """Get guidance based on confidence score"""
        if confidence >= 0.80:
            return "✅ 高度可信，推荐进行湿实验验证"
        elif confidence >= 0.70:
            return "⚠️ 中等可信，建议谨慎验证"
        else:
            return "❌ 可信度较低，建议优化参数后重新预测"

    def _get_bsj_guidance(self, bsj_dist):
        """Get guidance based on BSJ distance"""
        if 3.0 <= bsj_dist <= 4.0:
            return "✅ 符合磷酸二酯键理想长度（3.5Å），环化效率预期高"
        elif 2.8 <= bsj_dist <= 5.0:
            return "⚠️ 在合理范围内，环化可能成功但需验证"
        elif bsj_dist < 2.8:
            return "❌ 距离过短，可能存在结构冲突，环化困难"
        else:
            return "❌ 距离过长，环化效率预期低，建议重新设计"

    def _get_energy_guidance(self, energy):
        """Get guidance based on energy"""
        if energy < 500:
            return "✅ 结构稳定，热力学可靠性高"
        elif energy <= 800:
            return "⚠️ 中等稳定，建议实验验证稳定性"
        else:
            return "❌ 结构不稳定，可能存在冲突，需优化"

    def _get_rmsd_guidance(self, rmsd):
        """Get guidance based on RMSD"""
        if rmsd < 2.0:
            return "✅ 高质量预测，与真实结构高度相似"
        elif rmsd <= 5.0:
            return "⚠️ 中等质量，结构基本相似但有差异"
        else:
            return "❌ 低质量，结构差异较大，预测不可靠"

    def _render_experiment_guidance(self, result):
        """Render overall experiment guidance"""

        st.markdown("---")
        st.subheader("💡 实验设计建议")

        confidence = result.get('confidence', 0)
        bsj_dist = result.get('bsj_distance', 0)

        # Overall recommendation
        if confidence >= 0.80 and 3.0 <= bsj_dist <= 4.0:
            st.success("""
            🎯 **推荐进行湿实验验证**

            **建议实验：**
            1. circRNA环化验证（验证BSJ连接）
            2. 3D结构验证（冷冻电镜/核磁共振）
            3. 功能验证（功能性实验）

            **预期成功率：高**
            """)

        elif confidence >= 0.70:
            st.warning("""
            ⚠️ **谨慎验证**

            **建议：**
            1. 先优化计算参数
            2. 进行初步环化验证
            3. 根据验证结果调整

            **预期成功率：中等**
            """)

        else:
            st.error("""
            ❌ **建议暂不实验**

            **需要：**
            1. 优化计算参数（提高置信度）
            2. 检查序列质量
            3. 重新预测后再评估

            **预期成功率：低**
            """)

    def _render_terminology_help(self):
        """Render terminology help section"""

        st.markdown("---")
        st.subheader("📖 术语解释")

        with st.expander("查看术语字典"):
            for term, info in self.BIOLOGICAL_TERMS.items():
                st.markdown(f"""
                **{term}**
                - **中文**: {info['中文']}
                - **生物学意义**: {info['生物学意义']}
                - **通俗解释**: {info['通俗解释']}
                ---
                """)


# Convenience function
def interpret_results(result):
    """Interpret results with biological meaning"""
    interpreter = BiologicalInterpreter()
    return interpreter.interpret_prediction(result)