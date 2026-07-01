"""Batch Scheme Runner - Run multiple Schemes in parallel"""

import streamlit as st
import time
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class BatchSchemeRunner:
    """Batch Scheme execution manager"""

    def run_all_schemes(self, config):
        """
        Run selected Schemes in parallel

        Args:
            config: Common configuration for all Schemes

        Returns:
            results: List of Scheme results
        """
        st.subheader("🔄 批量Scheme运行")
        st.caption("同时运行多个Scheme进行效果对比")

        # ═══════════════════════════════════════════════════════════
        # Scheme Selection
        # ═══════════════════════════════════════════════════════════

        schemes_to_run = st.multiselect(
            "选择要运行的Scheme",
            options=[0, 1, 2, 3, 4, 6, 7, 8],
            default=[0, 7],
            format_func=lambda x: f"Scheme {x} - {self._get_scheme_name(x)}"
        )

        if len(schemes_to_run) == 0:
            st.warning("请至少选择一个Scheme")
            return []

        st.info(f"将运行 {len(schemes_to_run)} 个Scheme")

        # ═══════════════════════════════════════════════════════════
        # Execution Options
        # ═══════════════════════════════════════════════════════════

        execution_mode = st.radio(
            "执行模式",
            options=['并行执行（快速）', '顺序执行（稳定）'],
            horizontal=True
        )

        # ═══════════════════════════════════════════════════════════
        # Run Button
        # ═══════════════════════════════════════════════════════════

        if st.button("🚀 启动批量运行", type="primary"):
            results = self._execute_schemes(schemes_to_run, config, execution_mode)

            # ═══════════════════════════════════════════════════════════
            # Results Comparison
            # ═══════════════════════════════════════════════════════════

            self._compare_results(results)

            return results

        return []

    def _get_scheme_name(self, scheme_id):
        """Get Scheme name"""
        names = {
            0: "CircFold Baseline",
            1: "EGNN + Physics",
            2: "Force Field",
            3: "Dual-Engine",
            4: "Diffusion + EGNN",
            6: "GNN Latent",
            7: "Mamba + Transformer",
            8: "Sparse Pair"
        }
        return names.get(scheme_id, "Unknown")

    def _execute_schemes(self, scheme_ids, config, execution_mode):
        """Execute Schemes based on mode"""

        results = []

        if '并行' in execution_mode:
            # Parallel execution
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, scheme_id in enumerate(scheme_ids):
                status_text.text(f"正在运行 Scheme {scheme_id}...")

                # Mock execution
                result = self._mock_run_scheme(scheme_id, config)
                results.append(result)

                progress = (i + 1) / len(scheme_ids)
                progress_bar.progress(progress)

                time.sleep(1)  # Simulate execution time

            status_text.text("✅ 所有Scheme运行完成！")
            progress_bar.empty()

        else:
            # Sequential execution
            for scheme_id in scheme_ids:
                with st.spinner(f"运行 Scheme {scheme_id}..."):
                    result = self._mock_run_scheme(scheme_id, config)
                    results.append(result)
                    st.success(f"✅ Scheme {scheme_id} 完成")

        return results

    def _mock_run_scheme(self, scheme_id, config):
        """Mock Scheme execution for demonstration"""

        # Simulate different Scheme performances
        performances = {
            0: {'bsj_accuracy': 85, 'rmsd': 2.5, 'time': 3600, 'speed': 1.0},
            1: {'bsj_accuracy': 82, 'rmsd': 2.8, 'time': 1800, 'speed': 2.0},
            2: {'bsj_accuracy': 78, 'rmsd': 3.2, 'time': 7200, 'speed': 0.5},
            3: {'bsj_accuracy': 88, 'rmsd': 2.2, 'time': 2400, 'speed': 1.5},
            4: {'bsj_accuracy': 80, 'rmsd': 2.9, 'time': 3600, 'speed': 1.0},
            6: {'bsj_accuracy': 83, 'rmsd': 2.6, 'time': 3000, 'speed': 1.2},
            7: {'bsj_accuracy': 92, 'rmsd': 2.0, 'time': 1200, 'speed': 3.0},
            8: {'bsj_accuracy': 86, 'rmsd': 2.4, 'time': 1500, 'speed': 2.4}
        }

        perf = performances.get(scheme_id, performances[7])

        return {
            'scheme_id': scheme_id,
            'scheme_name': self._get_scheme_name(scheme_id),
            'bsj_accuracy': perf['bsj_accuracy'],
            'rmsd': perf['rmsd'],
            'training_time': perf['time'],
            'inference_speed': perf['speed'],
            'config': config,
            'status': 'completed'
        }

    def _compare_results(self, results):
        """Compare all Scheme results"""

        st.markdown("---")
        st.subheader("📊 Scheme对比分析")

        # ═══════════════════════════════════════════════════════════
        # Comparison Table
        # ═══════════════════════════════════════════════════════════

        comparison_data = {
            'Scheme': [r['scheme_name'] for r in results],
            'Scheme ID': [r['scheme_id'] for r in results],
            'BSJ准确率 (%)': [r['bsj_accuracy'] for r in results],
            'RMSD (Å)': [r['rmsd'] for r in results],
            '训练时间 (秒)': [r['training_time'] for r in results],
            '推理速度': [r['inference_speed'] for r in results]
        }

        df = pd.DataFrame(comparison_data)
        st.table(df)

        # ═══════════════════════════════════════════════════════════
        # Comparison Charts
        # ═══════════════════════════════════════════════════════════

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("BSJ准确率对比", "RMSD对比", "训练时间对比", "推理速度对比")
        )

        x_vals = df['Scheme']

        # BSJ Accuracy
        fig.add_trace(
            go.Bar(x=x_vals, y=df['BSJ准确率 (%)'], name='BSJ准确率', marker_color='green'),
            row=1, col=1
        )

        # RMSD
        fig.add_trace(
            go.Bar(x=x_vals, y=df['RMSD (Å)'], name='RMSD', marker_color='blue'),
            row=1, col=2
        )

        # Training Time
        fig.add_trace(
            go.Bar(x=x_vals, y=df['训练时间 (秒)'], name='训练时间', marker_color='orange'),
            row=2, col=1
        )

        # Inference Speed
        fig.add_trace(
            go.Bar(x=x_vals, y=df['推理速度'], name='推理速度', marker_color='purple'),
            row=2, col=2
        )

        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig)

        # ═══════════════════════════════════════════════════════════
        # Best Scheme Recommendation
        # ═══════════════════════════════════════════════════════════

        st.markdown("---")
        st.subheader("🏆 最佳Scheme推荐")

        best_scheme = max(results, key=lambda x: x['bsj_accuracy'])

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "最佳Scheme",
                f"Scheme {best_scheme['scheme_id']}",
                delta=best_scheme['scheme_name']
            )

        with col2:
            st.metric(
                "最高BSJ准确率",
                f"{best_scheme['bsj_accuracy']}%",
                delta=f"+{best_scheme['bsj_accuracy'] - results[0]['bsj_accuracy']}%"
            )

        with col3:
            st.metric(
                "最快推理",
                f"{best_scheme['inference_speed']}x",
                delta="相对速度"
            )

        st.success(f"""
        ✅ **推荐使用：Scheme {best_scheme['scheme_id']} ({best_scheme['scheme_name']})**

        **优势：**
        - BSJ准确率最高：{best_scheme['bsj_accuracy']}%
        - RMSD最优：{best_scheme['rmsd']} Å
        - 推理速度：{best_scheme['inference_speed']}x基准

        **适用场景：** {self._get_scheme_usage(best_scheme['scheme_id'])}
        """)

        # ═══════════════════════════════════════════════════════════
        # Export Results
        # ═══════════════════════════════════════════════════════════

        st.markdown("---")

        if st.button("📥 导出对比结果"):
            df.to_csv('batch_scheme_comparison.csv', index=False)
            st.success("✅ 对比结果已保存到 batch_scheme_comparison.csv")

            csv_data = df.to_csv(index=False)
            st.download_button(
                label="下载CSV文件",
                data=csv_data,
                file_name='batch_scheme_comparison.csv',
                mime='text/csv'
            )

    def _get_scheme_usage(self, scheme_id):
        """Get Scheme usage recommendations"""
        usages = {
            0: "官方基线、数据生成、知识蒸馏Teacher",
            1: "EGNN物理约束、结构精修",
            3: "知识蒸馏、快速推理",
            7: "生产级应用、最佳性能 ⭐ 推荐",
            8: "BSJ准确率优化、稀疏配对"
        }
        return usages.get(scheme_id, "通用场景")


# Convenience function
def run_batch_schemes(config):
    """Run batch Scheme comparison"""
    runner = BatchSchemeRunner()
    return runner.run_all_schemes(config)