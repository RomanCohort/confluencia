"""Data Exporter - Export results in multiple formats"""

import streamlit as st
import pandas as pd
import json
import zipfile
from pathlib import Path
from datetime import datetime


class DataExporter:
    """Multi-format data export module"""

    def __init__(self):
        self.export_dir = Path('exports')
        self.export_dir.mkdir(exist_ok=True)

    def render_export_panel(self, results):
        """
        Render data export interface

        Args:
            results: Dictionary containing prediction results
        """
        st.subheader("📥 数据导出")
        st.caption("导出预测结果和训练数据")

        # ═══════════════════════════════════════════════════════════
        # Export Format Selection
        # ═══════════════════════════════════════════════════════════

        export_format = st.radio(
            "选择导出格式",
            options=['PDB (3D结构)', 'JSON (质量指标)', 'CSV (训练曲线)', '全部 (ZIP压缩)'],
            horizontal=True
        )

        # ═══════════════════════════════════════════════════════════
        # Export Content Selection
        # ═══════════════════════════════════════════════════════════

        export_content = st.multiselect(
            "选择导出内容",
            options=['3D结构文件', '质量指标', '训练曲线', '配置文件', '实验日志'],
            default=['3D结构文件', '质量指标']
        )

        # ═══════════════════════════════════════════════════════════
        # Export Button
        # ═══════════════════════════════════════════════════════════

        if st.button("🚀 开始导出", type="primary"):
            with st.spinner("正在导出数据..."):
                export_path = self._export_data(
                    results,
                    export_format,
                    export_content
                )

                st.success(f"✅ 导出完成！")

                # Show export summary
                self._show_export_summary(export_path, export_content)

        # ═══════════════════════════════════════════════════════════
        # Batch Export (for multiple sequences)
        # ═══════════════════════════════════════════════════════════

        if 'batch_mode' in results:
            self._render_batch_export(results)

    def _export_data(self, results, export_format, export_content):
        """
        Export data in selected format

        Returns:
            export_path: Path to exported files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = self.export_dir / f'export_{timestamp}'
        export_path.mkdir(exist_ok=True)

        # Export PDB files
        if 'PDB' in export_format or '全部' in export_format:
            if '3D结构文件' in export_content:
                self._export_pdb(results, export_path)

        # Export JSON metrics
        if 'JSON' in export_format or '全部' in export_format:
            if '质量指标' in export_content:
                self._export_json(results, export_path)

        # Export CSV training curves
        if 'CSV' in export_format or '全部' in export_format:
            if '训练曲线' in export_content:
                self._export_csv(results, export_path)

        # Create ZIP if exporting all
        if '全部' in export_format:
            self._create_zip(export_path)

        return export_path

    def _export_pdb(self, results, export_path):
        """Export PDB structure files"""
        pdb_dir = export_path / 'pdb_files'
        pdb_dir.mkdir(exist_ok=True)

        if 'pdb_files' in results:
            for i, pdb_file in enumerate(results['pdb_files']):
                # Copy PDB file
                import shutil
                dest = pdb_dir / f'structure_{i}.pdb'
                shutil.copy(pdb_file, dest)

            st.info(f"📄 已导出 {len(results['pdb_files'])} 个PDB文件")

    def _export_json(self, results, export_path):
        """Export quality metrics as JSON"""
        metrics_json = {
            'timestamp': datetime.now().isoformat(),
            'sequence_id': results.get('sequence_id', 'unknown'),
            'quality_metrics': {
                'confidence': results.get('confidence', 0),
                'bsj_distance': results.get('bsj_distance', 0),
                'energy': results.get('energy', 0),
                'rmsd': results.get('rmsd', 0),
                'bsj_accuracy': results.get('bsj_accuracy', 0)
            },
            'training_metrics': {
                'train_loss': results.get('train_loss', []),
                'val_loss': results.get('val_loss', []),
                'best_epoch': results.get('best_epoch', 0)
            }
        }

        json_path = export_path / 'metrics.json'
        with open(json_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)

        st.info(f"📊 已导出质量指标到: {json_path.name}")

        # Download button
        with open(json_path, 'rb') as f:
            st.download_button(
                label="📥 下载JSON文件",
                data=f,
                file_name='metrics.json',
                mime='application/json'
            )

    def _export_csv(self, results, export_path):
        """Export training curves as CSV"""
        if 'training_history' in results:
            df = pd.DataFrame(results['training_history'])
            csv_path = export_path / 'training_curve.csv'
            df.to_csv(csv_path, index=False)

            st.info(f"📈 已导出训练曲线到: {csv_path.name}")

            # Download button
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 下载CSV文件",
                data=csv_data,
                file_name='training_curve.csv',
                mime='text/csv'
            )

    def _create_zip(self, export_path):
        """Create ZIP archive of all exported files"""
        zip_path = export_path.parent / f'{export_path.name}.zip'

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in export_path.rglob('*'):
                if file.is_file():
                    arcname = file.relative_to(export_path)
                    zipf.write(file, arcname)

        st.success(f"📦 已创建压缩包: {zip_path.name}")

        # Download ZIP button
        with open(zip_path, 'rb') as f:
            st.download_button(
                label="📥 下载所有文件 (ZIP)",
                data=f,
                file_name=f'{export_path.name}.zip',
                mime='application/zip'
            )

    def _show_export_summary(self, export_path, export_content):
        """Show export summary table"""
        summary_data = {
            '内容': export_content,
            '状态': ['✅ 已导出'] * len(export_content),
            '路径': [str(export_path)] * len(export_content)
        }

        df = pd.DataFrame(summary_data)
        st.table(df)

    def _render_batch_export(self, results):
        """Render batch export interface for multiple sequences"""
        st.markdown("---")
        st.subheader("📦 批量导出")

        num_sequences = len(results.get('batch_results', []))

        st.info(f"检测到 {num_sequences} 条序列结果")

        if st.button("批量导出所有结果"):
            with st.spinner(f"正在导出 {num_sequences} 条序列..."):
                batch_export_dir = self._export_batch(results)

            st.success(f"✅ 批量导出完成！共导出 {num_sequences} 条序列")

            # Download ZIP
            zip_path = batch_export_dir.parent / f'{batch_export_dir.name}.zip'
            with open(zip_path, 'rb') as f:
                st.download_button(
                    label=f"📥 下载批量结果 (ZIP, {num_sequences}序列)",
                    data=f,
                    file_name=f'batch_export_{num_sequences}_sequences.zip',
                    mime='application/zip'
                )

    def _export_batch(self, results):
        """Export batch results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = self.export_dir / f'batch_{timestamp}'
        batch_dir.mkdir(exist_ok=True)

        for i, result in enumerate(results.get('batch_results', [])):
            seq_dir = batch_dir / f'sequence_{i}'
            seq_dir.mkdir(exist_ok=True)

            # Export each sequence
            self._export_json(result, seq_dir)
            self._export_pdb(result, seq_dir)

        # Create batch summary CSV
        summary = []
        for i, result in enumerate(results.get('batch_results', [])):
            summary.append({
                'sequence_id': i,
                'confidence': result.get('confidence', 0),
                'bsj_distance': result.get('bsj_distance', 0),
                'energy': result.get('energy', 0)
            })

        df = pd.DataFrame(summary)
        df.to_csv(batch_dir / 'batch_summary.csv', index=False)

        # Create ZIP
        self._create_zip(batch_dir)

        return batch_dir


# Convenience function
def export_results(results):
    """Export results with default settings"""
    exporter = DataExporter()
    exporter.render_export_panel(results)