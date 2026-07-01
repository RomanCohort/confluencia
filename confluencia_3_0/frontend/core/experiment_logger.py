"""Experiment Logger - Record and retrieve experiment history"""

import streamlit as st
import sqlite3
import json
from pathlib import Path
from datetime import datetime
import pandas as pd


class ExperimentLogger:
    """Experiment history management system"""

    def __init__(self, db_path='experiments/experiments.db'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                scheme_id INTEGER NOT NULL,
                config_json TEXT,
                train_loss REAL,
                val_loss REAL,
                bsj_accuracy REAL,
                rmsd REAL,
                best_epoch INTEGER,
                output_path TEXT,
                checkpoint_path TEXT,
                notes TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def log_experiment(self, scheme_id, config, results):
        """
        Log experiment to database

        Args:
            scheme_id: Scheme number
            config: Configuration dictionary
            results: Results dictionary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO experiments (
                timestamp, scheme_id, config_json,
                train_loss, val_loss, bsj_accuracy, rmsd,
                best_epoch, output_path, checkpoint_path, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            scheme_id,
            json.dumps(config),
            results.get('train_loss', 0),
            results.get('val_loss', 0),
            results.get('bsj_accuracy', 0),
            results.get('rmsd', 0),
            results.get('best_epoch', 0),
            results.get('output_path', ''),
            results.get('checkpoint_path', ''),
            results.get('notes', '')
        ))

        exp_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # Also save to JSON
        self._save_json_log(exp_id, scheme_id, config, results)

        return exp_id

    def _save_json_log(self, exp_id, scheme_id, config, results):
        """Save experiment to JSON file"""
        log_dir = Path('experiments/logs')
        log_dir.mkdir(parents=True, exist_ok=True)

        log_data = {
            'id': exp_id,
            'timestamp': datetime.now().isoformat(),
            'scheme_id': scheme_id,
            'config': config,
            'results': results
        }

        log_path = log_dir / f'experiment_{exp_id}.json'
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

    def load_history(self, limit=50):
        """
        Load experiment history from database

        Returns:
            DataFrame with experiment history
        """
        conn = sqlite3.connect(self.db_path)

        df = pd.read_sql_query('''
            SELECT id, timestamp, scheme_id,
                   train_loss, val_loss, bsj_accuracy, rmsd,
                   best_epoch, notes
            FROM experiments
            ORDER BY timestamp DESC
            LIMIT ?
        ''', conn, params=(limit,))

        conn.close()
        return df

    def get_experiment(self, exp_id):
        """
        Retrieve specific experiment details

        Args:
            exp_id: Experiment ID

        Returns:
            Dictionary with experiment details
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM experiments WHERE id = ?
        ''', (exp_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                'id': row[0],
                'timestamp': row[1],
                'scheme_id': row[2],
                'config': json.loads(row[3]) if row[3] else {},
                'train_loss': row[4],
                'val_loss': row[5],
                'bsj_accuracy': row[6],
                'rmsd': row[7],
                'best_epoch': row[8],
                'output_path': row[9],
                'checkpoint_path': row[10],
                'notes': row[11]
            }
        return None

    def render_history_panel(self):
        """Render experiment history interface"""
        st.subheader("📚 实验历史记录")
        st.caption("查看和管理历史实验")

        # Load history
        history_df = self.load_history()

        if len(history_df) == 0:
            st.info("暂无实验记录")
            return

        # ═══════════════════════════════════════════════════════════
        # History Table
        # ═══════════════════════════════════════════════════════════

        st.dataframe(
            history_df,
            use_container_width=True,
            column_config={
                'id': st.column_config.NumberColumn('ID', width='small'),
                'timestamp': st.column_config.TextColumn('时间', width='medium'),
                'scheme_id': st.column_config.NumberColumn('Scheme', width='small'),
                'bsj_accuracy': st.column_config.NumberColumn('BSJ准确率', format='%.2f%%'),
                'rmsd': st.column_config.NumberColumn('RMSD (Å)', format='%.2f'),
            }
        )

        # ═══════════════════════════════════════════════════════════
        # Load Previous Experiment
        # ═══════════════════════════════════════════════════════════

        st.markdown("---")
        st.subheader("🔄 加载历史实验")

        col1, col2 = st.columns([3, 1])

        with col1:
            selected_id = st.selectbox(
                "选择实验ID",
                options=history_df['id'].tolist(),
                format_func=lambda x: f"实验 {x} - Scheme {history_df[history_df['id']==x]['scheme_id'].values[0]}"
            )

        with col2:
            if st.button("加载配置", type="primary"):
                experiment = self.get_experiment(selected_id)
                if experiment:
                    st.session_state['loaded_config'] = experiment['config']
                    st.success(f"✅ 已加载实验 {selected_id} 的配置")

        # Show selected experiment details
        if selected_id:
            experiment = self.get_experiment(selected_id)
            if experiment:
                with st.expander(f"📊 实验 {selected_id} 详细信息"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("训练损失", f"{experiment['train_loss']:.4f}")
                        st.metric("验证损失", f"{experiment['val_loss']:.4f}")
                        st.metric("最佳Epoch", experiment['best_epoch'])

                    with col2:
                        st.metric("BSJ准确率", f"{experiment['bsj_accuracy']:.2f}%")
                        st.metric("RMSD", f"{experiment['rmsd']:.2f} Å")

                    st.markdown("**配置参数:**")
                    st.json(experiment['config'])

                    if experiment['notes']:
                        st.markdown(f"**备注:** {experiment['notes']}")

        # ═══════════════════════════════════════════════════════════
        # Best Experiment
        # ═══════════════════════════════════════════════════════════

        st.markdown("---")
        st.subheader("🏆 最佳实验")

        best_exp = self.get_best_experiment()

        if best_exp:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("最佳BSJ准确率", f"{best_exp['bsj_accuracy']:.2f}%")

            with col2:
                st.metric("实验ID", best_exp['id'])

            with col3:
                st.metric("Scheme", best_exp['scheme_id'])

            if st.button("加载最佳配置"):
                st.session_state['loaded_config'] = best_exp['config']
                st.success("✅ 已加载最佳实验配置")

        # ═══════════════════════════════════════════════════════════
        # Clear History
        # ═══════════════════════════════════════════════════════════

        st.markdown("---")
        if st.button("🗑️ 清空历史记录", type="secondary"):
            if st.checkbox("确认清空"):
                self._clear_history()
                st.success("✅ 历史记录已清空")
                st.rerun()

    def get_best_experiment(self):
        """Get experiment with highest BSJ accuracy"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM experiments
            ORDER BY bsj_accuracy DESC
            LIMIT 1
        ''')

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                'id': row[0],
                'timestamp': row[1],
                'scheme_id': row[2],
                'config': json.loads(row[3]) if row[3] else {},
                'bsj_accuracy': row[6],
                'rmsd': row[7]
            }
        return None

    def _clear_history(self):
        """Clear all experiment history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM experiments')
        conn.commit()
        conn.close()

        # Also clear JSON logs
        log_dir = Path('experiments/logs')
        if log_dir.exists():
            for file in log_dir.glob('*.json'):
                file.unlink()


# Convenience function
def render_experiment_history():
    """Render experiment history interface"""
    logger = ExperimentLogger()
    logger.render_history_panel()