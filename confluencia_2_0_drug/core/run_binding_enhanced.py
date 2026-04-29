"""
完整 Binding 增强管线
整合: ESM-2 + Mamba3Lite + MHC 特征 + Binding 专项微调

执行顺序:
1. ESM-2 + Mamba3Lite 融合编码 (方案 C: Gating)
2. MHC 等位基因特征
3. Binding 专项微调

目标: 追平 NetMHCpan (AUC 0.92+)
"""

import os
import sys
import json
import numpy as np
from typing import Optional, Dict, List
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


def setup_environment():
    """设置环境"""
    os.environ['TRANSFORMERS_OFFLINE'] = '0'  # 首次需联网
    os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), '..', 'models', 'huggingface')
    os.makedirs(os.environ['HF_HOME'], exist_ok=True)


class FullBindingPipeline:
    """
    完整 Binding 增强管线

    特征组成:
    - ESM-2 嵌入: 320维 (全局语义)
    - Mamba3Lite: 528维 (多尺度局部)
    - MHC 特征: 979维 (伪序列 + HLA one-hot + 结合位置)
    - 总计: 1827维
    """

    def __init__(
        self,
        use_esm2: bool = True,
        use_mamba: bool = True,
        use_mhc: bool = True,
        esm2_model: str = "facebook/esm2_t6_8M_UR50D",
        random_state: int = 42
    ):
        self.use_esm2 = use_esm2
        self.use_mamba = use_mamba
        self.use_mhc = use_mhc
        self.esm2_model = esm2_model
        self.random_state = random_state

        self.esm2_encoder = None
        self.mamba_encoder = None
        self.mhc_encoder = None
        self.scaler = None
        self.classifier = None

        self.feature_dims = {}
        self.cv_results = {}

    def load_encoders(self):
        """加载所有编码器"""
        from core.mhc_features import MHCFeatureEncoder

        if self.use_mhc:
            print("[加载] MHC 特征编码器")
            self.mhc_encoder = MHCFeatureEncoder()
            self.feature_dims['mhc'] = self.mhc_encoder.feature_dim

    def encode_peptides(
        self,
        peptides: List[str],
        alleles: Optional[List[str]] = None,
        sequences: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        编码肽段为完整特征

        Returns:
            features: (N, total_dim) 融合特征
        """
        features = []

        # 1. ESM-2 编码
        if self.use_esm2:
            if self.esm2_encoder is None:
                from core.esm2_mamba_fusion import ESM2Encoder
                print("[编码] 加载 ESM-2 模型...")
                self.esm2_encoder = ESM2Encoder()
                self.esm2_encoder.load()

            print("[编码] ESM-2 嵌入...")
            esm2_feat = self.esm2_encoder.encode(peptides)
            features.append(esm2_feat)
            self.feature_dims['esm2'] = esm2_feat.shape[1]

        # 2. Mamba3Lite 编码
        if self.use_mamba:
            if self.mamba_encoder is None:
                from core.esm2_mamba_fusion import MambaEncoder
                print("[编码] 加载 Mamba3Lite...")
                self.mamba_encoder = MambaEncoder()
                self.mamba_encoder.load()

            print("[编码] Mamba3Lite 特征...")
            mamba_feat = self.mamba_encoder.encode(sequences or peptides)
            features.append(mamba_feat)
            self.feature_dims['mamba'] = mamba_feat.shape[1]

        # 3. MHC 特征编码
        if self.use_mhc and alleles:
            print("[编码] MHC 等位基因特征...")
            mhc_feat = self.mhc_encoder.encode_batch(peptides, alleles)
            features.append(mhc_feat)

        # 拼接
        X = np.concatenate(features, axis=1)
        print(f"[编码] 融合特征维度: {X.shape}")

        return X

    def train(
        self,
        peptides: List[str],
        labels: np.ndarray,
        alleles: Optional[List[str]] = None,
        sequences: Optional[List[str]] = None,
        n_cv: int = 5
    ) -> Dict:
        """
        训练完整管线

        Args:
            peptides: 肽段序列
            labels: (N,) 二分类标签
            alleles: HLA 等位基因 [可选]
            sequences: 表位序列 [可选]
            n_cv: 交叉验证折数

        Returns:
            results: 训练结果
        """
        print("=" * 60)
        print("完整 Binding 增强管线")
        print("=" * 60)
        print(f"样本数: {len(peptides)}")
        print(f"正例: {labels.sum()}, 负例: {len(labels)-labels.sum()}")
        print(f"特征配置: ESM2={self.use_esm2}, Mamba={self.use_mamba}, MHC={self.use_mhc}")
        print("=" * 60)

        # Step 1: 加载编码器
        self.load_encoders()

        # Step 2: 特征编码
        print("\n[Step 1] 特征编码")
        X = self.encode_peptides(peptides, alleles, sequences)

        # Step 3: 标准化
        print("\n[Step 2] 标准化")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Step 4: 交叉验证
        print(f"\n[Step 3] {n_cv} 折交叉验证")
        skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=self.random_state)

        cv_aucs = []
        cv_accs = []
        cv_f1s = []
        oof_preds = np.zeros(len(labels))
        oof_probas = np.zeros(len(labels))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, labels)):
            print(f"\n  Fold {fold+1}/{n_cv}...")

            X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_tr, y_val = labels[train_idx], labels[val_idx]

            # 训练 HGB
            clf = HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_depth=8,
                l2_regularization=0.1,
                early_stopping=True,
                n_iter_no_change=20,
                validation_fraction=0.1,
                random_state=self.random_state
            )
            clf.fit(X_tr, y_tr)

            # 预测
            y_proba = clf.predict_proba(X_val)[:, 1]
            y_pred = (y_proba > 0.5).astype(int)

            # 记录
            fold_auc = roc_auc_score(y_val, y_proba)
            fold_acc = accuracy_score(y_val, y_pred)
            fold_f1 = f1_score(y_val, y_pred)

            cv_aucs.append(fold_auc)
            cv_accs.append(fold_acc)
            cv_f1s.append(fold_f1)
            oof_preds[val_idx] = y_pred
            oof_probas[val_idx] = y_proba

            print(f"    AUC={fold_auc:.4f}, Acc={fold_acc:.4f}, F1={fold_f1:.4f}")

        # Step 5: 训练最终模型
        print("\n[Step 4] 训练最终模型...")
        self.classifier = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=8,
            l2_regularization=0.1,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.1,
            random_state=self.random_state
        )
        self.classifier.fit(X_scaled, labels)

        # 找最优阈值
        best_thresh = 0.5
        best_j = 0
        for thresh in np.arange(0.3, 0.7, 0.01):
            pred = (oof_probas > thresh).astype(int)
            tp = ((pred == 1) & (labels == 1)).sum()
            tn = ((pred == 0) & (labels == 0)).sum()
            fp = ((pred == 1) & (labels == 0)).sum()
            fn = ((pred == 0) & (labels == 1)).sum()

            sens = tp / (tp + fn + 1e-8)
            spec = tn / (tn + fp + 1e-8)
            j = sens + spec - 1

            if j > best_j:
                best_j = j
                best_thresh = thresh

        # 结果汇总
        self.cv_results = {
            'auc_mean': np.mean(cv_aucs),
            'auc_std': np.std(cv_aucs),
            'acc_mean': np.mean(cv_accs),
            'f1_mean': np.mean(cv_f1s),
            'best_threshold': best_thresh,
            'feature_dims': self.feature_dims,
            'n_samples': len(peptides),
            'n_positive': int(labels.sum()),
            'n_negative': int(len(labels) - labels.sum())
        }

        # 打印结果
        print("\n" + "=" * 60)
        print("最终结果")
        print("=" * 60)
        print(f"AUC:  {self.cv_results['auc_mean']:.4f} ± {self.cv_results['auc_std']:.4f}")
        print(f"Acc:  {self.cv_results['acc_mean']:.4f}")
        print(f"F1:   {self.cv_results['f1_mean']:.4f}")
        print(f"阈值: {best_thresh:.2f}")
        print(f"特征维度: {self.feature_dims}")
        print("=" * 60)

        return self.cv_results

    def predict(
        self,
        peptides: List[str],
        alleles: Optional[List[str]] = None,
        sequences: Optional[List[str]] = None
    ) -> np.ndarray:
        """预测"""
        X = self.encode_peptides(peptides, alleles, sequences)
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict(X_scaled)

    def predict_proba(
        self,
        peptides: List[str],
        alleles: Optional[List[str]] = None,
        sequences: Optional[List[str]] = None
    ) -> np.ndarray:
        """预测概率"""
        X = self.encode_peptides(peptides, alleles, sequences)
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict_proba(X_scaled)[:, 1]


def run_with_iedb_data():
    """
    使用 IEDB 数据运行完整管线
    """
    print("加载 IEDB 数据...")

    # 尝试加载 IEDB 数据
    try:
        # 假设数据在 data/ 目录
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'example_epitope.csv')
        df = pd.read_csv(data_path)

        # 检查是否有必要的列
        if 'sequence' in df.columns:
            peptides = df['sequence'].tolist()

            # 创建模拟标签 (如果有 efficacy 列) 或随机生成
            if 'efficacy' in df.columns:
                labels = (df['efficacy'] > df['efficacy'].median()).astype(int).values
            else:
                labels = np.random.randint(0, 2, len(peptides))

            # 模拟等位基因
            alleles = np.random.choice([
                'HLA-A*02:01', 'HLA-B*07:02', 'HLA-A*01:01',
                'HLA-B*08:01', 'HLA-A*03:01'
            ], len(peptides)).tolist()

            print(f"加载 {len(peptides)} 个样本")

            # 仅使用 MHC 特征测试 (ESM-2 需要联网)
            print("\n" + "=" * 60)
            print("测试: MHC-only 模式")
            print("=" * 60)

            pipeline = FullBindingPipeline(
                use_esm2=False,
                use_mamba=False,
                use_mhc=True
            )

            results = pipeline.train(
                peptides=peptides,
                labels=labels,
                alleles=alleles,
                n_cv=5
            )

            return results

    except Exception as e:
        print(f"加载数据失败: {e}")
        print("使用模拟数据...")

    # 模拟数据
    np.random.seed(42)
    peptides = [f"PEPTIDE{i:04d}" for i in range(500)]
    alleles = np.random.choice([
        'HLA-A*02:01', 'HLA-B*07:02', 'HLA-A*01:01',
        'HLA-B*08:01', 'HLA-A*03:01'
    ], 500).tolist()
    labels = np.random.randint(0, 2, 500)

    pipeline = FullBindingPipeline(use_esm2=False, use_mamba=False, use_mhc=True)
    results = pipeline.train(peptides, labels, alleles)

    return results


def main():
    """主函数"""
    print("=" * 60)
    print("Confluencia Binding 增强管线 v1.0")
    print("目标: 追平 NetMHCpan (AUC 0.92+)")
    print("=" * 60)

    # 运行管线
    results = run_with_iedb_data()

    # 保存结果
    output_path = os.path.join(
        os.path.dirname(__file__),
        '..', 'logs', 'binding_enhanced_results.json'
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存: {output_path}")

    # 打印下一步建议
    print("\n" + "=" * 60)
    print("下一步")
    print("=" * 60)
    print("1. 联网下载 ESM-2: pip install fair-esm && python -c 'import esm'")
    print("2. 启用 ESM-2: pipeline = FullBindingPipeline(use_esm2=True)")
    print("3. 运行完整管线")
    print("=" * 60)


if __name__ == '__main__':
    main()
