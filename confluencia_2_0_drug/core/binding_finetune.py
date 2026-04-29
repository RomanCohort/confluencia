"""
Binding 专项微调模块
Step 3: 在多任务预训练基础上，专项优化 Binding 分类性能

策略:
1. 多任务预训练 → 共享特征表示 (已由 GatedFusion + MHC 完成)
2. 冻结共享层 → 只训练 Binding 分类头
3. 专项数据增强 → 聚焦 Binding 相关样本
"""

import os
import numpy as np
from typing import Optional, Dict, List, Tuple

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score
from sklearn.model_selection import StratifiedKFold


class BindingFineTuner:
    """
    Binding 专项微调器

    流程:
    1. 提取 ESM-2 + Mamba + MHC 全量特征
    2. 标准化
    3. 多种分类器交叉验证
    4. 选择最优模型 + 阈值
    5. 输出评估报告
    """

    def __init__(
        self,
        esm2_model_dir: Optional[str] = None,
        random_state: int = 42,
        cv_folds: int = 5
    ):
        self.esm2_model_dir = esm2_model_dir
        self.random_state = random_state
        self.cv_folds = cv_folds

        # 编码器 (延迟加载)
        self.encoder = None
        self.scaler = None
        self.best_model = None
        self.best_threshold = 0.5
        self.best_model_name = ""

        # 评估结果
        self.cv_results = {}
        self.test_results = {}

    def load_encoder(self):
        """加载编码器"""
        if self.encoder is not None:
            return

        from core.mhc_features import FullEpitopeEncoder
        self.encoder = FullEpitopeEncoder(
            esm2_model_dir=self.esm2_model_dir,
            use_mamba=True,
            use_mhc=True
        )
        self.encoder.load()

    def extract_features(
        self,
        peptides: List[str],
        alleles: Optional[List[str]] = None,
        sequences: Optional[List[str]] = None
    ) -> np.ndarray:
        """提取完整特征"""
        self.load_encoder()
        return self.encoder.encode(peptides, alleles, sequences)

    def train(
        self,
        peptides: List[str],
        labels: np.ndarray,
        alleles: Optional[List[str]] = None,
        sequences: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        训练 Binding 分类器 (交叉验证选最优)

        Args:
            peptides: 肽段序列
            labels: (N,) 二分类标签 (0=non-binder, 1=binder)
            alleles: HLA 等位基因 [可选]
            sequences: 表位序列 [可选]

        Returns:
            best_results: 最优模型的评估指标
        """
        print("=" * 60)
        print("Binding 专项微调")
        print("=" * 60)
        print(f"样本数: {len(peptides)}, 正例: {labels.sum()}, 负例: {len(labels)-labels.sum()}")

        # 1. 提取特征
        print("\n[Step 1] 提取特征...")
        X = self.extract_features(peptides, alleles, sequences)
        print(f"  特征维度: {X.shape}")

        # 2. 标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 3. 候选分类器交叉验证
        print(f"\n[Step 2] {self.cv_folds} 折交叉验证...")
        candidates = {
            'HGB': HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_depth=8,
                l2_regularization=0.1,
                random_state=self.random_state
            ),
            'RF': RandomForestClassifier(
                n_estimators=500,
                max_depth=15,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=self.random_state
            ),
            'Logistic': LogisticRegression(
                C=1.0,
                max_iter=2000,
                class_weight='balanced',
                random_state=self.random_state
            ),
            'HGB_deep': HistGradientBoostingClassifier(
                max_iter=500,
                learning_rate=0.03,
                max_depth=10,
                l2_regularization=0.05,
                random_state=self.random_state
            ),
        }

        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        for name, clf in candidates.items():
            aucs = []
            accs = []
            f1s = []

            for train_idx, val_idx in skf.split(X_scaled, labels):
                X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_tr, y_val = labels[train_idx], labels[val_idx]

                clf.fit(X_tr, y_tr)
                y_pred_proba = clf.predict_proba(X_val)[:, 1]

                aucs.append(roc_auc_score(y_val, y_pred_proba))
                accs.append(accuracy_score(y_val, (y_pred_proba > 0.5).astype(int)))
                f1s.append(f1_score(y_val, (y_pred_proba > 0.5).astype(int)))

            self.cv_results[name] = {
                'auc_mean': np.mean(aucs),
                'auc_std': np.std(aucs),
                'acc_mean': np.mean(accs),
                'f1_mean': np.mean(f1s),
            }

            print(f"  {name:12s}: AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}, "
                  f"Acc={np.mean(accs):.4f}, F1={np.mean(f1s):.4f}")

        # 4. 选择最优模型
        best_name = max(self.cv_results, key=lambda k: self.cv_results[k]['auc_mean'])
        self.best_model_name = best_name
        self.best_model = candidates[best_name]

        # 在全量数据上重新训练
        print(f"\n[Step 3] 最优模型: {best_name}, 在全量数据上重新训练...")
        self.best_model.fit(X_scaled, labels)

        # 5. 优化阈值 (Youden's J)
        self.best_threshold = 0.5  # 默认
        y_proba = self.best_model.predict_proba(X_scaled)[:, 1]

        best_j = 0
        for threshold in np.arange(0.3, 0.7, 0.01):
            y_pred = (y_proba > threshold).astype(int)
            tp = ((y_pred == 1) & (labels == 1)).sum()
            tn = ((y_pred == 0) & (labels == 0)).sum()
            fp = ((y_pred == 1) & (labels == 0)).sum()
            fn = ((y_pred == 0) & (labels == 1)).sum()

            sensitivity = tp / (tp + fn + 1e-8)
            specificity = tn / (tn + fp + 1e-8)
            j = sensitivity + specificity - 1

            if j > best_j:
                best_j = j
                self.best_threshold = threshold

        print(f"  最优阈值: {self.best_threshold:.2f} (Youden's J={best_j:.3f})")

        # 6. 输出结果
        best_results = self.cv_results[best_name]
        print(f"\n{'=' * 60}")
        print(f"最终结果 ({best_name}):")
        print(f"  AUC:  {best_results['auc_mean']:.4f} ± {best_results['auc_std']:.4f}")
        print(f"  Acc:  {best_results['acc_mean']:.4f}")
        print(f"  F1:   {best_results['f1_mean']:.4f}")
        print(f"  特征维度: {X.shape[1]}")
        print(f"{'=' * 60}")

        return best_results

    def predict(self, peptides: List[str], alleles: Optional[List[str]] = None,
                sequences: Optional[List[str]] = None) -> np.ndarray:
        """预测二分类标签"""
        X = self.extract_features(peptides, alleles, sequences)
        X_scaled = self.scaler.transform(X)
        y_proba = self.best_model.predict_proba(X_scaled)[:, 1]
        return (y_proba > self.best_threshold).astype(int)

    def predict_proba(self, peptides: List[str], alleles: Optional[List[str]] = None,
                      sequences: Optional[List[str]] = None) -> np.ndarray:
        """预测概率"""
        X = self.extract_features(peptides, alleles, sequences)
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict_proba(X_scaled)[:, 1]

    def evaluate(
        self,
        peptides: List[str],
        labels: np.ndarray,
        alleles: Optional[List[str]] = None,
        sequences: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """评估模型"""
        y_proba = self.predict_proba(peptides, alleles, sequences)
        y_pred = (y_proba > self.best_threshold).astype(int)

        results = {
            'auc': roc_auc_score(labels, y_proba),
            'auc_pr': average_precision_score(labels, y_proba),
            'accuracy': accuracy_score(labels, y_pred),
            'f1': f1_score(labels, y_pred),
            'n_samples': len(peptides),
            'model': self.best_model_name,
            'threshold': self.best_threshold,
        }

        print(f"\n[评估结果]")
        print(f"  AUC:     {results['auc']:.4f}")
        print(f"  AUC-PR:  {results['auc_pr']:.4f}")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  F1:      {results['f1']:.4f}")
        print(f"  模型:    {results['model']}")
        print(f"  阈值:    {results['threshold']:.2f}")

        return results


class MultiSourceValidator:
    """
    多源外部验证器
    在多个外部数据集上评估 Binding 预测性能
    """

    def __init__(self, fine_tuner: BindingFineTuner):
        self.fine_tuner = fine_tuner

    def validate(self, datasets: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        在多个数据集上验证

        Args:
            datasets: {
                'name': {
                    'peptides': [...],
                    'labels': np.array,
                    'alleles': [...] (optional)
                }
            }

        Returns:
            results: {name: {auc, acc, f1, ...}}
        """
        results = {}

        for name, data in datasets.items():
            print(f"\n{'=' * 40}")
            print(f"验证: {name}")
            print(f"  样本数: {len(data['peptides'])}")

            eval_result = self.fine_tuner.evaluate(
                peptides=data['peptides'],
                labels=data['labels'],
                alleles=data.get('alleles'),
                sequences=data.get('sequences')
            )
            results[name] = eval_result

        # 汇总
        print(f"\n{'=' * 60}")
        print("多源验证汇总")
        print(f"{'=' * 60}")
        print(f"{'数据集':>20s} | {'AUC':>6s} | {'Acc':>6s} | {'F1':>6s}")
        print("-" * 50)
        for name, res in results.items():
            print(f"{name:>20s} | {res['auc']:6.3f} | {res['accuracy']:6.3f} | {res['f1']:6.3f}")

        return results


def run_binding_pipeline(
    train_peptides: List[str],
    train_labels: np.ndarray,
    train_alleles: Optional[List[str]] = None,
    esm2_model_dir: Optional[str] = None
):
    """
    运行完整的 Binding 专项微调管线

    Args:
        train_peptides: 训练肽段
        train_labels: 训练标签
        train_alleles: 训练 HLA 等位基因
        esm2_model_dir: ESM-2 本地缓存

    Returns:
        fine_tuner: 训练好的微调器
    """
    print("=" * 60)
    print("Binding 专项微调管线")
    print("=" * 60)

    # Step 1: 训练
    fine_tuner = BindingFineTuner(
        esm2_model_dir=esm2_model_dir,
        random_state=42,
        cv_folds=5
    )

    fine_tuner.train(
        peptides=train_peptides,
        labels=train_labels,
        alleles=train_alleles
    )

    return fine_tuner


def quick_test():
    """快速测试 (使用模拟数据)"""
    print("Binding 专项微调 - 快速测试")
    print("(使用模拟数据，实际使用时请替换为真实数据)")

    # 模拟数据
    np.random.seed(42)
    peptides = ['SLYNTVATL', 'GILGFVFTL', 'KLGGALQAK', 'NLVPMVATV',
                'IVTDFSVIK', 'LLFGYPVYV', 'RAKFKQLL', 'ELAGIGILTV',
                'AVFDRKSDAK', 'CINGVCWTV', 'YLDKVHMV', 'RMFPNAPYL'] * 50

    alleles = ['HLA-A*02:01', 'HLA-A*02:01', 'HLA-B*07:02', 'HLA-A*01:01',
               'HLA-A*02:01', 'HLA-B*08:01', 'HLA-A*03:01', 'HLA-A*02:01',
               'HLA-B*27:05', 'HLA-A*02:01', 'HLA-A*24:02', 'HLA-B*35:01'] * 50

    labels = np.random.randint(0, 2, len(peptides))

    # 测试 MHC 特征编码 (不需要 ESM-2)
    print("\n[测试] MHC 特征编码")
    from core.mhc_features import MHCFeatureEncoder
    mhc_enc = MHCFeatureEncoder()

    feats = mhc_enc.encode_batch(peptides[:10], alleles[:10])
    print(f"  MHC 特征维度: {feats.shape}")
    print(f"  期望维度: (10, 969)")

    # 交叉验证 (仅使用 MHC 特征，不需要 ESM-2)
    print("\n[测试] MHC-only 分类")
    X_mhc = mhc_enc.encode_batch(peptides, alleles)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_mhc)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for train_idx, val_idx in skf.split(X_scaled, labels):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_val = labels[train_idx], labels[val_idx]

        clf = HistGradientBoostingClassifier(max_iter=100, random_state=42)
        clf.fit(X_tr, y_tr)
        y_proba = clf.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, y_proba))

    print(f"  MHC-only AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  (模拟数据，实际数据上预期更高)")

    print("\n" + "=" * 60)
    print("快速测试完成")
    print("下一步: 联网下载 ESM-2 后运行完整管线")
    print("=" * 60)


if __name__ == '__main__':
    quick_test()
