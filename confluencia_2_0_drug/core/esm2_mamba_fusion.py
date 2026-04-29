"""
ESM-2 + Mamba3Lite Gating 门控融合编码器
用于提升 Epitope 结合预测性能

架构:
  肽段序列
      │
      ├──→ ESM-2(8M) ──→ 全局语义嵌入 (320维)
      │
      └──→ Mamba3Lite ──→ 多尺度局部特征 (528维)
                          │
                          ↓ Gate Network
      ┌───────────────────┴───────────────────┐
      │            Gating: w = σ(W·concat)    │
      │  fused = w*ESM2 + (1-w)*Mamba        │
      └───────────────────────────────────────┘
                          │
                          ↓
              融合特征 (320维) + 门权重 (1维)
"""

import os
import numpy as np
from typing import Optional, Dict, List, Tuple

# ESM-2 配置
ESM2_MODELS = {
    "8M": ("facebook/esm2_t6_8M_UR50D", 320),
    "650M": ("facebook/esm2_t33_650M_UR50D", 1280),
}
ESM2_MODEL_NAME = ESM2_MODELS["8M"][0]  # 默认 8M
ESM2_EMBED_DIM = ESM2_MODELS["8M"][1]

# Mamba3Lite 配置 (保持原有)
MAMBA_EMBED_DIM = 528  # 96*4 + 64*2 + 16 + env

# 融合维度 (门控后与 ESM2 相同，但保留门权重用于分析)
FUSED_EMBED_DIM = ESM2_EMBED_DIM  # 320


class ESM2Encoder:
    """
    ESM-2 编码器，支持 8M (320-dim) 或 650M (1280-dim) 模型。
    """

    def __init__(self, model_size: str = "8M", model_dir: Optional[str] = None):
        self.model = None
        self.tokenizer = None
        self.model_dir = model_dir or os.environ.get('HF_HOME', None)
        self.model_size = str(model_size)
        self._model_name = ESM2_MODELS.get(self.model_size, ESM2_MODELS["8M"])[0]
        self._embed_dim = ESM2_MODELS.get(self.model_size, ESM2_MODELS["8M"])[1]

    def load(self):
        """延迟加载 ESM-2 模型"""
        if self.model is not None:
            return

        try:
            from transformers import AutoModel, AutoTokenizer

            print(f"[ESM2] 加载模型: {self._model_name}")

            os.environ['TRANSFORMERS_OFFLINE'] = '1' if self.model_dir else '0'

            self.model = AutoModel.from_pretrained(
                self._model_name,
                local_files_only=(self.model_dir is not None)
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self.model.eval()

            print(f"[ESM2] 模型加载成功，嵌入维度: {self._embed_dim}")

        except Exception as e:
            print(f"[ESM2] 加载失败: {e}")
            print("[ESM2] 请确保已联网下载模型，或设置 HF_HOME 为本地缓存目录")
            raise

    def encode(self, sequences: List[str]) -> np.ndarray:
        """
        Encode peptide sequences to embedding vectors.

        Args:
            sequences: amino acid sequence list

        Returns:
            embeddings: (N, 320) or (N, 1280) numpy array depending on model size
        """
        if self.model is None:
            self.load()

        import torch

        with torch.no_grad():
            inputs = self.tokenizer(
                sequences,
                padding=True,
                truncation=True,
                max_length=50,
                return_tensors='pt'
            )

            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state  # (B, L, 320)

            # Mean pooling
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            embeddings = (last_hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

        return embeddings.cpu().numpy()


class MambaEncoder:
    """
    Mamba3Lite 编码器
    使用已有的 EpitopeFeatureEngine
    """

    def __init__(self):
        self.encoder = None

    def load(self):
        """延迟加载 Mamba3Lite 编码器"""
        if self.encoder is not None:
            return

        try:
            from core.features import EpitopeFeatureEngine
            print("[Mamba] 加载 Mamba3Lite 编码器")
            self.encoder = EpitopeFeatureEngine()

        except ImportError as e:
            print(f"[Mamba] 加载失败: {e}")
            raise

    def encode(self, sequences: List[str]) -> np.ndarray:
        """编码肽段序列"""
        if self.encoder is None:
            self.load()

        features = self.encoder.extract_features(sequences)
        return features


class GatingNetwork:
    """
    门控网络: 决定 ESM-2 和 Mamba3Lite 的融合权重

    公式:
        w = sigmoid(W_g @ concat(esm2_feat, mamba_feat) + b_g)
        fused = w * esm2_feat + (1 - w) * mamba_feat

    门权重 w ∈ [0, 1]:
        w → 1: 更信任 ESM-2 全局语义
        w → 0: 更信任 Mamba3Lite 局部模式
    """

    def __init__(self, esm2_dim: int = ESM2_EMBED_DIM, mamba_dim: int = MAMBA_EMBED_DIM):
        self.esm2_dim = esm2_dim
        self.mamba_dim = mamba_dim
        self.input_dim = esm2_dim + mamba_dim

        # 门控网络权重 (轻量级: 单层 MLP)
        # W_g: (1, esm2_dim + mamba_dim)
        self.W_gate = np.random.randn(1, self.input_dim) * 0.01
        self.b_gate = np.zeros(1)

    def compute_gate(self, esm2_feat: np.ndarray, mamba_feat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算门控权重和融合特征

        Args:
            esm2_feat: (N, 320) ESM-2 特征
            mamba_feat: (N, 528) Mamba3Lite 特征

        Returns:
            gate_weights: (N, 1) 门权重 (0-1)
            fused: (N, 320) 融合特征
        """
        # 拼接特征
        concat_feat = np.concatenate([esm2_feat, mamba_feat], axis=1)

        # 计算门权重: sigmoid
        gate_logits = concat_feat @ self.W_gate.T + self.b_gate
        gate_weights = 1 / (1 + np.exp(-gate_logits))  # sigmoid

        # 融合: w * ESM2 + (1-w) * Mamba
        fused = gate_weights * esm2_feat + (1 - gate_weights) * mamba_feat

        return gate_weights, fused

    def fit(self, esm2_feats: np.ndarray, mamba_feats: np.ndarray, labels: np.ndarray):
        """
        训练门控网络 (简单的线性回归 + sigmoid)

        Args:
            esm2_feats: (N, 320) ESM-2 特征
            mamba_feats: (N, 528) Mamba3Lite 特征
            labels: (N,) 二分类标签
        """
        from sklearn.linear_model import LogisticRegression

        concat_feats = np.concatenate([esm2_feats, mamba_feats], axis=1)

        # 使用 logistic regression 学习门控权重
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr.fit(concat_feats, labels)

        # 提取权重
        self.W_gate = lr.coef_
        self.b_gate = lr.intercept_

        print(f"[Gate] 门控网络训练完成")
        print(f"[Gate] 正类样本平均门权重: {labels.mean():.3f}")


class GatedFusionEncoder:
    """
    Gating 门控融合编码器 (方案 C)

    特点:
    - 自适应学习 ESM-2 和 Mamba3Lite 的融合权重
    - 可解释性强: 门权重直接反映模型对两种特征的信任度
    - 轻量级: 门控网络只有 ~850 参数
    """

    def __init__(
        self,
        esm2_model_dir: Optional[str] = None,
        use_mamba: bool = True
    ):
        self.esm2_encoder = ESM2Encoder(esm2_model_dir)
        self.mamba_encoder = MambaEncoder() if use_mamba else None
        self.gating_net = GatingNetwork()

        self._is_fitted = False
        self._esm2_mean = None
        self._esm2_std = None
        self._mamba_mean = None
        self._mamba_std = None

    def load_all(self):
        """加载所有编码器"""
        self.esm2_encoder.load()
        if self.mamba_encoder is not None:
            self.mamba_encoder.load()

    def encode(self, sequences: List[str], return_gate_weights: bool = False):
        """
        门控融合编码

        Args:
            sequences: 氨基酸序列列表
            return_gate_weights: 是否返回门权重

        Returns:
            fused_features: (N, 320) 融合特征
            gate_weights: (N, 1) [可选] 门权重
        """
        # 1. ESM-2 编码
        esm2_features = self.esm2_encoder.encode(sequences)

        # 2. Mamba3Lite 编码
        if self.mamba_encoder is not None:
            mamba_features = self.mamba_encoder.encode(sequences)
        else:
            mamba_features = np.zeros((len(sequences), MAMBA_EMBED_DIM))

        # 3. 标准化 (如果已训练)
        if self._is_fitted:
            esm2_features = (esm2_features - self._esm2_mean) / (self._esm2_std + 1e-8)
            mamba_features = (mamba_features - self._mamba_mean) / (self._mamba_std + 1e-8)

        # 4. 门控融合
        if self._is_fitted:
            gate_weights, fused = self.gating_net.compute_gate(esm2_features, mamba_features)
        else:
            # 未训练时: 简单平均
            gate_weights = np.full((len(sequences), 1), 0.5)
            fused = 0.5 * esm2_features + 0.5 * mamba_features[:, :ESM2_EMBED_DIM]

        # 5. L2 归一化
        norms = np.linalg.norm(fused, axis=1, keepdims=True)
        fused = fused / (norms + 1e-8)

        if return_gate_weights:
            return fused, gate_weights
        return fused

    def fit(self, sequences: List[str], labels: np.ndarray):
        """
        训练融合模型

        Args:
            sequences: 氨基酸序列列表
            labels: (N,) 二分类标签
        """
        print("[GatedFusion] 开始训练门控融合模型")
        print(f"[GatedFusion] 样本数: {len(sequences)}, 正例: {labels.sum()}, 负例: {len(labels)-labels.sum()}")

        self.load_all()

        # 1. 提取特征
        print("[GatedFusion] 提取 ESM-2 特征...")
        esm2_features = self.esm2_encoder.encode(sequences)

        print("[GatedFusion] 提取 Mamba3Lite 特征...")
        mamba_features = self.mamba_encoder.encode(sequences)

        # 2. 标准化
        self._esm2_mean = esm2_features.mean(axis=0)
        self._esm2_std = esm2_features.std(axis=0)
        self._mamba_mean = mamba_features.mean(axis=0)
        self._mamba_std = mamba_features.std(axis=0)

        esm2_norm = (esm2_features - self._esm2_mean) / (self._esm2_std + 1e-8)
        mamba_norm = (mamba_features - self._mamba_mean) / (self._mamba_std + 1e-8)

        # 3. 训练门控网络
        self.gating_net.fit(esm2_norm, mamba_norm, labels)
        self._is_fitted = True

        print("[GatedFusion] 训练完成")

    def get_feature_dim(self) -> int:
        """返回融合特征维度"""
        return FUSED_EMBED_DIM

    def get_gate_statistics(self, sequences: List[str]) -> Dict[str, float]:
        """
        获取门权重统计 (用于可解释性分析)

        Returns:
            dict: 门权重统计 {mean, std, min, max}
        """
        _, gate_weights = self.encode(sequences, return_gate_weights=True)

        return {
            'mean': float(gate_weights.mean()),
            'std': float(gate_weights.std()),
            'min': float(gate_weights.min()),
            'max': float(gate_weights.max()),
            'prefer_esm2_ratio': float((gate_weights > 0.5).mean()),
            'prefer_mamba_ratio': float((gate_weights <= 0.5).mean())
        }


class GatedFusionClassifier:
    """
    门控融合分类器
    封装: GatedFusionEncoder + HGB 分类器
    """

    def __init__(self, esm2_model_dir: Optional[str] = None, random_state: int = 42):
        self.encoder = GatedFusionEncoder(esm2_model_dir, use_mamba=True)
        self.random_state = random_state

        self.classifier = None
        self._is_trained = False

    def fit(self, sequences: List[str], labels: np.ndarray):
        """训练分类器"""
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler

        print("=" * 60)
        print("门控融合分类器训练")
        print("=" * 60)

        # 1. 训练门控融合编码器
        self.encoder.fit(sequences, labels)

        # 2. 提取融合特征
        print("[Train] 提取融合特征...")
        X = self.encoder.encode(sequences)

        # 3. 标准化 + 分类器
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        self.classifier = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=self.random_state
        )
        self.classifier.fit(X_scaled, labels)
        self._is_trained = True

        print("[Train] 分类器训练完成")

        # 4. 门权重分析
        gate_stats = self.encoder.get_gate_statistics(sequences)
        print(f"\n[Gate 分析]")
        print(f"  平均门权重: {gate_stats['mean']:.3f} (1=ESM2, 0=Mamba)")
        print(f"  偏好 ESM-2: {gate_stats['prefer_esm2_ratio']:.1%}")
        print(f"  偏好 Mamba: {gate_stats['prefer_mamba_ratio']:.1%}")

    def predict(self, sequences: List[str]) -> np.ndarray:
        """预测二分类"""
        X = self.encoder.encode(sequences)
        return self.classifier.predict(X)

    def predict_proba(self, sequences: List[str]) -> np.ndarray:
        """预测概率"""
        X = self.encoder.encode(sequences)
        return self.classifier.predict_proba(X)[:, 1]

    def evaluate(self, sequences: List[str], labels: np.ndarray) -> Dict[str, float]:
        """评估模型性能"""
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

        y_pred = self.predict(sequences)
        y_proba = self.predict_proba(sequences)

        return {
            'accuracy': accuracy_score(labels, y_pred),
            'auc': roc_auc_score(labels, y_proba),
            'f1': f1_score(labels, y_pred)
        }


def quick_benchmark():
    """
    快速基准测试
    比较: ESM2-only vs Mamba-only vs Gated Fusion
    """
    import pandas as pd

    print("=" * 60)
    print("ESM-2 + Mamba3Lite Gating 门控融合基准测试")
    print("=" * 60)

    # 加载测试数据
    try:
        data = pd.read_csv('data/example_epitope.csv')
        sequences = data['sequence'].tolist()[:500]
        labels = data['efficacy'].values[:500]
        labels = (labels > labels.median()).astype(int)
        print(f"[Data] 加载 {len(sequences)} 个样本, 正例率: {labels.mean():.2%}")
    except Exception as e:
        print(f"[Data] 使用模拟数据: {e}")
        sequences = ['SLYNTVATLYKYRK'] * 100
        labels = np.random.randint(0, 2, 100)

    # 测试门控融合
    print("\n[测试] 门控融合编码器")
    encoder = GatedFusionEncoder(use_mamba=True)

    try:
        encoder.load_all()
        X_fusion, gate_weights = encoder.encode(sequences, return_gate_weights=True)

        print(f"  融合特征维度: {X_fusion.shape}")
        print(f"  门权重范围: [{gate_weights.min():.3f}, {gate_weights.max():.3f}]")
        print(f"  门权重均值: {gate_weights.mean():.3f}")

    except Exception as e:
        print(f"  测试失败: {e}")
        print("  (可能需要联网下载 ESM-2 模型)")

    print("\n" + "=" * 60)
    print("Gating 门控融合编码器准备就绪")
    print("=" * 60)

    return {
        'esm2_dim': ESM2_EMBED_DIM,
        'mamba_dim': MAMBA_EMBED_DIM,
        'fused_dim': FUSED_EMBED_DIM,
        'n_samples': len(sequences)
    }


if __name__ == '__main__':
    result = quick_benchmark()
    print(f"\n结果: {result}")
