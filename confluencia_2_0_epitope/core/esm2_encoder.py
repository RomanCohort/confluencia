"""
ESM-2 蛋白质语言模型编码器 (Epitope模块专用)
===============================================
基于 facebook/esm2_t33_650M_UR50D (650M参数, 1280维嵌入)

改进点 (相对于 drug 模块的 8M 版本):
- 升级到 650M 模型 (1280维, 在 2.5 亿蛋白序列上预训练)
- 添加 mini-batch 处理避免大 OOM
- 添加磁盘缓存机制
- 支持 fallback 到较小模型

冻结特征提取: 不 fine-tune, 直接用预训练权重生成嵌入
"""

import os
import hashlib
import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

# 模型配置
_ESM2_MODELS = {
    "650M": {
        "name": "facebook/esm2_t33_650M_UR50D",
        "modelscope": "AI-ModelScope/esm2_t33_650M_UR50D",  # ModelScope 镜像
        "embed_dim": 1280,
        "layers": 33,
    },
    "150M": {
        "name": "facebook/esm2_t30_150M_UR50D",
        "modelscope": "AI-ModelScope/esm2_t30_150M_UR50D",
        "embed_dim": 640,
        "layers": 30,
    },
    "35M": {
        "name": "facebook/esm2_t12_35M_UR50D",
        "modelscope": "AI-ModelScope/esm2_t12_35M_UR50D",
        "embed_dim": 480,
        "layers": 12,
    },
    "8M": {
        "name": "facebook/esm2_t6_8M_UR50D",
        "modelscope": "AI-ModelScope/esm2_t6_8M_UR50D",
        "embed_dim": 320,
        "layers": 6,
    },
}

DEFAULT_MODEL = "650M"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 50


def _seq_hash(seq: str) -> str:
    """生成序列的稳定 hash 作为缓存 key"""
    return hashlib.sha256(seq.encode("utf-8")).hexdigest()[:16]


def _batch_encode(sequences: List[str], tokenizer, model, device, batch_size: int, max_length: int):
    """分批编码，支持大批量数据，带进度显示"""
    import torch
    import sys

    n = len(sequences)
    embeddings = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = sequences[start:end]

        with torch.no_grad():
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state  # (B, L, embed_dim)

            # Mean pooling
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            emb = (last_hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            embeddings.append(emb.cpu().numpy())

        # 进度显示
        progress = end / n * 100
        sys.stdout.write(f"\r[ESM-2] Encoding: {end}/{n} ({progress:.1f}%)")
        sys.stdout.flush()

    print()  # 换行
    return np.vstack(embeddings)


class ESM2Encoder:
    """
    ESM-2 蛋白质语言模型编码器 (冻结特征提取)

    使用方法:
        encoder = ESM2Encoder()  # 默认 650M
        embeddings = encoder.encode(["SLYNTVATL", "GILGFVFTL"])  # shape (2, 1280)

    可用模型:
        - "650M": facebook/esm2_t33_650M_UR50D (1280维, 默认)
        - "150M": facebook/esm2_t30_150M_UR50D (640维)
        - "35M":  facebook/esm2_t12_35M_UR50D (480维)
        - "8M":   facebook/esm2_t6_8M_UR50D (320维, 轻量)
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL,
        model_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
    ):
        if model_size not in _ESM2_MODELS:
            raise ValueError(
                f"Unknown model size '{model_size}'. Available: {list(_ESM2_MODELS.keys())}"
            )

        self.model_size = model_size
        self.model_name = _ESM2_MODELS[model_size]["name"]
        self.embed_dim = _ESM2_MODELS[model_size]["embed_dim"]
        self.batch_size = batch_size
        self.max_length = max_length

        self.model = None
        self.tokenizer = None
        self.device = None

        # 缓存配置
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._memory_cache: dict[str, np.ndarray] = {}

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # 加载已有缓存索引
            self._load_cache_index()

    def _load_cache_index(self):
        """从磁盘加载已有的缓存索引"""
        index_path = self.cache_dir / "esm2_cache_index.json"
        if index_path.exists():
            try:
                with open(index_path) as f:
                    self._cache_index = json.load(f)
            except Exception:
                self._cache_index = {}
        else:
            self._cache_index = {}

    def _save_cache_index(self):
        """保存缓存索引到磁盘"""
        if not self.cache_dir:
            return
        index_path = self.cache_dir / "esm2_cache_index.json"
        with open(index_path, "w") as f:
            json.dump(self._cache_index, f)

    def _get_cache_path(self, seq_hash: str) -> Path:
        return self.cache_dir / f"esm2_{self.model_size}_{seq_hash}.npy"

    def load(self):
        """延迟加载 ESM-2 模型 (优先使用本地缓存)"""
        if self.model is not None:
            return

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            from pathlib import Path as PPath

            print(f"[ESM-2] 加载模型: {self.model_name} ({self.model_size})")
            print(f"[ESM-2] 嵌入维度: {self.embed_dim}")

            # 1. 检查自定义缓存目录
            if self.cache_dir:
                local_model_dir = self.cache_dir / f"esm2_{self.model_size}_local"
                if (local_model_dir / "config.json").exists():
                    print(f"[ESM-2] 使用自定义缓存: {local_model_dir}")
                    self.tokenizer = AutoTokenizer.from_pretrained(str(local_model_dir))
                    self.model = AutoModel.from_pretrained(str(local_model_dir))
            else:
                local_model_dir = None

            # 2. 检查 HuggingFace 默认缓存 (~/.cache/huggingface/hub/)
            if self.model is None:
                hf_cache_root = PPath.home() / ".cache" / "huggingface" / "hub"
                model_cache_name = f"models--facebook--esm2_t33_650M_UR50D" if self.model_size == "650M" else f"models--facebook--esm2_t{ {'150M': '30', '35M': '12', '8M': '6'}[self.model_size] }_{self.model_size}_UR50D"
                hf_cache_dir = hf_cache_root / model_cache_name

                if hf_cache_dir.exists():
                    # 找到 snapshots 目录下的实际模型路径
                    snapshots_dir = hf_cache_dir / "snapshots"
                    if snapshots_dir.exists():
                        snapshot_hashes = list(snapshots_dir.iterdir())
                        if snapshot_hashes:
                            snapshot_path = snapshot_hashes[0]  # 取最新的 snapshot
                            if (snapshot_path / "config.json").exists():
                                print(f"[ESM-2] 使用 HuggingFace 缓存: {snapshot_path}")
                                self.tokenizer = AutoTokenizer.from_pretrained(str(snapshot_path), local_files_only=True)
                                self.model = AutoModel.from_pretrained(str(snapshot_path), local_files_only=True)
                                print(f"[ESM-2] 离线加载成功")

            # 3. 尝试 ModelScope 镜像
            if self.model is None:
                modelscope_name = _ESM2_MODELS[self.model_size].get("modelscope")
                loaded = False

                if modelscope_name:
                    try:
                        from modelscope import snapshot_download
                        print(f"[ESM-2] 尝试 ModelScope 镜像: {modelscope_name}")
                        model_dir = snapshot_download(modelscope_name)
                        print(f"[ESM-2] ModelScope 下载完成: {model_dir}")
                        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
                        self.model = AutoModel.from_pretrained(model_dir)
                        loaded = True
                    except ImportError:
                        print("[ESM-2] modelscope 未安装，跳过镜像下载")
                    except Exception as e:
                        print(f"[ESM-2] ModelScope 下载失败: {e}")

                # 4. 最后尝试 HuggingFace 在线下载
                if not loaded:
                    hf_endpoint = os.environ.get("HF_ENDPOINT", "")
                    if hf_endpoint:
                        print(f"[ESM-2] 使用 HuggingFace 镜像端点: {hf_endpoint}")
                    print(f"[ESM-2] 从 HuggingFace 下载: {self.model_name}")
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    self.model = AutoModel.from_pretrained(self.model_name)

            # 优先使用 GPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            self.model.eval()

            print(f"[ESM-2] 模型加载成功，设备: {self.device}")

        except Exception as e:
            print(f"[ESM-2] 加载失败: {e}")
            print("[ESM-2] 提示: 请确保网络连接或设置 HuggingFace 缓存目录")
            raise

    def encode(self, sequences: List[str]) -> np.ndarray:
        """
        编码肽段序列为嵌入向量

        Args:
            sequences: 氨基酸序列列表

        Returns:
            embeddings: (N, embed_dim) numpy array
        """
        if self.model is None:
            self.load()

        import torch

        # 1. 从缓存获取已编码的序列
        uncached_indices = []
        uncached_hashes = []
        results = [None] * len(sequences)

        for i, seq in enumerate(sequences):
            h = _seq_hash(seq)

            # 内存缓存优先
            if h in self._memory_cache:
                results[i] = self._memory_cache[h]
                continue

            # 磁盘缓存
            if self.cache_dir:
                cache_path = self._get_cache_path(h)
                if cache_path.exists():
                    try:
                        cached = np.load(cache_path)
                        self._memory_cache[h] = cached
                        results[i] = cached
                        continue
                    except Exception:
                        pass

            uncached_indices.append(i)
            uncached_hashes.append(h)

        # 2. 批量编码未缓存的序列
        if uncached_indices:
            seqs_to_encode = [sequences[i] for i in uncached_indices]
            new_embeddings = _batch_encode(
                seqs_to_encode,
                self.tokenizer,
                self.model,
                self.device,
                self.batch_size,
                self.max_length,
            )

            # 3. 存储到缓存
            for j, (idx, h) in enumerate(zip(uncached_indices, uncached_hashes)):
                emb = new_embeddings[j]
                results[idx] = emb

                # 内存缓存
                self._memory_cache[h] = emb

                # 磁盘缓存
                if self.cache_dir:
                    cache_path = self._get_cache_path(h)
                    np.save(cache_path, emb)
                    self._cache_index[h] = {
                        "model_size": self.model_size,
                        "embed_dim": self.embed_dim,
                        "seq": sequences[idx][:50],
                    }

            # 保存索引
            if self.cache_dir:
                self._save_cache_index()

        # 合并结果
        return np.vstack(results)

    def encode_single(self, sequence: str) -> np.ndarray:
        """编码单个序列 (便捷方法)"""
        return self.encode([sequence])[0]

    def clear_cache(self):
        """清空内存缓存"""
        self._memory_cache.clear()

    def cache_stats(self) -> dict:
        """返回缓存统计信息"""
        mem_count = len(self._memory_cache)
        disk_count = len(self._cache_index) if self.cache_dir else 0
        return {
            "memory_cached": mem_count,
            "disk_cached": disk_count,
            "model_size": self.model_size,
            "embed_dim": self.embed_dim,
        }


def quick_test():
    """快速测试"""
    print("=" * 60)
    print("ESM-2 编码器测试")
    print("=" * 60)

    encoder = ESM2Encoder(model_size="650M")

    try:
        encoder.load()
        test_seqs = ["SLYNTVATLY", "GILGFVFTL", "KKKKKKKKK"]
        emb = encoder.encode(test_seqs)
        print(f"[OK] 输出维度: {emb.shape} (期望: (3, 1280))")
        print(f"[OK] 缓存统计: {encoder.cache_stats()}")

        # 单序列测试
        single = encoder.encode_single("SLYNTVATL")
        print(f"[OK] 单序列: {single.shape} (期望: (1280,))")

    except Exception as e:
        print(f"[FAIL] 加载失败: {e}")
        print("  (需要网络连接下载模型，或检查 torch/transformers 是否安装)")

    print("=" * 60)
    return encoder


if __name__ == "__main__":
    quick_test()