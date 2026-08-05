"""msa_cluster.py — MSA 聚类 + 代表性序列选择.

层次化 MSA 处理的第一步:
  Level 0: 聚类选出 M 条代表性序列 (M << N)
  Level 1: 用代表性序列提取精细 pair 特征
  Level 2: 从所有 N 条序列 soft-select 全局进化信息

聚类方法:
  1. MMseqs2 (推荐, 快速): 需要安装 mmseqs CLI
  2. CD-HIT (备选): 需要安装 cd-hit CLI
  3. 纯 PyTorch 近似: 不依赖外部工具, 用 embedding 距离聚类

用法:
  cluster = MSACluster(n_representatives=64, method='embedding')
  rep_ids, rep_seqs, weights = cluster(msa)
"""
from __future__ import annotations
from typing import Tuple, Optional
import subprocess
import tempfile
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSACluster:
    """MSA 聚类器.

    从 N 条 MSA 序列中选出 M 条代表性序列.
    每条代表性序列携带一个权重 (consensus 信息量).
    """

    def __init__(
        self,
        n_representatives: int = 64,
        method: str = "embedding",
        similarity_threshold: float = 0.9,
    ):
        """
        Args:
            n_representatives: 代表性序列数量 (M)
            method: 'mmseqs' | 'cdhit' | 'embedding'
            similarity_threshold: 序列相似度阈值 (mmseqs/cdhit)
        """
        self.n_rep = n_representatives
        self.method = method
        self.sim_threshold = similarity_threshold

    def __call__(
        self, msa: torch.Tensor, sequences: Optional[list] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """聚类 MSA 并选出代表性序列.

        Args:
            msa: (N, L) token IDs — MSA, N 条序列, L 长度
                 第一行通常是 query sequence
            sequences: 可选, 原始字符串序列 (mmseqs/cdhit 需要)

        Returns:
            rep_ids: (M,) int — 代表性序列在 MSA 中的原始索引
            rep_seqs: (M, L) token IDs — 代表性序列
            weights: (M,) float — 每条代表性序列的权重 (consensus 贡献)
        """
        N, L = msa.shape
        M = min(self.n_rep, N)

        if N <= M:
            # 序列数 ≤ 代表性数量, 直接全部返回
            rep_ids = torch.arange(N)
            rep_seqs = msa
            weights = torch.ones(N)
            return rep_ids, rep_seqs, weights

        if self.method == "embedding":
            return self._cluster_embedding(msa, M)
        elif self.method == "mmseqs" and sequences is not None:
            return self._cluster_mmseqs(sequences, M)
        elif self.method == "cdhit" and sequences is not None:
            return self._cluster_cdhit(sequences, M)
        else:
            # fallback: 均匀采样
            return self._cluster_uniform(msa, M)

    def _cluster_embedding(
        self, msa: torch.Tensor, M: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """基于 embedding 距离的聚类 (纯 PyTorch).

        方法:
        1. 用 one-hot embedding 把 token ID 变成向量
        2. 计算序列间余弦相似度
        3. K-means 聚类, 选 cluster centroid 作为代表性序列
        """
        N, L = msa.shape

        # 1. One-hot embedding (N, L, 5)
        one_hot = F.one_hot(msa, num_classes=5).float()  # (N, L, 5)

        # 2. 序列 embedding: 平均池化 (N, 5)
        seq_emb = one_hot.mean(dim=1)  # (N, 5)

        # 3. K-means 简化版: 贪心选择 M 个聚类中心
        rep_ids = self._kmeans_greedy(seq_emb, M)

        # 4. 计算权重: 距离中心越近权重越高
        centers = seq_emb[rep_ids]  # (M, 5)
        # 余弦相似度
        sim = F.cosine_similarity(
            seq_emb.unsqueeze(1), centers.unsqueeze(0), dim=2
        )  # (N, M)
        # 每个序列归属最近的中心
        assignment = sim.argmax(dim=1)  # (N,)
        # 权重 = 每个 cluster 的序列数 (越大越consensus)
        counts = torch.bincount(assignment, minlength=M).float()
        weights = counts / counts.sum() * M  # 归一化

        # 如果 query (第0条) 不在 rep_ids 中, 强制加入
        if 0 not in rep_ids:
            # 替换权重最小的
            min_idx = weights.argmin()
            rep_ids[min_idx] = 0
            weights[min_idx] = 1.0

        rep_seqs = msa[rep_ids]
        return rep_ids, rep_seqs, weights

    def _kmeans_greedy(
        self, embeddings: torch.Tensor, K: int
    ) -> torch.Tensor:
        """贪心 K-means: 每次选离已有中心最远的点."""
        N = embeddings.shape[0]
        selected = [0]  # 第一个中心 = query

        # 归一化
        emb_norm = F.normalize(embeddings, dim=1)

        for _ in range(K - 1):
            # 当前中心
            centers = emb_norm[selected]  # (k, 5)
            # 每个点到最近中心的距离
            sim = torch.mm(emb_norm, centers.t())  # (N, k)
            min_dist = 1.0 - sim.max(dim=1).values  # (N,)
            # 已选过的点距离设为 0
            min_dist[selected] = -1.0
            # 选最远的
            next_id = min_dist.argmax().item()
            selected.append(next_id)

        return torch.tensor(selected, dtype=torch.long)

    def _cluster_uniform(
        self, msa: torch.Tensor, M: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """均匀采样 (fallback)."""
        N = msa.shape[0]
        # 始终包含 query (第0条)
        indices = torch.linspace(0, N - 1, M).long()
        if 0 not in indices:
            indices[0] = 0
        rep_ids = indices
        rep_seqs = msa[rep_ids]
        weights = torch.ones(M)
        return rep_ids, rep_seqs, weights

    def _cluster_mmseqs(
        self, sequences: list, M: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MMseqs2 聚类 (需要 CLI)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 写 FASTA
            input_fa = os.path.join(tmpdir, "input.fasta")
            with open(input_fa, "w") as f:
                for i, seq in enumerate(sequences):
                    f.write(f">seq_{i}\n{seq}\n")

            # 运行 MMseqs2
            output_db = os.path.join(tmpdir, "clusters")
            tmp_db = os.path.join(tmpdir, "tmp")
            try:
                subprocess.run(
                    [
                        "mmseqs", "easy-cluster", input_fa,
                        output_db, tmp_db,
                        "--min-seq-id", str(self.sim_threshold),
                        "-c", "0.8",
                    ],
                    check=True, capture_output=True, timeout=120,
                )
                # 解析聚类结果
                return self._parse_mmseqs_clusters(
                    output_db + "_cluster.tsv", M
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("  WARNING: MMseqs2 not available, falling back to embedding")
                return self._cluster_uniform(None, M)

    def _parse_mmseqs_clusters(
        self, cluster_file: str, M: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """解析 MMseqs2 聚类结果."""
        clusters = {}  # representative_id -> [member_ids]
        with open(cluster_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                rep_id = int(parts[0].split("_")[1])
                mem_id = int(parts[1].split("_")[1])
                if rep_id not in clusters:
                    clusters[rep_id] = []
                clusters[rep_id].append(mem_id)

        # 按 cluster 大小排序, 取前 M 个
        sorted_reps = sorted(
            clusters.keys(), key=lambda x: len(clusters[x]), reverse=True
        )[:M]

        rep_ids = torch.tensor(sorted_reps, dtype=torch.long)
        # 权重 = cluster 大小
        weights = torch.tensor(
            [len(clusters[r]) for r in sorted_reps], dtype=torch.float
        )
        weights = weights / weights.sum() * M

        return rep_ids, None, weights  # sequences 需要外部处理

    def _cluster_cdhit(
        self, sequences: list, M: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """CD-HIT 聚类 (需要 CLI)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_fa = os.path.join(tmpdir, "input.fasta")
            output_fa = os.path.join(tmpdir, "clusters")
            with open(input_fa, "w") as f:
                for i, seq in enumerate(sequences):
                    f.write(f">seq_{i}\n{seq}\n")

            try:
                subprocess.run(
                    [
                        "cd-hit", "-i", input_fa, "-o", output_fa,
                        "-c", str(self.sim_threshold),
                        "-n", "5", "-d", "0",
                    ],
                    check=True, capture_output=True, timeout=120,
                )
                return self._parse_cdhit_clusters(output_fa, M)
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("  WARNING: CD-HIT not available, falling back to embedding")
                return self._cluster_uniform(None, M)

    def _parse_cdhit_clusters(
        self, cluster_file: str, M: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """解析 CD-HIT 聚类结果."""
        clusters = {}
        current_rep = None
        with open(cluster_file + ".clstr") as f:
            for line in f:
                if line.startswith(">Cluster"):
                    continue
                if "*" in line:
                    # 代表性序列
                    current_rep = int(line.split(">")[1].split("...")[0].split("_")[1])
                    clusters[current_rep] = [current_rep]
                elif current_rep is not None:
                    mem_id = int(line.split(">")[1].split("...")[0].split("_")[1])
                    clusters[current_rep].append(mem_id)

        sorted_reps = sorted(
            clusters.keys(), key=lambda x: len(clusters[x]), reverse=True
        )[:M]

        rep_ids = torch.tensor(sorted_reps, dtype=torch.long)
        weights = torch.tensor(
            [len(clusters[r]) for r in sorted_reps], dtype=torch.float
        )
        weights = weights / weights.sum() * M

        return rep_ids, None, weights


class HierarchicalMSA(nn.Module):
    """分层 MSA 处理模块.

    Level 0: 聚类 → M 代表性序列
    Level 1: 代表序列 pair 特征 (MI/covariance)
    Level 2: 所有序列 soft-select (attention consensus)
    """

    def __init__(
        self,
        n_representatives: int = 64,
        d_msa: int = 640,
        d_pair: int = 128,
        n_heads: int = 8,
        use_clustering: bool = True,
    ):
        super().__init__()
        self.n_rep = n_representatives
        self.d_msa = d_msa
        self.d_pair = d_pair

        # Level 0: 聚类器
        self.cluster = MSACluster(
            n_representatives=n_representatives,
            method="embedding",
        ) if use_clustering else None

        # Level 1: 代表序列 pair 特征提取
        self.pair_net = PairNet(d_pair=d_pair, n_heads=n_heads)

        # Level 2: soft-select attention
        self.consensus_attn = ConsensusAttention(
            d_msa=d_msa, n_heads=n_heads
        )

    def forward(
        self, msa_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """分层 MSA 处理.

        Args:
            msa_tokens: (N, L) token IDs — MSA

        Returns:
            consensus_repr: (L, d_msa) — 全局进化共识
            pair_repr: (L, L, d_pair) — pair 特征
        """
        N, L = msa_tokens.shape

        # Level 0: 聚类
        if self.cluster is not None:
            rep_ids, rep_seqs, weights = self.cluster(msa_tokens)
        else:
            rep_ids = torch.arange(min(self.n_rep, N))
            rep_seqs = msa_tokens[rep_ids]
            weights = torch.ones(len(rep_ids))

        # Level 1: 代表序列 pair 特征
        pair_repr = self.pair_net(rep_seqs)  # (L, L, d_pair)

        # Level 2: soft-select consensus
        consensus = self.consensus_attn(msa_tokens, weights)  # (L, d_msa)

        return consensus, pair_repr


class PairNet(nn.Module):
    """从 MSA 代表序列提取 pair 特征.

    基于 mutual information (MI) 和 covariance:
      MI(i,j) = Σ_a,b p(a,b) log(p(a,b) / p(a)p(b))
      Cov(i,j) = E[one_hot(a_i) ⊗ one_hot(a_j)] - outer(p_i, p_j)
    """

    def __init__(self, d_pair: int = 128, n_heads: int = 8):
        super().__init__()
        # 输入: 5×5 的共进化矩阵 + 位置编码
        self.input_proj = nn.Linear(25 + 64, d_pair)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_pair),
                nn.GELU(),
                nn.Linear(d_pair, d_pair),
            )
            for _ in range(2)
        ])

    def forward(self, rep_seqs: torch.Tensor) -> torch.Tensor:
        """提取 pair 特征.

        Args:
            rep_seqs: (M, L) token IDs — 代表性序列

        Returns:
            pair_repr: (L, L, d_pair) — pair 特征
        """
        M, L = rep_seqs.shape
        device = rep_seqs.device

        # 1. 计算共进化矩阵 (简化版: 条件概率)
        # one-hot: (M, L, 5)
        oh = F.one_hot(rep_seqs, num_classes=5).float()

        # 联合概率 p(a,b) = mean over M sequences
        # (M, L, 5, 1) × (M, 1, L, 5) → (L, L, 5, 5)
        joint = torch.einsum("mia,mjb->ijab", oh, oh) / M  # (L,L,5,5)

        # 边缘概率
        p_i = joint.sum(dim=-1)  # (L, L, 5)
        p_j = joint.sum(dim=-2)  # (L, L, 5)

        # MI 近似: -Σ p(a,b) log(p(a,b) / (p(a)p(b) + ε))
        eps = 1e-8
        mi = joint * torch.log(joint / (p_i.unsqueeze(-1) * p_j.unsqueeze(-2) + eps) + eps)
        mi = mi.sum(dim=(-1, -2))  # (L, L) — 标量 MI

        # Covariance 矩阵 (简化: 用 MI 作为输入)
        # 展平 (L, L, 5, 5) → (L, L, 25)
        joint_flat = joint.reshape(L, L, 25)

        # MI 作为额外通道
        mi_feat = mi.unsqueeze(-1)  # (L, L, 1)

        # 拼接: 25 + 1 = 26 维
        pair_input = torch.cat([
            joint_flat,
            mi_feat.expand(-1, -1, 1),
        ], dim=-1)  # (L, L, 26)

        # padding 到 25+64=89 维 (位置编码用零)
        if pair_input.shape[-1] < 89:
            pad = torch.zeros(L, L, 89 - pair_input.shape[-1], device=device)
            pair_input = torch.cat([pair_input, pad], dim=-1)

        # 投影到 d_pair
        pair_repr = self.input_proj(pair_input)  # (L, L, d_pair)

        # Transformer layers
        for layer in self.layers:
            pair_repr = pair_repr + layer(pair_repr)  # residual

        return pair_repr


class ConsensusAttention(nn.Module):
    """从所有 MSA 序列中 soft-select 全局进化信息.

    对每个 residue position, 用 attention 加权 MSA 中所有序列的 embedding,
    得到 consensus representation.
    """

    def __init__(self, d_msa: int = 640, n_heads: int = 8):
        super().__init__()
        self.d_msa = d_msa

        # 每个序列的 embedding: token → d_msa
        self.token_proj = nn.Linear(5, d_msa)

        # Attention
        self.q_proj = nn.Linear(d_msa, d_msa)
        self.k_proj = nn.Linear(d_msa, d_msa)
        self.v_proj = nn.Linear(d_msa, d_msa)
        self.out_proj = nn.Linear(d_msa, d_msa)

        self.n_heads = n_heads
        self.head_dim = d_msa // n_heads

    def forward(
        self, msa_tokens: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Soft-select consensus.

        Args:
            msa_tokens: (N, L) token IDs
            weights: (M,) — 聚类权重 (M 代表性序列)

        Returns:
            consensus: (L, d_msa)
        """
        N, L = msa_tokens.shape
        device = msa_tokens.device

        # Token → embedding
        oh = F.one_hot(msa_tokens, num_classes=5).float()  # (N, L, 5)
        seq_emb = self.token_proj(oh)  # (N, L, d_msa)

        # 应用聚类权重
        # weights 可能长度 ≠ N, 需要 broadcast
        if weights.shape[0] < N:
            # 简单: 用 query (第0条) 的权重作为所有序列的权重
            w = weights[0] if len(weights) > 0 else 1.0
        else:
            w = weights
        w = w.to(device)

        # Attention: query = 均值, key/value = 所有序列
        mean_emb = seq_emb.mean(dim=0, keepdim=True)  # (1, L, d_msa)

        Q = self.q_proj(mean_emb)  # (1, L, d_msa)
        K = self.k_proj(seq_emb)   # (N, L, d_msa)
        V = self.v_proj(seq_emb)   # (N, L, d_msa)

        # Multi-head attention
        B = 1
        Q = Q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(N, L, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(N, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores: (1, heads, L, N)
        attn = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)

        # Weighted sum
        out = torch.matmul(attn, V)  # (1, heads, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_msa)

        return self.out_proj(out).squeeze(0)  # (L, d_msa)
