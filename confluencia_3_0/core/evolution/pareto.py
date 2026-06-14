"""Pareto 多目标优化 + REINFORCE RL 工具函数

提取自 confluencia-2.0-drug/core/evolution.py 的共享工具。
"""
from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """数值稳定 softmax。"""
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


def normalize_cols(X: np.ndarray) -> np.ndarray:
    """列方向 min-max 归一化到 [0, 1]。"""
    mn = X.min(axis=0, keepdims=True)
    mx = X.max(axis=0, keepdims=True)
    den = np.maximum(mx - mn, 1e-6)
    return (X - mn) / den


def pareto_front_mask(X: np.ndarray) -> np.ndarray:
    """识别非支配点 (最大化目标矩阵)。"""
    n = X.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dom = np.all(X >= X[i], axis=1) & np.any(X > X[i], axis=1)
        dom[i] = False
        if np.any(dom):
            keep[i] = False
    return keep


def select_weights_with_pareto(
    X_obj_norm: np.ndarray,
    top_k: int,
    n_samples: int,
    rng: np.random.Generator,
    prior: np.ndarray | None = None,
) -> np.ndarray:
    """Pareto 导向目标权重搜索。

    采样 Dirichlet 向量 + 可选手动先验，选择使 top-k 均值奖励最大的权重。
    """
    d = X_obj_norm.shape[1]
    if prior is not None:
        p = prior.astype(np.float32)
        if int(p.size) < int(d):
            p = np.pad(p, (0, int(d) - int(p.size)), constant_values=0.20)
        if int(p.size) > int(d):
            p = p[:int(d)]
    else:
        p = np.ones(d, dtype=np.float32) / d

    bank = [p]
    for _ in range(max(int(n_samples), 4) - 1):
        bank.append(rng.dirichlet(np.ones(d, dtype=np.float32)).astype(np.float32))

    best_w = bank[0]
    best_score = -1e9
    for w in bank:
        w = w / np.maximum(w.sum(), 1e-8)
        r = (X_obj_norm @ w).astype(np.float32)
        top = np.sort(r)[-max(int(top_k), 2):]
        score = float(top.mean())
        if score > best_score:
            best_score = score
            best_w = w
    return best_w.astype(np.float32)


def reward_from_weights(X_obj_norm: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """加权标量化奖励。"""
    w = np.asarray(weights, dtype=np.float32)
    w = w / np.maximum(w.sum(), 1e-8)
    return (X_obj_norm @ w).astype(np.float32)


def pick_actions(
    logits: np.ndarray,
    n: int,
    eps: float,
    n_actions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Epsilon-greedy 动作选择。"""
    probs = softmax(logits)
    acts = []
    for _ in range(n):
        if float(rng.random()) < float(eps):
            acts.append(int(rng.integers(0, n_actions)))
        else:
            acts.append(int(rng.choice(np.arange(n_actions), p=probs)))
    return np.array(acts, dtype=int)
