"""
检查点管理器 - 训练过程中的即时保存与恢复机制

功能：
1. 定期保存训练检查点（模型权重、优化器状态、训练进度）
2. 意外中断后自动恢复
3. 支持多版本检查点管理
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from confluencia_shared.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CheckpointMeta:
    """检查点元数据"""
    checkpoint_id: str
    timestamp: str
    epoch: int
    total_epochs: int
    train_loss: float
    val_loss: float
    best_val_loss: float
    bad_rounds: int
    is_best: bool
    file_size_mb: float


@dataclass
class CheckpointConfig:
    """检查点配置"""
    enabled: bool = True
    save_dir: str = "./checkpoints"
    save_every_n_epochs: int = 5  # 每 N 个 epoch 保存一次
    save_best_only: bool = False  # 是否只保存最佳模型
    keep_last_n: int = 3  # 保留最近 N 个检查点
    auto_resume: bool = True  # 是否自动从最新检查点恢复
    checkpoint_prefix: str = "ckpt"


class CheckpointManager:
    """
    检查点管理器

    使用示例：
    ```python
    manager = CheckpointManager(config, model_id="epitope_exp_001")

    # 训练前检查是否需要恢复
    latest = manager.get_latest_checkpoint()
    if latest:
        start_epoch, state = manager.load_checkpoint(latest)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])

    # 训练中保存
    for epoch in range(start_epoch, total_epochs):
        # ... 训练代码 ...
        manager.maybe_save(epoch, model, optimizer, metrics)

    # 训练完成后清理
    manager.cleanup()
    ```
    """

    def __init__(
        self,
        config: CheckpointConfig,
        model_id: str = "default",
        on_save_callback: Optional[Callable[[str, CheckpointMeta], None]] = None,
    ):
        self.config = config
        self.model_id = model_id
        self.on_save_callback = on_save_callback

        # 创建保存目录
        self.save_dir = Path(config.save_dir) / model_id
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 元数据文件
        self.meta_file = self.save_dir / "checkpoints_meta.json"
        self.checkpoints: List[CheckpointMeta] = self._load_meta()

    def _load_meta(self) -> List[CheckpointMeta]:
        """加载检查点元数据"""
        if not self.meta_file.exists():
            return []
        try:
            with open(self.meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [CheckpointMeta(**item) for item in data]
        except Exception as exc:
            logger.debug(f"Failed to load checkpoint metadata: {exc}")
            return []

    def _save_meta(self):
        """保存检查点元数据"""
        data = [asdict(ckpt) for ckpt in self.checkpoints]
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _get_checkpoint_path(self, epoch: int, is_best: bool = False) -> Path:
        """获取检查点文件路径"""
        suffix = "_best" if is_best else ""
        filename = f"{self.config.checkpoint_prefix}_epoch{epoch:04d}{suffix}.npz"
        return self.save_dir / filename

    def get_latest_checkpoint(self) -> Optional[CheckpointMeta]:
        """获取最新的检查点"""
        if not self.checkpoints:
            return None
        # 按时间戳排序，返回最新的
        sorted_ckpts = sorted(self.checkpoints, key=lambda x: x.timestamp, reverse=True)
        return sorted_ckpts[0]

    def get_best_checkpoint(self) -> Optional[CheckpointMeta]:
        """获取最佳验证损失的检查点"""
        best_ckpts = [c for c in self.checkpoints if c.is_best]
        if not best_ckpts:
            # 如果没有标记为 best，返回 val_loss 最小的
            if not self.checkpoints:
                return None
            return min(self.checkpoints, key=lambda x: x.val_loss)
        # 返回最新的 best 检查点
        return max(best_ckpts, key=lambda x: x.timestamp)

    def save_checkpoint(
        self,
        epoch: int,
        total_epochs: int,
        model_state: Dict[str, Any],
        optimizer_state: Dict[str, Any],
        train_loss: float,
        val_loss: float,
        best_val_loss: float,
        bad_rounds: int,
        extra_state: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
    ) -> CheckpointMeta:
        """
        保存检查点

        Args:
            epoch: 当前 epoch
            total_epochs: 总 epoch 数
            model_state: 模型状态字典
            optimizer_state: 优化器状态字典
            train_loss: 训练损失
            val_loss: 验证损失
            best_val_loss: 最佳验证损失
            bad_rounds: 早停计数器
            extra_state: 额外状态（如 scheduler、rng state）
            is_best: 是否为最佳模型

        Returns:
            CheckpointMeta: 检查点元数据
        """
        if not self.config.enabled:
            raise RuntimeError("Checkpoint is disabled")

        checkpoint_path = self._get_checkpoint_path(epoch, is_best)

        # 构建检查点数据
        checkpoint_data = {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "bad_rounds": bad_rounds,
            "is_best": is_best,
            "timestamp": datetime.now().isoformat(),
            "model_state": model_state,
            "optimizer_state": optimizer_state,
        }

        if extra_state:
            checkpoint_data["extra_state"] = extra_state

        # 保存为 npz 格式（适合 PyTorch state dict）
        # 将 state dict 转换为可序列化格式
        save_dict = {}
        for key, value in checkpoint_data.items():
            if isinstance(value, dict):
                # 对于 model_state 和 optimizer_state
                for sub_key, sub_value in value.items():
                    if hasattr(sub_value, 'numpy'):
                        save_dict[f"{key}.{sub_key}"] = sub_value.cpu().numpy()
                    elif isinstance(sub_value, np.ndarray):
                        save_dict[f"{key}.{sub_key}"] = sub_value
                    elif isinstance(sub_value, (int, float, str, bool)):
                        save_dict[f"{key}.{sub_key}"] = np.array(sub_value)
                    else:
                        # 尝试序列化
                        try:
                            save_dict[f"{key}.{sub_key}"] = np.array(sub_value, dtype=object)
                        except Exception as exc:
                            logger.debug(f"Skipping unserializable checkpoint field {key}.{sub_key}: {exc}")
            elif isinstance(value, (int, float, str, bool)):
                save_dict[key] = np.array(value)
            elif isinstance(value, np.ndarray):
                save_dict[key] = value
            else:
                save_dict[key] = np.array(str(value), dtype=object)

        np.savez_compressed(checkpoint_path, **save_dict)

        # 计算文件大小
        file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

        # 创建元数据
        meta = CheckpointMeta(
            checkpoint_id=checkpoint_path.stem,
            timestamp=datetime.now().isoformat(),
            epoch=epoch,
            total_epochs=total_epochs,
            train_loss=train_loss,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
            bad_rounds=bad_rounds,
            is_best=is_best,
            file_size_mb=round(file_size_mb, 2),
        )

        # 更新检查点列表
        self.checkpoints.append(meta)

        # 清理旧检查点
        self._cleanup_old_checkpoints()

        # 保存元数据
        self._save_meta()

        # 回调通知
        if self.on_save_callback:
            self.on_save_callback(str(checkpoint_path), meta)

        return meta

    def load_checkpoint(
        self,
        checkpoint_meta: CheckpointMeta,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        加载检查点

        Args:
            checkpoint_meta: 检查点元数据

        Returns:
            (start_epoch, state_dict): 恢复的起始 epoch 和状态字典
        """
        checkpoint_path = self.save_dir / f"{checkpoint_meta.checkpoint_id}.npz"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        data = np.load(checkpoint_path, allow_pickle=True)

        # 重建状态字典
        state = {
            "epoch": int(data["epoch"]),
            "total_epochs": int(data["total_epochs"]),
            "train_loss": float(data["train_loss"]),
            "val_loss": float(data["val_loss"]),
            "best_val_loss": float(data["best_val_loss"]),
            "bad_rounds": int(data["bad_rounds"]),
            "model_state": {},
            "optimizer_state": {},
        }

        # 重建 model_state 和 optimizer_state
        for key in data.files:
            if key.startswith("model_state."):
                param_name = key[len("model_state."):]
                state["model_state"][param_name] = data[key]
            elif key.startswith("optimizer_state."):
                param_name = key[len("optimizer_state."):]
                state["optimizer_state"][param_name] = data[key]

        # 返回下一个 epoch 作为起始点
        start_epoch = state["epoch"] + 1
        return start_epoch, state

    def maybe_save(
        self,
        epoch: int,
        model_state: Dict[str, Any],
        optimizer_state: Dict[str, Any],
        train_loss: float,
        val_loss: float,
        best_val_loss: float,
        bad_rounds: int,
        total_epochs: int,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[CheckpointMeta]:
        """
        根据配置决定是否保存检查点

        Returns:
            如果保存了检查点，返回 CheckpointMeta；否则返回 None
        """
        if not self.config.enabled:
            return None

        is_best = val_loss < best_val_loss

        # 判断是否需要保存
        should_save = False
        if is_best:
            should_save = True
        elif not self.config.save_best_only:
            if epoch % self.config.save_every_n_epochs == 0:
                should_save = True
            elif epoch == total_epochs - 1:  # 最后一个 epoch
                should_save = True

        if should_save:
            return self.save_checkpoint(
                epoch=epoch,
                total_epochs=total_epochs,
                model_state=model_state,
                optimizer_state=optimizer_state,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                bad_rounds=bad_rounds,
                extra_state=extra_state,
                is_best=is_best,
            )
        return None

    def _cleanup_old_checkpoints(self):
        """清理旧检查点，只保留最近的 N 个"""
        if self.config.keep_last_n <= 0:
            return

        # 分离 best 和普通检查点
        best_ckpts = [c for c in self.checkpoints if c.is_best]
        normal_ckpts = [c for c in self.checkpoints if not c.is_best]

        # 按时间戳排序普通检查点
        normal_ckpts.sort(key=lambda x: x.timestamp, reverse=True)

        # 删除超出数量的普通检查点
        to_remove = normal_ckpts[self.config.keep_last_n:]

        for ckpt in to_remove:
            ckpt_path = self.save_dir / f"{ckpt.checkpoint_id}.npz"
            try:
                if ckpt_path.exists():
                    ckpt_path.unlink()
            except Exception as exc:
                logger.debug(f"Failed to delete checkpoint {ckpt_path}: {exc}")
        self.checkpoints = best_ckpts + normal_ckpts[:self.config.keep_last_n]

    def cleanup(self, keep_best: bool = True):
        """
        清理检查点目录

        Args:
            keep_best: 是否保留最佳检查点
        """
        if keep_best:
            best = self.get_best_checkpoint()
            for ckpt in self.checkpoints:
                if best and ckpt.checkpoint_id == best.checkpoint_id:
                    continue
                ckpt_path = self.save_dir / f"{ckpt.checkpoint_id}.npz"
                try:
                    if ckpt_path.exists():
                        ckpt_path.unlink()
                except Exception as exc:
                    logger.debug(f"Failed to delete non-best checkpoint {ckpt_path}: {exc}")
            self.checkpoints = [best] if best else []
        else:
            shutil.rmtree(self.save_dir, ignore_errors=True)
            self.checkpoints = []

        self._save_meta()

    def list_checkpoints(self) -> List[CheckpointMeta]:
        """列出所有检查点"""
        return list(self.checkpoints)

    def get_disk_usage(self) -> float:
        """获取检查点目录的磁盘使用量（MB）"""
        total_size = 0
        for ckpt in self.checkpoints:
            total_size += ckpt.file_size_mb
        return round(total_size, 2)


class TrainingStateSaver:
    """
    训练状态保存器 - 用于 sklearn MOE 等非 PyTorch 模型

    支持：
    - 保存训练中间状态
    - 保存已完成的专家模型
    - 支持增量训练恢复
    """

    def __init__(
        self,
        save_dir: str = "./checkpoints",
        model_id: str = "default",
        save_every_n_seconds: float = 30.0,  # 每 N 秒保存一次
    ):
        self.save_dir = Path(save_dir) / model_id
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_seconds = save_every_n_seconds
        self.last_save_time = 0.0
        self.model_id = model_id

    def should_save(self) -> bool:
        """判断是否应该保存"""
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_every_n_seconds:
            return True
        return False

    def save_partial_state(
        self,
        trained_experts: List[Any],
        expert_names: List[str],
        expert_metrics: Dict[str, float],
        remaining_experts: List[str],
        current_step: int,
        total_steps: int,
    ) -> Path:
        """
        保存部分训练状态

        Args:
            trained_experts: 已训练完成的专家模型列表
            expert_names: 专家名称列表
            expert_metrics: 专家指标
            remaining_experts: 剩余待训练的专家
            current_step: 当前步骤
            total_steps: 总步骤数

        Returns:
            保存的文件路径
        """
        import pickle

        state = {
            "trained_experts": trained_experts,
            "expert_names": expert_names,
            "expert_metrics": expert_metrics,
            "remaining_experts": remaining_experts,
            "current_step": current_step,
            "total_steps": total_steps,
            "timestamp": datetime.now().isoformat(),
        }

        filename = f"partial_state_{current_step:04d}.pkl"
        filepath = self.save_dir / filename

        with open(filepath, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.last_save_time = time.time()
        return filepath

    def load_partial_state(self) -> Optional[Dict[str, Any]]:
        """加载最近的部分状态"""
        import pickle

        # 查找最新的部分状态文件
        state_files = list(self.save_dir.glob("partial_state_*.pkl"))
        if not state_files:
            return None

        # 按修改时间排序
        state_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_file = state_files[0]

        try:
            with open(latest_file, "rb") as f:
                state = pickle.load(f)
            return state
        except Exception as exc:
            logger.debug(f"Failed to load partial state from {latest_file}: {exc}")
            return None

    def clear_partial_states(self):
        """清理部分状态文件"""
        for f in self.save_dir.glob("partial_state_*.pkl"):
            try:
                f.unlink()
            except Exception as exc:
                logger.debug(f"Failed to delete partial state {f}: {exc}")
