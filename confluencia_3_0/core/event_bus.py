"""事件总线 (Event Bus)

轻量级同步 pub/sub 事件系统，用于模块间解耦通信。

设计原则：
1. 同步调用（与单线程模型一致）
2. handler 返回值自动收集
3. 优先级排序保证执行顺序
4. 可选事件日志用于调试

移植自 Civis Lucri-Faber (CLF) core/event_bus.py
"""
import time
from typing import Callable, Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Event:
    """事件对象"""
    type: str
    data: Dict[str, Any]
    source: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Subscription:
    """订阅记录"""
    handler: Callable
    priority: int = 0  # 数值越小越先执行
    name: str = ""


class EventBus:
    """轻量级同步事件总线

    用法:
        bus = EventBus()
        bus.subscribe("STEP_START", self.on_step_start, priority=0)
        bus.publish("STEP_START", {"elapsed": 1.0}, source="agent")
    """

    def __init__(self, log_enabled: bool = True):
        self._subscribers: Dict[str, List[Subscription]] = defaultdict(list)
        self._log_enabled = log_enabled
        self._log: List[Event] = []
        self._publish_count: Dict[str, int] = defaultdict(int)
        self._error_counts: Dict[str, int] = defaultdict(int)

    def subscribe(
        self,
        event_type: str,
        handler: Callable,
        priority: int = 0,
        name: str = "",
    ) -> None:
        """订阅事件

        Args:
            event_type: 事件类型
            handler: 处理函数，签名为 handler(event: Event) -> Optional[dict]
            priority: 优先级（数值越小越先执行）
            name: handler 名称（用于调试和返回值收集）
        """
        sub = Subscription(handler=handler, priority=priority, name=name or handler.__name__)
        self._subscribers[event_type].append(sub)
        self._subscribers[event_type].sort(key=lambda s: s.priority)

    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """取消订阅"""
        self._subscribers[event_type] = [
            s for s in self._subscribers[event_type]
            if s.handler is not handler
        ]

    def publish(
        self,
        event_type: str,
        data: Dict[str, Any] = None,
        source: str = "",
    ) -> Dict[str, Any]:
        """发布事件（同步调用所有订阅者）

        Args:
            event_type: 事件类型
            data: 事件数据
            source: 事件来源

        Returns:
            collected: 收集的所有 handler 返回值，key 为 handler name
        """
        data = data or {}
        event = Event(type=event_type, data=data, source=source)
        collected: Dict[str, Any] = {}

        self._publish_count[event_type] += 1

        if self._log_enabled:
            self._log.append(event)
            if len(self._log) > 200:
                self._log = self._log[-200:]

        for sub in self._subscribers.get(event_type, []):
            try:
                result = sub.handler(event)
                if result is not None and isinstance(result, dict):
                    collected[sub.name] = result
            except Exception:
                key = f"{event_type}->{sub.name}"
                self._error_counts[key] = self._error_counts.get(key, 0) + 1

        return collected

    def get_stats(self) -> Dict[str, Any]:
        """获取事件统计"""
        return {
            "publish_counts": dict(self._publish_count),
            "subscriber_counts": {
                etype: len(subs)
                for etype, subs in self._subscribers.items()
            },
            "log_size": len(self._log),
            "error_counts": dict(self._error_counts),
        }

    def clear_log(self) -> None:
        """清空事件日志"""
        self._log.clear()

    def reset(self) -> None:
        """重置总线"""
        self._subscribers.clear()
        self._log.clear()
        self._publish_count.clear()


# 全局单例（可选使用）
global_bus = EventBus()