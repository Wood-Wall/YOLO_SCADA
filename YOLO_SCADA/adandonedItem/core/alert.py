"""
告警管理模块
============
管理所有告警的触发、去重、通知。

设计模式：
  观察者模式 (Observer Pattern)
  - AlertManager 是被观察者（Subject）
  - 外部通过 attach() 注册回调函数，告警触发时自动通知
  - 支持冷却机制 — 同一物体不会频繁告警
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional
from enum import Enum, auto
import time


class AlertLevel(Enum):
    """告警级别"""
    INFO = auto()         # 信息
    WARNING = auto()      # 警告
    CRITICAL = auto()     # 严重


@dataclass
class AlertInfo:
    """告警信息数据结构"""
    track_id: int               # 触发告警的追踪器 ID
    class_name: str             # 物体类别名称
    level: AlertLevel           # 告警级别
    message: str                # 告警文字
    timestamp: float = field(default_factory=time.time)  # 触发时间
    duration: float = 0.0       # 已持续秒数
    acknowledged: bool = False  # 是否已确认


class AlertManager:
    """
    告警管理器（观察者模式）。

    用法:
        manager = AlertManager(cooldown=5.0)

        # 注册告警回调
        def on_alert(alert):
            print(f"告警: {alert.message}")
        manager.attach(on_alert)

        # 触发告警
        manager.trigger(track_id=1, class_name="背包")
    """

    def __init__(self, cooldown: float = 5.0):
        # 观察者列表
        self._observers: List[Callable[[AlertInfo], None]] = []

        # 活跃告警 {track_id: AlertInfo}
        self._active_alerts: dict[int, AlertInfo] = {}

        # 历史告警（用于统计）
        self._history: List[AlertInfo] = []

        # 同一物体最短告警间隔（秒）
        self._cooldown = cooldown

    # ──────────────── 属性 ────────────────

    @property
    def active_alerts(self) -> list[AlertInfo]:
        """当前活跃告警列表"""
        now = time.time()
        return [
            a for a in self._active_alerts.values()
            if now - a.timestamp < self._cooldown
        ]

    @property
    def alert_count(self) -> int:
        """告警数量（用于面板显示）"""
        return len(self._active_alerts)

    @property
    def history_count(self) -> int:
        """累计告警总数"""
        return len(self._history)

    # ──────────────── 观察者管理 ────────────────

    def attach(self, observer: Callable[[AlertInfo], None]):
        """注册观察者（告警时自动调用）"""
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer: Callable[[AlertInfo], None]):
        """移除观察者"""
        if observer in self._observers:
            self._observers.remove(observer)

    def _notify(self, alert: AlertInfo):
        """通知所有观察者"""
        for observer in self._observers:
            observer(alert)

    # ──────────────── 核心方法 ────────────────

    def trigger(
        self,
        track_id: int,
        class_name: str,
        level: AlertLevel = AlertLevel.WARNING,
        message: str = ""
    ) -> Optional[AlertInfo]:
        """
        触发告警。

        如果同一 track_id 在冷却期内，不会重复触发。

        参数:
            track_id:   追踪器 ID
            class_name: 类别名称
            level:      告警级别
            message:    自定义消息（不传则自动生成）

        返回:
            新创建的 AlertInfo，如果在冷却期内则返回 None
        """
        now = time.time()

        # 检查冷却期
        existing = self._active_alerts.get(track_id)
        if existing and (now - existing.timestamp) < self._cooldown:
            existing.duration = now - existing.timestamp
            return None

        # 自动生成消息
        if not message:
            message = f"{class_name} 已无人看管！"

        alert = AlertInfo(
            track_id=track_id,
            class_name=class_name,
            level=level,
            message=message,
            timestamp=now,
        )

        self._active_alerts[track_id] = alert
        self._history.append(alert)

        # 通知观察者
        self._notify(alert)

        return alert

    def clear_alert(self, track_id: int):
        """清除某个追踪器的告警（物体被人拿走了）"""
        if track_id in self._active_alerts:
            alert = self._active_alerts.pop(track_id)
            alert.acknowledged = True

    def clear_all(self):
        """清除所有告警"""
        for alert in self._active_alerts.values():
            alert.acknowledged = True
        self._active_alerts.clear()