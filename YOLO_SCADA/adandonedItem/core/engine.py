"""
核心检测引擎
============
协调检测器、追踪器、告警管理器，形成完整的异留置检测管线。

设计模式：
  中介者模式 (Mediator Pattern)
  - 引擎作为中介者，协调 detector / tracker / alert 三者的交互
  - 外部只需要调用 process_frame()，内部管线自动完成

管线流程（每帧）:
  1. YOLO 检测 → 得到人和可疑物
  2. 更新人检测状态（是否在画面中、连续缺席帧数）
  3. IOU 匹配追踪器 → 更新位置和状态
  4. 判断无人看管 → 计时 → 超时告警
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import time
import numpy as np

from config import AppConfig
from models.detector import YOLODetector, DetectionResult
from models.tracker import ObjectTracker, TrackInfo, TrackState
from core.alert import AlertManager, AlertInfo, AlertLevel


@dataclass
class FrameResult:
    """一帧处理完成后的结果"""
    detection: DetectionResult      # 原始检测结果
    person_present: bool            # 人是否在画面中
    fps: float                      # 当前 FPS
    frame_count: int                # 帧序号


class AbandonedDetectorEngine:
    """
    异物留置检测引擎（核心协调器）。

    用法:
        engine = AbandonedDetectorEngine(config)
        while True:
            result = engine.process_frame(frame)
            # result 包含所有需要的检测信息
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.detector = YOLODetector(config)
        self.tracker = ObjectTracker(config)
        self.alert_manager = AlertManager(cooldown=5.0)

        # ── 人检测状态 ──
        self._person_boxes: List[Tuple] = []  # 当前帧的人框
        self._person_miss_count: int = 0      # 人连续缺席帧数
        self._person_present: bool = False    # 人是否在画面中

        # ── 性能统计 ──
        self._fps: float = 0.0
        self._frame_times: List[float] = []   # 最近帧的时间戳
        self._frame_count: int = 0

    # ──────────────── 属性 ────────────────

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    # ──────────────── 核心方法 ────────────────

    def __init__(self, config: AppConfig):
        self.config = config
        self.detector = YOLODetector(config)
        self.tracker = ObjectTracker(config)
        self.alert_manager = AlertManager(cooldown=5.0)

        # ── 人检测状态 ──
        self._person_boxes: list[tuple] = []
        self._person_miss_count: int = 0
        self._person_present: bool = False

        # ── 性能统计 ──
        self._fps: float = 0.0
        self._frame_times: list[float] = []
        self._frame_count: int = 0

        # ── 日志 ──
        self._logger = logging.getLogger(__name__)
        self._logger.info("引擎初始化完成，模型: %s", config.model_path)

    # ──────────────── 属性 ────────────────

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    # ──────────────── 核心方法 ────────────────

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """
        处理单帧（完整的检测 → 追踪 → 告警管线）。

        参数:
            frame: BGR 图片 (numpy array)

        返回:
            FrameResult 包含当前帧的所有处理结果
        """
        self._frame_count += 1
        current_time = time.time()

        # ── 计算 FPS ──
        self._frame_times.append(current_time)
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)
        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            self._fps = (len(self._frame_times) - 1) / elapsed if elapsed > 0 else 0

        # ════════════════════════════════════════════
        # Step 1: YOLO 检测
        # ════════════════════════════════════════════
        detection = self.detector.detect(frame)

        # ════════════════════════════════════════════
        # Step 2: 人检测状态更新
        # ════════════════════════════════════════════
        if detection.has_person:
            if not self._person_present:
                self._logger.info("人进入画面，开始监控。")
            self._person_boxes = [p.bbox for p in detection.persons]
            self._person_miss_count = 0
            self._person_present = True
        else:
            self._person_miss_count += 1
            if self._person_miss_count >= self.config.person_clear_after:
                if self._person_present:
                    self._logger.info("人离开画面，进入无人监控模式。")
                self._person_boxes = []
                self._person_present = False

        # ════════════════════════════════════════════
        # Step 3: 追踪器更新 (IOU 匹配 + 状态机)
        # ════════════════════════════════════════════
        self.tracker.update(detection.objects, current_time)

        # ════════════════════════════════════════════
        # Step 4: 告警逻辑
        # ════════════════════════════════════════════
        for tid, track in self.tracker.tracks.items():
            # 只处理"已静止"或"已告警"的物体
            if track.state not in (TrackState.STATIONARY, TrackState.ABANDONED):
                continue

            # 检查物体附近是否有人
            nearby_person = ObjectTracker.get_nearby_person(
                track, self._person_boxes, self.config.dist_threshold
            )

            if nearby_person is None:
                # ── 无人看管 → 计时 ──
                if track.abandon_timer == 0:
                    track.abandon_timer = current_time  # 首次进入无人状态
                elapsed = current_time - track.abandon_timer

                if track.state == TrackState.STATIONARY and elapsed >= self.config.abandon_seconds:
                    # 超时！从"静止"变为"告警"
                    track.state = TrackState.ABANDONED
                    self.alert_manager.trigger(
                        track_id=tid,
                        class_name=track.class_name,
                        level=AlertLevel.WARNING,
                    )
            else:
                # ── 有人看管 → 重置计时 ──
                track.abandon_timer = 0
                if track.state == TrackState.ABANDONED:
                    # 如果之前已告警，现在有人回来了 → 恢复静止状态
                    track.state = TrackState.STATIONARY
                    self.alert_manager.clear_alert(tid)

        return FrameResult(
            detection=detection,
            person_present=self._person_present,
            fps=self._fps,
            frame_count=self._frame_count,
        )