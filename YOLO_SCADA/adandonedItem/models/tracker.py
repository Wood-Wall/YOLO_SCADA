"""
物体追踪器模块
==============
核心追踪逻辑：IOU 匹配 + 状态机。

设计模式：
  状态模式 (State Pattern) —— TrackState 枚举定义物体状态，
  状态迁移逻辑集中在 ObjectTracker.update() 中。

状态机:
  NEW → MOVING ↔ STATIONARY → ABANDONED
                    ↓              ↓
                 有人回来重置   持续无人告警
"""
from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Tuple, Optional
import math


class TrackState(Enum):
    """追踪状态枚举"""
    NEW = auto()           # 刚出现，等待确认
    MOVING = auto()        # 移动中（还没放下）
    STATIONARY = auto()    # 静止（已放下，开始计时）
    ABANDONED = auto()     # 已告警（留置超时）


@dataclass
class TrackInfo:
    """
    单个追踪器的完整状态。

    字段说明:
        track_id:      唯一 ID（自增）
        positions:     最近 N 帧的位置历史 (deque)，用于判断移动/静止
        stationary_counter: 连续静止帧数
        abandon_timer: 无人看管计时（秒）
        miss_count:    连续未被检测到的帧数（容错）
    """
    track_id: int
    class_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]

    # ── 状态 ──
    state: TrackState = TrackState.NEW

    # ── 位置追踪 ──
    positions: deque = field(
        default_factory=lambda: deque(maxlen=30)
    )
    stationary_counter: int = 0

    # ── 计时 ──
    abandon_timer: float = 0.0       # 无人看管累计秒数
    first_seen: float = 0.0          # 首次出现时间
    last_seen: float = 0.0           # 最后出现时间

    # ── 容错 ──
    miss_count: int = 0

    def update_position(self, bbox: Tuple[int, int, int, int]):
        """
        更新位置并判断是否静止。

        核心逻辑:
          计算当前帧与上一帧的中心点距离，
          如果 < 5 像素 → stationary_counter++（算静止）
          否则 → stationary_counter = 0（重置）
        """
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self.positions.append((cx, cy))
        self.bbox = bbox

        if len(self.positions) >= 2:
            prev = self.positions[-2]
            dist = math.sqrt((cx - prev[0]) ** 2 + (cy - prev[1]) ** 2)

            if dist < 5:       # 移动 < 5 像素 → 静止
                self.stationary_counter += 1
            else:
                self.stationary_counter = 0
                # 如果之前已是静止/告警态，被移动了 → 回到 MOVING
                if self.state in (TrackState.STATIONARY, TrackState.ABANDONED):
                    self.state = TrackState.MOVING


class ObjectTracker:
    """
    追踪器管理器。

    职责:
      1. 用 IOU 将当前检测框匹配到已有追踪器
      2. 管理追踪器的状态迁移
      3. 清理丢失的追踪器

    用法:
        tracker = ObjectTracker(config)
        matched, new_boxes = tracker.update(detections, current_time)
    """

    def __init__(self, config):
        self.config = config
        self.tracks: Dict[int, TrackInfo] = {}   # track_id → TrackInfo
        self._next_id: int = 0

    @property
    def active_count(self) -> int:
        """当前活跃追踪器数量"""
        return len(self.tracks)

    @property
    def stats(self) -> dict:
        """各状态数量统计（用于面板显示）"""
        moving = sum(1 for t in self.tracks.values() if t.state == TrackState.MOVING)
        stationary = sum(1 for t in self.tracks.values() if t.state == TrackState.STATIONARY)
        abandoned = sum(1 for t in self.tracks.values() if t.state == TrackState.ABANDONED)
        return {"moving": moving, "stationary": stationary, "abandoned": abandoned}

    # ──────────────── 私有方法 ────────────────

    @staticmethod
    def _compute_iou(box1: Tuple, box2: Tuple) -> float:
        """
        计算两个框的交并比 (IOU)。

        IOU = 交集面积 / 并集面积
        值域 [0, 1]，越高表示两个框重叠越多。
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0               # 不重叠

        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter / (area1 + area2 - inter)

    # ──────────────── 公开方法 ────────────────

    def update(
        self,
        detections: List,
        current_time: float
    ) -> Tuple[Dict[int, TrackInfo], List]:
        """
        用当前帧的检测结果更新所有追踪器。

        参数:
            detections:  本帧检测到的可疑物列表
            current_time: 当前时间戳 (time.time())

        返回:
            (matched_tracks, new_detections)
            matched_tracks: 已匹配到追踪器的 {track_id: TrackInfo}
            new_detections: 未匹配的新检测框列表（需要创建新追踪器）
        """
        matched: Dict[int, TrackInfo] = {}
        new_detections: List = []
        used_ids: set = set()

        # ── 第 1 步：IOU 匹配 ──
        # 对每个新检测框，找 IOU 最高的已有追踪器
        for det in detections:
            best_id = None
            best_iou = self.config.iou_threshold

            for tid, track in self.tracks.items():
                if tid in used_ids:
                    continue
                iou = self._compute_iou(det.bbox, track.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid

            if best_id is not None:
                # 匹配成功 → 更新追踪器
                used_ids.add(best_id)
                track = self.tracks[best_id]
                track.update_position(det.bbox)
                track.miss_count = 0
                track.last_seen = current_time
                matched[best_id] = track
            else:
                # 匹配失败 → 可能是新物体
                new_detections.append(det)

        # ── 第 2 步：处理未匹配的追踪器 ──
        # 没被匹配到 → miss_count +1
        # miss_count 超限 → 删除
        for tid in list(self.tracks.keys()):
            if tid not in matched:
                self.tracks[tid].miss_count += 1
                if self.tracks[tid].miss_count >= self.config.max_miss_count:
                    del self.tracks[tid]

        # ── 第 3 步：新检测框创建新追踪器 ──
        for det in new_detections:
            track = TrackInfo(
                track_id=self._next_id,
                class_id=det.class_id,
                class_name=det.class_name,
                bbox=det.bbox,
                first_seen=current_time,
                last_seen=current_time,
            )
            track.update_position(det.bbox)
            self.tracks[self._next_id] = track
            self._next_id += 1

        # ── 第 4 步：状态机迁移 ──
        # NEW → MOVING（只要出现了就算移动中）
        # MOVING → STATIONARY（连续 stationary_frames 帧没动）
        for track in self.tracks.values():
            if track.state == TrackState.NEW:
                if track.stationary_counter >= self.config.stationary_frames:
                    track.state = TrackState.STATIONARY
                else:
                    track.state = TrackState.MOVING

        return matched, new_detections

    @staticmethod
    def get_nearby_person(
        track: TrackInfo,
        person_boxes: List[Tuple],
        dist_threshold: int = 120
    ) -> Optional[Tuple]:
        """
        判断物体附近有没有人。

        计算物体的框底部中心 与 每个人的框底部中心 的距离，
        如果存在小于 dist_threshold 的人，返回该人框。

        参数:
            track:           追踪器
            person_boxes:    当前帧的人框列表 [(x1,y1,x2,y2), ...]
            dist_threshold:  "附近"的像素阈值

        返回:
            最近的人框，如果没有人在附近则返回 None
        """
        tx1, ty1, tx2, ty2 = track.bbox
        tc_x = (tx1 + tx2) / 2
        tc_y = ty2                               # 物体底部中心

        closest = None
        min_dist = float('inf')

        for pb in person_boxes:
            pc_x = (pb[0] + pb[2]) / 2
            pc_y = pb[3]                          # 人脚位置（框底部）

            dist = math.sqrt((tc_x - pc_x) ** 2 + (tc_y - pc_y) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest = pb

        if min_dist <= dist_threshold:
            return closest
        return None