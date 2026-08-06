"""
可视化工具模块
==============
负责在画面上绘制检测框、状态面板、告警横幅等。

设计模式：纯静态方法，无状态（函数式风格）
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import Dict, List, Tuple

from models.tracker import TrackInfo, TrackState
from core.alert import AlertInfo, AlertLevel


class ColorPalette:
    """颜色方案（BGR 格式）"""
    PERSON_BOX = (0, 255, 0)           # 绿色 —— 人
    MOVING_BOX = (255, 255, 0)          # 青色 —— 移动中
    STATIONARY_BOX = (0, 165, 255)      # 橙色 —— 静止待检
    ABANDONED_BOX = (0, 0, 255)         # 红色 —— 已告警
    PANEL_BG = (0, 0, 0)               # 黑色 —— 面板背景
    PANEL_BORDER = (100, 100, 100)      # 灰色 —— 面板边框
    TEXT_NORMAL = (255, 255, 255)       # 白色 —— 普通文字
    TEXT_GOOD = (0, 255, 0)             # 绿色 —— 正常状态
    TEXT_BAD = (0, 0, 255)              # 红色 —— 告警状态


def _compute_distance(box_a: Tuple, box_b: Tuple) -> float:
    """计算两个框底部中心的距离（判断人是否在物体旁）"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    a_cx = (ax1 + ax2) / 2
    a_cy = ay2                # 用框底部代表"脚的位置"
    b_cx = (bx1 + bx2) / 2
    b_cy = by2
    return np.sqrt((a_cx - b_cx) ** 2 + (a_cy - b_cy) ** 2)


class Visualizer:
    """画面绘制器 — 所有方法都是静态的"""

    @staticmethod
    def draw_tracks(
        frame: np.ndarray,
        tracks: Dict[int, TrackInfo],
        person_boxes: List[Tuple],
        dist_threshold: int
    ) -> np.ndarray:
        """在画面绘制追踪器和检测框"""
        vis = frame.copy()

        for track in tracks.values():
            x1, y1, x2, y2 = track.bbox

            # ── 按状态选颜色 ──
            color_map = {
                TrackState.MOVING: ColorPalette.MOVING_BOX,
                TrackState.STATIONARY: ColorPalette.STATIONARY_BOX,
                TrackState.ABANDONED: ColorPalette.ABANDONED_BOX,
            }
            color = color_map.get(track.state, ColorPalette.MOVING_BOX)

            # ── 静止/告警状态下画"看管距离圈" ──
            if track.state in (TrackState.STATIONARY, TrackState.ABANDONED):
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.circle(vis, (cx, cy), dist_threshold, color, 1, cv2.LINE_AA)

                # 如果有"看管人"，画连线
                for pb in person_boxes:
                    dist = _compute_distance(track.bbox, pb)
                    if dist <= dist_threshold:
                        pc_x = (pb[0] + pb[2]) // 2
                        pc_y = pb[3]                     # 人脚位置
                        cv2.line(vis, (cx, cy), (pc_x, pc_y),
                                 ColorPalette.PERSON_BOX, 1, cv2.LINE_AA)

            # ── 画框 ──
            thickness = 3 if track.state == TrackState.ABANDONED else 2
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

            # ── 标签 ──
            if track.state == TrackState.ABANDONED:
                label = f"【留置】{track.class_name}"
            elif track.state == TrackState.STATIONARY:
                elapsed = int(track.abandon_timer)
                label = f"{track.class_name} {elapsed}s"
            else:
                label = track.class_name

            cv2.putText(vis, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return vis

    @staticmethod
    def draw_detections(frame: np.ndarray, person_boxes: List[Tuple]) -> np.ndarray:
        """单独画人的检测框（绿色）"""
        vis = frame.copy()
        for pb in person_boxes:
            x1, y1, x2, y2 = pb
            cv2.rectangle(vis, (x1, y1), (x2, y2),
                          ColorPalette.PERSON_BOX, 2)
            label = "person"
            cv2.putText(vis, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        ColorPalette.PERSON_BOX, 2)
        return vis

    @staticmethod
    def draw_status_panel(frame: np.ndarray, engine) -> np.ndarray:
        """绘制左上角状态面板"""
        vis = frame.copy()
        h, w = frame.shape[:2]

        # ── 面板背景 ──
        px, py = 10, 10
        pw, ph = 220, 200
        cv2.rectangle(vis, (px, py), (px + pw, py + ph),
                      ColorPalette.PANEL_BG, -1)
        cv2.rectangle(vis, (px, py), (px + pw, py + ph),
                      ColorPalette.PANEL_BORDER, 1)

        y = py + 20
        lh = 22                      # line height

        def _text(msg, color=ColorPalette.TEXT_NORMAL, size=0.4):
            nonlocal y
            cv2.putText(vis, msg, (px + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, size, color, 1)
            y += lh

        # ── 标题 ──
        _text("异物留置检测系统", ColorPalette.TEXT_NORMAL, 0.45)
        y += 5

        # ── 追踪统计 ──
        stats = engine.tracker.stats
        _text(f"追踪器: {engine.tracker.active_count} 个")
        _text(f"  移动中:{stats['moving']}  静止:{stats['stationary']}  告警:{stats['abandoned']}")

        # ── 人检测状态 ──
        person_color = ColorPalette.TEXT_GOOD if engine._person_present else ColorPalette.TEXT_BAD
        person_text = f"人: {'在画面中' if engine._person_present else '已离开'} (连续未检测:{engine._person_miss_count})"
        _text(person_text, person_color)

        # ── FPS ──
        _text(f"FPS: {engine.fps:.1f}")

        # ── 帧数 ──
        _text(f"帧: {engine.frame_count}")

        y += 5
        # ── 告警数量 ──
        alert_color = ColorPalette.TEXT_BAD if engine.alert_manager.alert_count > 0 else ColorPalette.TEXT_NORMAL
        _text(f"告警: {engine.alert_manager.alert_count}", alert_color)

        return vis

    @staticmethod
    def draw_alerts(frame: np.ndarray, alerts: List[AlertInfo]) -> np.ndarray:
        """在画面底部绘制告警横幅"""
        if not alerts:
            return frame

        vis = frame.copy()
        h, w = frame.shape[:2]

        for i, alert in enumerate(alerts):
            bar_h = 36
            bar_y = h - (len(alerts) - i) * bar_h
            # 背景
            cv2.rectangle(vis, (0, bar_y), (w, bar_y + bar_h),
                          (0, 0, 180), -1)
            cv2.rectangle(vis, (0, bar_y), (w, bar_y + bar_h),
                          (0, 0, 255), 1)
            # 文字
            cv2.putText(vis, f"⚠ {alert.message}",
                        (20, bar_y + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        ColorPalette.TEXT_NORMAL, 2)

        return vis

    @staticmethod
    def draw_all(frame: np.ndarray, engine) -> np.ndarray:
        """全绘制：检测框 + 追踪器 + 面板 + 告警"""
        # 1. 先画追踪器（含检测框）
        vis = Visualizer.draw_tracks(
            frame, engine.tracker.tracks,
            engine._person_boxes, engine.config.dist_threshold
        )
        # 2. 状态面板
        vis = Visualizer.draw_status_panel(vis, engine)
        # 3. 告警横幅（如果有）
        vis = Visualizer.draw_alerts(vis, engine.alert_manager.active_alerts)

        return vis