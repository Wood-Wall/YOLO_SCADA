"""
YOLO 检测器封装模块
===================
封装 Ultralytics YOLO，提供结构化检测结果。

设计模式：外观模式 (Facade) —— 隐藏 YOLO 的复杂输出，
对外只提供简单的 DetectedObject 数据类。

关键设计：
  - 人和可疑物用不同置信度阈值分流
  - 只返回关心的类别（人 + 可疑留置物），其他忽略
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from ultralytics import YOLO


@dataclass
class DetectedObject:
    """单个检测结果的数据类"""
    class_id: int               # COCO 类别编号
    class_name: str             # 可读名称，如 "cell phone"
    confidence: float           # 置信度 0~1
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) 像素坐标


@dataclass
class DetectionResult:
    """一帧的完整检测结果"""
    objects: List[DetectedObject]   # 可疑留置物
    persons: List[DetectedObject]   # 检测到的人
    frame_shape: Tuple[int, int]    # (高, 宽)

    @property
    def has_person(self) -> bool:
        """这一帧是否检测到人"""
        return len(self.persons) > 0


class YOLODetector:
    """
    YOLO 检测器外观类。

    用法:
        detector = YOLODetector(config)
        result = detector.detect(frame)   # → DetectionResult
    """

    def __init__(self, config):
        self.config = config
        self.model = YOLO(config.model_path)
        self._names = self.model.names     # 类别号 → 名称 的映射

    @property
    def names(self) -> dict:
        """获取 COCO 类别名称映射"""
        return self._names

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        检测一帧画面。

        参数:
            frame: BGR 格式的图像 (numpy array)

        返回:
            DetectionResult 结构体，包含人和可疑物的分离列表
        """
        results = self.model(frame, verbose=False)
        result = results[0]

        objects: List[DetectedObject] = []
        persons: List[DetectedObject] = []
        h, w = frame.shape[:2]

        # 没有检测结果 → 直接返回空结果
        if result.boxes is None or len(result.boxes) == 0:
            return DetectionResult(
                objects=objects,
                persons=persons,
                frame_shape=(h, w)
            )

        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)

            detected = DetectedObject(
                class_id=cls_id,
                class_name=self._names.get(cls_id, "unknown"),
                confidence=conf,
                bbox=(x1, y1, x2, y2),
            )

            # ── 分流：人 vs 可疑留置物 ──
            if cls_id == self.config.person_class_id:
                if conf >= self.config.person_conf_threshold:
                    persons.append(detected)
            elif cls_id in self.config.suspicious_classes:
                if conf >= self.config.obj_conf_threshold:
                    objects.append(detected)

        return DetectionResult(
            objects=objects,
            persons=persons,
            frame_shape=(h, w),
        )