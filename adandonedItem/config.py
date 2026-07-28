"""
配置管理模块
============
用 dataclass 管理所有配置，支持从 YAML 文件加载。

设计模式：单例（通过模块级实例）+ 工厂（from_yaml）
"""
from __future__ import annotations
import yaml
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AppConfig:
    """应用配置（核心数据结构）"""

    # ── YOLO 模型 ──────────────────────────────────────────
    model_path: str = "yolov8n.pt"

    # ── 检测阈值 ──────────────────────────────────────────
    obj_conf_threshold: float = 0.25       # 物体置信度（低一点宁多勿漏）
    person_conf_threshold: float = 0.65    # 人的置信度（高一点减少误检）
    iou_threshold: float = 0.3             # IOU 匹配阈值（追踪用）

    # ── 追踪参数 ──────────────────────────────────────────
    stationary_frames: int = 5     # 连续几帧不动就算"静止放下"
    max_miss_count: int = 3        # 连续漏检几次才删除追踪器

    # ── 告警参数 ──────────────────────────────────────────
    abandon_seconds: int = 5       # 无人看管超过几秒触发告警

    # ── 人检测清除 ────────────────────────────────────────
    person_clear_after: int = 3    # 连续几帧无人 → 清空人框

    # ── 视频源 ────────────────────────────────────────────
    video_source: str = "rtsp://admin:Geis2015@192.168.1.125/Streaming/Channels/101"

    # ── 性能参数 ──────────────────────────────────────────
    detect_interval: int = 3       # 每 N 帧跑一次 YOLO 检测

    # ── 输出 ──────────────────────────────────────────────
    save_result: bool = False
    output_path: str = "output.avi"

    # ── YOLO 类别 ─────────────────────────────────────────
    person_class_id: int = 0                               # COCO 中"人"的编号
    suspicious_classes: list = field(default_factory=lambda: [
        24, 26, 28,    # 背包、手提包、行李箱
        39, 41,        # 水瓶、杯子
        43, 44, 45,    # 刀、勺子、碗
        63, 64, 65, 66, 67,  # 笔记本、鼠标、遥控器、键盘、手机
        73, 76, 77,    # 书、剪刀、玩具熊
    ])

    # ── 可视化 ────────────────────────────────────────────
    dist_threshold: int = 120      # "看管距离"（像素），在此距离内算有人看管

    # ──────────────── 方法 ────────────────

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "AppConfig":
        """
        从 YAML 文件加载配置。
        如果文件不存在，返回默认配置。
        文件中的字段会覆盖默认值（部分覆盖）。

        用法:
            config = AppConfig.from_yaml("config.yaml")
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data is None:        # 文件为空
                return cls()
            return cls(
                model_path=data.get("model_path", cls.model_path),
                video_source=str(data.get("video_source", cls.video_source)),
                abandon_seconds=int(data.get("abandon_seconds", cls.abandon_seconds)),
                save_result=bool(data.get("save_result", cls.save_result)),
                obj_conf_threshold=float(data.get("obj_conf", cls.obj_conf_threshold)),
                person_conf_threshold=float(data.get("person_conf", cls.person_conf_threshold)),
            )
        except FileNotFoundError:
            return cls()            # 没有配置文件就用默认值