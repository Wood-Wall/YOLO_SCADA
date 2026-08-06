"""
视频源抽象模块
==============
统一不同视频源（摄像头/文件/RTSP）的接口。

设计模式：
  - 抽象基类 (ABC)：定义统一的 read/release/fps/frame_size 接口
  - 工厂模式 (VideoSourceFactory)：根据字符串自动创建对应的视频源
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import cv2
import os


class VideoSource(ABC):
    """
    视频源抽象基类。
    所有视频源（摄像头、文件、RTSP）都继承这个类。

    用法:
        with VideoSourceFactory.create("test.mp4") as source:
            ret, frame = source.read()
    """

    def __init__(self, source: str | int):
        self._cap: Optional[cv2.VideoCapture] = None
        self._source = source
        self._fps: float = 0
        self._width: int = 0
        self._height: int = 0
        self._total_frames: int = 0

    @abstractmethod
    def _open(self) -> bool:
        """子类实现具体的打开逻辑"""
        ...

    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """读一帧，返回 (是否成功, 画面数据)"""
        if self._cap is None:
            return False, None
        return self._cap.read()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_size(self) -> Tuple[int, int]:
        return self._width, self._height

    @property
    def total_frames(self) -> int:
        return self._total_frames

    def release(self):
        """释放资源"""
        if self._cap:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *args):
        self.release()


class LocalCamera(VideoSource):
    """本地摄像头 (如 0, 1)"""

    def _open(self) -> bool:
        self._cap = cv2.VideoCapture(int(self._source))
        if self._cap.isOpened():
            self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30
            self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return True
        return False


class VideoFile(VideoSource):
    """本地视频文件 (如 .mp4, .avi)"""

    def _open(self) -> bool:
        self._cap = cv2.VideoCapture(str(self._source))
        if self._cap.isOpened():
            self._fps = self._cap.get(cv2.CAP_PROP_FPS)
            self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return True
        return False


class RTSPStream(VideoSource):
    """RTSP 网络推流 (监控摄像头)"""

    def _open(self) -> bool:
        # 用 TCP 传输，减少丢包
        self._cap = cv2.VideoCapture(self._source)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # 减小缓冲，降低延迟
        if self._cap.isOpened():
            self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 25
            self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return True
        return False


class VideoSourceFactory:
    """
    视频源工厂。
    根据字符串自动判断类型并创建对应的 VideoSource 对象。

    规则:
        "0", "1"...    → LocalCamera
        rtsp://...     → RTSPStream
        .mp4/.avi...   → VideoFile
        其他           → 默认 LocalCamera(0)
    """

    @staticmethod
    def create(source: str | int) -> VideoSource:
        if isinstance(source, int):
            return LocalCamera(source)

        s = str(source).strip()

        # 纯数字 → 摄像头
        if s.isdigit():
            return LocalCamera(int(s))

        # RTSP / RTMP → 网络流
        if s.startswith("rtsp://") or s.startswith("rtmp://"):
            return RTSPStream(s)

        # 本地文件
        if os.path.isfile(s):
            return VideoFile(s)

        # 默认
        return LocalCamera(0)