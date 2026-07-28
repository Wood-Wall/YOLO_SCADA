"""
异物留置检测系统 (Abandoned Object Detection System)
====================================================
工程化的异物留置检测系统，基于 YOLO + IOU 追踪。

架构:
  config.py         配置管理 (dataclass + YAML)
  models/
    detector.py     YOLO 检测器封装
    tracker.py      物体追踪器 (IOU匹配 + 状态机)
  core/
    engine.py       核心检测引擎 (协调器)
    alert.py        告警管理 (观察者模式)
  utils/
    video.py        视频源抽象 (工厂模式)
    visualization.py 可视化绘制
  main.py           程序入口
"""