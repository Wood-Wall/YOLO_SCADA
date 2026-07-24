"""
YOLO 训练脚本
用法: python train.py

Python 知识点:
  - argparse 解析命令行参数
  - os.path 操作文件路径
  - f-string 格式化输出
"""

import os
from ultralytics import YOLO


def main():
    # ============================================
    # 你可以在这里改参数（不用每次敲命令行）
    # ============================================

    # ---------- 模型 ----------
    # yolo11n.pt / yolo11s.pt / yolo11m.pt / yolo11l.pt / yolo11x.pt
    model_name = "yolov8n.pt"  # nano 版，训练最快

    # ---------- 数据 ----------
    # 选项 1: 用 COCO128 测试数据集（自动下载，先跑通流程）
    # 选项 2: 换成你自己的 dataset.yaml
    data_yaml = "dataset.yaml"  # ← 测试用，自动下载

    # ---------- 训练参数 ----------
    epochs = 100        # 训练轮数（越大越久，但也可能更准）
    workers = 1         # 数据加载线程数（默认 4）
    batch = 16          # 每批几张图（显存小就调小，如 8/4）
    imgsz = 640         # 输入图片尺寸（默认 640）
    device = "0"      # 用 CPU 还是 GPU ("0" = 第一张显卡，没有就写 cpu)

    # ---------- 其他 ----------
    project = "runs/train"  # 结果保存目录
    name = "exp"            # 本次实验名称

    # ============================================
    # 下面不用改
    # ============================================

    # 加载模型
    # YOLO() 会自动下载模型（如果本地没有）
    print(f"正在加载模型: {model_name}")
    model = YOLO(model_name)

    # 开始训练
    # 参数详解:
    #   data    → 数据集配置文件的路径
    #   epochs  → 训练多少轮
    #   batch   → 每批处理多少张图
    #   imgsz   → 输入图片的尺寸
    #   device  → cpu 或 cuda 设备号
    #   project → 结果保存到哪个目录
    #   name    → 本次实验的子文件夹名
    print(f"\n开始训练! 参数如下:")
    print(f"  模型: {model_name}")
    print(f"  数据: {data_yaml}")
    print(f"  轮数: {epochs}")
    print(f"  批次: {batch}")
    print(f"  尺寸: {imgsz}")
    print(f"  设备: {device}")
    print("-" * 40)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
    )

    print(f"\n训练完成! 结果保存在: {project}/{name}/")


if __name__ == "__main__":
    main()