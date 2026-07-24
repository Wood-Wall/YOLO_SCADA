from ultralytics import YOLO
import cv2

# 1. 加载模型（这行代码背后就完成了 Backbone + Neck + Head 的初始化）
#    'yolov8n.pt' 是官方预训练权重，第一次运行会自动下载
model = YOLO('yolov8m.pt')

results = model(source=0, show=True, conf=0.6, save=True)
 