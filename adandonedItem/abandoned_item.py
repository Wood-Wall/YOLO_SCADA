from timeit import main
"""
=================================================================
异物留置检测（Abandoned Object Detection）
=================================================================
思路：
  1. 用 YOLO 检测"人"和"可留置物"（行李箱/背包/手提包等）
  2. IOU 追踪每个物体，判断它是否"静止放下"了
  3. 如果物体静止且旁边没人超过阀值 → 告警

用法：
  python abandoned_item.py                         # 默认摄像头
  python abandoned_item.py --video test.mp4        # 指定视频文件
  python abandoned_item.py --threshold 10 --save   # 10秒+保存结果
"""

import cv2
import yaml
from ultralytics import YOLO

SKIP_FRAMES = 5 # 5帧检查一次
# --- YOLO 可检测的"可能被留置"的物体 ---
# COCO 数据集类别编号:
#   24=背包  26=手提包  28=行李箱
#   39=水瓶  41=杯子   43=刀  44=勺子  45=碗
#   63=笔记本  64=鼠标  65=遥控器  66=键盘  67=手机
#   73=书  76=剪刀  77=玩具熊
SUSPICIOUS_CLASSES = [24, 26, 28, 39, 41, 43, 44, 45,
                      63, 64, 65, 66, 67, 73, 76, 77]

PERSON_CLASS = 0       # COCO 中"人"的编号
STATIONARY_FRAMES = 5  # 连续几帧不动就算"放下了"（原来10，改5更快触发）
DIST_THRESHOLD = 120   # 人离物体多少像素内算"看管中"
CONF_THRESHOLD = 0.25  # 物体置信度阈值
PERSON_CONF_THRESHOLD = 0.65  # 人的置信度阈值（高一点减少误检）

person_boxes = []  # 存储所有检测到的人的框坐标
object_boxes = []  # 存储所有检测到的可留置物的框坐标

# ============================================================
# 加载配置文件
# ============================================================
def load_config(yaml_path="config.yaml"):
    with open(yaml_path,'r',encoding='utf-8') as f:
        data = yaml.safe_load()
      
    video = data.get("video_source","rtsp://admin:Geis2015@192.168.1.125/Streaming/Channels/101")
    seconds = data.get("abandon_seconds",30)
    save = data.get("save_result",False)

    config = {
          "视频源":video,
          "留置时间":seconds,
          "保存结果":save,
    }
    return config
    
# ============================================================
# 
# ============================================================
def process_frame(frame):
  results = model(frame)
  result = results[0]
  if len(result) <= 0 or result.boxes is None:
    return
  boxes = result.boxes
  for i in range(len(boxes)):
    clsid = int(boxes[i].cls)
    conf = boxes[i].conf
    # 有人存在
    if clsid == PERSON_CLASS:
      if conf < PERSON_CONF_THRESHOLD:
        continue



# ============================================================
# 
# ============================================================
def mian():
    # ---- 加载配置文件参数 ----
    config = load_config()
    model = YOLO("yolov8n.pt")
    seconds = config["留置时间"]
    save = config["保存结果"]
    video = config["视频源"]
    if video == "0":
        video = 0

    # ---- 打开视频源 ----
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    
    # ---- 循环处理视频帧 ----
    while True:
        ret,frame = cap.read()
        if not ret:   # 视频结束
            break
        frame_cnt += 1 # 帧计数
        if frame_cnt % SKIP_FRAMES == 0:
          # ---- 关键帧 检测物体 ----
          process_frame(frame)
          
if __name__ == "__main__":
    main()