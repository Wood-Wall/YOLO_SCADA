# my_demo.py — 自己实现异物留置检测

# ============================================================
# 第 1 步：导入需要用到的库
# ============================================================

import cv2          # OpenCV，处理摄像头/视频/画图
import numpy as np  # 科学计算，这里主要用来算平方根
from ultralytics import YOLO  # YOLO 模型
import argparse     # 解析命令行参数（--video, --threshold 等）
import time         # 计时
from collections import deque  # 双端队列，自动淘汰旧数据

# ============================================================
# 第 2 步：配置参数（写在文件顶部，方便以后修改）
# ============================================================

# --- COCO 数据集中"可能被留置"的物体编号 ---
# YOLO 是用 COCO 数据集训练的，能识别 80 种物体
# 每个物体有一个编号（class id）：
#   0 = 人
#   24 = 背包    26 = 手提包    28 = 行李箱
#   39 = 水瓶    41 = 杯子      67 = 手机
#   63 = 笔记本  64 = 鼠标      66 = 键盘
#   73 = 书      76 = 剪刀      77 = 玩具熊
SUSPICIOUS_CLASSES = [24, 26, 28, 39, 41, 43, 44, 45,
                      63, 64, 65, 66, 67, 73, 76, 77]

PERSON_CLASS = 0       # 人的编号

# --- 检测参数 ---
CONF_THRESHOLD = 0.25           # 物体的置信度阈值（低于这个值忽略）
PERSON_CONF_THRESHOLD = 0.65   # 人的置信度阈值（人用高阈值，减少误检）
SKIP_FRAMES = 2                # 每 3 帧检测一次（省 CPU，0=每帧都检）

# --- 追踪参数 ---
STATIONARY_FRAMES = 5   # 连续多少次检测位置没变 = "静止放下"
MAX_MISS_FRAMES = 3     # 追踪器连续几次没匹配到 = "物体消失了"→删除
DIST_THRESHOLD = 120    # 人离物体多远(像素)算"在看管中"

# --- 告警参数 ---
ABANDON_SECONDS = 5     # 无人看管超过几秒 → 告警

# --- 人消失判断 ---
PERSON_CLEAR_AFTER = 3  # 连续几次检测没人 → 清空人列表

# --- 调试开关 ---
DEBUG = True            # True=每帧打印 YOLO 检测到了什么

class Track:
    """单个物体的追踪记录"""
    def __init__(self, track_id, box, class_id, class_name):
        """
        __init__ 构造函数
        track_id    : 物体唯一编号
        box         : 此帧边框 (x1, y1, x2, y2)
        class_id    : 物体类型编号 (数据集编号)
        class_name  : 物体类型名称
        """
        self.id = track_id
        self.class_id = class_id
        self.class_name = class_name

        # 只保留最近 STATIONARY_FRAMES 帧
        self.positions = deque(maxlen=STATIONARY_FRAMES)

        # 存储x\y的中心点, 用来判断静止，不用完整边框
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        self.positions.append((cx, cy))

        self.last_box = box  # 最新一帧
        self.miss_count = 0  # 连续没检测到的帧数

        # 状态标记
        self.stationary_count = 0   # 累计"静止"次数
        self.is_stationary = False  # 是否已确认静止
        self.abandon_timer = 0.0    # 无人看管的累计秒数
        self.alert = False          # 是否已经触发告警

    def update(self, box):
        """
        返回 ：无
        参数：
            box ：当前帧内此追踪器物品边框
        方法：+
            更新物体中心点队列
            更新物体最新边框位置
            更新物体静止状态
        """
        # 算中心点
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        # 更新当前帧
        self.positions.append((cx,cy))
        self.last_box = box
        # 判断静止
        self.check_stationary()
        
    def check_stationary(self):
        """
            静止判断方法：
            遍历物品位置队列，保存的这几帧内边框移动的最大距离
            如果小于最大距离：算静止，连续X次判断是静止则更新物体状态为静止
            如果大于最大距离：算移动
        """
        # 为了防止误判，必须满X帧才能判断静止
        MAX_PIXEL = 10
        if len(self.positions) < STATIONARY_FRAMES:
            return
        xs = [x for x,y in self.positions]
        ys = [y for x,y in self.positions]
        max_movement = max(max(xs)-min(xs), max(ys)-min(ys))
        if max_movement < MAX_PIXEL:
            self.stationary_count += 1
            if self.stationary_count >= STATIONARY_FRAMES:
                self.is_stationary = True
        else:
            self.stationary_count = 0
            self.is_stationary = False

def is_person_near(obj_box, person_boxes, threshold = 120):
    """
    返回：
        bool- 物体身边是否有人
    参数：
        obj_box ：物体边框
        person_boxes: 当前帧内所有人的边框
        threshold: 人与物体之间允许的距离（像素个数）
    方法：
        判断边框与人体边框是否重叠
        判断人与物体边框距离
    """
    ox1,oy1,ox2,oy2 = obj_box
    obj_cx = (ox1 + ox2) // 2
    obj_cy = (oy1 + oy2) // 2
    for(px1,py1,px2,py2) in person_boxes:
        #判断人和物体是否有重叠，有直接认为人在旁边
        if not (px1 > ox2 or px2 < ox1 or py1 > oy2 or py2 < oy1):
            return True
        #判断人和物体的距离是否足够
        per_cx = (px1 + px2) // 2
        per_cy = (py1 + py2) // 2
        dist = np.sqrt((obj_cx - per_cx)**2 + (obj_cy - per_cy)**2)
        if dist < threshold:
            return True

if __name__ == "__main__":
    # 创建一个追踪器
    t = Track(1,(1,1,100,100),28,"suitcase")
    print(f"创建完成：ID={t.id}, 类别={t.class_name}")
    # 模拟 5 帧位置都没变
    for i in range(5):
        t.update((1, 1, 100, 100))
        print(f"第 {i+2} 帧: stationary_count={t.stationary_count}, is_stationary={t.is_stationary}")



