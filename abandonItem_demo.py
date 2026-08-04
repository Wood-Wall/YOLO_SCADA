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
        方法：
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

def main():
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description="异物滞留检测演示")
    parser.add_argument("--video",type=str,default="0",
                        help="视频源：默认0=本地摄像头")
    parser.add_argument("--threshold",type=int,default=ABANDON_SECONDS,
                        help=f"无人看管超过几秒告警（默认 {ABANDON_SECONDS} 秒）")
    parser.add_argument("--save", action="store_true",
                        help="加上这个参数就保存结果视频到文件")
    # action="store_true" 是 Python 特有用法
    # 意思是：只要命令行写了 --save，args.save 就等于 True
    # 不写就等于 False，不需要传值
    args = parser.parse_args()
    #参数：args.video,args.threshold,args.save

    # 2. 加载YOLO模型
    print("正在加载YOLO模型")
    model = YOLO("yolov8n.pt")
    abandon_seconds = args.threshold
    video_source = args.video
    if video_source == "0":
        video_source = 0
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("错误，视频无法打开")
        return

    # 3. 打印日志
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25 
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频信息: {width}×{height}, {fps:.1f} FPS")
    print(f"告警阈值: {abandon_seconds} 秒")
    print(f"检测物体: 背包/手提包/行李箱/笔记本/手机/书")
    print("按 'q' 退出 | 按 'r' 重置追踪\n")

    # 4. 初始化变量
    # 视频保存对象
    writer = None 
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # VideoWriter(文件名, 编码, 帧率, 尺寸)
        writer = cv2.VideoWriter("abandoned_result.mp4", fourcc, fps, (width, height))
    # tracks = 所有追踪器的字典
    # 字典是 Python 的核心数据结构，类似 C++ 的 unordered_map
    # 格式：{ key: value, key: value, ... }
    # 这里 key = 追踪器 id（整数），value = Track 对象
    tracks = {}

    next_id = 0                        # 下一个新追踪器的编号
    frame_count = 0                    # 当前处理到第几帧

    # model.names 是 YOLO 模型自带的类别名称字典
    # 比如 {0: "person", 24: "backpack", 28: "suitcase", ...}
    # hasattr(model, 'names') 判断 model 有没有 names 这个属性
    # 这是 Python 的动态特性——先检查再使用，避免报错
    coco_names = model.names if hasattr(model, 'names') else {}

    # person_boxes 保存当前帧中所有人的边框
    # 因为不是每帧都跑 YOLO，所以要把人的位置记住
    person_boxes = []

    # person_miss_count 记录"连续多少帧没检测到人"
    # 用于过滤背景误检：偶尔漏检一帧不影响
    person_miss_count = 0

    # 5. 主循环
    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频播放完毕")
            break
        frame_count += 1
        # 处理目标帧
        if frame_count % (SKIP_FRAMES + 1) == 1:
            results = model(frame, verbose=False)
            person_boxes = []
            obj_boxes = []
            is_dectected_person = False
            #检查YOLO是否检测到东西
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                #遍历每个物体
                for i in range(len(boxes)):
                    cls = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    if cls == PERSON_CLASS:
                        if conf < PERSON_CONF_THRESHOLD:    # 置信度低的人过滤掉
                            continue
                        is_dectected_person = True
                        x1,y1,x2,y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                        person_boxes.append((x1,y1,x2,y2))
                    elif cls in SUSPICIOUS_CLASSES:
                        if conf < CONF_THRESHOLD:   #置信度低的物品过滤掉
                            continue
                        x1,y1,x2,y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                        obj_boxes.append((x1,y1,x2,y2,cls))
        # 画人框
        for (px1,py1,px2,py2) in person_boxes:
            cv2.rectangle(frame,(px1,py1),(px2,py2),(0,255,0),2)
            cv2.putText(frame, "Person", (px1, py1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # 物体，画黄色框
        for (x1,y1,x2,y2,cls) in obj_boxes:
            class_name = coco_names.get(cls,"unknown")
            cv2.rectangle(frame,(x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        # 画可疑物体
        cv2.imshow("Abandoned Object Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):    # 按 q → 退出
            print("用户退出")
            break
    cap.release()              # 关闭摄像头（类似 C++ 的 close）
    if writer:
        writer.release()       # 关闭视频文件
    cv2.destroyAllWindows()    # 关闭所有 OpenCV 窗口
    print("程序已退出")

if __name__ == "__main__":
    """
    Python 特殊变量 __name__：
    - 直接运行本文件时，__name__ 自动变成 "__main__"
    - 如果是被其他文件 import，__name__ 就是文件名，不会执行 main()
    这行代码 = "只有直接运行时才执行 main()"
    """
    # 创建一个追踪器
    main()