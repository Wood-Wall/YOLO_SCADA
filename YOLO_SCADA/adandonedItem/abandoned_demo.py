"""
=================================================================
异物留置检测（Abandoned Object Detection）
=================================================================

思路：
  1. 用 YOLO 检测"人"和"可留置物"（行李箱/背包/手提包等）
  2. IOU 追踪每个物体，判断它是否"静止放下"了
  3. 如果物体静止且旁边没人超过阀值 → 告警

用法：
  python abandoned_demo.py                         # 默认摄像头
  python abandoned_demo.py --video test.mp4        # 指定视频文件
  python abandoned_demo.py --threshold 10 --save   # 10秒+保存结果
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import time
from collections import deque

# ============================================================
# 1. 配置参数（你可以随意改这些数字）
# ============================================================

# --- YOLO 可检测的"可能被留置"的物体 ---
# COCO 数据集类别编号:
#   24=背包  26=手提包  28=行李箱
#   39=水瓶  41=杯子   43=刀  44=勺子  45=碗
#   63=笔记本  64=鼠标  65=遥控器  66=键盘  67=手机
#   73=书  76=剪刀  77=玩具熊
SUSPICIOUS_CLASSES = [24, 26, 28, 39, 41, 43, 44, 45,
                      63, 64, 65, 66, 67, 73, 76, 77]

PERSON_CLASS = 0       # COCO 中"人"的编号

# 调试模式：设为 True 会在终端打印每一帧检测到的所有物体
DEBUG = True

ABANDON_SECONDS = 5    # 物体无人看管超过几秒就告警
STATIONARY_FRAMES = 5  # 连续几帧不动就算"放下了"（原来10，改5更快触发）
DIST_THRESHOLD = 120   # 人离物体多少像素内算"看管中"

SKIP_FRAMES = 2        # 每 N 帧检测一次（省 CPU，画框不跳帧）
CONF_THRESHOLD = 0.25  # 物体置信度阈值（原来0.4，你的手机0.3左右被过滤了，改低）
PERSON_CONF_THRESHOLD = 0.65  # 人的置信度阈值（高一点减少误检）
PERSON_CLEAR_AFTER = 3 # 连续 N 次检测无人，就清空 person_boxes
MAX_MISS_FRAMES = 3    # 追踪器连续 N 次没匹配到才删除（原来直接删，容错性差）

# ============================================================
# 2. 追踪器类（用来记住每个物体的状态）
# ============================================================
class Track:
    """单个物体的追踪记录"""
    def __init__(self, track_id, box, class_id, class_name):
        """
        track_id : 唯一编号
        box      : 当前边框 (x1, y1, x2, y2)
        class_id : 物体类别编号
        class_name: 物体类别名称
        """
        self.id = track_id
        self.class_id = class_id
        self.class_name = class_name

        # ---- 位置历史 ----
        # deque 是一个"双端队列"，maxlen=N 表示只保留最近 N 个元素
        # 超出 N 时，最早的那个会自动丢掉
        self.positions = deque(maxlen=STATIONARY_FRAMES)
        self.positions.append(self._center(box))
        self.last_box = box          # 最新一帧的边框

        # ---- 状态标记 ----
        self.stationary_count = 0    # 连续静止帧数（累计）
        self.is_stationary = False   # 是否已确认静止
        self.abandon_timer = 0.0     # 无人看管累计秒数
        self.alerted = False         # 是否已触发告警

        # 连续未匹配帧数（用于容错，避免 YOLO 偶尔漏检就删追踪器）
        self.miss_count = 0

        # 首次出现时间（备用）
        self.first_seen = time.time()

    # --------------------------------------------------------
    # 辅助方法：计算边框中心点
    # --------------------------------------------------------
    @staticmethod
    def _center(box):
        """返回边框的中心坐标 (cx, cy)"""
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    # --------------------------------------------------------
    # 更新方法：每检测到一帧就调用一次
    # --------------------------------------------------------
    def update(self, box):
        """
        用新的边框更新追踪器，判断物体是否静止

        box : (x1, y1, x2, y2) 当前帧检测到的边框
        """
        cx, cy = self._center(box)
        last_cx, last_cy = self.positions[-1]

        # 计算当前帧的位移
        # np.sqrt 算平方根（勾股定理算距离）
        dist = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)

        # 把新位置加入历史队列（deque 会自动淘汰最旧的那个）
        self.positions.append((cx, cy))
        self.last_box = box

        # ---- 判断是否静止 ----
        # 只有历史队列存够了 STATIONARY_FRAMES 帧才开始判断
        if len(self.positions) == STATIONARY_FRAMES:
            # 取出所有历史位置的 x 坐标和 y 坐标
            xs = [p[0] for p in self.positions]   # 列表推导式：循环取每个位置的 x
            ys = [p[1] for p in self.positions]   # 同上，取 y

            # 最大移动范围 = x 方向最大差值 和 y 方向最大差值 中较大的那个
            # max(xs) - min(xs) 就是"x 方向走了多少像素"
            max_movement = max(max(xs) - min(xs), max(ys) - min(ys))

            if max_movement < 10:   # 10 像素以内→算静止
                self.stationary_count += 1
                if self.stationary_count >= STATIONARY_FRAMES:
                    self.is_stationary = True
            else:
                # 动了→重置静止计数器
                self.stationary_count = 0
                self.is_stationary = False


# ============================================================
# 3. 工具函数
# ============================================================

def is_person_near(obj_box, person_boxes, threshold=100):
    """
    判断物体附近有没有人

    obj_box      : (x1, y1, x2, y2) 物体的边框
    person_boxes : 列表，每个元素是 (x1, y1, x2, y2) 人的边框
    threshold    : 人的脚离物体中心多近算"附近"

    返回 True/False
    """
    ox1, oy1, ox2, oy2 = obj_box
    obj_cx = (ox1 + ox2) // 2   # 物体中心 x
    obj_cy = (oy1 + oy2) // 2   # 物体中心 y

    for (px1, py1, px2, py2) in person_boxes:
        # --- 情况 1：边框直接重叠 ---
        # 如果物体框和人框有交集（即不满足"分离"条件）
        if not (px1 > ox2 or px2 < ox1 or py1 > oy2 or py2 < oy1):
            return True     # 重叠 = 人在旁边

        # --- 情况 2：人的脚离物体中心很近 ---
        # 人的脚 ≈ 边框底部中心
        foot_x = (px1 + px2) // 2   # 脚 x
        foot_y = py2                # 脚 y（边框底部 y 坐标）
        dist = np.sqrt((foot_x - obj_cx)**2 + (foot_y - obj_cy)**2)

        if dist < threshold:
            return True

    return False    # 没有人靠近


def match_tracks(current_boxes, tracks, iou_threshold=0.3):
    """
    把当前帧检测到的物体和已有的追踪器做"匹配"

    current_boxes : 当前帧检测到的所有边框 [(x1,y1,x2,y2), ...]
    tracks        : 现有的追踪器字典 { id: Track对象, ... }
    iou_threshold : IOU 大于此值才算匹配成功

    返回: (matched_dict, new_boxes_list)
          matched_dict   = { track_id: box, ... }
          new_boxes_list = [box, ...]   ← 没匹配到的新物体列表
    """
    matched = {}
    new_boxes = []
    used_ids = set()    # 记录已经被匹配的追踪器 id

    for box in current_boxes:
        best_id = None
        best_iou = iou_threshold    # 只有 IOU 超过这个值才算匹配

        for tid, track in tracks.items():
            if tid in used_ids:
                continue    # 这个追踪器已经匹配过了

            # ===== 计算 IOU（交并比） =====
            # IOU = 两个框的交集面积 ÷ 并集面积
            # 值越接近 1 说明两个框重合越好
            x1, y1, x2, y2 = box
            tx1, ty1, tx2, ty2 = track.last_box

            # 交集区域的左上角和右下角
            inter_x1 = max(x1, tx1)
            inter_y1 = max(y1, ty1)
            inter_x2 = min(x2, tx2)
            inter_y2 = min(y2, ty2)

            iou = 0
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                box_area = (x2 - x1) * (y2 - y1)
                track_area = (tx2 - tx1) * (ty2 - ty1)
                union_area = box_area + track_area - inter_area
                iou = inter_area / union_area if union_area > 0 else 0

            if iou > best_iou:
                best_iou = iou
                best_id = tid

        if best_id is not None:
            # 匹配成功→更新已有的追踪器
            matched[best_id] = box
            used_ids.add(best_id)
        else:
            # 匹配失败→这是一个新物体（放到列表里，不会互相覆盖）
            new_boxes.append(box)

    return matched, new_boxes


# ============================================================
# 4. 主函数
# ============================================================
def main():
    # ---- 4a. 解析命令行参数 ----
    # argparse 帮我们处理命令行输入的参数
    parser = argparse.ArgumentParser(description="异物留置检测")
    parser.add_argument("--video", type=str,
                    default="rtsp://admin:Geis2015@192.168.1.125/Streaming/Channels/101",
                    help="视频源（默认RTSP推流，填0=本地摄像头，填路径=视频文件）")
    parser.add_argument("--threshold", type=int, default=ABANDON_SECONDS,
                        help=f"无人看管超过多少秒告警（默认 {ABANDON_SECONDS}s）")
    parser.add_argument("--save", action="store_true",
                        help="保存结果视频到 abandoned_result.mp4")
    args = parser.parse_args()

    # ---- 4b. 加载 YOLO 模型 ----
    print("正在加载 YOLO 模型...")
    model = YOLO("yolov8n.pt")          # 首次运行会自动下载
    abandon_seconds = args.threshold

    # ---- 4c. 打开视频 ----
    # args.video 默认为 RTSP 地址；如果传了 --video 0 则用本地摄像头
    video_source = args.video
    if video_source == "0":
        video_source = 0  # 转成整数 0 = 本地摄像头
    cap = cv2.VideoCapture(video_source)    # 打开摄像头或视频文件
    if not cap.isOpened():
        print("错误：无法打开视频，请检查路径或摄像头")
        return

    # 获取视频的帧率和尺寸
    fps = cap.get(cv2.CAP_PROP_FPS)    # cap.get() 读视频属性
    if fps <= 0:                       # 摄像头可能读不到 FPS
        fps = 25                       # 默认 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"视频信息: {width}×{height}, {fps:.1f} FPS")
    print(f"告警阈值: {abandon_seconds} 秒")
    print(f"检测物体: 背包/手提包/行李箱/笔记本/手机/书")
    print("按 'q' 退出 | 按 'r' 重置追踪\n")

    # ---- 4d. 视频保存（可选） ----
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 编码格式
        writer = cv2.VideoWriter(
            "abandoned_result.mp4", fourcc, fps, (width, height)
        )

    # ---- 4e. 初始化追踪器 ----
    tracks = {}          # { id: Track对象, ... }  字典
    next_id = 0          # 下一个新物体的 id
    frame_count = 0      # 处理了多少帧

    # COCO 类别名（YOLO 模型自带的）
    coco_names = model.names if hasattr(model, 'names') else {}

    # 保存上一帧检测到的人（用于画框，因为不是每帧都检测）
    person_boxes = []

    # 人连续未出现的计数（用来清除背景误检）
    person_miss_count = 0

    # ============================================================
    # 5. 主循环（核心部分）
    # ============================================================
    while True:
        ret, frame = cap.read()     # 读取一帧
        if not ret:                 # 视频结束
            break
        frame_count += 1

        # ---- 5a. 每隔 SKIP_FRAMES+1 帧跑一次 YOLO（省 CPU） ----
        if frame_count % (SKIP_FRAMES + 1) == 1:
            # results 是 YOLO 返回的结果列表
            results = model(frame, verbose=False)

            # 分别存储"人"和"可疑物体"
            person_boxes = []
            object_boxes = []       # 每个元素: (x1,y1,x2,y2, class_id)

            # ---- 分别用不同置信度阈值筛选人和物体 ----
            # 人对人用高阈值（PERSON_CONF_THRESHOLD），减少背景误检
            # 物体用低阈值（CONF_THRESHOLD），宁多勿漏
            person_detected_this_frame = False  # 这一帧是否检测到了人

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes

                for i in range(len(boxes)):
                    cls = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())

                    # ---- 分类别使用不同置信度阈值 ----
                    if cls == PERSON_CLASS:
                        if conf < PERSON_CONF_THRESHOLD:
                            continue
                        person_detected_this_frame = True
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                        person_boxes.append((x1, y1, x2, y2))
                    elif cls in SUSPICIOUS_CLASSES:
                        if conf < CONF_THRESHOLD:
                            continue
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                        object_boxes.append((x1, y1, x2, y2, cls))

            # ---- 人消失判断：连续多帧没检测到人 → 清空 person_boxes ----
            if person_detected_this_frame:
                person_miss_count = 0
            else:
                person_miss_count += 1
                if person_miss_count >= PERSON_CLEAR_AFTER:
                    person_boxes.clear()  # 背景误检不再影响计时

            # ---- 调试输出：看看 YOLO 到底检测到了什么 ----
            if DEBUG:
                # 收集这一帧所有检测结果（包括被置信度过滤掉的）
                all_names = []
                if len(results) > 0 and results[0].boxes is not None:
                    all_boxes = results[0].boxes
                    for j in range(len(all_boxes)):
                        c = int(all_boxes.cls[j].item())
                        conf = float(all_boxes.conf[j].item())
                        name = coco_names.get(c, f"class{c}")
                        all_names.append(f"{name}({conf:.2f})")
                if all_names:
                    print(f"[帧 {frame_count}] 检测到: {', '.join(all_names)}")
                else:
                    print(f"[帧 {frame_count}] 未检测到任何物体")

            # ---- 5b. 匹配追踪 ----
            # 取出物体边框（去掉 class_id）
            current_objects = [(x1, y1, x2, y2) for (x1, y1, x2, y2, _) in object_boxes]
            matched, new_boxes = match_tracks(current_objects, tracks)

            # 更新已匹配的追踪器
            for tid, box in matched.items():
                # 已有追踪器 → 更新位置
                tracks[tid].update(box)

            # ---- 创建新物体的追踪器 ----
            for box in new_boxes:
                # 找到这个新物体对应的 class_id
                cls = None
                for obj_box in object_boxes:
                    # 如果边框四个坐标都相同，说明是同一个框
                    # 注意：这里用前 4 个值比较（去掉 class_id）
                    if (obj_box[0], obj_box[1], obj_box[2], obj_box[3]) == box:
                        cls = obj_box[4]
                        break

                # 查字典：class_id → 名称，找不到就返回 "unknown"
                class_name = coco_names.get(cls, "unknown") if cls is not None else "unknown"

                tracks[next_id] = Track(next_id, box, cls, class_name)
                next_id += 1

            # ---- 5c. 清理连续消失的物体 ----
            # 如果某个追踪器本轮没有匹配到检测框，有两种可能：
            #   1. 物体被拿走了（真消失）→ 超过 MAX_MISS_FRAMES 次才删除
            #   2. YOLO 这帧没检测到（跳检/置信度波动）→ 留着别删，容忍一下
            # list(tracks.keys()) 复制一份 key 列表，因为遍历时不能修改字典
            for tid in list(tracks.keys()):
                if tid not in matched:
                    tracks[tid].miss_count += 1
                    if tracks[tid].miss_count >= MAX_MISS_FRAMES:
                        del tracks[tid]
                else:
                    # 匹配上了 → 重置 miss_count
                    tracks[tid].miss_count = 0

            # ---- 5d. 计时判断（核心逻辑） ----
            for tid, track in tracks.items():
                if track.alerted:
                    continue        # 已经告警过了，跳过

                if not track.is_stationary:
                    continue        # 物体还在移动，不算"留置"

                # 有人靠近吗？
                if is_person_near(track.last_box, person_boxes, DIST_THRESHOLD):
                    track.abandon_timer = 0.0   # 有人看着 → 重置计时
                else:
                    # 没人 → 累计时间
                    # (SKIP_FRAMES+1)/fps 约等于"两帧之间的实际秒数"
                    track.abandon_timer += (SKIP_FRAMES + 1) / fps

                    if track.abandon_timer >= abandon_seconds:
                        track.alerted = True
                        print(f"🚨 告警！{track.class_name}(ID:{track.id}) "
                              f"已无人看管 {abandon_seconds} 秒！")

        # ============================================================
        # 6. 可视化（每帧都画）
        # ============================================================

        # ---- 6a. 画所有追踪的物体 ----
        for tid, track in tracks.items():
            x1, y1, x2, y2 = track.last_box

            if track.alerted:
                color = (0, 0, 255)         # 红色 - 已告警
                status = f"ABANDONED! {track.class_name}"
            elif track.is_stationary:
                color = (0, 165, 255)       # 橙色 - 静止观察中
                status = f"{track.class_name} {track.abandon_timer:.1f}s"
            else:
                color = (0, 255, 255)       # 黄色 - 还在移动
                status = f"{track.class_name} moving"

            # cv2.rectangle 画矩形：图片，左上角，右下角，颜色，线宽
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # cv2.putText 写文字：图片，文字，位置，字体，大小，颜色，线宽
            cv2.putText(frame, status, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ---- 6b. 画人（绿色框） ----
        for (px1, py1, px2, py2) in person_boxes:
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (px1, py1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # ---- 6c. 叠加状态面板（方便你看到追踪器状态） ----
        # 统计各类追踪器数量
        total = len(tracks)
        moving_cnt = sum(1 for t in tracks.values() if not t.is_stationary and not t.alerted)
        stationary_cnt = sum(1 for t in tracks.values() if t.is_stationary and not t.alerted)
        alerted_cnt = sum(1 for t in tracks.values() if t.alerted)

        # 叠加黑色半透明背景做信息面板
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (360, 130), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # 状态面板内容
        info_y = 22
        cv2.putText(frame, f"追踪器: {total} 个", (12, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"  移动中:{moving_cnt}  静止:{stationary_cnt}  告警:{alerted_cnt}",
                   (12, info_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                   (255, 255, 255), 1)

        # 人检测状态（帮你判断"人离开画面"有没有被正确识别）
        if person_boxes:
            cv2.putText(frame, f"  人检测到: {len(person_boxes)} 个  掉帧: {person_miss_count}",
                       (12, info_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                       (0, 255, 0), 1)  # 绿色=有人
        else:
            cv2.putText(frame, f"  无人 (掉帧: {person_miss_count}/{PERSON_CLEAR_AFTER})",
                       (12, info_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                       (0, 0, 255), 1)  # 红色=无人

        cv2.putText(frame, "按 q 退出 | 按 r 重置",
                   (12, info_y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                   (200, 200, 200), 1)

        # ---- 6d. 显示画面 ----
        cv2.imshow("Abandoned Object Detection", frame)

        if writer:
            writer.write(frame)     # 写入视频文件

        # ---- 6d. 按键处理 ----
        # cv2.waitKey(1) 等待 1 毫秒，返回按下的键的 ASCII 码
        # & 0xFF 取低 8 位（兼容不同操作系统）
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):          # 按 q 退出
            break
        elif key == ord('r'):        # 按 r 重置所有追踪
            tracks.clear()
            next_id = 0
            print("追踪器已重置")

    # ============================================================
    # 7. 释放资源
    # ============================================================
    cap.release()                    # 关闭摄像头/视频
    if writer:
        writer.release()             # 关闭视频文件
    cv2.destroyAllWindows()          # 关闭所有 OpenCV 窗口
    print("程序已退出")


# ============================================================
# 8. 程序入口
# ============================================================
if __name__ == "__main__":
    """
    Python 特殊变量：__name__
    - 当文件直接运行时，__name__ == "__main__"
    - 当文件被 import 时，__name__ == 文件名（不会执行 main）
    这行代码确保 main() 只在直接运行脚本时执行
    """
    main()