"""
主程序入口
==========
启动异物留置检测系统。

用法:
    python main.py                           # 默认（config.yaml）
    python main.py --video 0                # 本地摄像头
    python main.py --video test.mp4         # 视频文件
    python main.py --threshold 10 --save    # 10秒告警 + 保存结果

按键:
    q  — 退出
    r  — 重置所有追踪器
"""
from __future__ import annotations
import argparse
import sys
import cv2

from config import AppConfig
from core.engine import AbandonedDetectorEngine
from core.alert import AlertInfo
from utils.video import VideoSourceFactory
from utils.visualization import Visualizer


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="异物留置检测系统 (Abandoned Object Detection)"
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help='视频源（默认用config.yaml，填"0"=摄像头，填路径=视频文件）'
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="配置文件路径（默认: config.yaml）"
    )
    parser.add_argument(
        "--threshold", type=int, default=None,
        help="告警阈值（秒），会覆盖配置文件"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="保存结果视频到 output.avi"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="YOLO 模型路径，会覆盖配置文件"
    )
    return parser.parse_args()


def print_startup_info(config: AppConfig):
    """打印启动信息"""
    print("=" * 50)
    print("  异物留置检测系统启动")
    print("=" * 50)
    print(f"  模型: {config.model_path}")
    print(f"  视频源: {config.video_source}")
    print(f"  告警阈值: {config.abandon_seconds}s")
    print(f"  保存视频: {config.save_result}")
    print(f"  检测间隔: 每 {config.detect_interval} 帧")
    print("-" * 50)
    print("  按键:  q=退出  r=重置追踪器")
    print("=" * 50)
    print()


def main():
    """主函数"""
    args = parse_args()

    # ── 加载配置 ──
    config = AppConfig.from_yaml(args.config)

    # 命令行参数覆盖配置
    if args.video is not None:
        config.video_source = args.video
    if args.threshold is not None:
        config.abandon_seconds = args.threshold
    if args.save:
        config.save_result = True
    if args.model is not None:
        config.model_path = args.model

    # 打印启动信息
    print_startup_info(config)

    # ── 创建视频源 ──
    video_source = VideoSourceFactory.create(config.video_source)
    if not video_source._open():
        print(f"❌ 无法打开视频源: {config.video_source}")
        sys.exit(1)

    print(f"  视频信息: {video_source.frame_size[0]}×{video_source.frame_size[1]}, "
          f"{video_source.fps:.1f} FPS")
    if video_source.total_frames > 0:
        print(f"  总帧数: {video_source.total_frames}")
    print()

    # ── 创建引擎 ──
    engine = AbandonedDetectorEngine(config)

    # 注册告警回调 → 打印到终端
    def on_alert(alert: AlertInfo):
        print(f"⚠️ 告警: {alert.message}")
    engine.alert_manager.attach(on_alert)

    # ── 视频写入器（如果需要保存） ──
    writer = None
    if config.save_result:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(
            config.output_path, fourcc,
            video_source.fps,
            video_source.frame_size
        )
        print(f"  结果将保存到: {config.output_path}")
        print()

    # ── 帧间隔控制（跳帧检测，但逐帧显示） ──
    detect_counter = 0

    # ════════════════════════════════════════════
    # 主循环
    # ════════════════════════════════════════════
    try:
        while True:
            ret, frame = video_source.read()
            if not ret:
                print("视频源已断开，退出。")
                break

            # ── 检测（间隔执行，提高性能） ──
            detect_counter += 1
            if detect_counter % config.detect_interval == 0:
                result = engine.process_frame(frame)
            else:
                # 不跑 YOLO，只更新 FPS 统计
                engine._frame_count += 1
                engine._frame_times.append(time.time())
                if len(engine._frame_times) > 30:
                    engine._frame_times.pop(0)
                result = None

            # ── 画面绘制（每一帧都画） ──
            vis = Visualizer.draw_all(frame, engine)

            # ── 保存 ──
            if writer:
                writer.write(vis)

            # ── 显示 ──
            cv2.imshow("Abandoned Object Detection", vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("用户退出。")
                break
            elif key == ord('r'):
                engine.tracker.tracks.clear()
                engine.alert_manager.clear_all()
                print("已重置所有追踪器。")

    except KeyboardInterrupt:
        print("用户中断。")
    finally:
        # ── 清理资源 ──
        video_source.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("系统已安全关闭。")


if __name__ == "__main__":
    main()