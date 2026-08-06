"""
视频抽帧工具 - 每隔 N 帧保存一张图片
用法: python extract_frames.py <视频路径> [--interval 帧间隔] [--output 输出目录]

"""

# ========== 导入模块（import 语法）==========
# import 模块名         → 导入整个模块，用 模块名.函数名() 调用
# from 模块名 import 东西  → 只导入指定内容，可以直接用函数名()
import argparse   # argparse: 解析命令行参数的官方库
import os         # os: 和操作系统交互（文件路径、目录操作等）
import cv2        # cv2 (OpenCV): 计算机视觉库，处理图像和视频


def main():
    """
    def 函数名(参数1, 参数2=默认值):
        函数体
        return 返回值

    这是自定义函数，把主要逻辑包在函数里是 Python 的惯例写法。
    好处：变量不会污染全局，代码可复用。
    """

    # ===== argparse 用法 =====
    # 作用：让用户可以在命令行传参，比手动改代码方便得多
    #
    # 步骤:
    #   1. 创建解析器: ArgumentParser(描述文字)
    #   2. 添加参数:   add_argument(参数名, type=类型, default=默认值, help=说明)
    #      - 不加 -- 前缀 → 位置参数（必须按顺序传）
    #      - 加 -- 前缀   → 可选参数（不传就用 default）
    #   3. 解析参数:   parse_args() 返回一个对象，用 args.参数名 取值
    parser = argparse.ArgumentParser(description="视频抽帧")

    # 位置参数 - 必填，运行时直接提供
    parser.add_argument("--video_path", default= "D:/CodeExa/langchain/YOLO/test.mp4", help="输入视频路径")

    # 可选参数 - 带默认值，不传就用默认的
    # type=int 表示把传入的字符串转成整数（默认是字符串）
    parser.add_argument("--interval", type=int, default=30,
                        help="每隔多少帧抽取一张（默认 30，即每秒1张@30fps）")
    parser.add_argument("--output", default="frames",
                        help="输出目录（默认 ./frames）")

    # 解析！此时用户输入的参数已变成 args 的属性
    args = parser.parse_args()

    # ===== f-string 字符串格式化 =====
    # Python 3.6+ 特性，在字符串前加 f，可以用 {} 嵌入变量/表达式
    # 写法: f"文字 {变量名} 文字 {表达式}"
    # 对比旧写法: "文字 " + 变量名 + " 文字"  —— 又长又容易错

    # os.path.exists(路径) → 判断文件或目录是否存在，返回 True/False
    if not os.path.exists(args.video_path):
        print(f"[错误] 视频文件不存在: {args.video_path}")
        return  # return 提前结束函数，不执行后面代码

    # os.makedirs(路径, exist_ok=True)
    # → 递归创建目录，exist_ok=True 表示目录已存在也不报错（类似 mkdir -p）
    os.makedirs(args.output, exist_ok=True)

    # ===== OpenCV 读取视频 =====
    # cv2.VideoCapture(路径) → 打开视频文件，返回一个"视频对象"
    # 可以理解为把视频文件"打开"了，一帧一帧取画面
    cap = cv2.VideoCapture(args.video_path)

    # cap.isOpened() → 检查视频是否成功打开，返回 True/False
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {args.video_path}")
        return

    # ===== 获取视频属性 =====
    # cap.get(cv2.CAP_PROP_XXX) → 读取视频的元信息
    # cv2.CAP_PROP_FRAME_COUNT → 视频总帧数
    # cv2.CAP_PROP_FPS         → 帧率（每秒多少帧）
    # int() → 把浮点数转成整数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # fps 是浮点数，保留小数给用户看

    print(f"视频信息: {total_frames} 帧, {fps:.2f} fps")
    #                    ↑ {表达式:.2f} 表示浮点数保留2位小数
    print(f"抽帧间隔: 每 {args.interval} 帧")
    print(f"输出目录: {args.output}\n")
    # \n 是转义字符，表示换行

    # ===== 计数器变量 =====
    # Python 是动态类型，不用声明类型直接赋值
    # 变量名 = 值  即可
    frame_idx = 0  # 当前读到第几帧（从 0 开始计数）
    saved = 0      # 已经保存了多少张图片

    # ===== while 循环 =====
    # while 条件:
    #     循环体
    # 条件为 True 就一直循环，遇到 break 跳出
    #
    # Python 用 缩进（4空格） 表示代码块，不用 {} 或 end
    # 缩进不对会报错！
    while True:
        # cap.read() → 读取下一帧
        # 返回值是两个:
        #   ret   → True/False，True 表示成功读到一帧
        #   frame → 图片数据（三维数组: 高×宽×颜色通道）
        #
        # Python 的"多重赋值"语法：函数可以返回多个值，一次性拆包到多个变量
        ret, frame = cap.read()

        # not ret → 没读到帧了（视频播放完毕）
        # if 条件后面不用加括号 ( )
        # 用冒号 : 结尾
        if not ret:
            break  # break → 跳出当前循环，执行循环后面的代码

        # ===== 取模运算 =====
        # % 是取余数运算符
        # frame_idx % interval → 每 interval 帧余数为 0
        # 例: interval=30, 第0,30,60,90...帧的条件成立
        if frame_idx % args.interval == 0:
            # ===== f-string 高级用法 =====
            # f"frame_{saved:06d}.jpg"
            #     ↑             ↑
            #     变量名    {:06d} 格式说明
            #     :06d → 整数(d)，占6位，不够的前面补0
            #     例: saved=3  → "frame_000003.jpg"
            filename = f"frame_{saved:06d}.jpg"

            # os.path.join(目录, 文件名) → 拼接文件路径
            # 自动处理操作系统路径分隔符（Windows用\，Linux用/）
            filepath = os.path.join(args.output, filename)

            # cv2.imwrite(路径, 图片数据) → 把图片保存到文件
            # 支持 jpg/png 等格式，由文件扩展名自动判断
            cv2.imwrite(filepath, frame)

            saved += 1  # saved = saved + 1 的简写

            # print(end="\r") → 不换行，回到行首
            # 实现"原地刷新"效果，不会刷屏
            print(f"  [{saved}] 保存 {filename}", end="\r")

        frame_idx += 1  # 处理完一帧，计数器 +1

    # ===== 善后清理 =====
    # cap.release() → 释放视频文件，类似"关闭文件"
    cap.release()

    # 最后打印结果
    print(f"\n\n完成！共保存 {saved} 张图片到 {args.output}/")


# ===== Python 程序的入口惯例 =====
# __name__ 是 Python 内置变量:
#   - 当脚本被直接运行时，__name__ 等于 "__main__"
#   - 当脚本被 import 导入时，__name__ 等于文件名（不会自动执行）
#
# 这个 if 判断确保: 只有直接运行本文件时才执行 main()
# 如果是被别人 import，不会自动跑，防止副作用
if __name__ == "__main__":
    main()