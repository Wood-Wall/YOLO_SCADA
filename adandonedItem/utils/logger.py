"""
日志管理模块
============
统一管理控制台和文件日志输出。

功能：
  - 控制台彩色输出
  - 文件自动分割（RotatingFileHandler）
  - 分级输出（DEBUG / INFO / WARNING / ERROR）
  - 单例模式（只初始化一次）

用法:
    from utils.logger import LogManager

    LogManager.init(log_dir="logs", level="INFO")
    logger = LogManager.get_logger(__name__)

    logger.info("系统启动")
    logger.warning("配置文件不存在，使用默认值")
    logger.error("无法打开视频源")
"""
from __future__ import annotations
import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from typing import Optional


class _ColoredFormatter(logging.Formatter):
    """控制台彩色格式化器"""

    _COLORS = {
        logging.DEBUG: "\033[36m",       # 青色
        logging.INFO: "\033[32m",        # 绿色
        logging.WARNING: "\033[33m",     # 黄色
        logging.ERROR: "\033[31m",       # 红色
        logging.CRITICAL: "\033[1;31m",  # 亮红
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self._COLORS.get(record.levelno, self._RESET)
        record.levelname = f"{color}{record.levelname}{self._RESET}"
        return super().format(record)


class LogManager:
    """
    日志管理器（单例模式）。

    调用 init() 初始化一次，之后全局统一使用。
    """

    _initialized: bool = False
    _root_logger: logging.Logger = logging.getLogger()

    @classmethod
    def init(
        cls,
        log_dir: str = "logs",
        level: str = "INFO",
        max_bytes: int = 10 * 1024 * 1024,   # 10MB
        backup_count: int = 5,                # 保留 5 个备份
        console_level: str = "DEBUG",         # 控制台级别
        file_level: str = "DEBUG",            # 文件级别
    ) -> None:
        """
        初始化日志系统（只生效一次）。

        参数:
            log_dir:       日志文件存放目录（自动创建）
            level:         根记录器级别
            max_bytes:     每个日志文件最大字节数（默认 10MB）
            backup_count:  保留的备份文件数
            console_level: 控制台输出级别
            file_level:    文件输出级别
        """
        if cls._initialized:
            cls._root_logger.warning("日志系统已初始化，忽略重复调用。")
            return

        # ── 创建日志目录 ──
        os.makedirs(log_dir, exist_ok=True)

        # ── 根记录器 ──
        cls._root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        # 清除已有的 handlers（防止重复初始化）
        cls._root_logger.handlers.clear()

        # ── 文件格式（详细，带时间/级别/模块） ──
        file_fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # ── 控制台格式（简洁，彩色） ──
        console_fmt = _ColoredFormatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S",
        )

        # ── 文件 Handler (RotatingFileHandler → 自动分割) ──
        file_handler = RotatingFileHandler(
            filename=os.path.join(log_dir, "abandoned_detection.log"),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
        file_handler.setFormatter(file_fmt)

        # ── 控制台 Handler ──
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, console_level.upper(), logging.DEBUG))
        console_handler.setFormatter(console_fmt)

        # ── 注册 Handler ──
        cls._root_logger.addHandler(file_handler)
        cls._root_logger.addHandler(console_handler)

        cls._initialized = True

        cls._root_logger.info("=" * 50)
        cls._root_logger.info("日志系统初始化完成")
        cls._root_logger.info(f"  日志目录: {os.path.abspath(log_dir)}")
        cls._root_logger.info(f"  文件级别: {file_level.upper()} / 控制台级别: {console_level.upper()}")
        cls._root_logger.info(f"  文件分割: {max_bytes // 1024 // 1024}MB × {backup_count} 个备份")
        cls._root_logger.info("=" * 50)

    @classmethod
    def get_logger(cls, name: Optional[str] = None) -> logging.Logger:
        """
        获取指定名称的 logger。

        参数:
            name: 模块名称（通常是 __name__），为 None 时返回根记录器

        用法:
            logger = LogManager.get_logger(__name__)
        """
        if not cls._initialized:
            # 自动初始化（用默认参数）
            cls.init()
        return logging.getLogger(name)

    @classmethod
    def set_level(cls, level: str) -> None:
        """运行时动态修改日志级别"""
        level_num = getattr(logging, level.upper(), logging.INFO)
        cls._root_logger.setLevel(level_num)
        cls._root_logger.info(f"日志级别已切换为: {level.upper()}")