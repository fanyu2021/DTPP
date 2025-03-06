'''
Copyright (c) 2025 by GAC R&D Center, All Rights Reserved.
Author: 范雨
Date: 2025-03-05 11:52:55
LastEditTime: 2025-03-06 11:25:49
LastEditors: 范雨
Description: 
'''
import logging

class ColoredFormatter(logging.Formatter):
    """自定义带颜色的日志格式化器"""
    COLOR_CODES = {
        logging.DEBUG: "\033[36m",    # 青色
        logging.INFO: "\033[32m",     # 绿色
        logging.WARNING: "\033[33m",  # 黄色
        logging.ERROR: "\033[31m",    # 红色
        logging.CRITICAL: "\033[31;1m" # 红色加粗
    }
    RESET_CODE = "\033[0m"

    def format(self, record):
        color = self.COLOR_CODES.get(record.levelno, "")
        message = super().format(record)
        return f"{color}{message}{self.RESET_CODE}"

def create_colored_logger(name=__name__):
    """创建带颜色输出的logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # # 移除所有现有handler防止重复
    # if logger.handlers:
    #     logger.handlers = []

    # 控制台handler
    console_handler = logging.StreamHandler()
    # console_handler.setFormatter(ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    console_handler.setFormatter(ColoredFormatter("%(asctime)s %(name)s-[%(levelname)s] %(message)s"))
    logger.addHandler(console_handler)
    # logger.propagate = False  # 阻止传播到根logger
    return logger

logger = create_colored_logger()
