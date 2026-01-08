import logging
import sys
import os

def setup_logger(log_dir="logs", log_filename="simulation.log", level=logging.DEBUG, console_level=logging.INFO):
    """
    配置全局日志记录器。
    
    Args:
        log_dir (str): 日志文件目录。
        log_filename (str): 日志文件名。
        level (int): 文件日志级别 (e.g., logging.DEBUG, logging.INFO)。
        console_level (int): 控制台日志级别，默认为 INFO。
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_path = os.path.join(log_dir, log_filename)
    
    # 获取 root logger
    logger = logging.getLogger()
    # 设置为最低级别，以便 handlers 可以自行过滤
    logger.setLevel(min(level, console_level))
    
    # 清除旧的 handlers 避免重复
    if logger.handlers:
        logger.handlers = []
    
    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 1. 文件 Handler
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 2. 控制台 Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level) 
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logging.info(f"Logger initialized. Log file: {log_path}")

def get_logger(name):
    return logging.getLogger(name)
