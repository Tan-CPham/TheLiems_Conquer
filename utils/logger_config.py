import sys
import os
from loguru import logger

# Xóa handler mặc định
logger.remove()

# Cấu hình log ra console
logger.add(
    sys.stderr,
    level="DEBUG",
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True
)

log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "app.log")

logger.add(
    log_file_path,
    level="DEBUG",
    rotation="10 MB", # Xoay vòng file log khi đạt 10MB
    retention="5 days", # Giữ log trong 5 ngày
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    encoding="utf-8"
)
