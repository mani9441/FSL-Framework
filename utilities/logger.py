"""
Logging System for FSL Research Framework.
Provides structured console and file logging with automatic log rotation.
"""

import sys
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

from utilities.constants import LOGS_DIR

_LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# Global flag to track initialized loggers
_INITIALIZED_LOGGERS = set()


def setup_logging(
    log_file_name: str = "experiment.log",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> None:
    """
    Configures root logger with console and rotating file handlers.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file_path = LOGS_DIR / log_file_name

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if already configured
    if root_logger.handlers:
        return

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Rotating File Handler
    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def get_logger(name: str, log_level: Optional[int] = None) -> logging.Logger:
    """
    Returns a configured logger instance for the given component module.
    """
    # Ensure base logging is set up
    if not logging.getLogger().handlers:
        setup_logging()

    logger = logging.getLogger(name)
    if log_level is not None:
        logger.setLevel(log_level)
    return logger
