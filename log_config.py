import os
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from colorlog import ColoredFormatter

from config import CONFIG_PATH, SAVE_DIR


LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, f"{datetime.now():%Y-%m-%d}_log.log")

os.makedirs(f"{SAVE_DIR}/images", exist_ok=True)

IMAGE_DIR = os.path.join(SAVE_DIR, "images")

# Форматы логов
COLOR_FORMAT = "%(log_color)s[%(asctime)s] [%(levelname)s] %(message)s"
PLAIN_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def load_log_level() -> int:
    """Читает уровень логирования из config.json"""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            level = cfg.get("log_level", "INFO").upper()
            return getattr(logging, level, logging.INFO)
    except Exception:
        return logging.INFO


def setup_logging():
    """Настройка логирования с цветной консолью и ротацией файлов"""
    os.makedirs(LOG_DIR, exist_ok=True)
    # os.makedirs(IMAGE_DIR, exist_ok=True)

    log_level = load_log_level()
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Удалим старые обработчики
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    # Консоль с цветами
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        ColoredFormatter(COLOR_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(console_handler)

    # Ротация лог-файлов
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=2 * 1024 * 1024, backupCount=5, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(
        PLAIN_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(file_handler)

    logger.debug("✅ Логирование настроено на уровне %s",
                 logging.getLevelName(log_level))


def get_image_log_dir() -> str:
    """Возвращает путь к директории изображений логов"""
    return IMAGE_DIR
