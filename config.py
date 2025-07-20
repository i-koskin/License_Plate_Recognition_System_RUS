"""
Модуль config.py

Загружает переменные окружения из .env файла и предоставляет
константы для использования в других частях проекта.
"""

import os

# Путь к конфигурационному файлу
CONFIG_PATH = "config.json"

# Путь к .pt модели YOLO
VEHICLE_MODEL_PATH = "yolov8s.pt"
PLATE_MODEL_PATH = "yolov8_plate.pt"

# Путь к шрифтам (кириллица)
FONT_PATH = "fonts/AutoNumber_Regular.ttf"

# RTSP URL для подключения к IP-камере
RTSP_URL: str | None = os.getenv("RTSP_URL")

# Целевые классы объектов, распознаваемых моделью
TARGET_CLASSES: dict[int, str] = {
    2: 'car',
    5: 'bus',
    7: 'truck'
}

SAVE_DIR = "results"

CONFIDENCE_THRESHOLD = 0.4

PLATE_LOG_INTERVAL = 60  # сек - повторно сохранить номер

PLATE_HOLD_TIME = 2.0  # сек – сохранить номер за SID, если он не менялся

SID_TTL = 3.0  # сек - SID «живёт» без bbox‑а
