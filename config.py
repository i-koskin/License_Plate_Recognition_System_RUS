"""
Модуль config.py

Загружает переменные окружения из .env файла и предоставляет
константы для использования в других частях проекта.
"""

import os
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# Путь к конфигурационному файлу
CONFIG_PATH = "config.json"

# Путь к .pt модели YOLO
MODEL_PATH = "yolo_weights/yolov8s.pt"

# RTSP URL для подключения к IP-камере
RTSP_URL: str | None = os.getenv("RTSP_URL")

# Целевые классы объектов, распознаваемых моделью
TARGET_CLASSES: list[str] = [
    'car',
    'motorcycle',
    'bus',
    'truck'
]

# Фиксированная палитра для классов (BGR)
CLASS_COLOR_PALETTE = {
    "car":           (0, 255, 0),       # Зеленый
    'motorcycle':    (255, 0, 255),     # Розовый
    "bus":           (0, 255, 255),     # Желтый
    "truck":         (0, 0, 255),       # Красный
}

# Классы, по которым может происходить распознавание номеров
LICENSE_PLATE_KEYWORDS: list[str] = [
    'car', 'motorcycle', 'bus', 'truck']

# Минимальный порог уверенности для отбора объектов
CONFIDENCE_THRESHOLD = 0.5
