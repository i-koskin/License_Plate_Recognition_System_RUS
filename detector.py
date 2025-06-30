from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
import logging
from config import TARGET_CLASSES, CLASS_COLOR_PALETTE, CONFIDENCE_THRESHOLD

# Настройка logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)


def get_class_color(label: str) -> Tuple[int, int, int]:
    """
    Возвращает фиксированный цвет из палитры для заданного класса по его имени.

    Args:
        label (str): Название класса.

    Returns:
        Tuple[int, int, int]: Цвет в формате BGR.
    """
    if label.lower() in CLASS_COLOR_PALETTE:
        return CLASS_COLOR_PALETTE[label.lower()]


def draw_legend(frame: np.ndarray, class_counts: Dict[str, int]) -> None:
    """
    Отрисовывает легенду классов в левом нижнем углу кадра с полупрозрачной подложкой
    и счётчиком объектов, отсортированным по убыванию количества.

    Args:
        frame (np.ndarray): Кадр изображения.
        class_counts (Dict[str, int]): Словарь с количеством объектов по каждому классу.
    """
    height, width = frame.shape[:2]
    box_size = 20
    padding = 5
    font_scale = 0.7
    font_thickness = 2

    # Сортировка по количеству объектов (по убыванию)
    sorted_items = sorted(class_counts.items(),
                          key=lambda item: item[1], reverse=True)
    if not sorted_items:
        return

    total_height = len(sorted_items) * (box_size + padding) + padding
    max_text_width = max(
        cv2.getTextSize(
            f"{label} [{count}]", cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0][0]
        for label, count in sorted_items
    )
    legend_width = 10 + box_size + 10 + max_text_width + 10

    legend_x = 10
    legend_y = height - total_height - 10

    # Создание полупрозрачной подложки
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (legend_x - 5, legend_y - 5),
        (legend_x + legend_width, legend_y + total_height),
        (0, 0, 0),
        thickness=-1
    )
    alpha = 0.5
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Отрисовка квадратов и текста
    for i, (label, count) in enumerate(sorted_items):
        color = get_class_color(label)
        y_offset = legend_y + i * (box_size + padding)

        # Цветной квадрат
        cv2.rectangle(frame, (legend_x, y_offset),
                      (legend_x + box_size, y_offset + box_size),
                      color, -1)

        # Подпись с количеством
        text = f"{label} [{count}]"
        cv2.putText(frame, text,
                    (legend_x + box_size + 10, y_offset + box_size - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                    cv2.LINE_AA)


class ObjectDetector:
    """
    Детектор объектов на изображениях с использованием модели YOLO.
    """

    def __init__(self, model_path: str):
        """
        Инициализация модели YOLO.

        Args:
            model_path (str): Путь к весам модели YOLO (.pt).
        """
        self.model = YOLO(model_path)
        self.target_classes = TARGET_CLASSES
        self.conf_threshold = CONFIDENCE_THRESHOLD

    def detect(self, frame: np.ndarray, device: str) -> Tuple[np.ndarray, List[Dict[str, str | float | Tuple[int, int, int, int]]]]:
        """
        Выполняет детекцию объектов, фильтрует по классу и порогу уверенности, и опционально рисует боксы и легенду.

        Args:
            frame (np.ndarray): Кадр для анализа.
            draw (bool): Отрисовывать ли боксы и легенду на кадре.

        Returns:
            Tuple[np.ndarray, List[Dict]]: Кадр с отрисовкой и список найденных объектов.
        """
        results = self.model(frame, device=device, verbose=False)[0]
        detections: List[Dict[str, str | float |
                              Tuple[int, int, int, int]]] = []
        class_counts: Dict[str, int] = defaultdict(int)

        # Обработка каждого найденного объекта
        for box in results.boxes:
            cls = int(box.cls[0])
            label = self.model.names[cls]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label in self.target_classes and conf >= self.conf_threshold:
                detections.append({
                    "label": label,
                    "roi": (x1, y1, x2, y2),
                    "conf": conf
                })
                class_counts[label] += 1

        # Добавление легенды
        if class_counts:
            draw_legend(frame, class_counts)

        return frame, detections
