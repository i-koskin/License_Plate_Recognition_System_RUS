from ultralytics import YOLO
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
import logging

from config import CONFIDENCE_THRESHOLD

# Настройка logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)


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

            if conf >= self.conf_threshold:
                detections.append({
                    "label": label,
                    "roi": (x1, y1, x2, y2),
                    "conf": box.conf,
                    "cls": box.cls,
                    "boxes_xyxy": box.xyxy
                })
                # class_counts[label] += 1

        return frame, detections
