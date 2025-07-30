import os
import cv2
import numpy as np
import logging
from datetime import datetime
from pathlib import Path

from config import SAVE_DIR

logger = logging.getLogger(__name__)


def log_detection(
    frame: np.ndarray,
    object_type: str
) -> str:
    """
    Сохраняет кадр и логгирует событие.

    Args:
        frame (np.ndarray): Полный кадр с камеры.
        object_type (str): Тип обнаруженного объекта.
        
    Returns:
        str: Путь к сохраненному изображению.
    """
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    date_dir = Path(f"{SAVE_DIR}/images/{current_date_str}")
    os.makedirs(date_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    image_path = date_dir/f"{timestamp}_{object_type}.jpg"

    cv2.imwrite(str(image_path), frame)

    logger.info(f"Обнаружен объект: {object_type} | Сохранено: {image_path}")
    return image_path
