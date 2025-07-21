import os
import cv2
import numpy as np
import logging
from typing import Optional
from datetime import datetime

from log_config import get_image_log_dir

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
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}_{object_type}.jpg"
    image_path = os.path.join(get_image_log_dir(), filename)

    cv2.imwrite(image_path, frame)

    logger.info(f"Обнаружен объект: {object_type} | Сохранено: {image_path}")
    return filename
