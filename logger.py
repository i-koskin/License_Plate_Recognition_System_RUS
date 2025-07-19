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
    object_type: str,
    roi: Optional[np.ndarray] = None,
    save_full_frame: bool = False
) -> str:
    """
    Сохраняет кадр/ROI и логгирует событие.

    Args:
        frame (np.ndarray): Полный кадр с камеры.
        object_type (str): Тип обнаруженного объекта.
        roi (Optional[np.ndarray]): Область интереса (если есть).
        save_full_frame (bool): Сохранять ли весь кадр или только ROI.

    Returns:
        str: Путь к сохраненному изображению.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}_{object_type}.jpg"
    image_path = os.path.join(get_image_log_dir(), filename)

    if save_full_frame or roi is None:
        cv2.imwrite(image_path, frame)
    else:
        cv2.imwrite(image_path, roi)

    logger.info(f"Обнаружен объект: {object_type} | Сохранено: {image_path}")
    return filename
