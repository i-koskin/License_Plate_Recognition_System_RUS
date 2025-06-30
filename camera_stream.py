import cv2
from config import RTSP_URL


def get_camera_stream() -> cv2.VideoCapture:
    """
    Подключается к видеопотоку по RTSP и возвращает объект VideoCapture.

    Returns:
        cv2.VideoCapture: Объект захвата видео для дальнейшей работы с потоком.

    Raises:
        RuntimeError: Если не удалось подключиться к камере.
    """
    # Создаем объект захвата видео с использованием RTSP URL
    cap = cv2.VideoCapture(RTSP_URL)

    # Проверяем, удалось ли подключиться к камере
    if not cap.isOpened():
        raise RuntimeError("❌ Не удалось подключиться к камере")

    return cap
