import cv2
from pathlib import Path
from datetime import datetime
from typing import Tuple

from config import SAVE_DIR

def create_video_writer(
    frame_shape: Tuple[int, int],
    source_label: str,
    fps: float = 25.0,
    ext: str = ".avi"
) -> cv2.VideoWriter:
    """
    Создаёт объект cv2.VideoWriter для записи видео.

    :param frame_shape: Кортеж (высота, ширина) кадра.
    :param source_label: Метка источника (например, 'webcam', 'ipcam', видеофайл).
    :param fps: Частота кадров (по умолчанию 25.0).
    :param ext: Расширение файла ('.avi', '.mp4', '.mkv').
    :return: Объект VideoWriter.
    """
    h = int(frame_shape[0])  # Высота кадра
    w = int(frame_shape[1])  # Ширина кадра

    # Выбор кодека в зависимости от расширения
    ext = ext.lower()
    if ext == ".mp4":
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif ext == ".mkv":
        fourcc = cv2.VideoWriter_fourcc(*'X264')
    else:  # по умолчанию AVI
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        ext = ".avi"  # гарантировать корректность

    # Имя файла с меткой времени
    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"{now}_{source_label}{ext}"

    # Создание папки и объекта VideoWriter

    output_path = Path(SAVE_DIR)
    output_path.mkdir(exist_ok=True)

    return cv2.VideoWriter(str(output_path/filename), fourcc, fps, (w, h))
