import os
import time
import logging
import pandas as pd
from datetime import datetime
from typing import Dict

from config import PLATE_LOG_INTERVAL, SAVE_DIR

plate_log_times: Dict[str, float] = {}

logger = logging.getLogger(__name__)


def save_recognized_plate(plate_text: str, sid: int, source_label: str) -> None:
    """
    Сохраняет распознанный номерной знак в Excel-файл с разбивкой по дате.
    Добавляет только уникальные номера по интервалу времени.

    Args:
        plate_text (str): Распознанный номер.
        sid (int): Уникальный идентификатор отслеживания автомобиля (SID).
        source_label (str): Источник видео, например, "webcam" или имя файла
    """

    now = time.time()
    last_logged = plate_log_times.get(plate_text, 0)

    if now - last_logged < PLATE_LOG_INTERVAL:
        logger.info(f"[SKIP] Номер '{plate_text}' записан менее {PLATE_LOG_INTERVAL} секунд назад.")
        return

    plate_log_times[plate_text] = now

    recognized_path = os.path.join(SAVE_DIR, "recognized_plates.xlsx")

    # Создание строки для добавления
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "plate": plate_text,
        "sid": sid,
        "source": source_label,
    }

    # Чтение существующего файла, если есть
    if os.path.exists(recognized_path):
        try:
            df = pd.read_excel(recognized_path)
        except Exception as e:
            logger.error(f"[ERROR] Ошибка чтения {recognized_path}: {e}")
            df = pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    # Запись в файл
    try:
        df.to_excel(recognized_path, index=False)
        logger.info(f"[SAVE] Номер '{plate_text}' записан в {recognized_path}")
    except Exception as e:
        logger.error(
            f"[ERROR] Не удалось сохранить файл {recognized_path}: {e}")
