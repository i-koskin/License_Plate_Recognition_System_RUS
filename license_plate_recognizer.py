import re
from paddleocr import PaddleOCR
import numpy as np
from typing import Union


class PlateRecognizer:
    """
    Класс для распознавания автомобильных номеров с помощью PaddleOCR.
    """

    def __init__(self):
        """
        Инициализирует модуль OCR с классификацией угла поворота.
        """
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)

    def correct_plate_number(self, plate: str) -> str:
        """
        Корректирует возможные ошибки OCR, заменяя букву 'O' на цифру '0' в позициях, где должны быть цифры.

        Args:
            plate (str): Распознанная строка номера.

        Returns:
            str: Скорректированная строка номера.
        """
        plate = plate.strip()

        # Приводим к верхнему регистру и разбиваем на символы
        plate = list(plate.upper())

        # Индексы предполагаемых цифровых позиций (по ГОСТу): 2,3,4,7,8,9
        target_positions = [1, 2, 3, 6, 7, 8]

        for pos in target_positions:
            if pos < len(plate):
                char = plate[pos]
                # Заменяем 'O' (английскую) на '0'
                if char in ('O'):
                    plate[pos] = '0'

        return ''.join(plate)

    def is_license_plate(self, text: str) -> bool:
        """
        Проверяет, соответствует ли строка формату российского номерного знака.

        Args:
            text (str): Строка, которую нужно проверить.

        Returns:
            bool: True, если строка соответствует формату номера, иначе False.
        """
        # Удаляем пробелы и применяем регулярное выражение для шаблона номера
        text = text.upper().replace(" ", "")
        match = re.search(
            r'^[A-Z]{1}\d{3}[A-Z]{2}\d{2,3}$', text)
        return match.group(0) if match else None

    def recognize(self, roi: Union[np.ndarray, str]) -> str:
        """
        Выполняет распознавание номера на переданном изображении (ROI).

        Args:
            roi (np.ndarray | str): Область изображения, содержащая номер (или путь к изображению).

        Returns:
            str: Распознанный и откорректированный номерной знак, либо пустая строка.
        """

        result = self.ocr.ocr(roi, cls=True)

        # Проверяем, есть ли результат и текст
        if result and result[0]:
            # Извлекаем текст из результата OCR
            plate_raw = result[0][0][1][0]
            plate = self.correct_plate_number(plate_raw)

            if self.is_license_plate(plate):
                return plate

        return ""
