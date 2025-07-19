import re
from paddleocr import PaddleOCR
import numpy as np
import cv2
from typing import Union
from PIL import ImageFont, ImageDraw, Image

from config import FONT_PATH


class PlateRecognizer:
    """
    Класс для распознавания автомобильных номеров с помощью PaddleOCR.
    """

    def __init__(self, use_gpu=True):
        """
    Инициализирует модуль OCR с классификацией угла поворота.
    """
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=use_gpu,
            gpu_mem=4000,
            gpu_id=0
        )

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
        target_positions_number = [1, 2, 3, 6, 7, 8]
        target_positions_liter = [0, 4, 5]

        for pos in target_positions_liter:
            if pos < len(plate):
                char = plate[pos]
                # Заменяем 'V' на 'Y'
                if char in ('V'):
                    replacements = {'V': 'Y'}
                    plate[pos] = replacements[char]

        for pos in target_positions_number:
            if pos < len(plate):
                char = plate[pos]
                # Заменяем 'O' (английскую) на '0', 'I' -> '1'
                if char in ('O', 'I'):
                    replacements = {'O': '0', 'I': '1'}
                    plate[pos] = replacements[char]

        return ''.join(plate)

    def normalize_plate_text(self, text):
        LATIN_TO_CYRILLIC = {
            "A": "А", "B": "В", "E": "Е", "K": "К", "M": "М",
            "H": "Н", "O": "О", "P": "Р", "C": "С", "T": "Т",
            "Y": "У", "X": "Х",
            "a": "А", "b": "В", "e": "Е", "k": "К", "m": "М",
            "h": "Н", "o": "О", "p": "Р", "c": "С", "t": "Т",
            "y": "У", "x": "Х"
        }

        return "".join(LATIN_TO_CYRILLIC.get(c, c) for c in text)

    def is_license_plate(self, text: str) -> bool:
        """
        Проверяет, соответствует ли строка формату российского номерного знака.

        Args:
            text (str): Строка, которую нужно проверить.

        Returns:
            bool: True, если строка соответствует формату номера, иначе False.
        """
        LICENSE_PATTERN = re.compile(
            r"^[А-Я]{1}\d{3}[А-Я]{2}\d{2,3}$", re.IGNORECASE)

        # Удаляем пробелы и применяем регулярное выражение для шаблона номера
        text = text.upper().replace(" ", "").replace("-", "")
        plate_text = self.normalize_plate_text(text)
        is_valid = bool(LICENSE_PATTERN.match(plate_text))

        return is_valid, plate_text

    def draw_text_cyrillic(self, img_bgr,
                           text,
                           position,
                           font_size=30,
                           text_color=(0, 0, 0),
                           bg_color=(255, 255, 255),
                           padding=4):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(img_pil)

        try:
            font = ImageFont.truetype(FONT_PATH, font_size)
        except:
            font = ImageFont.load_default()

        # Размер текста
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        x, y = position
        box_x1 = x - padding
        box_y1 = y - padding
        box_x2 = x + text_width + padding
        box_y2 = y + text_height + padding

        # Рисуем белый прямоугольник
        draw.rectangle([(box_x1, box_y1), (box_x2, box_y2)], fill=bg_color)

        # Рисуем текст
        draw.text((x, y), text, font=font, fill=text_color)

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # def preprocess_roi(self, roi):
        # # Увеличение контраста
        # lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        # l, a, b = cv2.split(lab)
        # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        # cl = clahe.apply(l)
        # limg = cv2.merge((cl, a, b))
        # enhanced_roi = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # # Бинаризация
        # gray = cv2.cvtColor(enhanced_roi, cv2.COLOR_BGR2GRAY)

        # _, binary = cv2.threshold(
        #     gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # # Уменьшение шума
        # denoised = cv2.fastNlMeansDenoising(binary, h=10)

        # h, w = denoised.shape[:2]

        # return cv2.resize(denoised, (int(w * 2), int(h * 2)), interpolation=cv2.INTER_CUBIC)

        # Контрастирование (CLAHE)
        # lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        # l, a, b = cv2.split(lab)
        # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        # l = clahe.apply(l)
        # lab = cv2.merge((l, a, b))
        # image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # # Бинаризация
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                cv2.THRESH_BINARY_INV, 11, 2)
        # return thresh


# def upscale(img, scale=2.0):
#     h, w = img.shape[:2]
#     return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    def recognize(self, roi: Union[np.ndarray, str]) -> str:
        """
        Выполняет распознавание номера на переданном изображении (ROI).

        Args:
            roi (np.ndarray | str): Область изображения, содержащая номер (или путь к изображению).

        Returns:
            str: Распознанный и откорректированный номерной знак, либо пустая строка.
        """
        # denoised = self.preprocess_roi(roi)

        result = self.ocr.ocr(roi, cls=False)

        # Проверяем, есть ли результат и текст
        if result and result[0]:
            # Извлекаем текст из результата OCR
            plate_raw = result[0][0][1][0]
            plate = self.correct_plate_number(plate_raw)

            valid, normalized_plate = self.is_license_plate(plate)
            if valid:
                return normalized_plate

        return ""
