import cv2
from datetime import datetime


def add_timestamp(frame, fmt="%d.%m.%Y %H:%M:%S"):
    """
    Рисует текущие дату/время в правом нижнем углу кадра.
    frame : BGR‑кадр (numpy array)
    fmt   : формат времени (strftime)
    """
    timestamp_text = datetime.now().strftime(fmt)

    # параметры шрифта
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    # вычисляем ширину‑высоту текста
    (text_w,  text_h), baseline = cv2.getTextSize(
        timestamp_text, font, font_scale, thickness)

    # позиция правый‑нижний угол (немного отступаем)
    x = frame.shape[1] - text_w - 15        # 15 px справа
    y = frame.shape[0] - 10             # 10 px снизу

    # рисуем текст
    cv2.putText(frame, timestamp_text, (x, y),
                font, font_scale, (255, 255, 255),
                thickness, cv2.LINE_AA)
    return frame
