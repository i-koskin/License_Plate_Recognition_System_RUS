import cv2
import time
import json
import logging
import torch
from pathlib import Path
from datetime import datetime

from detector import ObjectDetector, get_class_color
from license_plate_recognizer import PlateRecognizer
from video_writer import create_video_writer
from logger import log_detection
from log_config import setup_logging
from config import MODEL_PATH, LICENSE_PLATE_KEYWORDS, CONFIDENCE_THRESHOLD

# Чтение конфигурационного файла
with open("config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

save_video = cfg.get("save_video", False)
recording_interval_minutes = max(
    int(cfg.get("recording_interval_minutes", 60)), 1)
recording_interval_seconds = recording_interval_minutes * 60
save_full_frame = cfg.get("save_full_frame", False)
frame_skip = cfg.get("frame_skip", 5)
device = "cuda" if torch.cuda.is_available() else "cpu"
video_source = cfg.get("video_source", "0")
if video_source == "0":
    video_source = 0  # Преобразовать строку в число для cv2

if str(video_source) == "0":
    source_label = "webcam"
elif str(video_source).startswith("rtsp"):
    source_label = "ipcam"
else:
    source_label = Path(str(video_source)).stem

# Запуск логирования
setup_logging()

logger = logging.getLogger(__name__)
logger.info("🚀 Приложение запущено")


def main():
    cap = cv2.VideoCapture(video_source)
    detector = ObjectDetector(MODEL_PATH)
    plate_reader = PlateRecognizer()

    cv2.namedWindow("YOLOv8 Detection", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("YOLOv8 Detection",
                          cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    logging.info("Система запущена")

    if not cap.isOpened():
        logging.warning("❌ Не удалось открыть источник видео")
        exit()

    print(f"📡 Источник: {video_source}")
    print(f"🧠 Устройство: {device.upper()}")

    is_file_source = isinstance(
        video_source, str) and video_source.lower().endswith((".avi", ".mp4", ".mkv"))

    # Инициализация записи видео
    if save_video:
        ret, frame = cap.read()
        if not ret:
            logging.error("❌ Не удалось получить первый кадр")
            return

        video_writer = create_video_writer(frame.shape, source_label)
        start_record_time = time.time()

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if is_file_source:
                logging.info("✅ Обработка файла завершена.")
                break  # Завершаем цикл для файлов
            else:
                logging.warning("🔁 Повторное подключение к потоку...")
                time.sleep(1)
                cap.release()
                cap = cv2.VideoCapture(video_source)
                continue

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame, detections = detector.detect(frame, device)

        for det in detections:
            x1, y1, x2, y2 = det['roi']
            label = det['label']
            conf = det['conf']
            roi = frame[int(y1):int(y2), int(x1):int(x2)]

            if roi.size == 0:
                continue

            color = get_class_color(label)

            if conf < CONFIDENCE_THRESHOLD:
                continue

            if any(k in label.lower() for k in LICENSE_PLATE_KEYWORDS):
                try:
                    plate = plate_reader.recognize(roi)
                    if plate:
                        label += f" [{plate}]"
                except Exception as e:
                    logging.warning(f"Ошибка распознавания номера: {e}")

            if label:
                log_detection(frame, label, roi, save_full_frame)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        # Добавление даты и времени
        now = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        text_size, _ = cv2.getTextSize(now, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_w, text_h = text_size
        x = frame.shape[1] - text_w - 15
        y = frame.shape[0] - 10
        cv2.putText(frame, now, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)

        if save_video:
            # Запись обработанного кадра
            video_writer.write(frame)

            # Перезапуск записи по времени
            if time.time() - start_record_time > recording_interval_seconds:
                video_writer.release()
                video_writer = create_video_writer(frame.shape, source_label)
                start_record_time = time.time()

        cv2.imshow("YOLOv8 Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:  # Остановка по клавишам "q" или "Esc"
            logging.info("🛠 Принудительная остановка пользователем")
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    logging.info("🛑 Захват остановлен. Файлы закрыты")


if __name__ == "__main__":
    is_file_source = isinstance(
        video_source, str) and video_source.lower().endswith((".avi", ".mp4", ".mkv"))

    while True:
        try:
            main()
            if is_file_source:
                logging.info("✅ Обработка видеофайла завершена")
                break
            else:
                logging.info("♻️ Перезапуск потока...")
        except KeyboardInterrupt:
            logging.info("Завершение по Ctrl+C")
            break
        except Exception as e:
            logging.exception(
                "❌ Критическая ошибка! Перезапуск через 5 сек...")
            time.sleep(5)
