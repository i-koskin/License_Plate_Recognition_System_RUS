import cv2
import time
import pandas as pd
import json
import logging
import torch
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
from supervision import Detections, ByteTrack

from logger import log_detection
from log_config import setup_logging
from video_writer import create_video_writer
from add_timestamp import add_timestamp
from license_plate_recognizer import PlateRecognizer
from plate_assignment import assign_plates_to_vehicles

from config import VEHICLE_MODEL_PATH, PLATE_MODEL_PATH, SAVE_DIR, TARGET_CLASSES, PLATE_LOG_INTERVAL, PLATE_HOLD_TIME, SID_TTL

# ---------------- CONFIG ----------------
with open("config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

save_video = cfg.get("save_video", False)
recording_interval_minutes = max(
    int(cfg.get("recording_interval_minutes", 60)), 1)
recording_interval_seconds = recording_interval_minutes * 60
frame_skip = cfg.get("frame_skip", 5)
device = "cuda" if torch.cuda.is_available() else "cpu"
video_source = cfg.get("video_source", "0")
video_source = 0 if video_source == "0" else video_source
source_label = "webcam" if str(video_source) == "0" else "ipcam" if str(
    video_source).startswith("rtsp") else Path(str(video_source)).stem

stable_boxes = {}
sid_last_seen = {}
plate_by_sid = defaultdict(lambda: "")
plate_to_sid = {}
sid_last_plate_time = defaultdict(lambda: 0)
recognized_plates = []
plate_last_log = defaultdict(lambda: 0)  # {plate_number: timestamp}


setup_logging()
logger = logging.getLogger(__name__)
logger.info("🚀 Приложение запущено")


def main():
    cap = cv2.VideoCapture(video_source)
    vehicle_model = YOLO(VEHICLE_MODEL_PATH)
    plate_model = YOLO(PLATE_MODEL_PATH)
    plate_reader = PlateRecognizer()
    tracker = ByteTrack()

    cv2.namedWindow("License Plate Recognition System RUS",
                    cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("License Plate Recognition System RUS",
                          cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    logging.info("Система запущена")

    if not cap.isOpened():
        logging.warning("❌ Не удалось открыть источник видео")
        exit()

    logger.info(f"📡 Источник: {video_source}")
    logger.info(f"🧠 Устройство: {device.upper()}")

    is_file_source = isinstance(
        video_source, str) and video_source.lower().endswith((".avi", ".mp4", ".mkv"))

    current_time = time.time()
    timestamp = int(time.time())
    log_ts = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

    # Инициализация записи видео
    if save_video:
        ret, frame = cap.read()
        if not ret:
            logging.error("❌ Не удалось получить первый кадр")
            return

        video_writer = create_video_writer(frame.shape, source_label)
        start_record_time = current_time

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if is_file_source:
                logger.info("✅ Обработка файла завершена.")
                break  # Завершаем цикл для файлов
            else:
                logger.warning("🔁 Повторное подключение к потоку...")
                time.sleep(1)
                cap.release()
                cap = cv2.VideoCapture(video_source)
                continue

        # Добавление даты и времени
        frame = add_timestamp(frame)

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        vehicle_results = vehicle_model(frame, device=device)[0]
        mask = [int(c) in TARGET_CLASSES for c in vehicle_results.boxes.cls]
        detections = Detections(
            xyxy=vehicle_results.boxes.xyxy.cpu().numpy()[mask],
            confidence=vehicle_results.boxes.conf.cpu().numpy()[mask],
            class_id=vehicle_results.boxes.cls.cpu().numpy()[mask].astype(int))

        tracks = tracker.update_with_detections(detections)

        # Детекция номеров по всему кадру
        plate_results = plate_model(frame, device=device)[0]
        plate_boxes = plate_results.boxes.xyxy.cpu().numpy()
        plate_crops = [frame[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
                       for b in plate_boxes]
        plate_texts = [plate_reader.recognize(crop) for crop in plate_crops]

        plate_assignments = assign_plates_to_vehicles(
            plate_boxes, plate_texts, tracks)

        for track in tracks:
            bbox, sid = track[0], int(track[4])
            vx1, vy1, vx2, vy2 = map(int, bbox.tolist())

            stable_boxes[sid] = bbox
            sid_last_seen[sid] = current_time

            plate_text = ""

            for assigned_sid, plate_text in plate_assignments.items():
                if plate_text:
                    last_plate = plate_by_sid.get(assigned_sid)
                    if plate_text != last_plate:
                        plate_by_sid[assigned_sid] = plate_text
                        sid_last_plate_time[assigned_sid] = current_time
                        last_time = plate_last_log.get(plate_text, 0)
                        if timestamp - last_time >= PLATE_LOG_INTERVAL:
                            # обновить время записи
                            plate_last_log[plate_text] = timestamp

                            recognized_plates.append({
                                "timestamp": log_ts,
                                "plate": plate_text
                            })

                elif assigned_sid in plate_by_sid:
                    # проверка: если номер старый, и SID не менялся долго → сохранить
                    if current_time - sid_last_plate_time.get(assigned_sid, 0) <= PLATE_HOLD_TIME:
                        continue  # удерживаем прежний номер
                    else:
                        plate_by_sid.pop(assigned_sid, None)

                else:
                    plate_text = ""
                    plate_by_sid[assigned_sid] = plate_text

            last_plate = plate_by_sid.get(sid, "")
            if last_plate:
                frame = plate_reader.draw_text_cyrillic(
                    frame, last_plate, (vx2 - 140, vy2 - 40))

            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (255, 0, 255), 2)
            cv2.putText(frame, f"SID {sid}", (vx1, vy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            log_detection(frame, last_plate)
       
        if save_video:
            # Запись обработанного кадра
            video_writer.write(frame)

            # Перезапуск записи по времени
            if current_time - start_record_time > recording_interval_seconds:
                video_writer.release()
                video_writer = create_video_writer(frame.shape, source_label)
                start_record_time = time.time()

        # Удаление устаревших SID
        expired_sids = [sid for sid, last_seen in sid_last_seen.items()
                        if current_time - last_seen > SID_TTL]
        for sid in expired_sids:
            stable_boxes.pop(sid, None)
            sid_last_seen.pop(sid, None)
            if sid in plate_by_sid:
                old_plate = plate_by_sid[sid]
                if old_plate in plate_to_sid:
                    del plate_to_sid[old_plate]
                del plate_by_sid[sid]

        cv2.imshow("License Plate Recognition System RUS", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:  # Остановка по клавишам "q" или "Esc"
            logger.info("🛠 Принудительная остановка пользователем")
            break

    df = pd.DataFrame(recognized_plates)
    df.to_excel(f"{SAVE_DIR}/recognized_plates.xlsx", index=False)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    logger.info("🛑 Захват остановлен. Окна закрыты")


if __name__ == "__main__":

    is_file_source = isinstance(
        video_source, str) and video_source.lower().endswith((".avi", ".mp4", ".mkv"))

    while True:
        try:
            main()
            if is_file_source:

                logger.info("✅ Обработка видеофайла завершена")
                break
            else:
                logger.info("♻️ Перезапуск потока...")
        except KeyboardInterrupt:
            logger.info("Завершение по Ctrl+C")
            break
        except Exception as e:
            logger.exception(
                "❌ Критическая ошибка! Перезапуск через 5 сек...")
            time.sleep(5)
