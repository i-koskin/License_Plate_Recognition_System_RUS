from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse
import json

from config import CONFIG_PATH

# Создаем приложение FastAPI
app = FastAPI()


def load_config():
    """
    Загружает настройки из конфигурационного файла config.json.

    Функция открывает файл, загружает его содержимое в формате JSON и возвращает
    как словарь. Если файл не найден или возникнут проблемы с загрузкой,
    возникнет исключение.

    Returns:
        dict: данные из конфигурационного файла.
    """
    if not CONFIG_PATH:
        return {
            "video_source": "0",
            "frame_skip": 5,
            "save_video": True,
            "recording_interval_minutes": 10,
            "save_full_frame": True,
            "log_level": "INFO"
        }
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def save_config(video_source, frame_skip, save_video, recording_interval_minutes, save_full_frame, log_level):
    """
    Сохраняет настройки в конфигурационный файл config.json.

    Эта функция записывает переданные параметры в конфигурационный файл. Настройки
    включают:
    - выбор источника видеосигнала;
    - количество кадров, которые будут пропускаться (для ускорения работы системы);
    - флаг для сохранения видео;
    - интервал записи в минутах;
    - флаг для сохранения полного кадра;
    - уровень логирования.

    Args:
        video_source (str): выбор источника видесигнала:
                            - 0 для веб-камеры;
                            - ввести путь к видеофайлу (*.avi) в веб-интерфейсе;
                            - указать rtsp://... для IP-камеры.
        frame_skip (int): количество кадров, которые будут пропускаться;
        save_video (bool): сохранять видео;
        recording_interval_minutes (int): интервал записи в минутах;
        save_full_frame (bool): сохранять полный кадр или только ROI;
        log_level (str): выбор уровеня логирования ("INFO", "WARNING", "ERROR", "DEBUG").

    """
    with open(CONFIG_PATH, "w") as f:
        # Сохраняем все параметры в JSON файл с отступами
        json.dump({
            "video_source": video_source,
            "frame_skip": frame_skip,
            "save_video": save_video,
            "recording_interval_minutes": recording_interval_minutes,
            "save_full_frame": save_full_frame,
            "log_level": log_level
        }, f, indent=2, ensure_ascii=False)


@app.get("/", response_class=HTMLResponse)
def read_form():
    """
    Отображает HTML-форму для настройки уведомлений и уровня логирования.

    Эта функция генерирует HTML-страницу, на которой пользователи могут настроить:
    - выбор источника видеосигнала (видеокамера, видеофайл);
    - количество кадров, которые будут пропускаться;
    - время записи в минутах;
    - сохранение полного кадра или только ROI;
    - уровень логирования.

    Returns:
        str: HTML-код страницы для настройки.
    """
    cfg = load_config()  # Загружаем текущие настройки из конфигурации
    # Выбираем источник видеосигнала
    video_source = cfg.get("video_source", "0")
    # Определяем количество кадров, которые будут пропускаться
    frame_skip = cfg.get("frame_skip", 5)
    # Определяем состояние чекбокса для сохранения видео
    video_checked = "checked" if cfg.get("save_video", False) else ""
    # Определяем время записи в минутах
    recording_interval_minutes = cfg.get("recording_interval_minutes", 10)
    # Определяем состояние чекбокса для сохранения кадра или ROI
    frame_checked = "checked" if cfg.get("save_full_frame", False) else ""
    # Получаем текущий уровень логирования
    log_level = cfg.get("log_level", "INFO")

    # Генерируем HTML-код для формы
    html = f"""
<html>
  <head>
    <style>
      body {{
        font-family: Arial, sans-serif;
        max-width: 550px;
        margin: 40px auto;
        padding: 20px;
        border: 1px solid #ccc;
        border-radius: 8px;
        background-color: #f9f9f9;
      }}
      input[type="submit"] {{
        padding: 8px 16px;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
      }}
      input[type="submit"]:hover {{
        background-color: #45a049;
      }}
    </style>
  </head>
  <body>
    <h2>🕒 Настройки уведомлений и сохранения</h2>
    <form method="post">

      <label>Источник видео (0, rtsp или путь к .avi):</label>
      <input type="text" name="video_source" value="{video_source}"><br><br>

      <label>Количество кадров, которые будут пропускаться:</label>
      <input type="number" name="frame_skip" value="{frame_skip}"><br><br>

      <label>
        <input type="checkbox" name="save_video" {video_checked}>
        Сохранять видео
      </label><br><br>

      <label>Время записи (в минутах):</label>
      <input type="number" name="recording_interval_minutes" value="{recording_interval_minutes}" min="1" max="60"><br><br>

      <label>
        <input type="checkbox" name="save_full_frame" {frame_checked}>
        Сохранять полный кадр (а не только ROI)
      </label><br><br>

      <label>Уровень логирования:</label>
      <select name="log_level">
        <option value="INFO" {'selected' if log_level == 'INFO' else ''}>INFO</option>
        <option value="WARNING" {'selected' if log_level == 'WARNING' else ''}>WARNING</option>
        <option value="ERROR" {'selected' if log_level == 'ERROR' else ''}>ERROR</option>
        <option value="DEBUG" {'selected' if log_level == 'DEBUG' else ''}>DEBUG</option>
      </select><br><br>

      <input type="submit" value="Сохранить">
    </form>
  </body>
</html>
"""
    return html


@app.post("/")
def update_config(
    video_source: str = Form(...),
    frame_skip: int = Form(...),
    save_video: str = Form(None),
    recording_interval_minutes: int = Form(...),
    save_full_frame: str = Form(None),
    log_level: str = Form(...)
):
    """
    Обновляет настройки конфигурации, включая уровень логирования.

    Эта функция получает данные из формы, обновляет конфигурационный файл и 
    перенаправляет пользователя обратно на страницу настроек.

    Args:
        video_source (str): сторка, указывающая источник видеосигнала;
        frame_skip (int): количество кадров, которые будут пропускаться;
        save_full_frame (str): строка, указывающая, сохранять ли полный кадр;
        log_level (str): уровень логирования, выбранный пользователем.

    Returns:
        RedirectResponse: перенаправление на страницу настроек.
    """
    # Преобразуем строки в логические значения
    save_video_flag = save_video is not None
    save_full_flag = save_full_frame is not None
    # Сохраняем новые настройки
    save_config(video_source, frame_skip, save_video_flag, recording_interval_minutes,
                save_full_flag, log_level)
    # Перенаправляем пользователя на страницу настроек
    return RedirectResponse("/", status_code=303)
