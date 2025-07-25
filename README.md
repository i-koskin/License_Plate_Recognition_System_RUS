# 🚔 License_Plate_Recognition_System_RUS

## Система автоматического распознавания государственных регистрационных знаков автомобилей

<p align="center">
<img src="assets/cvtest.gif")>
</p>

## 🔍 Возможности

- 🚗 Обнаружение транспортных средств (автомобили, автобусы, грузовики) с использованием модели YOLOv8 и трекингом автомобилей между кадрами с помощью ByteTrack из библиотеки Supervision
- 🔍 Детекция номерных знаков моделью **yolov8_plate.pt**, дополнительно обученной на 3000+ размеченнных изображениях автомобилей
- 🔠 Распознавание номерных знаков с поддержкой кириллицы с помощью PaddleOCR
- ⏱ Обработка видео в реальном времени
- 🛠 Работа с несколькими автомобилями в кадре
- 🖼️ Отображение результатов в реальном времени
- ✂️ Настройка продолжительности записи обработанного видео для последующего сохранения
- 💾 Сохранение изображений даже при неудачной попытке распознавания
- 📝 Сохранение распознанных номеров в Excel
- 🖥 Поддержка GPU (CUDA) для ускорения обработки
- 📹 Работа с различными источниками видео (веб-камера, IP-камера, видеофайлы)

## 📸 Примеры работы

<p align="center">
<img src="./assets/2025-07-19_07-52-23_К369НС777.jpg" width="900">
<img src="./assets/2025-07-19_07-54-00_.jpg" width="900">
<img src="./assets/2025-07-19_07-57-19_.jpg" width="900">
<img src="./assets/2025-07-19_07-59-17_Х402ТЕ750.jpg" width="900">  
</p>

## ⚙️ Установка

```bash
git clone https://github.com/i-koskin/License_Plate_Recognition_System_RUS.git
cd License_Plate_Recognition_System_RUS
py -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## 📁 Структура проекта

.

├── yolo_weights/                    *# Веса моделей*

├── fonts/                           *# Шрифты для отображения текста*

├── results/                         *# Результаты работы*

│   ├── images/                      *# Изображения транспортных средств*

│   ├── 2025-07-19_07-50_cvtest.avi  *# Обработанный видеофайл*

│   └── recognized_plates.xlsx       *# Лог распознанных номеров*

├── logger.py                        *# Модуль логирования*

├── video_writer.py                  *# Модуль записи видео*

└── ...                              *# Другие модули*

├── config.json                      *# Конфигурационный файл*

├── main.py                          *# Главный скрипт*

└── requirements.txt                 *# Зависимости*

## 🛠️ Интерфейс конфигурации

```bash
uvicorn web_interface:app --reload --port 8000
```
⚠️ **Укажите путь к источнику обрабатываемого видео!** 👇

<p align="center">
<img src="./assets/web_interface.JPG" width="500">
</p>

## 🚀 Запуск

```bash
py main.py
```

## 🛑 Остановка

Остановка по клавишам "q" или "Esc"
