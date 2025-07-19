## 🚗 License_Plate_Recognition_System_RUS

Система автоматического распознавания автомобильных номеров с поддержкой кириллицы на базе YOLOv8 и PaddleOCR.

![Demo](assets/cvtest.gif)

### 🔍 Возможности

- 🚗 Детекция транспортных средств (автомобили, автобусы, грузовики)
- 📸 Распознавание номерных знаков с поддержкой кириллицы
- ⏱ Обработка видео в реальном времени
- 📊 Логирование результатов в Excel
- 🖥 Поддержка GPU (CUDA) для ускорения обработки
- 📹 Работа с различными источниками видео (веб-камера, IP-камера, видеофайлы)

### 📸 Примеры работы

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
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 📁 Структура проекта

.

├── yolo_weights/                   # Веса моделей

├── fonts/                          # Шрифты для отображения текста

├── results/                        # Результаты работы

│   ├── images/                     # Изображения транспортных средств

│   ├── 2025-07-19_07-50_cvtest.avi # обработанный видеофайл

│   └── recognized_plates.xlsx      # Лог распознанных номеров

├── logger.py                       # Модуль логирования

├── video_writer.py                 # Модуль записи видео
└── ...                             # Другие модули

├── config.json                     # Основной конфигурационный файл

├── main.py                         # Главный скрипт

└── requirements.txt                # Зависимости

### 🛠️ Интерфейс конфигурации

```bash
uvicorn web_interface:app --reload --port 8000
```
<p align="center">
<img src="./docs/web_interface.JPG" width="500">
</p>

### 🚀 Запуск

```bash
python main.py
```

### 🛑 Остановка

Остановка по клавишам "q" или "Esc"
