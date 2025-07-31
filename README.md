# 🃏 Berserk TCG TensorFlow Lite

Проект для распознавания карт настольной игры Берсерк с использованием TensorFlow Lite и поддержкой GPU.

## 🐳 Запуск через Docker (Рекомендуется)

### Предварительные требования

1. **Docker Desktop** с поддержкой GPU:
   - Установите [Docker Desktop](https://www.docker.com/products/docker-desktop)
   - Для GPU поддержки: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

2. **Подготовка данных**:
   - Создайте папку `cards` в корне проекта
   - Поместите изображения карт в формате `.webp` в папку `cards`

### Быстрый старт

```bash
# 1. Сборка Docker образа
docker-build.bat

# 2. Полное обучение модели (аугментация + обучение + тестирование)
docker-train.bat

# 3. Запуск веб-демо
docker-web.bat
```

### Доступные команды

#### Основные скрипты:
- `docker-build.bat` - Сборка Docker образа
- `docker-train.bat` - Полное обучение модели
- `docker-web.bat` - Запуск веб-демонстрации
- `docker-run.bat [команда]` - Универсальный запуск

#### Команды CLI через Docker:

```bash
# Проверка окружения и данных
docker-run.bat check

# Создание аугментированного датасета
docker-run.bat augment

# Обучение новой модели
docker-run.bat train

# Дообучение существующей модели
docker-run.bat continue

# Тестирование модели
docker-run.bat test

# Запуск веб-демо
docker-run.bat web

# Полный пайплайн (все этапы сразу)
docker-run.bat full

# Интерактивная оболочка
docker-run.bat bash

# Справка
docker-run.bat help
```

### Пример использования с GPU

```bash
# Команда автоматически определяет наличие GPU и использует флаг --gpus all
docker run -it --gpus all -v %cd%:/app --name card-training card-recognition python cli.py train
```

## 💻 Локальный запуск (без Docker)

### Установка зависимостей

```bash
# Автоматическая настройка
python setup.py

# Активация виртуального окружения
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### Команды

```bash
# Проверяем датасет
python data_preparation.py

# Аугментируем все карты датасета
python cli.py augment 

# Запускаем обучение 
python cli.py train

# Проверяем в вебе
python cli.py web
```

## 📁 Структура проекта

```
berserk-tcg-tflite/
├── cards/                  # Исходные изображения карт
├── cards_augmented/        # Аугментированный датасет
├── models/                 # Сохраненные модели
├── web_assets/            # Ресурсы для веб-интерфейса
├── docker-*.bat           # Docker скрипты для Windows
├── Dockerfile             # Конфигурация Docker
├── docker-compose.yml     # Docker Compose конфигурация
├── requirements.txt       # Python зависимости
├── cli.py                 # CLI интерфейс
├── train_model.py         # Обучение модели
├── test_model.py          # Тестирование модели
├── web_demo.py           # Веб-демонстрация
└── data_preparation.py    # Подготовка данных
```

## 🎯 Особенности

- **GPU поддержка**: Автоматическое определение и использование GPU
- **TensorFlow Lite**: Оптимизированные модели для быстрого инференса
- **Веб-интерфейс**: Удобная демонстрация через браузер
- **Docker**: Изолированная среда выполнения
- **Аугментация данных**: Автоматическое увеличение датасета

## 🌐 Веб-демо

После запуска `docker-web.bat` или `docker-run.bat web`, веб-интерфейс будет доступен по адресу:
**http://localhost:5000**

## 🔧 Устранение неполадок

### GPU не определяется
1. Убедитесь, что установлен NVIDIA Container Toolkit
2. Перезапустите Docker Desktop
3. Проверьте: `docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi`

### Ошибки при сборке
1. Очистите Docker кэш: `docker system prune -a`
2. Пересоберите образ: `docker-build.bat`

### Недостаточно памяти
1. Увеличьте лимиты памяти в Docker Desktop
2. Закройте другие приложения
3. Используйте меньший batch_size в настройках обучения