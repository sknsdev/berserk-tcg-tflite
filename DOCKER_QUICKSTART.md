# 🐳 Docker Быстрый старт

## Минимальные требования

- **Docker Desktop** с поддержкой GPU
- **NVIDIA Container Toolkit** (для GPU)
- **Windows 10/11** с WSL2

## 🚀 Быстрый запуск (3 команды)

```bash
# 1. Сборка образа
docker-build.bat

# 2. Обучение модели (если есть данные)
docker-train.bat

# 3. Запуск веб-демо
docker-web.bat
```

## 📁 Подготовка данных

1. Создайте папку `cards` в корне проекта
2. Поместите изображения карт в формате `.webp`
3. Структура может быть:
   ```
   cards/
   ├── card1.webp
   ├── card2.webp
   └── ...
   ```
   или
   ```
   cards/
   ├── set1/
   │   ├── card1.webp
   │   └── card2.webp
   └── set2/
       ├── card3.webp
       └── card4.webp
   ```

## 🎯 Основные скрипты

| Скрипт | Описание |
|--------|----------|
| `docker-build.bat` | Сборка Docker образа |
| `docker-train.bat` | Полное обучение модели |
| `docker-web.bat` | Запуск веб-демо |
| `docker-run.bat [cmd]` | Универсальный запуск |
| `docker-clean.bat` | Очистка Docker ресурсов |
| `docker-demo.bat` | Демонстрация возможностей |

## 🔧 Команды CLI через Docker

```bash
# Проверка системы
docker-run.bat check

# Создание аугментированного датасета
docker-run.bat augment

# Обучение новой модели
docker-run.bat train

# Тестирование модели
docker-run.bat test

# Полный пайплайн
docker-run.bat full

# Интерактивная оболочка
docker-run.bat bash
```

## 🌐 Веб-демо

После запуска `docker-web.bat`:
- Откройте браузер
- Перейдите на http://localhost:5000
- Загрузите изображение карты
- Получите результат распознавания

## 🐳 Docker Compose

```bash
# Запуск через Docker Compose
docker-compose-run.bat web
docker-compose-run.bat train
docker-compose-run.bat bash

# Остановка всех сервисов
docker-compose-run.bat down
```

## 🎮 GPU vs CPU

**С GPU:**
- Автоматически определяется
- Использует флаг `--gpus all`
- Значительно быстрее обучение

**Без GPU:**
- Автоматически переключается на CPU
- Обучение займет больше времени
- Все функции доступны

## 🔍 Проверка GPU

```bash
# Проверка поддержки NVIDIA
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Если команда работает - GPU доступен
# Если ошибка - будет использоваться CPU
```

## 📊 Мониторинг

```bash
# Просмотр логов
docker logs card-training

# Статистика контейнера
docker stats card-training

# Подключение к работающему контейнеру
docker exec -it card-training /bin/bash
```

## 🛠️ Устранение неполадок

### GPU не работает
```bash
# Перезапуск Docker Desktop
# Проверка NVIDIA Container Toolkit
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

### Ошибки сборки
```bash
# Очистка Docker
docker-clean.bat

# Пересборка
docker-build.bat
```

### Недостаточно памяти
- Увеличьте лимиты в Docker Desktop
- Закройте другие приложения
- Используйте меньший batch_size

## 📝 Переменные окружения

Файл `.env` содержит настройки:
```env
BATCH_SIZE=32
EPOCHS=20
LEARNING_RATE=0.0001
TF_CPP_MIN_LOG_LEVEL=1
```

## 🎯 Рекомендуемый workflow

1. **Подготовка**: Поместите изображения в `cards/`
2. **Демо**: Запустите `docker-demo.bat` для обзора
3. **Сборка**: `docker-build.bat`
4. **Обучение**: `docker-train.bat`
5. **Тестирование**: `docker-web.bat`
6. **Очистка**: `docker-clean.bat` (при необходимости)

## 💡 Полезные команды

```bash
# Быстрый запуск веб-демо без обучения
docker-run.bat web

# Проверка только данных
docker-run.bat check

# Интерактивная работа
docker-run.bat bash

# Просмотр справки
docker-run.bat help
```