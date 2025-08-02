# Используем официальный образ TensorFlow с поддержкой GPU
FROM tensorflow/tensorflow:2.13.0-gpu

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgdal-dev \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Создаем пользователя для безопасности
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Создаем необходимые директории
RUN mkdir -p cards cards_augmented models logs && \
    chown -R appuser:appuser /app

# Копируем только необходимые Python файлы
COPY --chown=appuser:appuser *.py ./
COPY --chown=appuser:appuser web_assets/ ./web_assets/

# Устанавливаем переменные окружения
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV TF_CPP_MIN_LOG_LEVEL=1
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Переключаемся на пользователя appuser
USER appuser

# Открываем порт для веб-демо
EXPOSE 5000

# Проверка здоровья контейнера
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)" || exit 1

# Команда по умолчанию
CMD ["python", "cli.py", "--help"]