@echo off
chcp 65001 >nul
echo 🎯 Демонстрация Berserk TCG Docker интеграции
echo ============================================

REM Проверяем наличие Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker не установлен или недоступен
    echo Установите Docker Desktop: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

echo 🔍 Проверка системы...
echo.

REM Проверяем GPU
echo 🎮 Проверка GPU поддержки:
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo   ❌ GPU недоступен - будет использоваться CPU
    set GPU_STATUS=CPU
) else (
    echo   ✅ GPU доступен
    set GPU_STATUS=GPU
)

REM Проверяем наличие данных
echo 📁 Проверка данных:
if exist "cards" (
    for /f %%i in ('dir /b /s cards\*.webp 2^>nul ^| find /c /v ""') do set CARD_COUNT=%%i
    if !CARD_COUNT! gtr 0 (
        echo   ✅ Найдено !CARD_COUNT! изображений карт
    ) else (
        echo   ❌ Изображения карт не найдены
        set CARD_COUNT=0
    )
) else (
    echo   ❌ Папка cards не найдена
    set CARD_COUNT=0
)

echo.
echo 📋 Статус системы:
echo   🖥️  Вычисления: %GPU_STATUS%
echo   🃏 Карт найдено: %CARD_COUNT%
echo.

if %CARD_COUNT% equ 0 (
    echo ⚠️  Для полной демонстрации поместите изображения карт в папку 'cards'
    echo.
)

echo 🔨 Сборка Docker образа...
docker build -t card-recognition . --quiet
if %errorlevel% neq 0 (
    echo ❌ Ошибка при сборке образа
    pause
    exit /b 1
)
echo ✅ Образ собран успешно

echo.
echo 🚀 Доступные команды для запуска:
echo.
echo 📊 Основные операции:
echo   docker-run.bat check     - Проверить окружение
echo   docker-run.bat augment   - Создать аугментированный датасет
echo   docker-run.bat train     - Обучить модель
echo   docker-run.bat test      - Протестировать модель
echo   docker-run.bat full      - Полный пайплайн
echo.
echo 🌐 Веб-интерфейс:
echo   docker-run.bat web       - Запустить веб-демо
echo   docker-web.bat           - Быстрый запуск веб-демо
echo.
echo 🛠️  Утилиты:
echo   docker-run.bat bash      - Интерактивная оболочка
echo   docker-clean.bat         - Очистка Docker ресурсов
echo.
echo 🐳 Docker Compose:
echo   docker-compose-run.bat web    - Веб-демо через Compose
echo   docker-compose-run.bat train  - Обучение через Compose
echo.
echo 💡 Пример команды с GPU:
echo   docker run -it --gpus all -v %%cd%%:/app --name card-training card-recognition python cli.py train
echo.

if %CARD_COUNT% gtr 0 (
    echo 🎯 Рекомендуемый порядок действий:
    echo   1. docker-train.bat     (полное обучение)
    echo   2. docker-web.bat       (запуск веб-демо)
) else (
    echo 📝 Для начала работы:
    echo   1. Поместите изображения карт в папку 'cards'
    echo   2. Запустите docker-train.bat
    echo   3. Запустите docker-web.bat
)

echo.
echo ✅ Демонстрация завершена!
echo 📖 Подробная документация в README.md
pause