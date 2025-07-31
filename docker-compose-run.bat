@echo off
chcp 65001 >nul
echo 🐳 Запуск Berserk TCG через Docker Compose
echo =========================================

REM Проверяем наличие Docker Compose
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose не установлен или недоступен
    echo Установите Docker Desktop: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

REM Определяем команду
set COMMAND=%1
if "%COMMAND%"=="" set COMMAND=help

echo 🔍 Проверка поддержки GPU...
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  GPU поддержка недоступна, используем CPU версию
    set SERVICE=card-recognition-cpu
    set PROFILE=--profile cpu-only
) else (
    echo ✅ GPU поддержка доступна
    set SERVICE=card-recognition
    set PROFILE=
)

echo 🔨 Сборка образов...
docker-compose build

if "%COMMAND%"=="web" (
    echo 🌐 Запуск веб-демо...
    echo Веб-интерфейс: http://localhost:5000
    docker-compose up web-demo
) else if "%COMMAND%"=="bash" (
    echo 💻 Запуск интерактивной оболочки...
    docker-compose run --rm %PROFILE% %SERVICE% /bin/bash
) else if "%COMMAND%"=="down" (
    echo 🛑 Остановка всех сервисов...
    docker-compose down
) else if "%COMMAND%"=="logs" (
    echo 📋 Просмотр логов...
    docker-compose logs -f
) else if "%COMMAND%"=="help" (
    echo 📋 Доступные команды:
    echo   docker-compose-run.bat web      - Запуск веб-демо
    echo   docker-compose-run.bat train    - Обучение модели
    echo   docker-compose-run.bat test     - Тестирование модели
    echo   docker-compose-run.bat bash     - Интерактивная оболочка
    echo   docker-compose-run.bat down     - Остановка сервисов
    echo   docker-compose-run.bat logs     - Просмотр логов
    echo.
    docker-compose run --rm %PROFILE% %SERVICE% python cli.py --help
) else (
    echo 🚀 Выполнение команды: %*
    docker-compose run --rm %PROFILE% %SERVICE% python cli.py %*
)

echo ✅ Готово!
pause