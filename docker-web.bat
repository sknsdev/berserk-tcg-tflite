@echo off
chcp 65001 >nul
echo 🌐 Запуск веб-демо Berserk TCG в Docker
echo ======================================

REM Проверяем наличие Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker не установлен или недоступен
    echo Установите Docker Desktop: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

REM Проверяем наличие образа
docker image inspect card-recognition >nul 2>&1
if %errorlevel% neq 0 (
    echo 🔨 Образ не найден, собираем...
    docker build -t card-recognition .
    if %errorlevel% neq 0 (
        echo ❌ Ошибка при сборке образа
        pause
        exit /b 1
    )
)

REM Проверяем поддержку GPU
echo 🔍 Проверка поддержки GPU...
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  GPU поддержка недоступна, будет использоваться CPU
    set GPU_FLAG=
) else (
    echo ✅ GPU поддержка доступна
    set GPU_FLAG=--gpus all
)

REM Останавливаем существующий контейнер если есть
docker stop card-web-demo >nul 2>&1
docker rm card-web-demo >nul 2>&1

echo 🚀 Запуск веб-демо...
echo 🌐 Веб-интерфейс будет доступен по адресу: http://localhost:5000
echo 📱 Для остановки нажмите Ctrl+C
echo.

docker run -it %GPU_FLAG% -v "%cd%":/app -p 5000:5000 --name card-web-demo card-recognition python web_demo.py

REM Очищаем контейнер после завершения
echo 🧹 Очистка...
docker rm card-web-demo >nul 2>&1

echo ✅ Веб-демо остановлено!
pause