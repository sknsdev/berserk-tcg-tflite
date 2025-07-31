@echo off
chcp 65001 >nul
echo 🎯 Обучение модели Berserk TCG в Docker
echo ====================================

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
    echo ⚠️  Обучение может занять значительно больше времени
    set GPU_FLAG=
) else (
    echo ✅ GPU поддержка доступна
    set GPU_FLAG=--gpus all
)

REM Проверяем наличие папки с картами
if not exist "cards" (
    echo ❌ Папка 'cards' не найдена
    echo Создайте папку 'cards' и поместите в неё изображения карт
    pause
    exit /b 1
)

REM Останавливаем существующий контейнер если есть
docker stop card-training >nul 2>&1
docker rm card-training >nul 2>&1

echo 🚀 Запуск полного пайплайна обучения...
echo 📊 Это включает: проверку данных, аугментацию, обучение и тестирование
echo ⏱️  Процесс может занять от 30 минут до нескольких часов
echo.

docker run -it %GPU_FLAG% -v "%cd%":/app --name card-training card-recognition python cli.py full

if %errorlevel% equ 0 (
    echo ✅ Обучение завершено успешно!
    echo 📁 Результаты сохранены в текущей папке
    echo 🌐 Запустите веб-демо: docker-web.bat
) else (
    echo ❌ Ошибка при обучении
)

REM Очищаем контейнер после завершения
echo 🧹 Очистка...
docker rm card-training >nul 2>&1

pause