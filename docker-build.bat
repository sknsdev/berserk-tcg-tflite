@echo off
chcp 65001 >nul
echo 🔨 Сборка Docker образа для Berserk TCG
echo ========================================

REM Проверяем наличие Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker не установлен или недоступен
    echo Установите Docker Desktop: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

echo 🔨 Сборка образа card-recognition...
docker build -t card-recognition .

if %errorlevel% equ 0 (
    echo ✅ Образ успешно собран!
    echo.
    echo 📋 Доступные команды:
    echo   docker-run.bat help     - Показать справку CLI
    echo   docker-run.bat check    - Проверить окружение
    echo   docker-run.bat augment  - Создать аугментированный датасет
    echo   docker-run.bat train    - Обучить модель
    echo   docker-run.bat test     - Протестировать модель
    echo   docker-run.bat web      - Запустить веб-демо
    echo   docker-run.bat bash     - Интерактивная оболочка
) else (
    echo ❌ Ошибка при сборке образа
)

pause