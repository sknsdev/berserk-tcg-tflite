@echo off
chcp 65001 >nul
echo 🧹 Очистка Docker ресурсов для Berserk TCG
echo ==========================================

REM Проверяем наличие Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker не установлен или недоступен
    pause
    exit /b 1
)

echo 🛑 Остановка и удаление контейнеров...
docker stop card-training card-web-demo >nul 2>&1
docker rm card-training card-web-demo >nul 2>&1

echo 🗑️  Удаление образа card-recognition...
docker rmi card-recognition >nul 2>&1

echo 🧽 Очистка неиспользуемых ресурсов...
docker system prune -f

echo 📊 Освобождение места на диске...
docker system prune -a -f

echo ✅ Очистка завершена!
echo 💡 Для пересборки образа запустите: docker-build.bat

pause