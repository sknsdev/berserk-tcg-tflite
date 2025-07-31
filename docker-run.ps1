# PowerShell скрипт для запуска проекта Berserk TCG в Docker

Write-Host "🐳 Запуск проекта Berserk TCG в Docker" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan

# Проверяем наличие Docker
try {
    docker --version | Out-Null
    Write-Host "✅ Docker найден" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker не установлен или недоступен" -ForegroundColor Red
    Write-Host "Установите Docker Desktop: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    Read-Host "Нажмите Enter для выхода"
    exit 1
}

# Проверяем поддержку GPU
Write-Host "🔍 Проверка поддержки GPU..." -ForegroundColor Yellow
try {
    docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi 2>$null | Out-Null
    Write-Host "✅ GPU поддержка доступна" -ForegroundColor Green
    $gpuFlag = "--gpus all"
} catch {
    Write-Host "⚠️  GPU поддержка недоступна, будет использоваться CPU" -ForegroundColor Yellow
    $gpuFlag = ""
}

# Собираем образ если его нет
try {
    docker image inspect card-recognition 2>$null | Out-Null
    Write-Host "✅ Образ card-recognition найден" -ForegroundColor Green
} catch {
    Write-Host "🔨 Образ не найден, собираем..." -ForegroundColor Yellow
    docker build -t card-recognition .
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Ошибка при сборке образа" -ForegroundColor Red
        Read-Host "Нажмите Enter для выхода"
        exit 1
    }
}

# Определяем команду для запуска
$command = $args[0]
if (-not $command) { $command = "help" }

# Останавливаем существующий контейнер
docker stop card-training 2>$null | Out-Null
docker rm card-training 2>$null | Out-Null

# Запускаем контейнер
Write-Host "🚀 Запуск контейнера..." -ForegroundColor Green
Write-Host "Команда: $command" -ForegroundColor Cyan
Write-Host ""

switch ($command) {
    "web" {
        Write-Host "🌐 Запуск веб-демо на http://localhost:5000" -ForegroundColor Green
        if ($gpuFlag) {
            docker run -it --gpus all -v "${PWD}:/app" -p 5000:5000 --name card-training card-recognition python web_demo.py
        } else {
            docker run -it -v "${PWD}:/app" -p 5000:5000 --name card-training card-recognition python web_demo.py
        }
    }
    "bash" {
        Write-Host "💻 Запуск интерактивной оболочки" -ForegroundColor Green
        if ($gpuFlag) {
            docker run -it --gpus all -v "${PWD}:/app" --name card-training card-recognition /bin/bash
        } else {
            docker run -it -v "${PWD}:/app" --name card-training card-recognition /bin/bash
        }
    }
    "help" {
        if ($gpuFlag) {
            docker run --rm --gpus all -v "${PWD}:/app" card-recognition python cli.py --help
        } else {
            docker run --rm -v "${PWD}:/app" card-recognition python cli.py --help
        }
    }
    default {
        if ($gpuFlag) {
            docker run -it --gpus all -v "${PWD}:/app" --name card-training card-recognition python cli.py $args
        } else {
            docker run -it -v "${PWD}:/app" --name card-training card-recognition python cli.py $args
        }
    }
}

# Очищаем контейнер после завершения
Write-Host "🧹 Очистка..." -ForegroundColor Yellow
docker rm card-training 2>$null | Out-Null

Write-Host "✅ Готово!" -ForegroundColor Green
Read-Host "Нажмите Enter для выхода"