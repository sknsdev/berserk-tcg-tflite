@echo off
chcp 65001 >nul
echo Docker Berserk TCG Project Runner
echo ===================================

REM Check Docker availability
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker not installed or unavailable
    echo Install Docker Desktop: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

REM Check GPU support
echo Checking GPU support...
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: GPU support unavailable, using CPU
    set GPU_FLAG=
) else (
    echo SUCCESS: GPU support available
    set GPU_FLAG=--gpus all
)

REM Build image if needed
echo Building Docker image...
docker build -t card-recognition .
if %errorlevel% neq 0 (
    echo ERROR: Failed to build image
    pause
    exit /b 1
)

REM Determine command to run
set COMMAND=%1
if "%COMMAND%"=="" set COMMAND=help

REM Stop and remove existing container
echo Stopping existing container...
docker stop card-training >nul 2>&1
docker rm card-training >nul 2>&1

REM Run container
echo Starting container...
echo Command: %COMMAND%
echo.

if "%COMMAND%"=="web" (
    echo Starting web demo at http://localhost:5000
    docker run -it %GPU_FLAG% -v "%cd%":/app -p 5000:5000 --name card-training card-recognition python web_demo.py
) else if "%COMMAND%"=="bash" (
    echo Starting interactive shell
    docker run -it %GPU_FLAG% -v "%cd%":/app --name card-training card-recognition /bin/bash
) else if "%COMMAND%"=="help" (
    docker run --rm %GPU_FLAG% -v "%cd%":/app card-recognition python cli.py --help
) else (
    docker run -it %GPU_FLAG% -v "%cd%":/app --name card-training card-recognition python cli.py %*
)

REM Cleanup container after completion
echo Cleaning up container...
docker rm card-training >nul 2>&1

echo Done!
pause