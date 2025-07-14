#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт автоматической настройки проекта Berserk TCG TFLite
Автоматизирует создание виртуального окружения и установку зависимостей
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, shell=True):
    """Выполняет команду в терминале"""
    try:
        result = subprocess.run(command, shell=shell, check=True, 
                              capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def check_python():
    """Проверяет версию Python"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Требуется Python 3.8 или выше")
        print(f"Текущая версия: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
    return True

def create_venv():
    """Создает виртуальное окружение"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("📁 Виртуальное окружение уже существует")
        return True
    
    print("🔧 Создание виртуального окружения...")
    
    # Определяем команду для создания venv
    if platform.system() == "Windows":
        python_cmd = "python"
    else:
        python_cmd = "python3"
    
    success, output = run_command(f"{python_cmd} -m venv venv")
    
    if success:
        print("✅ Виртуальное окружение создано")
        return True
    else:
        print(f"❌ Ошибка создания виртуального окружения: {output}")
        return False

def get_activation_command():
    """Возвращает команду активации для текущей ОС"""
    if platform.system() == "Windows":
        return "venv\\Scripts\\activate"
    else:
        return "source venv/bin/activate"

def get_requirements_file():
    """Определяет подходящий файл requirements в зависимости от версии Python"""
    version = sys.version_info
    
    # Для Python 3.13+ используем специальный файл
    if version.major == 3 and version.minor >= 13:
        if Path("requirements-py313.txt").exists():
            return "requirements-py313.txt"
    
    # Для остальных версий используем основной файл
    if Path("requirements.txt").exists():
        return "requirements.txt"
    
    return None

def install_requirements():
    """Устанавливает зависимости в виртуальное окружение"""
    print("📦 Установка зависимостей...")
    
    # Определяем путь к pip в виртуальном окружении
    if platform.system() == "Windows":
        pip_path = "venv\\Scripts\\pip"
    else:
        pip_path = "venv/bin/pip"
    
    # Обновляем pip
    print("🔄 Обновление pip...")
    success, output = run_command(f"{pip_path} install --upgrade pip")
    if not success:
        print(f"⚠️ Предупреждение при обновлении pip: {output}")
    
    # Определяем подходящий файл requirements
    requirements_file = get_requirements_file()
    if not requirements_file:
        print("❌ Файл requirements не найден")
        return False
    
    print(f"📋 Используется файл: {requirements_file}")
    
    # Устанавливаем зависимости
    success, output = run_command(f"{pip_path} install -r {requirements_file}")
    
    if success:
        print("✅ Зависимости установлены")
        return True
    else:
        print(f"❌ Ошибка установки зависимостей: {output}")
        print("\n💡 Попробуйте следующие решения:")
        print("1. Обновите Python до версии 3.9-3.12")
        print("2. Установите зависимости вручную: pip install tensorflow numpy pillow")
        print("3. Используйте conda вместо pip")
        return False

def check_cards_directory():
    """Проверяет наличие папки с картами"""
    cards_path = Path("cards")
    
    if not cards_path.exists():
        print("❌ Папка 'cards' не найдена")
        print("Создайте папку 'cards' и поместите в неё изображения карт")
        return False
    
    # Подсчитываем количество файлов
    card_files = list(cards_path.glob("*.webp"))
    
    if len(card_files) == 0:
        print("❌ В папке 'cards' нет файлов .webp")
        print("Поместите изображения карт в формате WebP в папку 'cards'")
        return False
    
    print(f"✅ Найдено {len(card_files)} изображений карт")
    return True

def print_next_steps():
    """Выводит инструкции для следующих шагов"""
    activation_cmd = get_activation_command()
    
    print("\n" + "="*60)
    print("🎉 НАСТРОЙКА ЗАВЕРШЕНА УСПЕШНО!")
    print("="*60)
    print("\n📋 СЛЕДУЮЩИЕ ШАГИ:")
    print("\n1. Активируйте виртуальное окружение:")
    print(f"   {activation_cmd}")
    print("\n2. Проверьте данные:")
    print("   python main.py check")
    print("\n3. Запустите обучение модели:")
    print("   python main.py full")
    print("\n4. Запустите веб-демонстрацию:")
    print("   python web_demo.py")
    print("\n5. Для деактивации окружения:")
    print("   deactivate")
    print("\n" + "="*60)

def main():
    """Основная функция настройки"""
    print("🚀 АВТОМАТИЧЕСКАЯ НАСТРОЙКА ПРОЕКТА BERSERK TCG TFLITE")
    print("="*60)
    
    # Проверяем Python
    if not check_python():
        return False
    
    # Создаем виртуальное окружение
    if not create_venv():
        return False
    
    # Устанавливаем зависимости
    if not install_requirements():
        return False
    
    # Проверяем данные
    check_cards_directory()
    
    # Выводим инструкции
    print_next_steps()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print("\n❌ Настройка завершилась с ошибками")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Настройка прервана пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}")
        sys.exit(1)