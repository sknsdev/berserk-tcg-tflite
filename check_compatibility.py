#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт проверки совместимости системы для проекта Berserk TCG TFLite
Автоматически определяет проблемы и предлагает решения
"""

import sys
import platform
import subprocess
from pathlib import Path

def check_python_version():
    """Проверяет версию Python и совместимость с TensorFlow"""
    version = sys.version_info
    print(f"🐍 Python версия: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3:
        print("❌ Требуется Python 3.x")
        return False
    
    if version.minor < 8:
        print("❌ Требуется Python 3.8 или выше")
        return False
    
    if version.minor >= 13:
        print("⚠️ Python 3.13+ может иметь проблемы совместимости с TensorFlow")
        print("💡 Рекомендуется использовать requirements-py313.txt")
        return "warning"
    
    if 9 <= version.minor <= 12:
        print("✅ Оптимальная версия Python для TensorFlow")
        return True
    
    print("⚠️ Версия Python может работать, но не тестировалась")
    return "warning"

def check_system_info():
    """Выводит информацию о системе"""
    print(f"💻 Операционная система: {platform.system()} {platform.release()}")
    print(f"🏗️ Архитектура: {platform.machine()}")
    
    # Проверка доступности GPU (опционально)
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🎮 Найдено GPU: {len(gpus)} устройств")
        else:
            print("🔧 GPU не найден, будет использоваться CPU устройства")
    except ImportError:
        print("📦 TensorFlow не установлен")

def check_requirements_files():
    """Проверяет наличие файлов requirements"""
    files = {
        "requirements.txt": "Основной файл зависимостей",
        "requirements-py313.txt": "Файл для Python 3.13+"
    }
    
    print("\n📋 Проверка файлов зависимостей:")
    for file, description in files.items():
        if Path(file).exists():
            print(f"✅ {file} - {description}")
        else:
            print(f"❌ {file} - {description} (отсутствует)")

def check_cards_directory():
    """Проверяет наличие и содержимое папки cards"""
    cards_dir = Path("cards")
    
    if not cards_dir.exists():
        print("❌ Папка 'cards' не найдена")
        return False
    
    webp_files = list(cards_dir.glob("*.webp"))
    if not webp_files:
        print("❌ В папке 'cards' нет файлов .webp")
        return False
    
    print(f"✅ Найдено {len(webp_files)} изображений карт")
    
    # Анализ структуры файлов
    sets = set()
    for file in webp_files:
        parts = file.stem.split('_')
        if len(parts) >= 2:
            sets.add(parts[0])
    
    print(f"📊 Найдены наборы карт: {', '.join(sorted(sets))}")
    return True

def suggest_installation_method():
    """Предлагает оптимальный метод установки зависимостей"""
    version = sys.version_info
    
    print("\n💡 Рекомендуемый метод установки:")
    
    if version.minor >= 13:
        print("1. Используйте специальный файл для Python 3.13+:")
        print("   pip install -r requirements-py313.txt")
        print("\n2. Альтернативно, установите основные пакеты:")
        print("   pip install tensorflow>=2.16.0 numpy pillow scikit-learn matplotlib")
    else:
        print("1. Используйте основной файл зависимостей:")
        print("   pip install -r requirements.txt")
        print("\n2. Или запустите автоматическую настройку:")
        print("   python setup.py")
    
    print("\n3. Если возникают проблемы, попробуйте conda:")
    print("   conda install tensorflow numpy pillow scikit-learn matplotlib opencv")

def test_tensorflow_installation():
    """Тестирует установку TensorFlow"""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} установлен")
        
        # Простой тест
        x = tf.constant([1, 2, 3])
        print("✅ TensorFlow работает корректно")
        return True
    except ImportError:
        print("❌ TensorFlow не установлен")
        return False
    except Exception as e:
        print(f"⚠️ Проблема с TensorFlow: {e}")
        return False

def main():
    """Основная функция проверки совместимости"""
    print("🔍 Проверка совместимости системы для Berserk TCG TFLite\n")
    
    # Проверка Python
    python_ok = check_python_version()
    
    # Информация о системе
    print("\n" + "="*50)
    check_system_info()
    
    # Проверка файлов
    print("\n" + "="*50)
    check_requirements_files()
    
    # Проверка данных
    print("\n" + "="*50)
    print("📁 Проверка данных:")
    cards_ok = check_cards_directory()
    
    # Проверка TensorFlow
    print("\n" + "="*50)
    print("🧠 Проверка TensorFlow:")
    tf_ok = test_tensorflow_installation()
    
    # Рекомендации
    print("\n" + "="*50)
    if not tf_ok:
        suggest_installation_method()
    
    # Итоговый статус
    print("\n" + "="*50)
    print("📊 Итоговый статус:")
    
    if python_ok == True and cards_ok and tf_ok:
        print("🎉 Система готова к работе!")
        print("\n🚀 Следующие шаги:")
        print("   python main.py full  # Полный цикл обучения")
        print("   python web_demo.py   # Запуск веб-демо")
    else:
        print("⚠️ Требуется дополнительная настройка")
        if not cards_ok:
            print("   - Добавьте изображения карт в папку 'cards'")
        if not tf_ok:
            print("   - Установите TensorFlow и зависимости")
        if python_ok != True:
            print("   - Проверьте версию Python")

if __name__ == "__main__":
    main()