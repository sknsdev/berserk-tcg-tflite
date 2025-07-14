#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт автоматического исправления проблем с установкой зависимостей
Пробует различные методы установки и выбирает рабочий
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

def get_pip_path():
    """Определяет путь к pip в виртуальном окружении"""
    if platform.system() == "Windows":
        return "venv\\Scripts\\pip"
    else:
        return "venv/bin/pip"

def check_venv():
    """Проверяет и создает виртуальное окружение если нужно"""
    venv_path = Path("venv")
    
    if not venv_path.exists():
        print("🔧 Создание виртуального окружения...")
        python_cmd = "python" if platform.system() == "Windows" else "python3"
        success, output = run_command(f"{python_cmd} -m venv venv")
        
        if success:
            print("✅ Виртуальное окружение создано")
        else:
            print(f"❌ Ошибка создания venv: {output}")
            return False
    
    return True

def method_1_use_global():
    """Метод 1: Использование глобально установленных пакетов"""
    print("\n🔧 Метод 1: Проверка глобальной установки")
    
    # Проверяем глобальную установку
    try:
        import tensorflow as tf
        import numpy as np
        import PIL
        print(f"✅ Найдены глобальные пакеты: TensorFlow {tf.__version__}, NumPy {np.__version__}")
        
        # Копируем в виртуальное окружение
        pip_path = get_pip_path()
        success, output = run_command(f"{pip_path} install tensorflow=={tf.__version__} numpy=={np.__version__} pillow")
        
        if success:
            print("✅ Пакеты скопированы в виртуальное окружение")
            return True
        else:
            print(f"❌ Ошибка копирования: {output}")
            return False
            
    except ImportError:
        print("❌ Глобальные пакеты не найдены")
        return False

def method_2_flexible_versions():
    """Метод 2: Установка с гибкими версиями"""
    print("\n🔧 Метод 2: Установка с гибкими версиями")
    pip_path = get_pip_path()
    
    # Обновляем pip
    run_command(f"{pip_path} install --upgrade pip")
    
    success, output = run_command(f"{pip_path} install 'tensorflow>=2.16.0' 'numpy>=1.24.0' 'pillow>=10.0.0'")
    if success:
        print("✅ Установка с гибкими версиями успешна")
        return test_tensorflow()
    else:
        print(f"❌ Ошибка: {output}")
        return False

def method_3_requirements_file():
    """Метод 3: Установка из файла requirements"""
    print("\n🔧 Метод 3: Установка из requirements.txt")
    pip_path = get_pip_path()
    
    success, output = run_command(f"{pip_path} install -r requirements.txt")
    if success:
        print("✅ Установка из requirements.txt успешна")
        return test_tensorflow()
    else:
        print(f"❌ Ошибка: {output}")
        return False



def method_4_conda():
    """Метод 4: Установка через conda"""
    print("\n🔧 Метод 4: Установка через conda")
    
    # Проверяем наличие conda
    success, output = run_command("conda --version")
    if not success:
        print("❌ Conda не установлена")
        return False
    
    success, output = run_command("conda install tensorflow numpy pillow -y")
    if success:
        print("✅ Установка через conda успешна")
        return test_tensorflow()
    else:
        print(f"❌ Ошибка: {output}")
        return False

def method_5_cpu_only():
    """Метод 5: Установка CPU-версии TensorFlow"""
    print("\n🔧 Метод 5: Установка CPU-версии TensorFlow")
    pip_path = get_pip_path()
    
    success, output = run_command(f"{pip_path} install tensorflow-cpu numpy pillow")
    if success:
        print("✅ Установка CPU-версии успешна")
        return test_tensorflow()
    else:
        print(f"❌ Ошибка: {output}")
        return False

def method_6_recreate_venv():
    """Метод 6: Пересоздание виртуального окружения"""
    print("\n🔧 Метод 6: Пересоздание виртуального окружения")
    
    # Удаляем старое окружение
    import shutil
    venv_path = Path("venv")
    if venv_path.exists():
        shutil.rmtree(venv_path)
        print("🗑️ Старое окружение удалено")
    
    # Создаем новое
    if not check_venv():
        return False
    
    # Устанавливаем пакеты
    pip_path = get_pip_path()
    run_command(f"{pip_path} install --upgrade pip")
    
    success, output = run_command(f"{pip_path} install tensorflow numpy pillow")
    if success:
        print("✅ Новое окружение создано и настроено")
        return test_tensorflow()
    else:
        print(f"❌ Ошибка: {output}")
        return False

def test_tensorflow():
    """Тестирует установку TensorFlow"""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} работает")
        return True
    except ImportError:
        print("❌ TensorFlow не установлен")
        return False
    except Exception as e:
        print(f"❌ Ошибка TensorFlow: {e}")
        return False



def main():
    """Основная функция исправления"""
    print("🔧 АВТОМАТИЧЕСКОЕ ИСПРАВЛЕНИЕ УСТАНОВКИ")
    print("="*50)
    
    # Проверяем виртуальное окружение
    if not check_venv():
        print("❌ Не удалось создать виртуальное окружение")
        return False
    
    # Проверяем текущее состояние
    print("\n📋 Проверка текущего состояния...")
    if test_tensorflow():
        print("\n🎉 TensorFlow уже работает! Исправление не требуется.")
        return True
    
    # Список методов для попытки
    methods = [
        method_1_use_global,
        method_2_flexible_versions,
        method_3_requirements_file,
        method_4_conda,
        method_5_cpu_only,
        method_6_recreate_venv
    ]
    
    print("\n🚀 Начинаем попытки исправления...")
    
    for i, method in enumerate(methods, 1):
        try:
            if method():
                print(f"\n🎉 Успех! Метод {i} сработал.")
                print("\n📋 Рекомендации:")
                print("1. Запустите: python main.py check")
                print("2. Если всё работает, начните обучение: python main.py full")
                return True
        except Exception as e:
            print(f"❌ Метод {i} завершился с ошибкой: {e}")
    
    print("\n❌ Все методы исправления не сработали")
    print("\n💡 Рекомендации:")
    print("1. Обновите Python до версии 3.9-3.12")
    print("2. Переустановите Python с официального сайта")
    print("3. Используйте Anaconda/Miniconda")
    print("4. Обратитесь за помощью с указанием версии Python и ОС")
    
    return False

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Исправление прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}")
        sys.exit(1)