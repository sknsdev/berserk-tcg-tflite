#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Главный скрипт для проекта распознавания карт Берсерк
Автор: AI Assistant
Описание: Полный пайплайн обучения и тестирования TensorFlow Lite модели
"""

import os
import sys
import argparse
import time
from data_preparation import BerserkCardDataset
from train_model import main as train_main
from test_model import main as test_main

def check_requirements():
    """Проверяет системные требования"""
    print("🔍 Проверка системных требований...")
    
    # Проверка Python версии
    version = sys.version_info
    if version < (3, 8):
        print("❌ Требуется Python 3.8 или выше")
        return False
    
    if version >= (3, 13):
        print("⚠️ Python 3.13+ может иметь проблемы совместимости")
        print("💡 Запустите 'python check_compatibility.py' для диагностики")
    
    # Проверка TensorFlow
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} найден")
        
        # Проверка GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🎮 Найдено GPU: {len(gpus)} устройств")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("🔧 GPU не найден, будет использоваться CPU компьютера")
            print("💡 Для диагностики GPU запустите: python gpu_diagnostic.py")
            
    except ImportError:
        print("❌ TensorFlow не установлен")
        print("💡 Запустите 'python check_compatibility.py' для решения")
        return False
    except Exception as e:
        print(f"⚠️ Проблема с TensorFlow: {e}")
        print("💡 Запустите 'python check_compatibility.py' для диагностики")
        return False
    
    # Проверка других зависимостей
    required_packages = {
        'numpy': 'numpy',
        'PIL': 'Pillow', 
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'cv2': 'opencv-python',
        'pandas': 'pandas',
        'tqdm': 'tqdm'
    }
    missing_packages = []
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"❌ Отсутствуют пакеты: {', '.join(missing_packages)}")
        print("💡 Запустите 'python setup.py' или 'pip install -r requirements.txt'")
        return False
    
    print("✅ Все системные требования выполнены")
    return True

def check_data():
    """Проверяет наличие данных"""
    cards_dir = './cards'
    
    if not os.path.exists(cards_dir):
        print(f"❌ Папка с картами не найдена: {cards_dir}")
        return False
    
    # Проверяем подпапки
    subdirs = [d for d in os.listdir(cards_dir) if os.path.isdir(os.path.join(cards_dir, d))]
    
    # Проверяем изображения в корне папки
    root_images = [f for f in os.listdir(cards_dir) 
                   if f.lower().endswith(('.webp', '.jpg', '.jpeg', '.png'))]
    
    total_images = 0
    
    if len(subdirs) > 0:
        # Структура с подпапками
        print("📁 Найдена структура с подпапками:")
        for subdir in subdirs:
            subdir_path = os.path.join(cards_dir, subdir)
            image_files = [f for f in os.listdir(subdir_path) 
                          if f.lower().endswith(('.webp', '.jpg', '.jpeg', '.png'))]
            total_images += len(image_files)
            print(f"📂 {subdir}: {len(image_files)} изображений")
    
    if len(root_images) > 0:
        # Изображения в корне папки
        total_images += len(root_images)
        print(f"📄 В корне папки: {len(root_images)} изображений")
        
        if len(subdirs) == 0:
            print("💡 Изображения находятся в корне папки cards")
            print("💡 Для организации по классам запустите: python organize_cards.py")
    
    if total_images == 0:
        print(f"❌ В папке {cards_dir} не найдено изображений")
        return False
    
    print(f"✅ Всего найдено {total_images} изображений карт")
    return True

def prepare_data():
    """Подготавливает данные для обучения"""
    print("\n=== ПОДГОТОВКА ДАННЫХ ===")
    
    try:
        dataset = BerserkCardDataset()
        df = dataset.load_dataset()
        df = dataset.prepare_labels(df)
        dataset.get_dataset_info(df)
        dataset.save_label_encoders()
        df.to_csv('cards_dataset.csv', index=False, encoding='utf-8')
        
        print("✅ Данные успешно подготовлены")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при подготовке данных: {e}")
        return False

def create_augmented_data():
    """Создает аугментированный датасет"""
    print("\n=== СОЗДАНИЕ АУГМЕНТИРОВАННОГО ДАТАСЕТА ===")
    
    if not os.path.exists('cards_dataset.csv'):
        print("❌ Базовый датасет не найден. Сначала запустите подготовку данных")
        return False
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, "data_augmentation.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Аугментированный датасет успешно создан")
            print(result.stdout)
            return True
        else:
            print(f"❌ Ошибка при создании аугментированного датасета: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка при запуске аугментации: {e}")
        return False

def check_dataset():
    """Запускает диагностику датасета"""
    print("\n=== ДИАГНОСТИКА ДАТАСЕТА ===")
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, "check_dataset.py"], 
                              capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Предупреждения:", result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Ошибка при запуске диагностики: {e}")
        return False

def train_model():
    """Запускает обучение модели"""
    print("\n=== ОБУЧЕНИЕ МОДЕЛИ ===")
    
    try:
        start_time = time.time()
        train_main()
        end_time = time.time()
        
        training_time = end_time - start_time
        print(f"\n✅ Обучение завершено за {training_time/60:.1f} минут")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при обучении модели: {e}")
        return False

def train_augmented_model():
    """Запускает обучение модели с аугментированными данными"""
    print("\n=== ОБУЧЕНИЕ МОДЕЛИ С АУГМЕНТИРОВАННЫМИ ДАННЫМИ ===")
    
    if not os.path.exists('augmented_cards_dataset.csv'):
        print("❌ Аугментированный датасет не найден")
        print("💡 Сначала запустите создание аугментированного датасета")
        return False
    
    try:
        import subprocess
        start_time = time.time()
        
        result = subprocess.run([sys.executable, "train_model_augmented.py"], 
                              capture_output=False, text=True)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        if result.returncode == 0:
            print(f"\n✅ Обучение с аугментированными данными завершено за {training_time/60:.1f} минут")
            return True
        else:
            print(f"❌ Ошибка при обучении с аугментированными данными")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка при запуске обучения: {e}")
        return False

def test_model():
    """Запускает тестирование модели"""
    print("\n=== ТЕСТИРОВАНИЕ МОДЕЛИ ===")
    
    if not os.path.exists('berserk_card_model.tflite'):
        print("❌ Модель не найдена. Сначала запустите обучение")
        return False
    
    try:
        test_main()
        print("\n✅ Тестирование завершено")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании модели: {e}")
        return False

def run_compatibility_check():
    """Запускает полную проверку совместимости"""
    try:
        import subprocess
        result = subprocess.run([sys.executable, "check_compatibility.py"], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Предупреждения:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Ошибка запуска проверки совместимости: {e}")
        return False

def show_project_info():
    """Показывает информацию о проекте"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                 ПРОЕКТ РАСПОЗНАВАНИЯ КАРТ БЕРСЕРК            ║
╠══════════════════════════════════════════════════════════════╣
║ Описание: TensorFlow Lite модель для распознавания карт      ║
║           из коллекционной карточной игры Берсерк            ║
║                                                              ║
║ Возможности:                                                 ║
║ • Распознавание сета карты (s1, s2, s3, s4, s5, laar)       ║
║ • Определение номера карты                                   ║
║ • Определение варианта (normal, pf, alt)                    ║
║ • Оптимизированная модель для мобильных устройств           ║
║                                                              ║
║ Архитектура: MobileNetV2 + Transfer Learning                ║
║ Формат модели: TensorFlow Lite (.tflite)                    ║
║                                                              ║
║ 🔧 Диагностика и настройка:                                 ║
║ python check_compatibility.py  - Проверка совместимости     ║
║ python setup.py               - Автоматическая настройка    ║
╚══════════════════════════════════════════════════════════════╝
    """)

def main():
    parser = argparse.ArgumentParser(
        description='Проект распознавания карт Берсерк',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'action',
        choices=['info', 'check', 'compatibility', 'prepare', 'augment', 'diagnose', 'train', 'train-aug', 'test', 'full', 'full-aug'],
        help='''
Доступные действия:
  info          - Показать информацию о проекте
  check         - Проверить зависимости и данные
  compatibility - Полная проверка совместимости
  prepare       - Подготовить данные для обучения
  augment       - Создать аугментированный датасет
  diagnose      - Диагностика датасета
  train         - Обучить модель (базовый датасет)
  train-aug     - Обучить модель (аугментированный датасет)
  test          - Протестировать модель
  full          - Полный цикл (prepare + train + test)
  full-aug      - Полный цикл с аугментацией (prepare + augment + train-aug + test)
        '''
    )
    
    args = parser.parse_args()
    
    if args.action == 'info':
        show_project_info()
        
    elif args.action == 'check':
        print("\n=== ПРОВЕРКА СИСТЕМЫ ===")
        deps_ok = check_requirements()
        data_ok = check_data()
        
        if deps_ok and data_ok:
            print("\n✅ Система готова к работе")
        else:
            print("\n❌ Система не готова. Исправьте ошибки выше")
            sys.exit(1)
    
    elif args.action == 'compatibility':
        print("\n=== ПОЛНАЯ ПРОВЕРКА СОВМЕСТИМОСТИ ===")
        if not run_compatibility_check():
            sys.exit(1)
    
    elif args.action == 'prepare':
        if not check_requirements() or not check_data():
            sys.exit(1)
        
        if not prepare_data():
            sys.exit(1)
    
    elif args.action == 'augment':
        if not check_requirements() or not check_data():
            sys.exit(1)
        
        # Подготавливаем базовые данные если нужно
        if not os.path.exists('cards_dataset.csv'):
            if not prepare_data():
                sys.exit(1)
        
        if not create_augmented_data():
            sys.exit(1)
    
    elif args.action == 'diagnose':
        if not check_requirements():
            sys.exit(1)
        
        if not check_dataset():
            sys.exit(1)
    
    elif args.action == 'train':
        if not check_requirements() or not check_data():
            sys.exit(1)
        
        # Подготавливаем данные если нужно
        if not os.path.exists('cards_dataset.csv'):
            if not prepare_data():
                sys.exit(1)
        
        if not train_model():
            sys.exit(1)
    
    elif args.action == 'train-aug':
        if not check_requirements() or not check_data():
            sys.exit(1)
        
        # Подготавливаем базовые данные если нужно
        if not os.path.exists('cards_dataset.csv'):
            if not prepare_data():
                sys.exit(1)
        
        # Создаем аугментированные данные если нужно
        if not os.path.exists('augmented_cards_dataset.csv'):
            if not create_augmented_data():
                sys.exit(1)
        
        if not train_augmented_model():
            sys.exit(1)
    
    elif args.action == 'test':
        if not check_requirements():
            sys.exit(1)
        
        if not test_model():
            sys.exit(1)
    
    elif args.action == 'full':
        print("\n🚀 ЗАПУСК ПОЛНОГО ЦИКЛА ОБУЧЕНИЯ")
        
        # Проверки
        if not check_requirements() or not check_data():
            sys.exit(1)
        
        # Подготовка данных
        if not prepare_data():
            sys.exit(1)
        
        # Обучение
        if not train_model():
            sys.exit(1)
        
        # Тестирование
        if not test_model():
            sys.exit(1)
        
        print("\n🎉 ПОЛНЫЙ ЦИКЛ УСПЕШНО ЗАВЕРШЕН!")
        print("\nСозданные файлы:")
        print("• berserk_card_model.tflite - TensorFlow Lite модель")
        print("• berserk_card_model.h5 - Keras модель")
        print("• model_info.json - Информация о модели")
        print("• label_encoders.json - Энкодеры меток")
        print("• cards_dataset.csv - Подготовленный датасет")
        print("• training_history.png - График обучения")
        print("• test_results.png - Результаты тестирования")
    
    elif args.action == 'full-aug':
        print("\n🚀 ЗАПУСК ПОЛНОГО ЦИКЛА С АУГМЕНТАЦИЕЙ")
        
        # Проверки
        if not check_requirements() or not check_data():
            sys.exit(1)
        
        # Подготовка данных
        if not prepare_data():
            sys.exit(1)
        
        # Диагностика
        print("\n📊 Диагностика базового датасета...")
        check_dataset()
        
        # Создание аугментированного датасета
        if not create_augmented_data():
            sys.exit(1)
        
        # Диагностика аугментированного датасета
        print("\n📊 Диагностика аугментированного датасета...")
        check_dataset()
        
        # Обучение с аугментированными данными
        if not train_augmented_model():
            sys.exit(1)
        
        # Тестирование
        if not test_model():
            sys.exit(1)
        
        print("\n🎉 ПОЛНЫЙ ЦИКЛ С АУГМЕНТАЦИЕЙ УСПЕШНО ЗАВЕРШЕН!")
        print("\nСозданные файлы:")
        print("• berserk_card_model_augmented.tflite - TensorFlow Lite модель")
        print("• berserk_card_model_augmented.h5 - Keras модель")
        print("• model_info_augmented.json - Информация о модели")
        print("• augmented_label_encoders.json - Энкодеры меток")
        print("• cards_dataset.csv - Базовый датасет")
        print("• augmented_cards_dataset.csv - Аугментированный датасет")
        print("• training_history_augmented.png - График обучения")
        print("• dataset_statistics.png - Статистика датасета")
        print("• dataset_report.json - Отчет о датасете")

if __name__ == "__main__":
    main()