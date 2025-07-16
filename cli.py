#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI интерфейс для проекта Berserk TCG TFLite
Позволяет поэтапно выполнять все операции от подготовки данных до готовой модели
"""

import argparse
import os
import sys
from pathlib import Path

def check_environment():
    """Проверяет готовность окружения"""
    print("🔍 Проверка окружения...")
    
    # Проверяем виртуальное окружение
    if not Path("venv").exists():
        print("❌ Виртуальное окружение не найдено")
        print("Запустите: python setup.py")
        return False
    
    # Проверяем папку с картами
    if not Path("cards").exists():
        print("❌ Папка 'cards' не найдена")
        print("Создайте папку 'cards' и поместите в неё изображения карт")
        return False
    
    # Подсчитываем карты
    card_files = list(Path("cards").glob("**/*.webp"))
    if len(card_files) == 0:
        print("❌ В папке 'cards' нет файлов .webp")
        return False
    
    print(f"✅ Найдено {len(card_files)} изображений карт")
    return True

def check_augmented_data():
    """Проверяет наличие аугментированных данных"""
    if not Path("cards_augmented").exists():
        return False
    
    # Проверяем наличие файлов в новой структуре (cards_augmented/<set>/<variant>/)
    augmented_files = list(Path("cards_augmented").rglob("*.webp"))
    csv_file = Path("cards_augmented/augmented_dataset.csv")
    
    return len(augmented_files) > 0 and csv_file.exists()

def create_augmented_data():
    """Создает аугментированный датасет"""
    print("\n=== СОЗДАНИЕ АУГМЕНТИРОВАННОГО ДАТАСЕТА ===")
    
    if check_augmented_data():
        response = input("Аугментированные данные уже существуют. Пересоздать? (y/N): ")
        if response.lower() != 'y':
            print("Используем существующие аугментированные данные")
            return True
    
    try:
        from data_augmentation import AdvancedDataAugmentator, AugmentationConfig
        
        # Спрашиваем количество аугментаций
        try:
            num_augs = int(input("Количество аугментированных версий на изображение (рекомендуется 4-6): ") or "4")
            if num_augs < 1:
                num_augs = 4
        except ValueError:
            num_augs = 4
        
        # Создаем конфигурацию
        config = AugmentationConfig(num_augmentations=num_augs)
        augmentator = AdvancedDataAugmentator(config=config)
        
        # Создаем аугментированный датасет
        aug_df = augmentator.create_augmented_dataset(mode='full')
        
        if not aug_df.empty:
            # Обновляем CSV и создаем энкодеры
            augmentator.update_csv_dataset(aug_df)
            aug_df = augmentator.create_labels_and_encoders(aug_df)
        
        print("✅ Аугментированный датасет создан")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при создании аугментированного датасета: {e}")
        return False

def train_new_model():
    """Обучает новую модель"""
    print("\n=== ОБУЧЕНИЕ НОВОЙ МОДЕЛИ ===")
    
    # Проверяем аугментированные данные
    if not check_augmented_data():
        print("❌ Аугментированные данные не найдены")
        print("Сначала выполните: python cli.py augment")
        return False
    
    try:
        # Удаляем старые файлы модели если есть
        old_files = ['berserk_card_model.h5', 'berserk_card_model.tflite', 'model_info.json']
        for file in old_files:
            if Path(file).exists():
                Path(file).unlink()
                print(f"Удален старый файл: {file}")
        
        from train_model import main as train_main
        train_main()
        
        print("✅ Обучение завершено")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при обучении: {e}")
        return False

def continue_training():
    """Дообучает существующую модель"""
    print("\n=== ДООБУЧЕНИЕ СУЩЕСТВУЮЩЕЙ МОДЕЛИ ===")
    
    # Проверяем наличие модели
    if not Path("berserk_card_model.h5").exists():
        print("❌ Существующая модель не найдена")
        print("Сначала обучите новую модель: python cli.py train")
        return False
    
    # Проверяем аугментированные данные
    if not check_augmented_data():
        print("❌ Аугментированные данные не найдены")
        print("Сначала выполните: python cli.py augment")
        return False
    
    try:
        from train_model import continue_training_main
        continue_training_main()
        
        print("✅ Дообучение завершено")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при дообучении: {e}")
        return False

def test_model():
    """Тестирует модель"""
    print("\n=== ТЕСТИРОВАНИЕ МОДЕЛИ ===")
    
    if not Path("berserk_card_model.tflite").exists():
        print("❌ TensorFlow Lite модель не найдена")
        print("Сначала обучите модель: python cli.py train")
        return False
    
    try:
        from test_model import main as test_main
        test_main()
        
        print("✅ Тестирование завершено")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        return False

def check_dataset():
    """Проверяет датасет"""
    print("\n=== ПРОВЕРКА ДАТАСЕТА ===")
    
    try:
        from check_dataset import main as check_main
        check_main()
        
        print("✅ Проверка датасета завершена")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при проверке датасета: {e}")
        return False

def run_web_demo():
    """Запускает веб-демонстрацию"""
    print("\n=== ЗАПУСК ВЕБ-ДЕМОНСТРАЦИИ ===")
    
    if not Path("berserk_card_model.tflite").exists():
        print("❌ TensorFlow Lite модель не найдена")
        print("Сначала обучите модель: python cli.py train")
        return False
    
    try:
        from web_demo import main as web_main
        web_main()
        
    except Exception as e:
        print(f"❌ Ошибка при запуске веб-демонстрации: {e}")
        return False

def full_pipeline():
    """Выполняет полный пайплайн от начала до конца"""
    print("\n=== ПОЛНЫЙ ПАЙПЛАЙН ОБУЧЕНИЯ ===")
    
    if not check_environment():
        return False
    
    # 1. Создаем аугментированные данные
    if not create_augmented_data():
        return False
    
    # 2. Обучаем модель
    if not train_new_model():
        return False
    
    # 3. Тестируем модель
    if not test_model():
        return False
    
    print("\n🎉 Полный пайплайн завершен успешно!")
    print("Модель готова к использованию.")
    print("Запустите веб-демонстрацию: python cli.py web")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="CLI для проекта Berserk TCG TFLite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python cli.py check          # Проверить окружение и данные
  python cli.py augment        # Создать аугментированный датасет
  python cli.py train          # Обучить новую модель
  python cli.py continue       # Дообучить существующую модель
  python cli.py test           # Протестировать модель
  python cli.py web            # Запустить веб-демонстрацию
  python cli.py full           # Выполнить полный пайплайн
        """
    )
    
    parser.add_argument(
        'command',
        choices=['check', 'augment', 'train', 'continue', 'test', 'web', 'full'],
        help='Команда для выполнения'
    )
    
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    args = parser.parse_args()
    
    print("🚀 BERSERK TCG TFLITE - CLI ИНТЕРФЕЙС")
    print("=" * 50)
    
    if args.command == 'check':
        check_environment()
        check_dataset()
    elif args.command == 'augment':
        if check_environment():
            create_augmented_data()
    elif args.command == 'train':
        if check_environment():
            train_new_model()
    elif args.command == 'continue':
        if check_environment():
            continue_training()
    elif args.command == 'test':
        if check_environment():
            test_model()
    elif args.command == 'web':
        if check_environment():
            run_web_demo()
    elif args.command == 'full':
        full_pipeline()

if __name__ == "__main__":
    main()