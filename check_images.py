#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для проверки и очистки поврежденных изображений в датасете
"""

import os
import sys
from PIL import Image
import pandas as pd
from tqdm import tqdm

def check_image(image_path):
    """Проверяет, можно ли открыть и обработать изображение"""
    try:
        with Image.open(image_path) as img:
            # Пытаемся загрузить изображение полностью
            img.load()
            # Проверяем, что изображение имеет разумные размеры
            if img.size[0] < 10 or img.size[1] < 10:
                return False, "Слишком маленькое изображение"
            # Пытаемся конвертировать в RGB
            if img.mode != 'RGB':
                img.convert('RGB')
            return True, "OK"
    except Exception as e:
        return False, str(e)

def check_dataset_images(cards_dir='./cards_augmented', csv_file='augmented_cards_dataset.csv'):
    """Проверяет все изображения в датасете"""
    print("=== ПРОВЕРКА ИЗОБРАЖЕНИЙ В ДАТАСЕТЕ ===")
    
    # Загружаем CSV файл если он существует
    if os.path.exists(csv_file):
        print(f"Загружаем данные из {csv_file}...")
        df = pd.read_csv(csv_file)
        total_files = len(df)
    else:
        print(f"CSV файл {csv_file} не найден. Сканируем директорию {cards_dir}...")
        # Сканируем директорию
        image_files = []
        for root, dirs, files in os.walk(cards_dir):
            for file in files:
                if file.lower().endswith(('.webp', '.jpg', '.jpeg', '.png')):
                    rel_path = os.path.relpath(os.path.join(root, file), cards_dir)
                    image_files.append(rel_path)
        
        df = pd.DataFrame({'filepath': image_files})
        total_files = len(image_files)
    
    print(f"Найдено {total_files} файлов для проверки")
    
    corrupted_files = []
    valid_files = []
    
    # Проверяем каждое изображение
    for idx, row in tqdm(df.iterrows(), total=total_files, desc="Проверка изображений"):
        if 'filepath' in row and row['filepath']:
            image_path = os.path.join(cards_dir, row['filepath'])
        elif 'filename' in row:
            image_path = os.path.join(cards_dir, row['filename'])
        else:
            print(f"Не удается определить путь для строки {idx}")
            continue
            
        # Проверяем существование файла
        if not os.path.exists(image_path):
            corrupted_files.append((image_path, "Файл не найден"))
            continue
            
        # Проверяем изображение
        is_valid, error_msg = check_image(image_path)
        
        if is_valid:
            valid_files.append(image_path)
        else:
            corrupted_files.append((image_path, error_msg))
    
    # Выводим результаты
    print(f"\n=== РЕЗУЛЬТАТЫ ПРОВЕРКИ ===")
    print(f"Всего файлов: {total_files}")
    print(f"Валидных изображений: {len(valid_files)}")
    print(f"Поврежденных файлов: {len(corrupted_files)}")
    print(f"Процент валидных: {(len(valid_files) / total_files * 100):.1f}%")
    
    if corrupted_files:
        print(f"\nПервые 10 поврежденных файлов:")
        for i, (filepath, error) in enumerate(corrupted_files[:10]):
            print(f"{i+1}. {filepath}: {error}")
        
        # Спрашиваем, удалить ли поврежденные файлы
        if len(corrupted_files) > 0:
            response = input(f"\nУдалить {len(corrupted_files)} поврежденных файлов? (y/N): ")
            if response.lower() in ['y', 'yes', 'да']:
                deleted_count = 0
                for filepath, error in corrupted_files:
                    try:
                        if os.path.exists(filepath):
                            os.remove(filepath)
                            deleted_count += 1
                    except Exception as e:
                        print(f"Ошибка при удалении {filepath}: {e}")
                print(f"Удалено {deleted_count} поврежденных файлов")
    
    return valid_files, corrupted_files

def main():
    """Основная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Проверка изображений в датасете')
    parser.add_argument('--cards-dir', default='./cards_augmented', 
                       help='Путь к директории с изображениями')
    parser.add_argument('--csv-file', default='augmented_cards_dataset.csv',
                       help='Путь к CSV файлу с данными')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.cards_dir):
        print(f"Директория {args.cards_dir} не найдена!")
        return
    
    valid_files, corrupted_files = check_dataset_images(args.cards_dir, args.csv_file)
    
    print("\nПроверка завершена!")

if __name__ == "__main__":
    main()