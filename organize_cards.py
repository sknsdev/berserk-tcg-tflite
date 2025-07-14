#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для организации карт в подпапки по классам
Автоматически создает структуру папок на основе названий файлов
"""

import os
import shutil
from pathlib import Path
import re

def organize_cards(cards_dir='./cards'):
    """
    Организует карты в подпапки по классам
    
    Args:
        cards_dir (str): Путь к папке с картами
    """
    cards_path = Path(cards_dir)
    
    if not cards_path.exists():
        print(f"❌ Папка {cards_dir} не найдена")
        return False
    
    # Получаем все изображения в корне папки
    image_extensions = {'.webp', '.jpg', '.jpeg', '.png'}
    image_files = [f for f in cards_path.iterdir() 
                  if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print("❌ Изображения в корне папки не найдены")
        return False
    
    print(f"📁 Найдено {len(image_files)} изображений для организации")
    
    # Анализируем названия файлов и группируем по классам
    class_groups = {}
    
    for image_file in image_files:
        filename = image_file.stem  # Имя без расширения
        
        # Определяем класс на основе названия файла
        class_name = extract_class_from_filename(filename)
        
        if class_name not in class_groups:
            class_groups[class_name] = []
        class_groups[class_name].append(image_file)
    
    print(f"\n📊 Обнаружено {len(class_groups)} классов:")
    for class_name, files in class_groups.items():
        print(f"  • {class_name}: {len(files)} файлов")
    
    # Создаем папки и перемещаем файлы
    moved_count = 0
    
    for class_name, files in class_groups.items():
        class_dir = cards_path / class_name
        
        # Создаем папку класса если её нет
        class_dir.mkdir(exist_ok=True)
        print(f"\n📂 Обрабатываем класс '{class_name}'...")
        
        for image_file in files:
            target_path = class_dir / image_file.name
            
            # Проверяем, не существует ли уже файл в целевой папке
            if target_path.exists():
                print(f"  ⚠️  Файл {image_file.name} уже существует в {class_name}/")
                continue
            
            try:
                shutil.move(str(image_file), str(target_path))
                moved_count += 1
                print(f"  ✅ {image_file.name} → {class_name}/")
            except Exception as e:
                print(f"  ❌ Ошибка при перемещении {image_file.name}: {e}")
    
    print(f"\n🎉 Организация завершена!")
    print(f"📁 Перемещено {moved_count} файлов")
    print(f"📂 Создано {len(class_groups)} папок классов")
    
    return True

def extract_class_from_filename(filename):
    """
    Извлекает класс карты из названия файла
    
    Args:
        filename (str): Название файла без расширения
        
    Returns:
        str: Название класса
    """
    # Паттерны для разных типов названий
    patterns = [
        # laar_1 -> laar
        (r'^([a-zA-Z]+)_\d+$', lambda m: m.group(1)),
        
        # s1_1, s2_100 -> series_1, series_2
        (r'^s(\d+)_\d+$', lambda m: f'series_{m.group(1)}'),
        
        # card_warrior_1 -> warrior
        (r'^\w+_([a-zA-Z]+)_\d+$', lambda m: m.group(1)),
        
        # warrior_fire_1 -> warrior_fire
        (r'^([a-zA-Z]+_[a-zA-Z]+)_\d+$', lambda m: m.group(1)),
        
        # warrior1, mage2 -> warrior, mage
        (r'^([a-zA-Z]+)\d+$', lambda m: m.group(1)),
    ]
    
    for pattern, extractor in patterns:
        match = re.match(pattern, filename)
        if match:
            return extractor(match)
    
    # Если паттерн не найден, используем первую часть до первого числа или подчеркивания
    parts = re.split(r'[_\d]', filename)
    if parts and parts[0]:
        return parts[0]
    
    # Последний вариант - используем 'unknown'
    return 'unknown'

def show_current_structure(cards_dir='./cards'):
    """
    Показывает текущую структуру папки с картами
    """
    cards_path = Path(cards_dir)
    
    if not cards_path.exists():
        print(f"❌ Папка {cards_dir} не найдена")
        return
    
    print(f"📁 Структура папки {cards_dir}:")
    
    # Подсчитываем файлы в корне
    image_extensions = {'.webp', '.jpg', '.jpeg', '.png'}
    root_images = [f for f in cards_path.iterdir() 
                  if f.is_file() and f.suffix.lower() in image_extensions]
    
    if root_images:
        print(f"  📄 В корне: {len(root_images)} изображений")
    
    # Подсчитываем файлы в подпапках
    subdirs = [d for d in cards_path.iterdir() if d.is_dir()]
    
    if subdirs:
        print(f"  📂 Подпапки:")
        for subdir in sorted(subdirs):
            subdir_images = [f for f in subdir.iterdir() 
                           if f.is_file() and f.suffix.lower() in image_extensions]
            print(f"    • {subdir.name}: {len(subdir_images)} изображений")
    else:
        print(f"  📂 Подпапок нет")

def main():
    """
    Главная функция
    """
    print("🃏 ОРГАНИЗАЦИЯ КАРТ ПО КЛАССАМ")
    print("=" * 40)
    
    # Показываем текущую структуру
    print("\n📊 Текущая структура:")
    show_current_structure()
    
    # Проверяем, есть ли файлы в корне для организации
    cards_path = Path('./cards')
    image_extensions = {'.webp', '.jpg', '.jpeg', '.png'}
    root_images = [f for f in cards_path.iterdir() 
                  if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not root_images:
        print("\n✅ Все изображения уже организованы в подпапки!")
        return
    
    # Спрашиваем подтверждение
    print(f"\n❓ Найдено {len(root_images)} изображений в корне папки.")
    print("Хотите организовать их в подпапки по классам? (y/n): ", end="")
    
    try:
        response = input().strip().lower()
        if response in ['y', 'yes', 'да', 'д']:
            print("\n🚀 Начинаем организацию...")
            if organize_cards():
                print("\n📊 Новая структура:")
                show_current_structure()
            else:
                print("\n❌ Организация не удалась")
        else:
            print("\n❌ Организация отменена")
    except KeyboardInterrupt:
        print("\n\n❌ Организация прервана пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")

if __name__ == "__main__":
    main()