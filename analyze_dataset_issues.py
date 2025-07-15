#!/usr/bin/env python3
"""
Скрипт для анализа проблем с датасетом карт Берсерк
"""

import os
import pandas as pd
import numpy as np
from collections import Counter
import json
from data_preparation import BerserkCardDataset

def analyze_dataset_structure():
    """Анализирует структуру папок с картами"""
    print("=== АНАЛИЗ СТРУКТУРЫ ДАТАСЕТА ===")
    
    # Проверяем основные папки
    folders_to_check = ['./cards', './cards_augmented']
    
    for folder in folders_to_check:
        if os.path.exists(folder):
            print(f"\n📁 Папка: {folder}")
            
            # Подсчитываем файлы в корне
            root_files = [f for f in os.listdir(folder) 
                         if f.lower().endswith(('.webp', '.jpg', '.jpeg', '.png'))]
            print(f"   Файлов в корне: {len(root_files)}")
            
            # Проверяем подпапки
            subdirs = [d for d in os.listdir(folder) 
                      if os.path.isdir(os.path.join(folder, d))]
            
            if subdirs:
                print(f"   Подпапок: {len(subdirs)}")
                total_subdir_files = 0
                
                for subdir in subdirs[:10]:  # Показываем первые 10
                    subdir_path = os.path.join(folder, subdir)
                    subdir_files = [f for f in os.listdir(subdir_path) 
                                   if f.lower().endswith(('.webp', '.jpg', '.jpeg', '.png'))]
                    total_subdir_files += len(subdir_files)
                    print(f"     {subdir}: {len(subdir_files)} файлов")
                
                if len(subdirs) > 10:
                    print(f"     ... и еще {len(subdirs) - 10} подпапок")
                    
                    # Подсчитываем общее количество в оставшихся папках
                    for subdir in subdirs[10:]:
                        subdir_path = os.path.join(folder, subdir)
                        subdir_files = [f for f in os.listdir(subdir_path) 
                                       if f.lower().endswith(('.webp', '.jpg', '.jpeg', '.png'))]
                        total_subdir_files += len(subdir_files)
                
                print(f"   Всего файлов в подпапках: {total_subdir_files}")
                print(f"   Общее количество файлов: {len(root_files) + total_subdir_files}")
            else:
                print(f"   Общее количество файлов: {len(root_files)}")
        else:
            print(f"\n❌ Папка {folder} не найдена")

def analyze_csv_files():
    """Анализирует CSV файлы с данными"""
    print("\n=== АНАЛИЗ CSV ФАЙЛОВ ===")
    
    csv_files = ['cards_dataset.csv', 'augmented_cards_dataset.csv']
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            print(f"\n📄 Файл: {csv_file}")
            df = pd.read_csv(csv_file)
            
            print(f"   Строк в CSV: {len(df)}")
            print(f"   Столбцы: {list(df.columns)}")
            
            if 'card_id_encoded' in df.columns:
                unique_classes = df['card_id_encoded'].nunique()
                max_class = df['card_id_encoded'].max()
                min_class = df['card_id_encoded'].min()
                print(f"   Уникальных классов: {unique_classes}")
                print(f"   Диапазон классов: {min_class} - {max_class}")
                
                # Проверяем пропуски в нумерации классов
                all_classes = set(range(min_class, max_class + 1))
                actual_classes = set(df['card_id_encoded'].unique())
                missing_classes = all_classes - actual_classes
                
                if missing_classes:
                    print(f"   ⚠️  Пропущенные классы: {sorted(missing_classes)}")
                else:
                    print(f"   ✅ Нумерация классов непрерывная")
            
            # Анализируем распределение по классам
            if 'card_id' in df.columns:
                class_counts = df['card_id'].value_counts()
                print(f"   Классов с 1 примером: {sum(class_counts == 1)}")
                print(f"   Классов с 2-5 примерами: {sum((class_counts >= 2) & (class_counts <= 5))}")
                print(f"   Классов с >5 примерами: {sum(class_counts > 5)}")
                print(f"   Максимум примеров в классе: {class_counts.max()}")
                print(f"   Минимум примеров в классе: {class_counts.min()}")
        else:
            print(f"\n❌ Файл {csv_file} не найден")

def analyze_saved_arrays():
    """Анализирует сохраненные массивы данных"""
    print("\n=== АНАЛИЗ СОХРАНЕННЫХ МАССИВОВ ===")
    
    if os.path.exists('X_data.npy') and os.path.exists('y_data.npy'):
        print("\n📊 Загружаем сохраненные массивы...")
        
        X = np.load('X_data.npy')
        y = np.load('y_data.npy')
        
        print(f"   Форма X: {X.shape}")
        print(f"   Форма y: {y.shape}")
        print(f"   Тип данных X: {X.dtype}")
        print(f"   Тип данных y: {y.dtype}")
        
        unique_classes = len(np.unique(y))
        max_class = np.max(y)
        min_class = np.min(y)
        
        print(f"   Уникальных классов: {unique_classes}")
        print(f"   Диапазон классов: {min_class} - {max_class}")
        
        # Проверяем пропуски в нумерации
        all_classes = set(range(min_class, max_class + 1))
        actual_classes = set(y)
        missing_classes = all_classes - actual_classes
        
        if missing_classes:
            print(f"   ⚠️  Пропущенные классы: {sorted(missing_classes)}")
            print(f"   ❗ Это может вызвать ошибку при обучении!")
        else:
            print(f"   ✅ Нумерация классов непрерывная")
        
        # Анализируем распределение классов
        class_counts = Counter(y)
        counts_distribution = Counter(class_counts.values())
        
        print(f"\n   Распределение количества примеров:")
        for count, num_classes in sorted(counts_distribution.items()):
            print(f"     {count} примеров: {num_classes} классов")
        
        return X, y
    else:
        print("\n❌ Сохраненные массивы не найдены")
        return None, None

def check_label_encoders():
    """Проверяет энкодеры меток"""
    print("\n=== АНАЛИЗ ЭНКОДЕРОВ МЕТОК ===")
    
    if os.path.exists('label_encoders.json'):
        print("\n🔧 Загружаем энкодеры...")
        
        with open('label_encoders.json', 'r', encoding='utf-8') as f:
            encoders = json.load(f)
        
        for encoder_name, encoder_data in encoders.items():
            classes = encoder_data['classes']
            print(f"   {encoder_name}: {len(classes)} классов")
            
            if encoder_name == 'card_id':
                print(f"     Примеры классов: {classes[:5]}...")
                print(f"     Максимальный индекс: {len(classes) - 1}")
    else:
        print("\n❌ Файл энкодеров не найден")

def suggest_fixes():
    """Предлагает способы исправления проблем"""
    print("\n=== РЕКОМЕНДАЦИИ ПО ИСПРАВЛЕНИЮ ===")
    
    print("\n🔧 Возможные решения:")
    print("\n1. Проблема с малым количеством изображений:")
    print("   - Проверьте, что используется правильная папка с данными")
    print("   - Убедитесь, что аугментация данных работает корректно")
    print("   - Запустите: python data_augmentation.py")
    
    print("\n2. Проблема с пропущенными классами:")
    print("   - Пересоздайте энкодеры меток")
    print("   - Удалите файлы X_data.npy, y_data.npy и пересоздайте их")
    print("   - Проверьте целостность исходных данных")
    
    print("\n3. Проблема с несоответствием классов:")
    print("   - Убедитесь, что все классы представлены в обучающей выборке")
    print("   - Используйте stratified split для разделения данных")
    print("   - Проверьте, что нумерация классов начинается с 0")
    
    print("\n4. Команды для исправления:")
    print("   # Очистка и пересоздание данных")
    print("   rm X_data.npy y_data.npy label_encoders.json")
    print("   python data_preparation.py")
    print("   python train_model.py")

def main():
    """Основная функция анализа"""
    print("🔍 ДИАГНОСТИКА ПРОБЛЕМ С ДАТАСЕТОМ КАРТ БЕРСЕРК")
    print("=" * 60)
    
    # Анализируем структуру папок
    analyze_dataset_structure()
    
    # Анализируем CSV файлы
    analyze_csv_files()
    
    # Анализируем сохраненные массивы
    X, y = analyze_saved_arrays()
    
    # Проверяем энкодеры
    check_label_encoders()
    
    # Предлагаем решения
    suggest_fixes()
    
    print("\n" + "=" * 60)
    print("✅ Анализ завершен")

if __name__ == "__main__":
    main()