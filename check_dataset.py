#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для проверки и диагностики датасета карт
Помогает выявить проблемы перед обучением модели
"""

import os
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import json
from pathlib import Path

def check_directory_structure():
    """Проверяет структуру директорий"""
    print("=== ПРОВЕРКА СТРУКТУРЫ ДИРЕКТОРИЙ ===")
    
    # Проверяем основную папку с картами
    cards_dir = './cards'
    if not os.path.exists(cards_dir):
        print(" Папка './cards' не найдена!")
        return False
    
    print(f"Папка '{cards_dir}' найдена")
    
    # Проверяем подпапки
    subdirs = [d for d in os.listdir(cards_dir) if os.path.isdir(os.path.join(cards_dir, d))]
    
    # Проверяем изображения в корне папки
    root_images = [f for f in os.listdir(cards_dir) 
                   if f.lower().endswith(('.webp', '.jpg', '.jpeg', '.png'))]
    
    total_files = 0
    
    if len(subdirs) > 0:
        print(f" Найдено подпапок: {len(subdirs)}")
        for subdir in subdirs:
            subdir_path = os.path.join(cards_dir, subdir)
            files = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.webp', '.jpg', '.jpeg', '.png'))]
            total_files += len(files)
            print(f"    {subdir}: {len(files)} файлов")
    
    if len(root_images) > 0:
        total_files += len(root_images)
        print(f"📄 В корне папки: {len(root_images)} изображений")
        
        if len(subdirs) == 0:
            print(" Изображения находятся в корне папки cards")
            print(" Для организации по классам запустите: python organize_cards.py")
    
    print(f" Всего изображений: {total_files}")
    
    if total_files == 0:
        print(" Изображения не найдены!")
        return False
    
    return True

def check_base_dataset():
    """Проверяет базовый датасет"""
    print("\n=== ПРОВЕРКА БАЗОВОГО ДАТАСЕТА ===")
    
    csv_file = './cards_dataset.csv'
    if not os.path.exists(csv_file):
        print(" Файл 'cards_dataset.csv' не найден!")
        return False, None
    
    try:
        df = pd.read_csv(csv_file)
        print(f" Файл '{csv_file}' загружен")
        print(f" Всего записей: {len(df)}")
        
        # Проверяем структуру
        required_columns = ['filename', 'card_name', 'set_name', 'split']
        # Проверяем наличие поля filepath (новая структура)
        if 'filepath' in df.columns:
            print(" Обнаружена новая структура данных с полем 'filepath'")
            required_columns.append('filepath')
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f" Отсутствуют столбцы: {missing_columns}")
            return False, None
        
        print(f" Все необходимые столбцы присутствуют: {list(df.columns)}")
        
        # Проверяем распределение по сплитам
        split_counts = df['split'].value_counts()
        print("\n📈 Распределение по сплитам:")
        for split, count in split_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {split}: {count} ({percentage:.1f}%)")
        
        # Проверяем уникальные карты
        unique_cards = df['card_name'].nunique()
        print(f"\n🃏 Уникальных карт: {unique_cards}")
        
        # Проверяем распределение по картам
        card_counts = df['card_name'].value_counts()
        print(f" Среднее количество примеров на карту: {card_counts.mean():.1f}")
        print(f" Минимальное количество примеров: {card_counts.min()}")
        print(f" Максимальное количество примеров: {card_counts.max()}")
        
        # Проверяем карты с малым количеством примеров
        low_count_cards = card_counts[card_counts < 5]
        if len(low_count_cards) > 0:
            print(f"\n  Карты с менее чем 5 примерами ({len(low_count_cards)} карт):")
            for card, count in low_count_cards.head(10).items():
                print(f"   {card}: {count} примеров")
            if len(low_count_cards) > 10:
                print(f"   ... и еще {len(low_count_cards) - 10} карт")
        
        # Проверяем существование файлов
        missing_files = []
        for _, row in df.head(20).iterrows():  # Проверяем только первые 20 для скорости
            # Используем filepath если есть, иначе filename
            if 'filepath' in df.columns and pd.notna(row['filepath']):
                file_path = os.path.join('./cards', row['filepath'])
            else:
                file_path = os.path.join('./cards', row['filename'])
            
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print(f"\n Не найдены файлы (проверено первые 20):")
            for file in missing_files:
                print(f"   {file}")
        else:
            print("\n Файлы существуют (проверено первые 20)")
        
        return True, df
        
    except Exception as e:
        print(f" Ошибка при загрузке датасета: {e}")
        return False, None

def check_augmented_dataset():
    """Проверяет аугментированный датасет"""
    print("\n=== ПРОВЕРКА АУГМЕНТИРОВАННОГО ДАТАСЕТА ===")
    
    csv_file = 'augmented_cards_dataset.csv'
    aug_dir = './cards_augmented'
    
    if not os.path.exists(csv_file):
        print(f" Файл '{csv_file}' не найден")
        print(" Запустите: python data_augmentation.py")
        return False, None
    
    if not os.path.exists(aug_dir):
        print(f" Папка '{aug_dir}' не найдена")
        print(" Запустите: python data_augmentation.py")
        return False, None
    
    print(f" Аугментированный датасет найден")
    
    try:
        df = pd.read_csv(csv_file)
        print(f" Записей в аугментированном датасете: {len(df)}")
        
        # Проверяем структуру
        required_columns = ['filename', 'card_name', 'set_name', 'split']
        # Проверяем наличие поля filepath (новая структура)
        if 'filepath' in df.columns:
            print(" Обнаружена новая структура данных с полем 'filepath'")
            required_columns.append('filepath')
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f" Отсутствуют столбцы: {missing_columns}")
            return False, None
        
        print(f" Все необходимые столбцы присутствуют: {list(df.columns)}")
        
        # Проверяем распределение по картам
        if 'card_name' in df.columns:
            card_counts = df['card_name'].value_counts()
            min_count = card_counts.min()
            max_count = card_counts.max()
            
            print(f" Минимум примеров на карту: {min_count}")
            print(f" Максимум примеров на карту: {max_count}")
            print(f" Среднее примеров на карту: {card_counts.mean():.1f}")
            
            if min_count < 2:
                problem_cards = card_counts[card_counts < 2]
                print(f"\n  ПРОБЛЕМА: {len(problem_cards)} карт все еще с единичными примерами")
                print(" Увеличьте количество аугментаций в data_augmentation.py")
                return False, None
            else:
                print("\n Все карты имеют достаточно примеров для обучения")
        
        # Проверяем типы аугментации
        if 'augmentation_type' in df.columns:
            aug_types = df['augmentation_type'].value_counts()
            print("\n Типы аугментации:")
            for aug_type, count in aug_types.items():
                print(f"   {aug_type}: {count} изображений")
        
        return True, df
        
    except Exception as e:
        print(f" Ошибка при чтении аугментированного датасета: {e}")
        return False, None

def check_model_files():
    """Проверяет наличие файлов модели"""
    print("\n=== ПРОВЕРКА ФАЙЛОВ МОДЕЛИ ===")
    
    model_files = [
        'berserk_card_model_augmented.tflite',
        'berserk_card_model_augmented.h5',
        'model_info_augmented.json',
        'training_history_augmented.png'
    ]
    
    found_files = []
    for file in model_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            if file.endswith('.png'):
                print(f" {file} ({size} байт)")
            else:
                print(f" {file} ({size / 1024 / 1024:.1f} MB)")
            found_files.append(file)
        else:
            print(f" {file} не найден")
    
    if len(found_files) == 0:
        print(" Запустите: python train_model_augmented.py")
    elif len(found_files) < len(model_files):
        print("  Некоторые файлы модели отсутствуют")
    else:
        print(" Все файлы модели найдены")
    
    return found_files

def plot_dataset_statistics(df_base, df_aug):
    """Строит графики статистики датасета"""
    print("\n=== ПОСТРОЕНИЕ ГРАФИКОВ ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Статистика датасета карт Berserk TCG', fontsize=16)
    
    # График 1: Распределение по сетам (базовый датасет)
    if df_base is not None and 'set' in df_base.columns:
        set_counts = df_base['set'].value_counts()
        axes[0, 0].bar(set_counts.index, set_counts.values)
        axes[0, 0].set_title('Распределение по сетам (базовый датасет)')
        axes[0, 0].set_xlabel('Сет')
        axes[0, 0].set_ylabel('Количество карт')
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # График 2: Распределение примеров на карту (базовый)
    if df_base is not None and 'card_name' in df_base.columns:
        card_counts = df_base['card_name'].value_counts()
        axes[0, 1].hist(card_counts.values, bins=20, alpha=0.7, color='blue')
        axes[0, 1].set_title('Распределение примеров на карту (базовый)')
        axes[0, 1].set_xlabel('Количество примеров')
        axes[0, 1].set_ylabel('Количество карт')
    
    # График 3: Распределение по сетам (аугментированный датасет)
    if df_aug is not None and 'set' in df_aug.columns:
        set_counts_aug = df_aug['set'].value_counts()
        axes[1, 0].bar(set_counts_aug.index, set_counts_aug.values, color='orange')
        axes[1, 0].set_title('Распределение по сетам (аугментированный)')
        axes[1, 0].set_xlabel('Сет')
        axes[1, 0].set_ylabel('Количество изображений')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # График 4: Распределение примеров на карту (аугментированный)
    if df_aug is not None and 'card_name' in df_aug.columns:
        card_counts_aug = df_aug['card_name'].value_counts()
        axes[1, 1].hist(card_counts_aug.values, bins=20, alpha=0.7, color='orange')
        axes[1, 1].set_title('Распределение примеров на карту (аугментированный)')
        axes[1, 1].set_xlabel('Количество примеров')
        axes[1, 1].set_ylabel('Количество карт')
    
    plt.tight_layout()
    plt.savefig('dataset_statistics.png', dpi=300, bbox_inches='tight')
    print("Графики сохранены в dataset_statistics.png")
    plt.show()

def generate_report(df_base, df_aug):
    """Генерирует отчет о датасете"""
    print("\n=== ГЕНЕРАЦИЯ ОТЧЕТА ===")
    
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'base_dataset': {},
        'augmented_dataset': {},
        'recommendations': []
    }
    
    # Базовый датасет
    if df_base is not None:
        report['base_dataset'] = {
            'total_images': len(df_base),
            'unique_cards': df_base['card_name'].nunique() if 'card_name' in df_base.columns else 0,
            'sets': df_base['set'].value_counts().to_dict() if 'set' in df_base.columns else {},
            'min_examples_per_card': df_base['card_name'].value_counts().min() if 'card_name' in df_base.columns else 0
        }
    
    # Аугментированный датасет
    if df_aug is not None:
        report['augmented_dataset'] = {
            'total_images': len(df_aug),
            'unique_cards': df_aug['card_name'].nunique() if 'card_name' in df_aug.columns else 0,
            'min_examples_per_card': df_aug['card_name'].value_counts().min() if 'card_name' in df_aug.columns else 0,
            'augmentation_types': df_aug['augmentation_type'].value_counts().to_dict() if 'augmentation_type' in df_aug.columns else {}
        }
    
    # Рекомендации
    if df_base is not None and 'card_name' in df_base.columns:
        min_count = df_base['card_name'].value_counts().min()
        if min_count < 2:
            report['recommendations'].append("Запустите data_augmentation.py для решения проблемы с единичными примерами")
    
    if df_aug is not None and 'card_name' in df_aug.columns:
        min_count_aug = df_aug['card_name'].value_counts().min()
        if min_count_aug < 2:
            report['recommendations'].append("Увеличьте количество аугментаций в data_augmentation.py")
        elif min_count_aug >= 2:
            report['recommendations'].append("Датасет готов для обучения! Запустите train_model_augmented.py")
    
    # Сохраняем отчет
    with open('dataset_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(" Отчет сохранен в dataset_report.json")
    return report

def main():
    """Основная функция проверки"""
    print("ДИАГНОСТИКА ДАТАСЕТА КАРТ BERSERK TCG")
    print("=" * 50)
    
    # Проверяем структуру директорий
    if not check_directory_structure():
        print("\n Критическая ошибка: проблемы со структурой директорий")
        return
    
    # Проверяем базовый датасет
    base_success, df_base = check_base_dataset()
    
    # Проверяем аугментированный датасет
    aug_success, df_aug = check_augmented_dataset()
    
    # Проверяем файлы модели
    model_files = check_model_files()
    
    # Строим графики если есть данные
    if (base_success and df_base is not None) or (aug_success and df_aug is not None):
        try:
            plot_dataset_statistics(df_base if base_success else None, df_aug if aug_success else None)
        except Exception as e:
            print(f" Не удалось построить графики: {e}")
    
    # Генерируем отчет
    report = generate_report(df_base if base_success else None, df_aug if aug_success else None)
    
    # Итоговые рекомендации
    print("\n" + "=" * 50)
    print(" ИТОГОВЫЕ РЕКОМЕНДАЦИИ:")
    
    if len(report['recommendations']) == 0:
        print(" Проблем не обнаружено!")
    else:
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")
    
    print("\n Подробная информация сохранена в:")
    print("   - dataset_report.json (JSON отчет)")
    if os.path.exists('dataset_statistics.png'):
        print("   - dataset_statistics.png (графики)")
    
    print("\n Удачи в обучении модели!")

if __name__ == "__main__":
    main()