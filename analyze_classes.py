#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from collections import Counter

def analyze_class_distribution():
    """Анализирует распределение классов в аугментированном датасете"""
    print("=== АНАЛИЗ РАСПРЕДЕЛЕНИЯ КЛАССОВ ===")
    
    # Загружаем аугментированный датасет
    try:
        df = pd.read_csv('augmented_cards_dataset.csv')
        print(f"Загружен датасет: {len(df)} записей")
    except FileNotFoundError:
        print("❌ Файл 'augmented_cards_dataset.csv' не найден!")
        return
    
    # Анализируем распределение по card_id (основной класс для классификации)
    if 'card_id' in df.columns:
        print("\n=== РАСПРЕДЕЛЕНИЕ ПО CARD_ID ===")
        card_counts = df['card_id'].value_counts().sort_values()
        
        print(f"Всего уникальных card_id: {len(card_counts)}")
        print(f"Минимальное количество примеров: {card_counts.min()}")
        print(f"Максимальное количество примеров: {card_counts.max()}")
        print(f"Среднее количество примеров: {card_counts.mean():.2f}")
        
        # Показываем классы с малым количеством примеров
        single_examples = card_counts[card_counts == 1]
        if len(single_examples) > 0:
            print(f"\n❌ ПРОБЛЕМА: Найдено {len(single_examples)} классов с единственным примером:")
            for card_id in single_examples.index[:10]:  # Показываем первые 10
                print(f"  - {card_id}")
            if len(single_examples) > 10:
                print(f"  ... и еще {len(single_examples) - 10} классов")
        
        # Показываем классы с 2 примерами (тоже проблематично для stratify)
        two_examples = card_counts[card_counts == 2]
        if len(two_examples) > 0:
            print(f"\n⚠️  ВНИМАНИЕ: Найдено {len(two_examples)} классов с двумя примерами:")
            for card_id in two_examples.index[:5]:  # Показываем первые 5
                print(f"  - {card_id}")
            if len(two_examples) > 5:
                print(f"  ... и еще {len(two_examples) - 5} классов")
        
        # Показываем распределение количества примеров
        print("\n=== РАСПРЕДЕЛЕНИЕ КОЛИЧЕСТВА ПРИМЕРОВ ===")
        count_distribution = Counter(card_counts.values)
        for count, num_classes in sorted(count_distribution.items()):
            print(f"  {count} примеров: {num_classes} классов")
    
    # Анализируем другие поля
    if 'card_id_encoded' in df.columns:
        print("\n=== РАСПРЕДЕЛЕНИЕ ПО CARD_ID_ENCODED ===")
        encoded_counts = df['card_id_encoded'].value_counts().sort_values()
        print(f"Всего уникальных encoded классов: {len(encoded_counts)}")
        print(f"Минимальное количество примеров: {encoded_counts.min()}")
        print(f"Максимальное количество примеров: {encoded_counts.max()}")
        
        single_encoded = encoded_counts[encoded_counts == 1]
        if len(single_encoded) > 0:
            print(f"❌ Классов с единственным примером: {len(single_encoded)}")
    
    return df, card_counts

def suggest_solutions(card_counts):
    """Предлагает решения проблемы"""
    print("\n=== РЕКОМЕНДАЦИИ ПО РЕШЕНИЮ ПРОБЛЕМЫ ===")
    
    single_examples = card_counts[card_counts == 1]
    two_examples = card_counts[card_counts == 2]
    
    if len(single_examples) > 0 or len(two_examples) > 0:
        print("\n🔧 ВОЗМОЖНЫЕ РЕШЕНИЯ:")
        print("\n1. Удалить классы с малым количеством примеров:")
        print("   - Удалить все классы с 1 примером")
        print("   - Возможно удалить классы с 2 примерами")
        
        print("\n2. Увеличить аугментацию для проблемных классов:")
        print("   - Создать больше вариантов для классов с малым количеством примеров")
        
        print("\n3. Изменить стратегию разделения данных:")
        print("   - Убрать параметр stratify из train_test_split")
        print("   - Использовать другой подход к разделению")
        
        print("\n4. Объединить похожие классы:")
        print("   - Группировать варианты одной карты в один класс")
        
        # Подсчитываем, сколько данных потеряем
        total_to_remove = len(single_examples) + len(two_examples) * 2
        total_samples = card_counts.sum()
        percentage = (total_to_remove / total_samples) * 100
        
        print(f"\n📊 СТАТИСТИКА:")
        print(f"   Всего примеров: {total_samples}")
        print(f"   Примеров к удалению: {total_to_remove} ({percentage:.1f}%)")
        print(f"   Останется примеров: {total_samples - total_to_remove}")
        print(f"   Останется классов: {len(card_counts) - len(single_examples) - len(two_examples)}")
    else:
        print("✅ Проблем с распределением классов не обнаружено!")

if __name__ == "__main__":
    df, card_counts = analyze_class_distribution()
    if df is not None:
        suggest_solutions(card_counts)