#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для аугментации данных карт Берсерк
Создает дополнительные варианты изображений для решения проблемы
с единичными примерами каждого класса
"""

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import random
from tqdm import tqdm
import json
from data_preparation import BerserkCardDataset

class DataAugmentator:
    def __init__(self, cards_dir='./cards', augmented_dir='./cards_augmented'):
        self.cards_dir = cards_dir
        self.augmented_dir = augmented_dir
        self.ensure_augmented_dir()
        
    def ensure_augmented_dir(self):
        """Создает папку для аугментированных данных"""
        if not os.path.exists(self.augmented_dir):
            os.makedirs(self.augmented_dir)
            print(f"Создана папка: {self.augmented_dir}")
    
    def rotate_image(self, image, angle_range=(-15, 15)):
        """Поворачивает изображение на случайный угол"""
        angle = random.uniform(*angle_range)
        return image.rotate(angle, fillcolor=(255, 255, 255))
    
    def adjust_brightness(self, image, factor_range=(0.8, 1.2)):
        """Изменяет яркость изображения"""
        factor = random.uniform(*factor_range)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    def adjust_contrast(self, image, factor_range=(0.8, 1.2)):
        """Изменяет контрастность изображения"""
        factor = random.uniform(*factor_range)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def adjust_saturation(self, image, factor_range=(0.8, 1.2)):
        """Изменяет насыщенность изображения"""
        factor = random.uniform(*factor_range)
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)
    
    def add_noise(self, image, noise_factor=0.1):
        """Добавляет шум к изображению"""
        image_array = np.array(image)
        noise = np.random.normal(0, noise_factor * 255, image_array.shape)
        noisy_image = image_array + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_image)
    
    def slight_blur(self, image, radius_range=(0.5, 1.5)):
        """Добавляет легкое размытие"""
        radius = random.uniform(*radius_range)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def crop_and_resize(self, image, crop_factor_range=(0.9, 0.95)):
        """Обрезает и изменяет размер изображения"""
        width, height = image.size
        crop_factor = random.uniform(*crop_factor_range)
        
        new_width = int(width * crop_factor)
        new_height = int(height * crop_factor)
        
        left = random.randint(0, width - new_width)
        top = random.randint(0, height - new_height)
        
        cropped = image.crop((left, top, left + new_width, top + new_height))
        return cropped.resize((width, height), Image.Resampling.LANCZOS)
    
    def flip_horizontal(self, image):
        """Отражает изображение по горизонтали (осторожно с текстом!)"""
        return image.transpose(Image.Flip.HORIZONTAL)
    
    def augment_image(self, image, augmentation_type):
        """Применяет определенный тип аугментации"""
        if augmentation_type == 'rotate':
            return self.rotate_image(image)
        elif augmentation_type == 'brightness':
            return self.adjust_brightness(image)
        elif augmentation_type == 'contrast':
            return self.adjust_contrast(image)
        elif augmentation_type == 'saturation':
            return self.adjust_saturation(image)
        elif augmentation_type == 'noise':
            return self.add_noise(image)
        elif augmentation_type == 'blur':
            return self.slight_blur(image)
        elif augmentation_type == 'crop':
            return self.crop_and_resize(image)
        elif augmentation_type == 'combined':
            # Комбинированная аугментация
            aug_image = image
            # Применяем 2-3 случайные аугментации
            augmentations = random.sample(['noise', 'brightness', 'contrast', 'saturation'], 
                                        random.randint(2, 3))
            for aug in augmentations:
                aug_image = self.augment_image(aug_image, aug)
            return aug_image
        else:
            return image
    
    def create_augmented_dataset(self, num_augmentations_per_image=4):
        """Создает аугментированный датасет"""
        print("=== СОЗДАНИЕ АУГМЕНТИРОВАННОГО ДАТАСЕТА ===")
        
        # Загружаем оригинальный датасет
        dataset = BerserkCardDataset(self.cards_dir)
        df = dataset.load_dataset()
        
        augmentation_types = ['rotate', 'brightness', 'contrast', 'saturation', 'combined']
        
        augmented_files = []
        
        print(f"Создаем {num_augmentations_per_image} аугментированных версий для каждого изображения...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Аугментация"):
            # Используем filepath если есть, иначе filename
            if 'filepath' in row and row['filepath']:
                original_path = os.path.join(self.cards_dir, row['filepath'])
            else:
                original_path = os.path.join(self.cards_dir, row['filename'])
            
            try:
                # Загружаем оригинальное изображение
                original_image = Image.open(original_path)
                if original_image.mode != 'RGB':
                    original_image = original_image.convert('RGB')
                
                # Копируем оригинал в новую папку
                original_new_path = os.path.join(self.augmented_dir, row['filename'])
                original_image.save(original_new_path, 'WEBP', quality=95)
                augmented_files.append({
                    'filename': row['filename'],
                    'original_filename': row['filename'],
                    'augmentation_type': 'original'
                })
                
                # Создаем аугментированные версии
                for aug_idx in range(num_augmentations_per_image):
                    aug_type = augmentation_types[aug_idx % len(augmentation_types)]
                    
                    # Создаем аугментированное изображение
                    augmented_image = self.augment_image(original_image, aug_type)
                    
                    # Генерируем имя файла
                    name_without_ext = os.path.splitext(row['filename'])[0]
                    aug_filename = f"{name_without_ext}_aug_{aug_idx + 1}.webp"
                    aug_path = os.path.join(self.augmented_dir, aug_filename)
                    
                    # Сохраняем аугментированное изображение
                    augmented_image.save(aug_path, 'WEBP', quality=95)
                    
                    augmented_files.append({
                        'filename': aug_filename,
                        'original_filename': row['filename'],
                        'augmentation_type': aug_type
                    })
                    
            except Exception as e:
                print(f"Ошибка при обработке {row['filename']}: {e}")
                continue
        
        # Создаем DataFrame с информацией об аугментации
        aug_df = pd.DataFrame(augmented_files)
        aug_df.to_csv('augmentation_info.csv', index=False, encoding='utf-8')
        
        print(f"\n=== РЕЗУЛЬТАТЫ АУГМЕНТАЦИИ ===")
        print(f"Оригинальных изображений: {len(df)}")
        print(f"Всего изображений после аугментации: {len(augmented_files)}")
        print(f"Коэффициент увеличения: {len(augmented_files) / len(df):.1f}x")
        print(f"Аугментированные данные сохранены в: {self.augmented_dir}")
        print(f"Информация об аугментации сохранена в: augmentation_info.csv")
        
        return aug_df
    
    def create_augmented_labels(self):
        """Создает метки для аугментированного датасета"""
        print("\n=== СОЗДАНИЕ МЕТОК ДЛЯ АУГМЕНТИРОВАННОГО ДАТАСЕТА ===")
        
        # Создаем датасет для аугментированных данных
        aug_dataset = BerserkCardDataset(self.augmented_dir)
        aug_df = aug_dataset.load_dataset()
        aug_df = aug_dataset.prepare_labels(aug_df)
        aug_dataset.get_dataset_info(aug_df)
        
        # Сохраняем энкодеры и датасет
        aug_dataset.save_label_encoders('augmented_label_encoders.json')
        aug_df.to_csv('augmented_cards_dataset.csv', index=False, encoding='utf-8')
        
        print("Аугментированный датасет готов!")
        return aug_df

def main():
    print("=== АУГМЕНТАЦИЯ ДАННЫХ КАРТ БЕРСЕРК ===")
    print("Этот скрипт создаст дополнительные варианты каждого изображения")
    print("для решения проблемы с единичными примерами классов.\n")
    
    # Создаем аугментатор
    augmentator = DataAugmentator()
    
    # Спрашиваем количество аугментаций
    try:
        num_augs = int(input("Сколько аугментированных версий создать для каждого изображения? (рекомендуется 4-6): "))
        if num_augs < 1:
            num_augs = 4
    except ValueError:
        num_augs = 4
        print("Используем значение по умолчанию: 4")
    
    # Создаем аугментированный датасет
    aug_info = augmentator.create_augmented_dataset(num_augs)
    
    # Создаем метки для аугментированного датасета
    aug_df = augmentator.create_augmented_labels()
    
    print("\n=== РЕКОМЕНДАЦИИ ===")
    print("1. Теперь используйте папку 'cards_augmented' вместо 'cards' для обучения")
    print("2. В train_model.py измените путь к данным:")
    print("   dataset = BerserkCardDataset('./cards_augmented')")
    print("3. Используйте файл 'augmented_cards_dataset.csv' вместо 'cards_dataset.csv'")
    print("4. Теперь у каждого класса будет достаточно примеров для обучения!")

if __name__ == "__main__":
    main()