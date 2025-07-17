#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Улучшенный скрипт для аугментации данных карт Берсерк
Поддерживает идемпотентность, современные библиотеки аугментации,
гибкую структуру хранения и CLI интерфейс
"""

import os
import sys
import argparse
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations не установлен. Используются базовые аугментации.")

from data_preparation import BerserkCardDataset


@dataclass
class AugmentationConfig:
    """Конфигурация аугментации"""
    num_augmentations: int = 4
    image_quality: int = 95
    target_size: Tuple[int, int] = (224, 224)
    augmentation_types: List[str] = None
    use_albumentations: bool = True
    seed: int = 42
    
    def __post_init__(self):
        if self.augmentation_types is None:
            if ALBUMENTATIONS_AVAILABLE and self.use_albumentations:
                self.augmentation_types = [
                    'rotate', 'brightness_contrast', 'hue_saturation', 
                    'noise', 'blur', 'elastic', 'optical_distortion', 'combined'
                ]
            else:
                self.augmentation_types = [
                    'rotate', 'brightness', 'contrast', 'saturation', 'combined'
                ]


class AdvancedDataAugmentator:
    """Улучшенный аугментатор данных с поддержкой идемпотентности"""
    
    def __init__(self, 
                 cards_dir: str = './cards',
                 augmented_dir: str = './cards_augmented',
                 config: Optional[AugmentationConfig] = None):
        self.cards_dir = Path(cards_dir)
        self.augmented_dir = Path(augmented_dir)
        self.config = config or AugmentationConfig()
        
        # Настройка логирования
        self.setup_logging()
        
        # Файлы для отслеживания состояния
        self.state_file = self.augmented_dir / 'augmentation_state.json'
        self.csv_file = self.augmented_dir / 'augmented_dataset.csv'
        self.encoders_file = self.augmented_dir / 'augmented_label_encoders.json'
        
        # Инициализация аугментаций
        self.setup_augmentations()
        
        # Создание директорий
        self.ensure_directories()
        
        # Загрузка состояния
        self.state = self.load_state()
        
    def setup_logging(self):
        """Настройка логирования"""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_format)
        self.logger = logging.getLogger(__name__)
        
        # Добавляем файловый логгер
        if not self.augmented_dir.exists():
            self.augmented_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(self.augmented_dir / 'augmentation.log')
        file_handler.setFormatter(logging.Formatter(log_format))
        self.logger.addHandler(file_handler)
    
    def setup_augmentations(self):
        """Настройка аугментаций"""
        if ALBUMENTATIONS_AVAILABLE and self.config.use_albumentations:
            self.augmentations = {
                'rotate': A.Rotate(limit=15, p=1.0),
                'brightness_contrast': A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=1.0
                ),
                'hue_saturation': A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1.0
                ),
                'noise': A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                'blur': A.OneOf([
                    A.MotionBlur(blur_limit=3),
                    A.GaussianBlur(blur_limit=3),
                ], p=1.0),
                'elastic': A.ElasticTransform(
                    alpha=50, sigma=5, alpha_affine=5, p=1.0
                ),
                'optical_distortion': A.OpticalDistortion(
                    distort_limit=0.1, shift_limit=0.1, p=1.0
                ),
                'combined': A.Compose([
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.8),
                    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=0.8),
                    A.GaussNoise(var_limit=(5.0, 25.0), p=0.5),
                ])
            }
        else:
            # Fallback к базовым аугментациям
            self.augmentations = self._setup_basic_augmentations()
    
    def _setup_basic_augmentations(self) -> Dict[str, Any]:
        """Базовые аугментации без albumentations"""
        return {
            'rotate': self._rotate_basic,
            'brightness': self._brightness_basic,
            'contrast': self._contrast_basic,
            'saturation': self._saturation_basic,
            'combined': self._combined_basic
        }
    
    def _rotate_basic(self, image: np.ndarray) -> np.ndarray:
        """Базовый поворот"""
        angle = np.random.uniform(-15, 15)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), borderValue=(255, 255, 255))
    
    def _brightness_basic(self, image: np.ndarray) -> np.ndarray:
        """Базовое изменение яркости"""
        factor = np.random.uniform(0.8, 1.2)
        return np.clip(image * factor, 0, 255).astype(np.uint8)
    
    def _contrast_basic(self, image: np.ndarray) -> np.ndarray:
        """Базовое изменение контраста"""
        factor = np.random.uniform(0.8, 1.2)
        mean = image.mean()
        return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
    
    def _saturation_basic(self, image: np.ndarray) -> np.ndarray:
        """Базовое изменение насыщенности"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        factor = np.random.uniform(0.8, 1.2)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    def _combined_basic(self, image: np.ndarray) -> np.ndarray:
        """Комбинированная базовая аугментация"""
        image = self._brightness_basic(image)
        image = self._contrast_basic(image)
        return image
    
    def ensure_directories(self):
        """Создание необходимых директорий"""
        self.augmented_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Рабочая директория: {self.augmented_dir}")
    
    def load_state(self) -> Dict[str, Any]:
        """Загрузка состояния аугментации"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                self.logger.info(f"Загружено состояние: {len(state.get('processed_files', {}))} файлов")
                return state
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки состояния: {e}")
        
        return {
            'processed_files': {},
            'last_update': None,
            'config': self.config.__dict__
        }
    
    def save_state(self):
        """Сохранение состояния аугментации"""
        self.state['last_update'] = datetime.now().isoformat()
        self.state['config'] = self.config.__dict__
        
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
    
    def get_file_hash(self, filepath: Path) -> str:
        """Получение хеша файла для проверки изменений"""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def parse_card_info(self, filename: str, row_data: Optional[Dict] = None) -> Optional[Dict[str, str]]:
        """Парсинг информации о карте из имени файла или данных строки"""
        # Если есть данные строки с уже распарсенной информацией, используем их
        if row_data and all(key in row_data for key in ['set_name', 'card_number', 'variant']):
            return {
                'set_name': str(row_data['set_name']),
                'card_number': str(row_data['card_number']),
                'variant': str(row_data['variant']),
                'base_name': f"{row_data['set_name']}_{row_data['card_number']}_{row_data['variant']}"
            }
        
        # Fallback: парсинг из имени файла (старая логика)
        name = Path(filename).stem
        
        # Убираем суффикс аугментации если есть
        parts = name.split('_')
        if len(parts) >= 4 and parts[-2] == 'aug':
            parts = parts[:-2]
        
        if len(parts) < 2:
            return None
        
        set_name = parts[0]
        card_number = parts[1]
        variant = parts[2] if len(parts) > 2 else 'normal'
        
        return {
            'set_name': set_name,
            'card_number': card_number,
            'variant': variant,
            'base_name': '_'.join(parts)
        }
    
    def get_augmented_structure_path(self, card_info: Dict[str, str], 
                                   filename: str, aug_type: str = None, 
                                   aug_idx: int = None) -> Path:
        """Получение пути для аугментированного файла в структуре"""
        set_dir = self.augmented_dir / card_info['set_name']
        variant_dir = set_dir / card_info['variant']
        
        if aug_type and aug_idx is not None:
            # Аугментированный файл
            base_name = Path(filename).stem
            ext = Path(filename).suffix or '.webp'
            aug_filename = f"{base_name}_aug_{aug_idx + 1}{ext}"
        else:
            # Оригинальный файл
            aug_filename = filename
        
        return variant_dir / aug_filename
    
    def apply_augmentation(self, image: np.ndarray, aug_type: str) -> np.ndarray:
        """Применение аугментации к изображению"""
        np.random.seed(None)  # Сброс seed для случайности
        
        if ALBUMENTATIONS_AVAILABLE and self.config.use_albumentations:
            if aug_type in self.augmentations:
                transform = self.augmentations[aug_type]
                result = transform(image=image)
                return result['image']
        else:
            if aug_type in self.augmentations:
                return self.augmentations[aug_type](image)
        
        return image
    
    def is_file_processed(self, original_path: Path, aug_type: str, aug_idx: int) -> bool:
        """Проверка, был ли файл уже обработан"""
        file_key = str(original_path)
        
        if file_key not in self.state['processed_files']:
            return False
        
        file_info = self.state['processed_files'][file_key]
        current_hash = self.get_file_hash(original_path)
        
        # Проверяем, изменился ли файл
        if file_info.get('hash') != current_hash:
            return False
        
        # Проверяем, есть ли эта аугментация
        aug_key = f"{aug_type}_{aug_idx}"
        return aug_key in file_info.get('augmentations', {})
    
    def mark_file_processed(self, original_path: Path, aug_type: str, 
                          aug_idx: int, output_path: Path):
        """Отметка файла как обработанного"""
        file_key = str(original_path)
        
        if file_key not in self.state['processed_files']:
            self.state['processed_files'][file_key] = {
                'hash': self.get_file_hash(original_path),
                'augmentations': {}
            }
        
        aug_key = f"{aug_type}_{aug_idx}"
        self.state['processed_files'][file_key]['augmentations'][aug_key] = {
            'output_path': str(output_path),
            'created_at': datetime.now().isoformat()
        }
    
    def create_augmented_dataset(self, mode: str = 'full') -> pd.DataFrame:
        """Создание аугментированного датасета"""
        self.logger.info(f"=== СОЗДАНИЕ АУГМЕНТИРОВАННОГО ДАТАСЕТА (режим: {mode}) ===")
        
        # Загрузка оригинального датасета
        dataset = BerserkCardDataset(str(self.cards_dir))
        df = dataset.load_dataset()
        
        if df.empty:
            raise ValueError(f"Не найдено изображений в {self.cards_dir}")
        
        augmented_records = []
        processed_count = 0
        skipped_count = 0
        
        self.logger.info(f"Обработка {len(df)} изображений...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Аугментация"):
            try:
                # Определение пути к оригинальному файлу
                if 'filepath' in row and row['filepath']:
                    original_path = self.cards_dir / row['filepath']
                else:
                    original_path = self.cards_dir / row['filename']
                
                if not original_path.exists():
                    self.logger.warning(f"Файл не найден: {original_path}")
                    continue
                
                # Парсинг информации о карте
                card_info = self.parse_card_info(row['filename'], row.to_dict())
                if not card_info:
                    self.logger.warning(f"Не удалось распарсить: {row['filename']}")
                    continue
                
                # Загрузка изображения
                image = cv2.imread(str(original_path))
                if image is None:
                    self.logger.warning(f"Не удалось загрузить: {original_path}")
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Сохранение оригинала
                original_output_path = self.get_augmented_structure_path(
                    card_info, row['filename']
                )
                original_output_path.parent.mkdir(parents=True, exist_ok=True)
                
                if not original_output_path.exists() or mode == 'full':
                    pil_image = Image.fromarray(image)
                    pil_image.save(original_output_path, 'WEBP', quality=self.config.image_quality)
                    
                    augmented_records.append({
                        'filename': original_output_path.name,
                        'filepath': str(original_output_path.relative_to(self.augmented_dir)),
                        'original_filename': row['filename'],
                        'augmentation_type': 'original',
                        'set_name': card_info['set_name'],
                        'card_number': card_info['card_number'],
                        'variant': card_info['variant'],
                        'full_path': str(original_output_path)
                    })
                
                # Создание аугментированных версий
                for aug_idx in range(self.config.num_augmentations):
                    aug_type = self.config.augmentation_types[aug_idx % len(self.config.augmentation_types)]
                    
                    # Проверка идемпотентности
                    if mode == 'incremental' and self.is_file_processed(original_path, aug_type, aug_idx):
                        skipped_count += 1
                        continue
                    
                    # Применение аугментации
                    augmented_image = self.apply_augmentation(image.copy(), aug_type)
                    
                    # Сохранение аугментированного изображения
                    aug_output_path = self.get_augmented_structure_path(
                        card_info, row['filename'], aug_type, aug_idx
                    )
                    aug_output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    pil_image = Image.fromarray(augmented_image)
                    pil_image.save(aug_output_path, 'WEBP', quality=self.config.image_quality)
                    
                    # Запись в реестр
                    augmented_records.append({
                        'filename': aug_output_path.name,
                        'filepath': str(aug_output_path.relative_to(self.augmented_dir)),
                        'original_filename': row['filename'],
                        'augmentation_type': aug_type,
                        'set_name': card_info['set_name'],
                        'card_number': card_info['card_number'],
                        'variant': card_info['variant'],
                        'full_path': str(aug_output_path)
                    })
                    
                    # Отметка как обработанного
                    self.mark_file_processed(original_path, aug_type, aug_idx, aug_output_path)
                    processed_count += 1
                
            except Exception as e:
                self.logger.error(f"Ошибка при обработке {row['filename']}: {e}")
                continue
        
        # Сохранение состояния
        self.save_state()
        
        # Создание DataFrame
        aug_df = pd.DataFrame(augmented_records)
        
        self.logger.info(f"\n=== РЕЗУЛЬТАТЫ АУГМЕНТАЦИИ ===")
        self.logger.info(f"Оригинальных изображений: {len(df)}")
        self.logger.info(f"Создано аугментаций: {processed_count}")
        self.logger.info(f"Пропущено (уже существуют): {skipped_count}")
        self.logger.info(f"Всего записей в датасете: {len(aug_df)}")
        
        return aug_df
    
    def update_csv_dataset(self, aug_df: pd.DataFrame):
        """Обновление CSV файла с датасетом"""
        # Загрузка существующего CSV если есть
        if self.csv_file.exists():
            try:
                existing_df = pd.read_csv(self.csv_file, encoding='utf-8')
                # Объединение с новыми данными (удаление дубликатов)
                combined_df = pd.concat([existing_df, aug_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['filepath'], keep='last')
                aug_df = combined_df
            except Exception as e:
                self.logger.warning(f"Ошибка при загрузке существующего CSV: {e}")
        
        # Сохранение обновленного CSV
        aug_df.to_csv(self.csv_file, index=False, encoding='utf-8')
        self.logger.info(f"Датасет сохранен в: {self.csv_file}")
    
    def create_labels_and_encoders(self, aug_df: pd.DataFrame):
        """Создание меток и энкодеров для аугментированного датасета"""
        self.logger.info("Создание меток и энкодеров...")
        
        # Создание датасета для аугментированных данных
        aug_dataset = BerserkCardDataset(str(self.augmented_dir))
        
        # Подготовка меток
        aug_df = aug_dataset.prepare_labels(aug_df)
        
        # Сохранение энкодеров
        aug_dataset.save_label_encoders(str(self.encoders_file))
        
        self.logger.info(f"Энкодеры сохранены в: {self.encoders_file}")
        return aug_df
    
    def cleanup_orphaned_files(self) -> int:
        """Удаление аугментированных файлов без оригиналов"""
        self.logger.info("Поиск и удаление устаревших аугментаций...")
        
        removed_count = 0
        
        # Загрузка списка оригинальных файлов
        dataset = BerserkCardDataset(str(self.cards_dir))
        df = dataset.load_dataset()
        original_files = set()
        
        for _, row in df.iterrows():
            if 'filepath' in row and row['filepath']:
                original_files.add(row['filepath'])
            else:
                original_files.add(row['filename'])
        
        # Проход по аугментированным файлам
        for aug_file in self.augmented_dir.rglob('*'):
            if aug_file.is_file() and aug_file.suffix.lower() in ['.webp', '.jpg', '.jpeg', '.png']:
                # Проверка, есть ли оригинал
                relative_path = aug_file.relative_to(self.augmented_dir)
                
                # Если это аугментированный файл (содержит _aug_)
                if '_aug_' in aug_file.stem:
                    # Извлекаем базовое имя
                    base_name = '_'.join(aug_file.stem.split('_')[:-2]) + aug_file.suffix
                    
                    # Ищем оригинал
                    original_found = False
                    for orig_file in original_files:
                        if Path(orig_file).stem == Path(base_name).stem:
                            original_found = True
                            break
                    
                    if not original_found:
                        aug_file.unlink()
                        removed_count += 1
                        self.logger.info(f"Удален устаревший файл: {relative_path}")
        
        # Удаление пустых директорий
        for dir_path in self.augmented_dir.rglob('*'):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                dir_path.rmdir()
        
        self.logger.info(f"Удалено устаревших файлов: {removed_count}")
        return removed_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики аугментированного датасета"""
        if not self.csv_file.exists():
            return {}
        
        try:
            df = pd.read_csv(self.csv_file, encoding='utf-8')
            
            stats = {
                'total_files': len(df),
                'original_files': len(df[df['augmentation_type'] == 'original']),
                'augmented_files': len(df[df['augmentation_type'] != 'original']),
                'sets': df['set_name'].nunique(),
                'variants': df['variant'].nunique(),
                'augmentation_types': df['augmentation_type'].value_counts().to_dict()
            }
            
            return stats
        except Exception as e:
            self.logger.error(f"Ошибка при получении статистики: {e}")
            return {}


def load_config_from_file(config_path: str) -> AugmentationConfig:
    """Загрузка конфигурации из файла"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        return AugmentationConfig(**config_data)
    except Exception as e:
        print(f"Ошибка загрузки конфигурации: {e}")
        return AugmentationConfig()


def create_default_config(config_path: str):
    """Создание файла конфигурации по умолчанию"""
    config = AugmentationConfig()
    config_data = {
        'num_augmentations': config.num_augmentations,
        'image_quality': config.image_quality,
        'target_size': config.target_size,
        'augmentation_types': config.augmentation_types,
        'use_albumentations': config.use_albumentations,
        'seed': config.seed
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)
    
    print(f"Создан файл конфигурации: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Улучшенный скрипт аугментации данных карт Берсерк',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python data_augmentation.py --mode full                    # Полная аугментация
  python data_augmentation.py --mode incremental            # Только новые файлы
  python data_augmentation.py --mode cleanup                # Очистка устаревших
  python data_augmentation.py --config config.json          # Использование конфигурации
  python data_augmentation.py --create-config config.json   # Создание конфигурации
        """
    )
    
    parser.add_argument('--cards-dir', default='./cards',
                       help='Путь к папке с оригинальными картами')
    parser.add_argument('--augmented-dir', default='./cards_augmented',
                       help='Путь к папке для аугментированных данных')
    parser.add_argument('--mode', choices=['full', 'incremental', 'cleanup', 'stats'],
                       default='incremental',
                       help='Режим работы: full - полная аугментация, incremental - только новые, cleanup - очистка, stats - статистика')
    parser.add_argument('--config', help='Путь к файлу конфигурации JSON')
    parser.add_argument('--create-config', help='Создать файл конфигурации по умолчанию')
    parser.add_argument('--num-augmentations', type=int, default=4,
                       help='Количество аугментаций на изображение')
    parser.add_argument('--no-albumentations', action='store_true',
                       help='Не использовать библиотеку albumentations')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed для воспроизводимости')
    
    args = parser.parse_args()
    
    # Создание конфигурации
    if args.create_config:
        create_default_config(args.create_config)
        return
    
    # Загрузка конфигурации
    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = AugmentationConfig(
            num_augmentations=args.num_augmentations,
            use_albumentations=not args.no_albumentations,
            seed=args.seed
        )
    
    # Установка seed
    np.random.seed(config.seed)
    
    # Создание аугментатора
    augmentator = AdvancedDataAugmentator(
        cards_dir=args.cards_dir,
        augmented_dir=args.augmented_dir,
        config=config
    )
    
    try:
        if args.mode == 'stats':
            # Показ статистики
            stats = augmentator.get_statistics()
            if stats:
                print("\n=== СТАТИСТИКА АУГМЕНТИРОВАННОГО ДАТАСЕТА ===")
                print(f"Всего файлов: {stats['total_files']}")
                print(f"Оригинальных: {stats['original_files']}")
                print(f"Аугментированных: {stats['augmented_files']}")
                print(f"Сетов: {stats['sets']}")
                print(f"Вариантов: {stats['variants']}")
                print("\nТипы аугментаций:")
                for aug_type, count in stats['augmentation_types'].items():
                    print(f"  {aug_type}: {count}")
            else:
                print("Статистика недоступна. Запустите аугментацию.")
        
        elif args.mode == 'cleanup':
            # Очистка устаревших файлов
            removed = augmentator.cleanup_orphaned_files()
            print(f"Удалено устаревших файлов: {removed}")
        
        else:
            # Аугментация
            aug_df = augmentator.create_augmented_dataset(mode=args.mode)
            
            if not aug_df.empty:
                # Обновление CSV
                augmentator.update_csv_dataset(aug_df)
                
                # Создание меток и энкодеров
                aug_df = augmentator.create_labels_and_encoders(aug_df)
                
                print("\n=== РЕКОМЕНДАЦИИ ===")
                print(f"1. Используйте папку '{args.augmented_dir}' для обучения")
                print(f"2. CSV файл: {augmentator.csv_file}")
                print(f"3. Энкодеры: {augmentator.encoders_file}")
                print("4. Структура: cards_augmented/<set>/<variant>/<filename>")
            else:
                print("Нет новых файлов для обработки.")
    
    except KeyboardInterrupt:
        print("\nПрервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()