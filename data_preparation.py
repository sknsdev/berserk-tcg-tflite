import os
import re
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json

class BerserkCardDataset:
    def __init__(self, cards_dir='./cards'):
        self.cards_dir = cards_dir
        self.data = []
        self.label_encoders = {
            'set_name': LabelEncoder(),
            'card_number': LabelEncoder(),
            'variant': LabelEncoder()
        }
        
    def parse_filename(self, filename):
        """Парсит название файла карты и извлекает информацию"""
        # Убираем расширение
        name = os.path.splitext(filename)[0]
        
        # Разбиваем по подчеркиванию
        parts = name.split('_')
        
        if len(parts) < 2:
            return None
            
        set_name = parts[0]  # s1, s2, laar и т.д.
        card_number = parts[1]  # номер карты
        variant = parts[2] if len(parts) > 2 else 'normal'  # pf, alt или normal
        
        return {
            'filename': filename,
            'set_name': set_name,
            'card_number': card_number,
            'variant': variant,
            'full_name': name
        }
    
    def load_dataset(self):
        """Загружает и парсит все файлы карт"""
        print("Загрузка датасета...")
        
        # Проверяем структуру папки
        subdirs = [d for d in os.listdir(self.cards_dir) 
                   if os.path.isdir(os.path.join(self.cards_dir, d))]
        
        if len(subdirs) > 0:
            # Структура с подпапками - загружаем из подпапок
            print("Найдена структура с подпапками, загружаем из подпапок...")
            for subdir in subdirs:
                subdir_path = os.path.join(self.cards_dir, subdir)
                for filename in os.listdir(subdir_path):
                    if filename.lower().endswith(('.webp', '.jpg', '.jpeg', '.png')):
                        card_info = self.parse_filename(filename)
                        if card_info:
                            # Добавляем путь к подпапке
                            card_info['filepath'] = os.path.join(subdir, filename)
                            card_info['class_from_folder'] = subdir
                            self.data.append(card_info)
        else:
            # Изображения в корне папки
            print("Загружаем изображения из корня папки...")
            for filename in os.listdir(self.cards_dir):
                if filename.lower().endswith(('.webp', '.jpg', '.jpeg', '.png')):
                    card_info = self.parse_filename(filename)
                    if card_info:
                        card_info['filepath'] = filename
                        card_info['class_from_folder'] = None
                        self.data.append(card_info)
        
        print(f"Найдено {len(self.data)} карт")
        return pd.DataFrame(self.data)
    
    def prepare_labels(self, df):
        """Подготавливает метки для обучения"""
        # Кодируем категориальные переменные
        df['set_encoded'] = self.label_encoders['set_name'].fit_transform(df['set_name'])
        df['card_encoded'] = self.label_encoders['card_number'].fit_transform(df['card_number'])
        df['variant_encoded'] = self.label_encoders['variant'].fit_transform(df['variant'])
        
        # Создаем уникальный ID для каждой карты
        df['card_id'] = df['set_name'] + '_' + df['card_number'] + '_' + df['variant']
        
        # Кодируем полный ID карты
        card_id_encoder = LabelEncoder()
        df['card_id_encoded'] = card_id_encoder.fit_transform(df['card_id'])
        
        self.label_encoders['card_id'] = card_id_encoder
        
        return df
    
    def load_and_preprocess_image(self, image_path, target_size=(224, 224)):
        """Загружает и предобрабатывает изображение"""
        try:
            image = Image.open(image_path)
            # Конвертируем в RGB если нужно
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Изменяем размер
            image = image.resize(target_size)
            
            # Конвертируем в numpy array и нормализуем
            image_array = np.array(image) / 255.0
            
            return image_array
        except Exception as e:
            print(f"Ошибка при загрузке {image_path}: {e}")
            return None
    
    def create_dataset_arrays(self, df, target_size=(224, 224)):
        """Создает массивы изображений и меток"""
        images = []
        labels = []
        
        print("Загрузка изображений...")
        for idx, row in df.iterrows():
            # Используем filepath если есть, иначе filename
            if 'filepath' in row and row['filepath']:
                image_path = os.path.join(self.cards_dir, row['filepath'])
            else:
                image_path = os.path.join(self.cards_dir, row['filename'])
                
            image = self.load_and_preprocess_image(image_path, target_size)
            
            if image is not None:
                images.append(image)
                labels.append(row['card_id_encoded'])
            
            if (idx + 1) % 100 == 0:
                print(f"Обработано {idx + 1}/{len(df)} изображений")
        
        return np.array(images), np.array(labels)
    
    def save_label_encoders(self, filepath='label_encoders.json'):
        """Сохраняет энкодеры меток"""
        encoders_data = {}
        for name, encoder in self.label_encoders.items():
            encoders_data[name] = {
                'classes': encoder.classes_.tolist()
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(encoders_data, f, ensure_ascii=False, indent=2)
        
        print(f"Энкодеры сохранены в {filepath}")
    
    def get_dataset_info(self, df):
        """Выводит информацию о датасете"""
        print("\n=== ИНФОРМАЦИЯ О ДАТАСЕТЕ ===")
        print(f"Общее количество карт: {len(df)}")
        print(f"\nКоличество карт по сетам:")
        print(df['set_name'].value_counts().sort_index())
        print(f"\nВарианты карт:")
        print(df['variant'].value_counts())
        print(f"\nУникальных карт: {df['card_id'].nunique()}")
        
if __name__ == "__main__":
    # Создаем экземпляр класса
    dataset = BerserkCardDataset()
    
    # Загружаем данные
    df = dataset.load_dataset()
    
    # Подготавливаем метки
    df = dataset.prepare_labels(df)
    
    # Выводим информацию
    dataset.get_dataset_info(df)
    
    # Сохраняем энкодеры
    dataset.save_label_encoders()
    
    # Сохраняем DataFrame
    df.to_csv('cards_dataset.csv', index=False, encoding='utf-8')
    print("\nДатасет сохранен в cards_dataset.csv")
    
    print("\nПример данных:")
    print(df.head())