import tensorflow as tf
import numpy as np
import json
from PIL import Image
import os
import random
from data_preparation import BerserkCardDataset
import matplotlib.pyplot as plt

class BerserkCardPredictor:
    def __init__(self, model_path='berserk_card_model.tflite', model_info_path='model_info.json'):
        self.model_path = model_path
        self.model_info_path = model_info_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.model_info = None
        self.label_encoders = None
        
        self.load_model()
        self.load_model_info()
    
    def load_model(self):
        """Загружает TensorFlow Lite модель"""
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print(f"Модель загружена: {self.model_path}")
            print(f"Входной тензор: {self.input_details[0]['shape']}")
            print(f"Выходной тензор: {self.output_details[0]['shape']}")
            
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            raise
    
    def load_model_info(self):
        """Загружает информацию о модели и энкодеры"""
        try:
            with open(self.model_info_path, 'r', encoding='utf-8') as f:
                self.model_info = json.load(f)
            
            if 'label_encoders' in self.model_info:
                self.label_encoders = self.model_info['label_encoders']
            else:
                # Загружаем из отдельного файла
                with open('label_encoders.json', 'r', encoding='utf-8') as f:
                    self.label_encoders = json.load(f)
            
            print("Информация о модели загружена")
            
        except Exception as e:
            print(f"Ошибка при загрузке информации о модели: {e}")
            raise
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """Предобрабатывает изображение для предсказания"""
        try:
            # Загружаем изображение
            image = Image.open(image_path)
            
            # Конвертируем в RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Изменяем размер
            image = image.resize(target_size)
            
            # Конвертируем в numpy array и нормализуем
            image_array = np.array(image, dtype=np.float32) / 255.0
            
            # Добавляем batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array, image
            
        except Exception as e:
            print(f"Ошибка при обработке изображения {image_path}: {e}")
            return None, None
    
    def predict(self, image_path):
        """Делает предсказание для изображения"""
        # Предобрабатываем изображение
        processed_image, original_image = self.preprocess_image(image_path)
        
        if processed_image is None:
            return None
        
        # Устанавливаем входной тензор
        self.interpreter.set_tensor(self.input_details[0]['index'], processed_image)
        
        # Запускаем инференс
        self.interpreter.invoke()
        
        # Получаем результат
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Получаем предсказанный класс и вероятность
        predicted_class = np.argmax(output_data[0])
        confidence = float(np.max(output_data[0]))
        
        # Декодируем предсказание
        card_info = self.decode_prediction(predicted_class)
        
        return {
            'predicted_class': int(predicted_class),
            'confidence': confidence,
            'card_info': card_info,
            'probabilities': output_data[0].tolist(),
            'original_image': original_image
        }
    
    def decode_prediction(self, predicted_class):
        """Декодирует предсказанный класс в информацию о карте"""
        try:
            # Получаем список всех card_id
            card_ids = self.label_encoders['card_id']['classes']
            
            if predicted_class < len(card_ids):
                card_id = card_ids[predicted_class]
                
                # Парсим card_id
                parts = card_id.split('_')
                if len(parts) >= 3:
                    set_name = parts[0]
                    card_number = parts[1]
                    variant = parts[2]
                    
                    return {
                        'card_id': card_id,
                        'set_name': set_name,
                        'card_number': card_number,
                        'variant': variant
                    }
            
            return {'card_id': 'unknown', 'set_name': 'unknown', 'card_number': 'unknown', 'variant': 'unknown'}
            
        except Exception as e:
            print(f"Ошибка при декодировании предсказания: {e}")
            return {'card_id': 'error', 'set_name': 'error', 'card_number': 'error', 'variant': 'error'}
    
    def test_random_images(self, cards_dir='./cards', num_images=5):
        """Тестирует модель на случайных изображениях"""
        # Получаем список всех изображений (включая подпапки)
        image_files = []
        image_paths = []
        
        # Проверяем изображения в корне папки
        for f in os.listdir(cards_dir):
            if f.lower().endswith(('.webp', '.jpg', '.jpeg', '.png')):
                image_files.append(f)
                image_paths.append(os.path.join(cards_dir, f))
        
        # Проверяем изображения в подпапках
        for subdir in os.listdir(cards_dir):
            subdir_path = os.path.join(cards_dir, subdir)
            if os.path.isdir(subdir_path):
                for f in os.listdir(subdir_path):
                    if f.lower().endswith(('.webp', '.jpg', '.jpeg', '.png')):
                        image_files.append(f)
                        image_paths.append(os.path.join(subdir_path, f))
        
        if not image_files:
            print("Изображения не найдены!")
            return []
        
        # Выбираем случайные изображения
        indices = random.sample(range(len(image_files)), min(num_images, len(image_files)))
        
        results = []
        
        print(f"\n=== ТЕСТИРОВАНИЕ НА {len(indices)} СЛУЧАЙНЫХ ИЗОБРАЖЕНИЯХ ===")
        
        for i, idx in enumerate(indices):
            image_file = image_files[idx]
            image_path = image_paths[idx]
            
            print(f"\n--- Изображение {i+1}: {image_file} ---")
            
            # Получаем истинную информацию из названия файла
            dataset = BerserkCardDataset()
            true_info = dataset.parse_filename(image_file)
            
            # Делаем предсказание
            prediction = self.predict(image_path)
            
            if prediction:
                print(f"Истинная карта: {true_info['set_name']}_{true_info['card_number']}_{true_info['variant']}")
                print(f"Предсказанная карта: {prediction['card_info']['card_id']}")
                print(f"Уверенность: {prediction['confidence']:.4f}")
                
                # Проверяем правильность предсказания
                true_card_id = f"{true_info['set_name']}_{true_info['card_number']}_{true_info['variant']}"
                predicted_card_id = prediction['card_info']['card_id']
                
                is_correct = true_card_id == predicted_card_id
                print(f"Правильно: {'ДА' if is_correct else 'НЕТ'}")
                
                results.append({
                    'image_file': image_file,
                    'image_path': image_path,
                    'true_card_id': true_card_id,
                    'predicted_card_id': predicted_card_id,
                    'confidence': prediction['confidence'],
                    'is_correct': is_correct
                })
            else:
                print("Ошибка при предсказании")
        
        # Выводим общую статистику
        if results:
            correct_predictions = sum(1 for r in results if r['is_correct'])
            accuracy = correct_predictions / len(results)
            avg_confidence = sum(r['confidence'] for r in results) / len(results)
            
            print(f"\n=== РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ===")
            print(f"Точность: {accuracy:.4f} ({correct_predictions}/{len(results)})")
            print(f"Средняя уверенность: {avg_confidence:.4f}")
        
        return results
    
    def visualize_predictions(self, results, cards_dir='./cards'):
        """Визуализирует результаты предсказаний"""
        if not results:
            print("Нет результатов для визуализации")
            return
        
        # Создаем сетку изображений
        num_images = len(results)
        cols = min(3, num_images)
        rows = (num_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, result in enumerate(results):
            if i >= len(axes):
                break
                
            # Загружаем изображение
            # Используем image_path если есть, иначе строим путь из cards_dir и image_file
            if 'image_path' in result:
                image_path = result['image_path']
            else:
                image_path = os.path.join(cards_dir, result['image_file'])
            
            image = Image.open(image_path)
            
            # Отображаем изображение
            axes[i].imshow(image)
            axes[i].axis('off')
            
            # Заголовок с результатами
            color = 'green' if result['is_correct'] else 'red'
            title = f"True: {result['true_card_id']}\nОжидаемая карта: {result['predicted_card_id']}\nConf: {result['confidence']:.3f}"
            axes[i].set_title(title, color=color, fontsize=10)
        
        # Скрываем лишние subplot'ы
        for i in range(len(results), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('test_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    print("=== ТЕСТИРОВАНИЕ МОДЕЛИ РАСПОЗНАВАНИЯ КАРТ БЕРСЕРК ===")
    
    # Проверяем наличие модели
    if not os.path.exists('berserk_card_model.tflite'):
        print("Модель не найдена. Сначала запустите train_model.py")
        return
    
    # Создаем предиктор
    try:
        predictor = BerserkCardPredictor()
    except Exception as e:
        print(f"Ошибка при инициализации предиктора: {e}")
        return
    
    # Тестируем на случайных изображениях
    results = predictor.test_random_images(num_images=10)
    
    # Визуализируем результаты
    if results:
        predictor.visualize_predictions(results)
    
    # Интерактивное тестирование
    print("\n=== ИНТЕРАКТИВНОЕ ТЕСТИРОВАНИЕ ===")
    print("Введите путь к изображению карты (или 'quit' для выхода):")
    
    while True:
        image_path = input("> ").strip()
        
        if image_path.lower() in ['quit', 'exit', 'q']:
            break
        
        if not os.path.exists(image_path):
            print("Файл не найден")
            continue
        
        prediction = predictor.predict(image_path)
        
        if prediction:
            print(f"Предсказанная карта: {prediction['card_info']['card_id']}")
            print(f"Сет: {prediction['card_info']['set_name']}")
            print(f"Номер карты: {prediction['card_info']['card_number']}")
            print(f"Вариант: {prediction['card_info']['variant']}")
            print(f"Уверенность: {prediction['confidence']:.4f}")
        else:
            print("Ошибка при предсказании")
        
        print()

if __name__ == "__main__":
    main()