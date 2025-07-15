#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import os
import random

def test_model():
    """Быстрый тест модели TensorFlow Lite"""
    
    # Загружаем модель
    print("Загружаем модель...")
    interpreter = tf.lite.Interpreter('berserk_card_model.tflite')
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Входной тензор: {input_details[0]['shape']}")
    print(f"Выходной тензор: {output_details[0]['shape']}")
    
    # Загружаем энкодеры меток
    try:
        with open('augmented_label_encoders.json', 'r', encoding='utf-8') as f:
            label_encoders = json.load(f)
        print("Энкодеры меток загружены")
    except Exception as e:
        print(f"Ошибка загрузки энкодеров: {e}")
        return
    
    # Ищем тестовые изображения
    test_dirs = ['./cards', './cards_augmented']
    image_files = []
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for file in os.listdir(test_dir):
                if file.lower().endswith(('.webp', '.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(test_dir, file))
            break
    
    if not image_files:
        print("Тестовые изображения не найдены!")
        return
    
    # Тестируем на нескольких случайных изображениях
    num_tests = min(3, len(image_files))
    test_images = random.sample(image_files, num_tests)
    
    print(f"\nТестируем на {num_tests} изображениях:")
    
    for i, image_path in enumerate(test_images):
        print(f"\n--- Тест {i+1}: {os.path.basename(image_path)} ---")
        
        try:
            # Загружаем и предобрабатываем изображение
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image = image.resize((224, 224))
            image_array = np.array(image, dtype=np.float32) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            
            # Делаем предсказание
            interpreter.set_tensor(input_details[0]['index'], image_array)
            interpreter.invoke()
            
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_class = np.argmax(output_data[0])
            confidence = float(np.max(output_data[0]))
            
            # Декодируем результат
            card_ids = label_encoders['card_id']['classes']
            if predicted_class < len(card_ids):
                predicted_card_id = card_ids[predicted_class]
            else:
                predicted_card_id = 'unknown'
            
            print(f"Предсказанная карта: {predicted_card_id}")
            print(f"Уверенность: {confidence:.4f}")
            
            # Показываем топ-3 предсказания
            top_3_indices = np.argsort(output_data[0])[-3:][::-1]
            print("Топ-3 предсказания:")
            for j, idx in enumerate(top_3_indices):
                if idx < len(card_ids):
                    card_id = card_ids[idx]
                    prob = output_data[0][idx]
                    print(f"  {j+1}. {card_id} ({prob:.4f})")
            
        except Exception as e:
            print(f"Ошибка при обработке {image_path}: {e}")
    
    print("\nТест завершен!")

if __name__ == "__main__":
    test_model()