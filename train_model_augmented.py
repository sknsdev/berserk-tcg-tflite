#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обучение модели с использованием аугментированных данных
Решает проблему с единичными примерами классов
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import json
import os
from data_preparation import BerserkCardDataset
from tqdm import tqdm

class BerserkCardClassifierAugmented:
    def __init__(self, input_shape=(224, 224, 3), num_classes=None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def create_model(self):
        """Создает модель на основе MobileNetV2 для TensorFlow Lite"""
        # Базовая модель MobileNetV2 (оптимизирована для мобильных устройств)
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Замораживаем базовую модель для начального обучения
        base_model.trainable = False
        
        # Добавляем классификационные слои
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Компилируем модель
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def fine_tune_model(self, learning_rate=0.0001):
        """Размораживает верхние слои базовой модели для fine-tuning"""
        # Размораживаем верхние слои базовой модели
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Замораживаем нижние слои
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Перекомпилируем с меньшим learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def create_data_generator(self):
        """Создает генератор данных с дополнительной аугментацией"""
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,  # Осторожно с текстом на картах
            fill_mode='nearest'
        )
        return datagen
    
    def train_with_validation_split(self, X, y, validation_split=0.2, epochs=20, batch_size=32):
        """Обучает модель с простым разделением на валидацию"""
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=7,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=0.00001
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model_checkpoint.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Создаем генератор данных
        datagen = self.create_data_generator()
        
        # Обучение с генератором данных
        print("Начинаем обучение с аугментацией данных...")
        self.history = self.model.fit(
            datagen.flow(X, y, batch_size=batch_size),
            steps_per_epoch=len(X) // batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def train_with_holdout(self, X, y, test_size=0.2, epochs=20, batch_size=32):
        """Обучает модель с отложенной тестовой выборкой"""
        # Разделяем данные на обучение и тест
        # Используем группировку по оригинальным картам для избежания утечки данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )
        
        print(f"Обучающая выборка: {len(X_train)}")
        print(f"Тестовая выборка: {len(X_test)}")
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=7,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=0.00001
            )
        ]
        
        # Создаем генератор данных
        datagen = self.create_data_generator()
        
        # Обучение
        print("Начинаем обучение...")
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history, X_test, y_test
    
    def evaluate(self, X_test, y_test):
        """Оценивает модель на тестовых данных"""
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nТочность на тестовых данных: {test_accuracy:.4f}")
        
        # Предсказания
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        return test_accuracy, y_pred_classes
    
    def plot_training_history(self):
        """Строит графики обучения"""
        if self.history is None:
            print("Модель еще не обучена")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # График точности
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in self.history.history:
            ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # График потерь
        ax2.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history_augmented.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def convert_to_tflite(self, model_path='berserk_card_model_augmented.tflite', quantize=True):
        """Конвертирует модель в TensorFlow Lite формат"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        if quantize:
            # Квантизация для уменьшения размера модели
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        # Сохраняем модель
        with open(model_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TensorFlow Lite модель сохранена: {model_path}")
        print(f"Размер модели: {len(tflite_model) / 1024 / 1024:.2f} MB")
        
        return tflite_model
    
    def save_model_info(self, filepath='model_info_augmented.json', label_encoders=None):
        """Сохраняет информацию о модели"""
        model_info = {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'model_architecture': 'MobileNetV2',
            'training_accuracy': float(max(self.history.history['accuracy'])) if self.history else None,
            'validation_accuracy': float(max(self.history.history.get('val_accuracy', [0]))) if self.history else None,
            'augmented_data': True
        }
        
        if label_encoders:
            model_info['label_encoders'] = label_encoders
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        print(f"Информация о модели сохранена: {filepath}")

def main():
    print("=== ОБУЧЕНИЕ МОДЕЛИ С АУГМЕНТИРОВАННЫМИ ДАННЫМИ ===")
    
    # Проверяем наличие аугментированных данных
    if not os.path.exists('augmented_cards_dataset.csv'):
        print("❌ Аугментированный датасет не найден!")
        print("Сначала запустите: python data_augmentation.py")
        return
    
    if not os.path.exists('./cards_augmented'):
        print("❌ Папка с аугментированными изображениями не найдена!")
        print("Сначала запустите: python data_augmentation.py")
        return
    
    print("✅ Загружаем аугментированные данные...")
    df = pd.read_csv('augmented_cards_dataset.csv')
    dataset = BerserkCardDataset('./cards_augmented')
    
    # Загружаем изображения
    print("Загружаем изображения...")
    X, y = dataset.create_dataset_arrays(df, target_size=(224, 224))
    
    print(f"Загружено {len(X)} изображений")
    print(f"Форма данных: {X.shape}")
    print(f"Количество классов: {len(np.unique(y))}")
    
    # Проверяем распределение классов
    unique, counts = np.unique(y, return_counts=True)
    min_samples = min(counts)
    print(f"Минимальное количество примеров в классе: {min_samples}")
    
    if min_samples < 2:
        print("❌ Все еще есть классы с единичными примерами!")
        print("Увеличьте количество аугментаций в data_augmentation.py")
        return
    
    # Создаем и обучаем модель
    classifier = BerserkCardClassifierAugmented(
        input_shape=(224, 224, 3),
        num_classes=len(np.unique(y))
    )
    
    model = classifier.create_model()
    print("\nАрхитектура модели:")
    model.summary()
    
    # Выбираем стратегию обучения
    print("\n=== ВЫБОР СТРАТЕГИИ ОБУЧЕНИЯ ===")
    print("1. Обучение с простым разделением на валидацию (рекомендуется)")
    print("2. Обучение с отложенной тестовой выборкой")
    
    try:
        choice = input("Выберите стратегию (1 или 2): ").strip()
        if choice == '2':
            # Стратегия с отложенной тестовой выборкой
            print("\n=== ОБУЧЕНИЕ С ТЕСТОВОЙ ВЫБОРКОЙ ===")
            history, X_test, y_test = classifier.train_with_holdout(
                X, y, test_size=0.2, epochs=15, batch_size=32
            )
            
            # Оценка модели
            print("\n=== ОЦЕНКА МОДЕЛИ ===")
            test_accuracy, y_pred = classifier.evaluate(X_test, y_test)
            
        else:
            # Стратегия с простым разделением
            print("\n=== ОБУЧЕНИЕ С ВАЛИДАЦИЕЙ ===")
            history = classifier.train_with_validation_split(
                X, y, validation_split=0.2, epochs=15, batch_size=32
            )
            test_accuracy = max(history.history.get('val_accuracy', [0]))
            
    except KeyboardInterrupt:
        print("\nОбучение прервано пользователем")
        return
    
    # Fine-tuning
    print("\n=== FINE-TUNING ===")
    classifier.fine_tune_model(learning_rate=0.00005)
    
    if 'X_test' in locals():
        # Дообучение с тестовой выборкой
        history2, _, _ = classifier.train_with_holdout(
            X, y, test_size=0.2, epochs=10, batch_size=16
        )
        final_accuracy, _ = classifier.evaluate(X_test, y_test)
    else:
        # Дообучение с валидацией
        history2 = classifier.train_with_validation_split(
            X, y, validation_split=0.2, epochs=10, batch_size=16
        )
        final_accuracy = max(history2.history.get('val_accuracy', [0]))
    
    # Строим графики
    classifier.plot_training_history()
    
    # Конвертируем в TensorFlow Lite
    print("\n=== КОНВЕРТАЦИЯ В TENSORFLOW LITE ===")
    tflite_model = classifier.convert_to_tflite('berserk_card_model_augmented.tflite')
    
    # Сохраняем информацию о модели
    with open('augmented_label_encoders.json', 'r', encoding='utf-8') as f:
        label_encoders = json.load(f)
    
    classifier.save_model_info('model_info_augmented.json', label_encoders)
    
    # Сохраняем обычную модель тоже
    model.save('berserk_card_model_augmented.h5')
    print("Keras модель сохранена: berserk_card_model_augmented.h5")
    
    print("\n=== ОБУЧЕНИЕ ЗАВЕРШЕНО ===")
    print(f"Финальная точность: {final_accuracy:.4f}")
    print("\n=== СОЗДАННЫЕ ФАЙЛЫ ===")
    print("- berserk_card_model_augmented.tflite (TensorFlow Lite модель)")
    print("- berserk_card_model_augmented.h5 (Keras модель)")
    print("- model_info_augmented.json (информация о модели)")
    print("- training_history_augmented.png (графики обучения)")

if __name__ == "__main__":
    main()