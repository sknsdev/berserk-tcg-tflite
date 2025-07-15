# Исправление кодировки для корректного вывода эмодзи в Windows
import sys
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import json
import os
from data_preparation import BerserkCardDataset
from tqdm import tqdm

class BerserkCardClassifier:
    def __init__(self, input_shape=(224, 224, 3), num_classes=None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def create_model(self):
        """Создает модель на основе MobileNetV2 для TensorFlow Lite"""
        # Базовая модель MobileNetV2 (оптимизирована для мобильных устройств)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Замораживаем базовую модель для transfer learning
        base_model.trainable = False
        
        # Добавляем классификационные слои
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Компилируем модель
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def fine_tune_model(self, learning_rate=0.0001/10):
        """Размораживает верхние слои для fine-tuning"""
        # Размораживаем верхние слои базовой модели
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Замораживаем нижние слои
        fine_tune_at = 100
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Перекомпилируем с меньшим learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        """Обучает модель"""
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=0.0001
            )
        ]
        
        # Обучение
        print("Начинаем обучение...")
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
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
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # График потерь
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def convert_to_tflite(self, model_path='berserk_card_model.tflite', quantize=True):
        """Конвертирует модель в TensorFlow Lite формат"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        if quantize:
            # Квантизация для уменьшения размера модели
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Можно добавить representative dataset для лучшей квантизации
            # converter.representative_dataset = representative_data_gen
            # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            # converter.inference_input_type = tf.uint8
            # converter.inference_output_type = tf.uint8
        
        tflite_model = converter.convert()
        
        # Сохраняем модель
        with open(model_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TensorFlow Lite модель сохранена: {model_path}")
        print(f"Размер модели: {len(tflite_model) / 1024 / 1024:.2f} MB")
        
        return tflite_model
    
    def save_model_info(self, filepath='model_info.json', label_encoders=None):
        """Сохраняет информацию о модели"""
        model_info = {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'model_architecture': 'MobileNetV2',
            'training_accuracy': float(max(self.history.history['accuracy'])) if self.history else None,
            'validation_accuracy': float(max(self.history.history['val_accuracy'])) if self.history else None
        }
        
        if label_encoders:
            model_info['label_encoders'] = label_encoders
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        print(f"Информация о модели сохранена: {filepath}")

def main():
    print("=== ОБУЧЕНИЕ МОДЕЛИ РАСПОЗНАВАНИЯ КАРТ БЕРСЕРК ===")
    
    try:
        # Проверяем наличие аугментированных данных
        if not os.path.exists('augmented_cards_dataset.csv'):
            print("Подготавливаем аугментированные данные...")
            dataset = BerserkCardDataset('./cards_augmented')
            df = dataset.load_dataset()
            df = dataset.prepare_labels(df)
            dataset.save_label_encoders()
            df.to_csv('augmented_cards_dataset.csv', index=False, encoding='utf-8')
        else:
            print("Загружаем подготовленные аугментированные данные...")
            df = pd.read_csv('augmented_cards_dataset.csv')
            dataset = BerserkCardDataset('./cards_augmented')
        
        # Проверяем наличие готовых массивов данных
        if os.path.exists('X_data.npy') and os.path.exists('y_data.npy'):
            print("Загружаем сохраненные массивы данных...")
            X = np.load('X_data.npy')
            y = np.load('y_data.npy')
            print(f"Загружено {len(X)} изображений из сохраненных файлов")
        else:
            # Загружаем изображения
            print("Загружаем изображения...")
            X, y = dataset.create_dataset_arrays(df, target_size=(224, 224))
            
            # Сохраняем массивы для будущего использования
            print("Сохраняем массивы данных...")
            np.save('X_data.npy', X)
            np.save('y_data.npy', y)
            print("Массивы данных сохранены в X_data.npy и y_data.npy")
        
        print(f"Загружено {len(X)} изображений")
        print(f"Форма данных: {X.shape}")
        
        # Определяем количество классов ДО разделения данных
        num_classes = len(np.unique(y))
        print(f"Количество классов: {num_classes}")
        print(f"Диапазон меток: {np.min(y)} - {np.max(y)}")
        
        # Разделяем данные (убираем stratify из-за классов с единичными примерами)
        print("Разделяем данные на обучающую, валидационную и тестовую выборки...")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        # Освобождаем память от исходных массивов
        del X, y
        import gc
        gc.collect()
    
        print(f"Обучающая выборка: {len(X_train)}")
        print(f"Валидационная выборка: {len(X_val)}")
        print(f"Тестовая выборка: {len(X_test)}")
        
        # Проверяем диапазон меток в каждой выборке
        print(f"Диапазон меток в обучающей выборке: {np.min(y_train)} - {np.max(y_train)}")
        print(f"Диапазон меток в валидационной выборке: {np.min(y_val)} - {np.max(y_val)}")
        print(f"Диапазон меток в тестовой выборке: {np.min(y_test)} - {np.max(y_test)}")
        
        # Создаем и обучаем модель
        classifier = BerserkCardClassifier(
            input_shape=(224, 224, 3),
            num_classes=num_classes
        )
        
        model = classifier.create_model()
        print("\nАрхитектура модели:")
        model.summary()
        
        # Первый этап обучения
        print("\n=== ПЕРВЫЙ ЭТАП ОБУЧЕНИЯ ===")
        history1 = classifier.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=32)
        
        # Fine-tuning
        print("\n=== FINE-TUNING ===")
        classifier.fine_tune_model()
        history2 = classifier.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=16)
        
        # Оценка модели
        print("\n=== ОЦЕНКА МОДЕЛИ ===")
        test_accuracy, y_pred = classifier.evaluate(X_test, y_test)
        
        # Строим графики
        classifier.plot_training_history()
        
        # Конвертируем в TensorFlow Lite
        print("\n=== КОНВЕРТАЦИЯ В TENSORFLOW LITE ===")
        tflite_model = classifier.convert_to_tflite('berserk_card_model.tflite')
        
        # Сохраняем информацию о модели
        with open('label_encoders.json', 'r', encoding='utf-8') as f:
            label_encoders = json.load(f)
        
        classifier.save_model_info('model_info.json', label_encoders)
        
        # Сохраняем обычную модель тоже
        model.save('berserk_card_model.h5')
        print("Keras модель сохранена: berserk_card_model.h5")
        
        print("\n=== ОБУЧЕНИЕ ЗАВЕРШЕНО ===")
        print(f"Финальная точность: {test_accuracy:.4f}")
        
    except KeyboardInterrupt:
        print("\n=== ОБУЧЕНИЕ ПРЕРВАНО ПОЛЬЗОВАТЕЛЕМ ===")
        print("Процесс был остановлен. Частично обработанные данные могут быть сохранены.")
        return
    except Exception as e:
        print(f"\n=== ОШИБКА ПРИ ОБУЧЕНИИ ===")
        print(f"Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()