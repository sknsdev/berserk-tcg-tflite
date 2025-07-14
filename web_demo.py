#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Веб-демонстрация модели распознавания карт Берсерк
Простой Flask веб-интерфейс для тестирования модели
"""

import os
import io
import base64
from flask import Flask, render_template_string, request, jsonify
from PIL import Image
import numpy as np
from test_model import BerserkCardPredictor

app = Flask(__name__)

# HTML шаблон для веб-интерфейса
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Распознавание карт Берсерк</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .upload-area:hover {
            border-color: #764ba2;
            background-color: #f8f9ff;
        }
        .upload-area.dragover {
            border-color: #764ba2;
            background-color: #f0f2ff;
        }
        #file-input {
            display: none;
        }
        .upload-text {
            font-size: 1.2em;
            color: #666;
            margin-bottom: 10px;
        }
        .upload-hint {
            color: #999;
            font-size: 0.9em;
        }
        .preview-container {
            text-align: center;
            margin: 20px 0;
        }
        .preview-image {
            max-width: 300px;
            max-height: 400px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .result-container {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            display: none;
        }
        .result-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #e9ecef;
        }
        .result-item:last-child {
            border-bottom: none;
        }
        .result-label {
            font-weight: bold;
            color: #495057;
        }
        .result-value {
            color: #667eea;
            font-weight: 600;
        }
        .confidence-bar {
            width: 100%;
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 10px;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.5s ease;
        }
        .loading {
            text-align: center;
            color: #667eea;
            font-style: italic;
            display: none;
        }
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
            display: none;
        }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            transition: transform 0.2s ease;
        }
        .btn:hover {
            transform: translateY(-2px);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🃏 Распознавание карт Берсерк</h1>
        
        <div class="upload-area" onclick="document.getElementById('file-input').click()">
            <div class="upload-text">📁 Выберите изображение карты</div>
            <div class="upload-hint">Поддерживаются форматы: JPG, PNG, WebP</div>
            <input type="file" id="file-input" accept="image/*">
        </div>
        
        <div class="preview-container" id="preview-container"></div>
        
        <div class="loading" id="loading">
            🔄 Анализируем изображение...
        </div>
        
        <div class="error" id="error"></div>
        
        <div class="result-container" id="result-container">
            <h3>📊 Результат распознавания:</h3>
            <div class="result-item">
                <span class="result-label">Сет:</span>
                <span class="result-value" id="set-name">-</span>
            </div>
            <div class="result-item">
                <span class="result-label">Номер карты:</span>
                <span class="result-value" id="card-number">-</span>
            </div>
            <div class="result-item">
                <span class="result-label">Вариант:</span>
                <span class="result-value" id="variant">-</span>
            </div>
            <div class="result-item">
                <span class="result-label">Полный ID:</span>
                <span class="result-value" id="card-id">-</span>
            </div>
            <div class="result-item">
                <span class="result-label">Уверенность:</span>
                <span class="result-value" id="confidence">-</span>
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill" id="confidence-fill" style="width: 0%"></div>
            </div>
        </div>
    </div>

    <script>
        const fileInput = document.getElementById('file-input');
        const uploadArea = document.querySelector('.upload-area');
        const previewContainer = document.getElementById('preview-container');
        const resultContainer = document.getElementById('result-container');
        const loading = document.getElementById('loading');
        const error = document.getElementById('error');

        // Drag and drop functionality
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });

        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                showError('Пожалуйста, выберите изображение');
                return;
            }

            // Show preview
            const reader = new FileReader();
            reader.onload = (e) => {
                previewContainer.innerHTML = `<img src="${e.target.result}" class="preview-image" alt="Preview">`;
            };
            reader.readAsDataURL(file);

            // Send to server
            const formData = new FormData();
            formData.append('image', file);

            loading.style.display = 'block';
            resultContainer.style.display = 'none';
            error.style.display = 'none';

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                loading.style.display = 'none';
                if (data.success) {
                    showResult(data.result);
                } else {
                    showError(data.error || 'Ошибка при обработке изображения');
                }
            })
            .catch(err => {
                loading.style.display = 'none';
                showError('Ошибка соединения с сервером');
            });
        }

        function showResult(result) {
            document.getElementById('set-name').textContent = result.card_info.set_name;
            document.getElementById('card-number').textContent = result.card_info.card_number;
            document.getElementById('variant').textContent = result.card_info.variant;
            document.getElementById('card-id').textContent = result.card_info.card_id;
            document.getElementById('confidence').textContent = (result.confidence * 100).toFixed(1) + '%';
            document.getElementById('confidence-fill').style.width = (result.confidence * 100) + '%';
            
            resultContainer.style.display = 'block';
        }

        function showError(message) {
            error.textContent = message;
            error.style.display = 'block';
        }
    </script>
</body>
</html>
"""

class WebDemo:
    def __init__(self):
        self.predictor = None
        self.load_model()
    
    def load_model(self):
        """Загружает модель для предсказаний"""
        try:
            if os.path.exists('berserk_card_model.tflite'):
                self.predictor = BerserkCardPredictor()
                print("✅ Модель загружена успешно")
            else:
                print("❌ Модель не найдена. Запустите обучение сначала.")
        except Exception as e:
            print(f"❌ Ошибка при загрузке модели: {e}")
    
    def predict_image(self, image_file):
        """Делает предсказание для загруженного изображения"""
        if not self.predictor:
            return {'success': False, 'error': 'Модель не загружена'}
        
        try:
            # Сохраняем временный файл
            temp_path = 'temp_image.jpg'
            image = Image.open(image_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(temp_path)
            
            # Делаем предсказание
            result = self.predictor.predict(temp_path)
            
            # Удаляем временный файл
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if result:
                return {'success': True, 'result': result}
            else:
                return {'success': False, 'error': 'Ошибка при предсказании'}
                
        except Exception as e:
            return {'success': False, 'error': f'Ошибка обработки: {str(e)}'}

# Создаем экземпляр демо
demo = WebDemo()

@app.route('/')
def index():
    """Главная страница"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """API для предсказания"""
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'Изображение не найдено'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Файл не выбран'})
    
    return jsonify(demo.predict_image(file))

@app.route('/health')
def health():
    """Проверка состояния сервера"""
    return jsonify({
        'status': 'ok',
        'model_loaded': demo.predictor is not None
    })

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    ВЕБ-ДЕМО ЗАПУЩЕНА                        ║
╠══════════════════════════════════════════════════════════════╣
║ Откройте браузер и перейдите по адресу:                     ║
║ http://localhost:5000                                        ║
║                                                              ║
║ Для остановки нажмите Ctrl+C                                ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    if not demo.predictor:
        print("\n⚠️  ВНИМАНИЕ: Модель не загружена!")
        print("Сначала обучите модель командой: python main.py train")
        print("Веб-интерфейс будет работать, но предсказания будут недоступны.\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main()