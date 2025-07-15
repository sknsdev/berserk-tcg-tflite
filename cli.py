#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Интерактивный CLI для проекта распознавания карт Берсерк
Автор: AI Assistant
Описание: Единый интерфейс для всех операций проекта
"""

import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import time
import subprocess
from pathlib import Path

class BerserkCLI:
    def __init__(self):
        self.project_root = Path.cwd()
        self.clear_screen()
        
    def clear_screen(self):
        """Очищает экран"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def show_header(self):
        """Показывает заголовок приложения"""
        print("""╔══════════════════════════════════════════════════════════════╗
║                 🃏 БЕРСЕРК TCG - РАСПОЗНАВАНИЕ КАРТ           ║
║                     TensorFlow Lite Project                  ║
╚══════════════════════════════════════════════════════════════╝""")
        print()
    
    def show_main_menu(self):
        """Показывает главное меню"""
        print("📋 ГЛАВНОЕ МЕНЮ:")
        print()
        print("🔧 НАСТРОЙКА И ДИАГНОСТИКА:")
        print("  1. Проверить системные требования")
        print("  2. Проверить совместимость")
        print("  3. Установить зависимости")
        print("  4. Диагностика GPU")
        print()
        print("📊 РАБОТА С ДАННЫМИ:")
        print("  5. Организовать карты по папкам")
        print("  6. Подготовить базовый датасет")
        print("  7. Создать аугментированный датасет")
        print("  8. Диагностика датасета")
        print("  9. Анализ распределения классов")
        print()
        print("🤖 ОБУЧЕНИЕ И ТЕСТИРОВАНИЕ:")
        print(" 10. Обучить модель (базовый датасет)")
        print(" 11. Обучить модель (аугментированный датасет)")
        print(" 12. Тестировать модель")
        print(" 13. Запустить веб-демо")
        print()
        print("🚀 АВТОМАТИЧЕСКИЕ РЕЖИМЫ:")
        print(" 14. Полный цикл (базовый)")
        print(" 15. Полный цикл (с аугментацией)")
        print()
        print("ℹ️  ИНФОРМАЦИЯ:")
        print(" 16. Показать информацию о проекте")
        print(" 17. Показать статус проекта")
        print()
        print("  0. Выход")
        print()
    
    def run_script(self, script_name, description=None):
        """Запускает скрипт и показывает результат"""
        if description:
            print(f"\n🔄 {description}...")
        else:
            print(f"\n🔄 Запуск {script_name}...")
        
        try:
            result = subprocess.run([sys.executable, script_name], 
                                  capture_output=False, text=True)
            
            if result.returncode == 0:
                print(f"\n✅ {script_name} выполнен успешно")
            else:
                print(f"\n❌ Ошибка при выполнении {script_name}")
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"\n❌ Ошибка запуска {script_name}: {e}")
            return False
    
    def check_file_exists(self, filename, description=None):
        """Проверяет существование файла"""
        if os.path.exists(filename):
            if description:
                print(f"✅ {description}: найден")
            return True
        else:
            if description:
                print(f"❌ {description}: не найден")
            return False
    
    def show_project_status(self):
        """Показывает текущий статус проекта"""
        print("\n📊 СТАТУС ПРОЕКТА:")
        print()
        
        # Проверка данных
        print("📁 ДАННЫЕ:")
        self.check_file_exists('./cards', 'Папка с картами')
        self.check_file_exists('./cards_dataset.csv', 'Базовый датасет')
        self.check_file_exists('./augmented_cards_dataset.csv', 'Аугментированный датасет')
        self.check_file_exists('./cards_augmented', 'Папка аугментированных карт')
        print()
        
        # Проверка моделей
        print("🤖 МОДЕЛИ:")
        self.check_file_exists('./berserk_card_model.tflite', 'TensorFlow Lite модель (базовая)')
        self.check_file_exists('./berserk_card_model.h5', 'Keras модель (базовая)')
        self.check_file_exists('./berserk_card_model_augmented.tflite', 'TensorFlow Lite модель (аугментированная)')
        self.check_file_exists('./berserk_card_model_augmented.h5', 'Keras модель (аугментированная)')
        print()
        
        # Проверка конфигурации
        print("⚙️ КОНФИГУРАЦИЯ:")
        self.check_file_exists('./label_encoders.json', 'Энкодеры меток (базовые)')
        self.check_file_exists('./augmented_label_encoders.json', 'Энкодеры меток (аугментированные)')
        self.check_file_exists('./model_info.json', 'Информация о модели (базовая)')
        self.check_file_exists('./model_info_augmented.json', 'Информация о модели (аугментированная)')
        print()
        
        # Проверка отчетов
        print("📈 ОТЧЕТЫ:")
        self.check_file_exists('./training_history.png', 'График обучения (базовый)')
        self.check_file_exists('./training_history_augmented.png', 'График обучения (аугментированный)')
        self.check_file_exists('./dataset_report.json', 'Отчет о датасете')
        self.check_file_exists('./dataset_statistics.png', 'Статистика датасета')
    
    def show_project_info(self):
        """Показывает информацию о проекте"""
        print("""\n📖 ИНФОРМАЦИЯ О ПРОЕКТЕ:

🎯 ЦЕЛЬ:
Создание TensorFlow Lite модели для распознавания карт
из коллекционной карточной игры "Берсерк"

🔍 ВОЗМОЖНОСТИ РАСПОЗНАВАНИЯ:
• Сет карты (s1, s2, s3, s4, s5, laar)
• Номер карты (1-200+)
• Вариант карты (normal, pf, alt)

🏗️ АРХИТЕКТУРА:
• Базовая модель: MobileNetV2
• Метод обучения: Transfer Learning
• Формат модели: TensorFlow Lite (.tflite)
• Оптимизация: Квантизация для мобильных устройств

📊 ДАННЫЕ:
• Поддержка структуры папок по сетам
• Автоматическая аугментация данных
• Диагностика качества датасета

🚀 ПРИМЕНЕНИЕ:
• Мобильные приложения
• Веб-интерфейсы
• Автоматизация каталогизации коллекций""")
    
    def wait_for_enter(self):
        """Ждет нажатия Enter"""
        input("\nНажмите Enter для продолжения...")
    
    def run(self):
        """Основной цикл приложения"""
        while True:
            self.clear_screen()
            self.show_header()
            self.show_main_menu()
            
            try:
                choice = input("Выберите действие (0-17): ").strip()
                
                if choice == '0':
                    print("\n👋 До свидания!")
                    break
                
                elif choice == '1':
                    self.run_script('main.py check', 'Проверка системных требований')
                    
                elif choice == '2':
                    self.run_script('check_compatibility.py', 'Проверка совместимости')
                    
                elif choice == '3':
                    self.run_script('setup.py', 'Установка зависимостей')
                    
                elif choice == '4':
                    self.run_script('gpu_diagnostic.py', 'Диагностика GPU')
                    
                elif choice == '5':
                    self.run_script('organize_cards.py', 'Организация карт по папкам')
                    
                elif choice == '6':
                    self.run_script('main.py prepare', 'Подготовка базового датасета')
                    
                elif choice == '7':
                    self.run_script('main.py augment', 'Создание аугментированного датасета')
                    
                elif choice == '8':
                    self.run_script('check_dataset.py', 'Диагностика датасета')
                    
                elif choice == '9':
                    self.run_script('analyze_classes.py', 'Анализ распределения классов')
                    
                elif choice == '10':
                    self.run_script('main.py train', 'Обучение модели (базовый датасет)')
                    
                elif choice == '11':
                    self.run_script('main.py train-aug', 'Обучение модели (аугментированный датасет)')
                    
                elif choice == '12':
                    self.run_script('test_model.py', 'Тестирование модели')
                    
                elif choice == '13':
                    print("\n🌐 Запуск веб-демо...")
                    print("Веб-сервер будет доступен по адресу: http://localhost:5000")
                    print("Для остановки нажмите Ctrl+C")
                    self.run_script('web_demo.py', 'Запуск веб-демо')
                    
                elif choice == '14':
                    self.run_script('main.py full', 'Полный цикл (базовый)')
                    
                elif choice == '15':
                    self.run_script('main.py full-aug', 'Полный цикл (с аугментацией)')
                    
                elif choice == '16':
                    self.show_project_info()
                    
                elif choice == '17':
                    self.show_project_status()
                    
                else:
                    print("\n❌ Неверный выбор. Попробуйте снова.")
                
                if choice != '0':
                    self.wait_for_enter()
                    
            except KeyboardInterrupt:
                print("\n\n👋 Программа прервана пользователем")
                break
            except Exception as e:
                print(f"\n❌ Ошибка: {e}")
                self.wait_for_enter()

if __name__ == "__main__":
    cli = BerserkCLI()
    cli.run()