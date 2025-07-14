#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Детальная диагностика GPU для TensorFlow
Помогает выявить причины, по которым TensorFlow не видит видеокарту
"""

import sys
import platform
import subprocess
import os

def check_nvidia_driver():
    """Проверяет наличие и версию драйвера NVIDIA"""
    print("🔍 Проверка драйвера NVIDIA...")
    
    try:
        # Проверяем nvidia-smi
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version:' in line:
                    driver_version = line.split('Driver Version:')[1].split()[0]
                    print(f"✅ Драйвер NVIDIA найден: версия {driver_version}")
                    return True
            print("✅ nvidia-smi работает, но версия драйвера не определена")
            return True
    except FileNotFoundError:
        print("❌ nvidia-smi не найден - драйвер NVIDIA не установлен")
        return False
    except subprocess.TimeoutExpired:
        print("⚠️ nvidia-smi не отвечает (возможно, проблемы с драйвером)")
        return False
    except Exception as e:
        print(f"⚠️ Ошибка при проверке nvidia-smi: {e}")
        return False

def check_cuda_installation():
    """Проверяет установку CUDA"""
    print("\n🔍 Проверка CUDA...")
    
    # Проверяем переменные окружения
    cuda_path = os.environ.get('CUDA_PATH')
    cuda_home = os.environ.get('CUDA_HOME')
    
    if cuda_path:
        print(f"✅ CUDA_PATH найден: {cuda_path}")
    elif cuda_home:
        print(f"✅ CUDA_HOME найден: {cuda_home}")
    else:
        print("⚠️ Переменные окружения CUDA не найдены")
    
    # Проверяем nvcc
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    print(f"✅ CUDA Compiler найден: {line.strip()}")
                    return True
    except FileNotFoundError:
        print("❌ nvcc не найден - CUDA Toolkit не установлен")
        return False
    except Exception as e:
        print(f"⚠️ Ошибка при проверке nvcc: {e}")
        return False
    
    return False

def check_cudnn():
    """Проверяет установку cuDNN"""
    print("\n🔍 Проверка cuDNN...")
    
    try:
        import tensorflow as tf
        
        # Проверяем, может ли TensorFlow найти cuDNN
        if hasattr(tf.config.experimental, 'list_physical_devices'):
            gpus = tf.config.experimental.list_physical_devices('GPU')
        else:
            gpus = tf.config.list_physical_devices('GPU')
            
        if gpus:
            print("✅ TensorFlow видит GPU устройства")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
            return True
        else:
            print("❌ TensorFlow не видит GPU устройства")
            return False
            
    except ImportError:
        print("❌ TensorFlow не установлен")
        return False
    except Exception as e:
        print(f"⚠️ Ошибка при проверке через TensorFlow: {e}")
        return False

def check_tensorflow_gpu_support():
    """Детальная проверка поддержки GPU в TensorFlow"""
    print("\n🔍 Детальная проверка TensorFlow GPU...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow версия: {tf.__version__}")
        
        # Проверяем, собран ли TensorFlow с поддержкой CUDA
        if tf.test.is_built_with_cuda():
            print("✅ TensorFlow собран с поддержкой CUDA")
        else:
            print("❌ TensorFlow собран БЕЗ поддержки CUDA")
            print("💡 Установите tensorflow-gpu или используйте conda")
            return False
        
        # Проверяем доступность GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ Найдено GPU устройств: {len(gpus)}")
            
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
                
                # Пытаемся получить детали GPU
                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    if details:
                        print(f"      Детали: {details}")
                except:
                    pass
            
            # Проверяем, можем ли мы использовать GPU
            try:
                with tf.device('/GPU:0'):
                    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                    c = tf.matmul(a, b)
                print("✅ GPU успешно используется для вычислений")
                return True
            except Exception as e:
                print(f"❌ Ошибка при использовании GPU: {e}")
                return False
        else:
            print("❌ GPU устройства не найдены")
            return False
            
    except ImportError:
        print("❌ TensorFlow не установлен")
        return False
    except Exception as e:
        print(f"⚠️ Ошибка при проверке TensorFlow: {e}")
        return False

def check_system_info():
    """Выводит информацию о системе"""
    print("💻 Информация о системе:")
    print(f"   ОС: {platform.system()} {platform.release()}")
    print(f"   Архитектура: {platform.machine()}")
    print(f"   Python: {sys.version}")
    
    # Проверяем PATH
    path_dirs = os.environ.get('PATH', '').split(os.pathsep)
    cuda_in_path = any('cuda' in dir.lower() for dir in path_dirs)
    if cuda_in_path:
        print("✅ CUDA директории найдены в PATH")
    else:
        print("⚠️ CUDA директории не найдены в PATH")

def provide_solutions():
    """Предлагает решения для типичных проблем"""
    print("\n" + "="*60)
    print("💡 ВОЗМОЖНЫЕ РЕШЕНИЯ:")
    print("\n1. 🎮 Если у вас NVIDIA видеокарта:")
    print("   • Установите последний драйвер NVIDIA")
    print("   • Установите CUDA Toolkit (рекомендуется 11.8 или 12.x)")
    print("   • Установите cuDNN")
    print("   • Переустановите TensorFlow: pip install tensorflow[and-cuda]")
    
    print("\n2. 🔧 Если у вас AMD видеокарта:")
    print("   • TensorFlow не поддерживает AMD GPU напрямую")
    print("   • Используйте TensorFlow с CPU (что и происходит сейчас)")
    print("   • Рассмотрите использование ROCm (экспериментально)")
    
    print("\n3. 💻 Если у вас интегрированная графика Intel:")
    print("   • TensorFlow не поддерживает Intel GPU")
    print("   • Используйте CPU версию (что и происходит сейчас)")
    
    print("\n4. 🐍 Проблемы с установкой TensorFlow:")
    print("   • pip uninstall tensorflow")
    print("   • pip install tensorflow[and-cuda]")
    print("   • Или используйте conda: conda install tensorflow-gpu")
    
    print("\n5. 🔄 Переменные окружения:")
    print("   • Добавьте CUDA в PATH")
    print("   • Установите CUDA_PATH и CUDA_HOME")
    print("   • Перезагрузите компьютер после установки CUDA")
    
    print("\n6. ✅ Проверка после исправлений:")
    print("   • python gpu_diagnostic.py")
    print("   • python main.py check")

def main():
    """Основная функция диагностики"""
    print("🔍 ДИАГНОСТИКА GPU ДЛЯ TENSORFLOW")
    print("="*60)
    
    check_system_info()
    print("\n" + "="*60)
    
    # Последовательная проверка компонентов
    nvidia_ok = check_nvidia_driver()
    cuda_ok = check_cuda_installation()
    cudnn_ok = check_cudnn()
    tf_gpu_ok = check_tensorflow_gpu_support()
    
    print("\n" + "="*60)
    print("📊 ИТОГОВЫЙ СТАТУС:")
    
    if tf_gpu_ok:
        print("🎉 GPU успешно настроен и работает с TensorFlow!")
    else:
        print("⚠️ GPU не работает с TensorFlow")
        print("\nПроблемы:")
        if not nvidia_ok:
            print("   ❌ Драйвер NVIDIA не найден или не работает")
        if not cuda_ok:
            print("   ❌ CUDA Toolkit не установлен или не настроен")
        if not cudnn_ok:
            print("   ❌ TensorFlow не может использовать GPU")
        
        provide_solutions()
    
    print("\n" + "="*60)
    print("ℹ️ Примечание: Использование CPU вместо GPU не критично")
    print("   для данного проекта, но GPU ускорит обучение модели.")

if __name__ == "__main__":
    main()