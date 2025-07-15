#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Запускатель CLI для проекта Берсерк TCG
"""

import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import subprocess
import os

def main():
    """Запускает CLI интерфейс"""
    try:
        # Проверяем наличие cli.py
        if not os.path.exists('cli.py'):
            print("❌ Файл cli.py не найден!")
            return
        
        # Запускаем CLI
        subprocess.run([sys.executable, 'cli.py'])
        
    except KeyboardInterrupt:
        print("\n👋 Программа завершена")
    except Exception as e:
        print(f"❌ Ошибка запуска: {e}")

if __name__ == "__main__":
    main()