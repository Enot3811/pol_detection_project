"""Пакет с модулями yolov7."""

# Оригинальный yolov7 репозиторий располагал пакет yolo в корневой папке
# потому для работы импортов необходимо добавить этот пакет в область видимости
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
