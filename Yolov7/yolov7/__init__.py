# Оригинальный yolov7 репозиторий располагал пакет yolo в корневой папке
# потому для работы импортов необходимо добавить этот пакет в область видимости
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

from yolov7.models.model_factory import MODEL_CONFIGS, create_yolov7_model
from yolov7.loss_factory import create_yolov7_loss

AVAILABLE_MODELS = list(MODEL_CONFIGS.keys())
