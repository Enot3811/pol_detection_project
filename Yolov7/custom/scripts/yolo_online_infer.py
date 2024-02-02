"""Запустить yolo на сканирование папки и обработку всех новых кадров."""

import sys
import time
from pathlib import Path
import argparse
from typing import Optional
import json

import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import torch

sys.path.append(str(Path(__file__).parents[3]))
from Yolov7.custom.model_utils import (
    create_yolo, load_yolo_checkpoint, yolo_inference, coco_idx2label)
from Yolov7.yolov7.dataset import create_yolov7_transforms
from utils.torch_utils.torch_functions import (
    image_tensor_to_numpy, draw_bounding_boxes)
from mako_camera.cameras_utils import split_raw_pol
from utils.image_utils.image_functions import read_image


def main(
    frames_dir: Path, config_pth: Optional[Path], conf_thresh: float,
    iou_thresh: float, show_time: bool
):
    """Запустить yolo на сканирование папки и обработку всех новых кадров.

    Parameters
    ----------
    frames_dir : Path
        Папка сканирования.
    config_pth : Path
        Путь к конфигу с параметрами датасета и модели. Если не указан,
        то грузятся официальные веса для COCO.
    conf_thresh : float, optional
        Порог уверенности для срабатывания модели.
    iou_thresh : float, optional
        Порог перекрытия рамок для срабатывания модели.
    show_time : bool, optional
        Показывать время выполнения.
    """
    if config_pth:
        # Read config
        with open(config_pth, 'r') as f:
            config_str = f.read()
        config = json.loads(config_str)
        
        cls_id_to_name = {val: key for key, val in config['cls_to_id'].items()}
        num_classes = len(cls_id_to_name)
        polarized = config['polarization']
        num_channels = 4 if polarized else 3
        weights_pth = Path(
            config['work_dir']) / 'ckpts' / 'best_checkpoint.pth'
    else:
        # Set COCO parameters
        cls_id_to_name = coco_idx2label
        num_classes = len(coco_idx2label)
        num_channels = 3
    
    pad_colour = (114,) * num_channels

    img_paths = set(frames_dir.glob('*.*'))
    model = None
    image = np.zeros((500, 500, 3), np.uint8)
    while True:
        # Читаем все пути
        if polarized:
            updated_paths = set(frames_dir.glob('*.npy'))
        else:
            updated_paths = []
            for ext in ('jpg', 'JPG', 'png', 'PNG', 'npy'):
                updated_paths += list(frames_dir.glob(f'*.{ext}'))
            updated_paths = set(updated_paths)
        # Отсеиваем старые для быстродействия
        new_paths = updated_paths - img_paths
        img_paths = updated_paths

        new_paths = list(new_paths)
        
        if len(new_paths) != 0:
            # Из оставшихся новых берём 1 самый последний
            new_paths.sort()
            pth = new_paths[-1]
            # Небольшая задержка, чтобы избежать чтения ещё не сформированного
            # файла
            time.sleep(0.1)

            if show_time:
                start = time.time()

            # Читаем картинку
            if polarized:
                raw_pol = np.load(pth)
                image = split_raw_pol(raw_pol)  # ndarray (h, w, 4)
            else:
                # Raw rgb
                if pth.name.split('.')[-1] == 'npy':
                    # ndarray(h, w) -> ndarray(h/2, w/2, 3)
                    bayer = np.load(pth)
                    image = cv2.cvtColor(bayer, cv2.COLOR_BAYER_RG2BGR)
                else:
                    image = read_image(pth)  # ndarray (h, w, 3)

            # Загрузка модели и предобработки
            if model is None:
                if config_pth:
                    model = load_yolo_checkpoint(weights_pth, num_classes)
                else:
                    # Создать COCO модель
                    model = create_yolo(num_classes)
                process_transforms = create_yolov7_transforms(
                    pad_colour=pad_colour)
                normalize_transforms = A.Compose(
                    [ToTensorV2(transpose_mask=True)])

            # Предобработка данных
            image = process_transforms(
                image=image, bboxes=[], labels=[])['image']
            tensor_image = normalize_transforms(image=image)['image']
            tensor_image = tensor_image.to(torch.float32) / 255
            tensor_image = tensor_image[None, ...]  # Add batch dim

            # Запуск модели
            boxes, class_ids, confidences = yolo_inference(
                model, tensor_image, conf_thresh, iou_thresh)

            if show_time:
                print('Время обработки:', time.time() - start)

            image = image_tensor_to_numpy(tensor_image)
            image = image[..., :3]

            labels = list(map(lambda idx: cls_id_to_name[idx],
                              class_ids.tolist()[:30]))
            image = draw_bounding_boxes(
                image[0],
                boxes.tolist()[:30],
                labels,
                confidences.tolist()[:30])
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
        cv2.imshow('yolo_predict', image)
        k = cv2.waitKey(1) & 0xFF
        # Exit
        if k == 27:  # esc
            cv2.destroyAllWindows()
            break


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'frames_dir', help='Директория для сканирования.', type=Path)
    parser.add_argument(
        '--config_pth', help='Путь к конфигу с параметрами датасета и модели. '
        'Если не указан, то грузятся официальные веса для COCO.', type=Path,
        default=None)
    parser.add_argument(
        '--conf_thresh', help='Порог уверенности для срабатывания модели.',
        type=float, default=0.6)
    parser.add_argument(
        '--iou_thresh', help='Порог перекрытия рамок для срабатывания модели.',
        type=float, default=0.2)
    parser.add_argument(
        '--show_time', help='Показывать время выполнения.',
        action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    frames_dir = args.frames_dir
    config_pth = args.config_pth
    conf_thresh = args.conf_thresh
    iou_thresh = args.iou_thresh
    show_time = args.show_time
    main(frames_dir=frames_dir, config_pth=config_pth, conf_thresh=conf_thresh,
         iou_thresh=iou_thresh, show_time=show_time)
