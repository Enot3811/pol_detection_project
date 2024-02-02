"""Скрипт запуска usb камеры и обработки полученных фреймов в yolov7."""


from pathlib import Path
import sys
import time
import json
import argparse
from typing import Optional

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import cv2

sys.path.append(str(Path(__file__).parents[3]))
from utils.torch_utils.torch_functions import (
    image_tensor_to_numpy, draw_bounding_boxes)
from Yolov7.yolov7.dataset import create_yolov7_transforms
from Yolov7.custom.model_utils import (
    create_yolo, load_yolo_checkpoint, yolo_inference, coco_idx2label)


def main(
    config_pth: Optional[Path], conf_thresh: float, iou_thresh: float,
    show_time: bool
):
    """Запустить yolo на сканирование папки и обработку всех новых кадров.

    Parameters
    ----------
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
        if polarized:
            raise NotImplementedError(
                'USB камера работает только в RGB режиме, а загружаемая модель'
                ' в поляризационном')
        num_channels = 3
        weights_pth = Path(
            config['work_dir']) / 'ckpts' / 'best_checkpoint.pth'
    else:
        # Set COCO parameters
        cls_id_to_name = coco_idx2label
        num_classes = len(coco_idx2label)
        num_channels = 3

    pad_colour = (114,) * num_channels

    # Загрузить модель
    if config_pth:
        model = load_yolo_checkpoint(weights_pth, num_classes)
    else:
        # Создать COCO модель
        model = create_yolo(num_classes)

    # Обработка семплов
    process_transforms = create_yolov7_transforms(pad_colour=pad_colour)
    normalize_transforms = A.Compose([ToTensorV2(transpose_mask=True)])

    # Берём камеру
    cap = cv2.VideoCapture(0)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            if show_time:
                start = time.time()

            image = process_transforms(
                image=frame, bboxes=[], labels=[])['image']
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

            cv2.imshow('Yolo inference (press any key)', image)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # esc
                break
    cv2.destroyAllWindows()
    cap.release()


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

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
    config_pth = args.config_pth
    conf_thresh = args.conf_thresh
    iou_thresh = args.iou_thresh
    show_time = args.show_time
    main(config_pth=config_pth, conf_thresh=conf_thresh,
         iou_thresh=iou_thresh, show_time=show_time)
