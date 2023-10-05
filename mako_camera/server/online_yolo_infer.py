"""Запустить yolo на сканирование папки и обработку всех новых кадров.

Пока работает только с официальными весами,
и для поляризации только в псевдо RGB.
"""

import sys
import time
from pathlib import Path
import argparse

import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import torch

sys.path.append(str(Path(__file__).parents[2]))
from Yolov7.custom.model_utils import (
    create_yolo, load_yolo_checkpoint, yolo_inference, idx2label)
from Yolov7.yolov7.dataset import create_yolov7_transforms
from utils.torch_utils.torch_functions import (
    image_tensor_to_numpy, draw_bounding_boxes)
from mako_camera.cameras_utils import split_raw_pol
from utils.image_utils.image_functions import read_image


def main(**kwargs):
    """Запустить yolo на сканирование папки и обработку всех новых кадров.

    Parameters
    ----------
    frames_dir : Path
        Папка сканирования.
    weights : Path
        Путь к весам модели.
    pretrained : bool
        Загрузить ли официальные предобученные веса.
        Игнорируется, если указан аргумент "weights".
    polarized : bool
        Поляризационная ли съёмка. Если False, то RGB.
    conf_thresh : float, optional
        Confidence threshold, by default 0.1
    iou_thresh : float, optional
        IoU threshold, by default 0.2
    show_time : bool, optional
        Показывать время выполнения, by default False
    """
    frames_dir = kwargs['frames_dir']
    weights = kwargs['weights']
    pretrained = kwargs['pretrained']
    polarized = kwargs['polarized']
    conf_thresh = kwargs['conf_thresh']
    iou_thresh = kwargs['iou_thresh']
    show_time = kwargs['show_time']

    # TODO как-то переделать это
    cls_id_to_name = {
        0: 'Tank'
    }
    num_classes = len(cls_id_to_name)
    # cls_id_to_name = idx2label

    img_paths = set(frames_dir.glob('*.*'))
    model = None
    image = np.zeros((500, 500, 3), np.uint8)
    while True:
        # Читаем все пути
        if polarized:
            updated_paths = set(frames_dir.glob('*.npy'))
        else:
            updated_paths = []
            for ext in ('jpg', 'JPG', 'png', 'PNG'):
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
                image = image[..., :3]
            else:
                image = read_image(pth)  # ndarray (h, w, 3)

            # Загрузка модели и предобработки
            if model is None:
                if weights:
                    model = load_yolo_checkpoint(weights, num_classes)
                else:
                    num_ch = 4 if polarized else 3
                    num_ch = 3  # нет весов для 4-х каналов
                    model = create_yolo(num_ch, pretrained)
                process_transforms = create_yolov7_transforms()
                normalize_transforms = A.Compose(
                    # [A.Normalize(), ToTensorV2(transpose_mask=True)])
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

            labels = list(map(lambda idx: cls_id_to_name[idx],
                              class_ids.tolist()[:30]))
            image = draw_bounding_boxes(
                image[0],
                boxes.tolist()[:30],
                labels,
                confidences.tolist()[:30])
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
        cv2.imshow('yolo', image)
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

    parser.add_argument('frames_dir',
                        help='Директория для сканирования.', type=Path)
    parser.add_argument('--weights',
                        help='Путь к весам модели.', type=Path, default=None)
    parser.add_argument('--pretrained', action='store_true',
                        help='Загрузить ли официальные предобученные веса. '
                        'Игнорируется, если указан аргумент "--weights".')
    parser.add_argument('--polarized',
                        help='Поляризационная съёмка. Если не указан, то RGB.',
                        action='store_true')
    parser.add_argument('--conf_thresh',
                        help='Порог уверенности модели.', type=float,
                        default=0.6)
    parser.add_argument('--iou_thresh',
                        help='Порог перекрытия рамок.', type=float,
                        default=0.2)
    parser.add_argument('--show_time',
                        help='Показывать время выполнения.',
                        action='store_true')
    args = parser.parse_args([
        'test',
        '--weights', 'Yolov7/work_dir/train_1/ckpts/best_model.pt'
    ])
    return args


if __name__ == '__main__':
    args = parse_args()
    frames_dir = args.frames_dir
    weights = args.weights
    pretrained = args.pretrained
    polarized = args.polarized
    conf_thresh = args.conf_thresh
    iou_thresh = args.iou_thresh
    show_time = args.show_time
    main(frames_dir=frames_dir, weights=weights, pretrained=pretrained,
         polarized=polarized, conf_thresh=conf_thresh,
         iou_thresh=iou_thresh, show_time=show_time)
