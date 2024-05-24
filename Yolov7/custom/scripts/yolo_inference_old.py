"""Старый inference для моделей из старого пайплайна.

Запустить yolo на указанных семплах.
Путь может указывать как на один .jpg, .png или .npy,
так и на директорию с несколькими.
"""


from pathlib import Path
import sys
import argparse

import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import cv2

sys.path.append(str(Path(__file__).parents[3]))
from mako_camera.cameras_utils import split_raw_pol
from utils.image_utils.image_functions import read_image, IMAGE_EXTENSIONS
from utils.data_utils.data_functions import collect_paths
from utils.torch_utils.torch_functions import draw_bounding_boxes
from Yolov7.yolov7.dataset import create_yolov7_transforms
from Yolov7.custom.model_utils import (  # noqa
    create_yolo, load_yolo_checkpoint, yolo_inference, coco_idx2label)


def main(
    samples_pth: Path, weights: Path, pretrained: bool, polarized: bool,
    conf_thresh: float, iou_thresh: float, show_time: bool
):
    """Запустить yolo на указанных семплах.

    Путь может указывать как на один .jpg, .png или .npy,
    так и на директорию с несколькими.

    Parameters
    ----------
    samples_pth : Path
        Путь к семплу или директории.
    weights : Path
        Путь к весам модели.
    pretrained : bool
        Загрузить ли официальные предобученные веса.
        Игнорируется, если указан аргумент "weights".
    polarized : bool
        Поляризационная ли съёмка. Если False, то RGB.
    conf_thresh : float
        Порог уверенности модели.
        Пропускать только те предсказания, уверенность которых выше порога.
    iou_thresh : float
        Порог перекрытия рамок.
        Пропускать только те предсказания, чей коэффициент перекрытия с другим
        более уверенным предсказанием этого же класса меньше порога.
    show_time : bool
        Показывать время выполнения.
    """
    # cls_id_to_name = {
    #     0: 'Tank'
    # }
    # cls_id_to_name = {
    #     0: "fire",
    #     1: "smoke"
    # }
    cls_id_to_name = coco_idx2label
    num_classes = len(cls_id_to_name)

    # Получить все пути
    if samples_pth.is_dir():
        # Поляризация
        if polarized:
            samples_pths = list(samples_pth.glob('*.npy'))
        # RGB
        else:
            samples_pths = collect_paths(samples_pth, IMAGE_EXTENSIONS)

    elif samples_pth.is_file():
        samples_pths = [samples_pth]

    else:
        raise
    samples_pths.sort()

    # Загрузить модель
    if weights:
        model = load_yolo_checkpoint(weights, num_classes)
    else:
        num_ch = 4 if polarized else 3
        num_ch = 3  # нет весов для 4-х каналов
        model = create_yolo(num_classes, num_ch, pretrained)

    # Обработка семплов
    process_transforms = create_yolov7_transforms()
    normalize_transforms = A.Compose(
        [ToTensorV2(transpose_mask=True)])

    for sample_pth in samples_pths:
        if polarized:
            raw_pol = np.load(sample_pth)
            image = split_raw_pol(raw_pol)  # ndarray (h, w, 4)
            image = image[..., :3]
        else:
            image = read_image(sample_pth)  # ndarray (h, w, 3)

        image = process_transforms(image=image, bboxes=[], labels=[])['image']
        tensor_image = normalize_transforms(image=image)['image']
        tensor_image = tensor_image.to(torch.float32) / 255
        tensor_image = tensor_image[None, ...]  # Add batch dim

        boxes, class_ids, confidences = yolo_inference(
            model, tensor_image, conf_thresh, iou_thresh)

        bboxes = boxes.tolist()[:30]
        class_ids = class_ids.tolist()[:30]
        confs = confidences.tolist()[:30]

        classes = list(map(lambda idx: cls_id_to_name[idx],
                           class_ids))
        bbox_img = draw_bounding_boxes(
            image, bboxes, class_labels=classes, confidences=confs)
        
        print(sample_pth.name)
        cv2.imshow('Yolo inference (press any key)',
                   cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(0)
        if key == 27:  # esc
            break
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('samples_pth',
                        help='Путь к семплу или директории.', type=Path)
    parser.add_argument('--weights',
                        help='Путь к весам модели.', type=Path, default=None)
    parser.add_argument('--pretrained', action='store_true',
                        help='Загрузить ли официальные предобученные веса. '
                        'Игнорируется, если указан аргумент "--weights".')
    parser.add_argument('--polarized',
                        help='Поляризационная ли съёмка. '
                        'Если не указан, то RGB.',
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
        # 'data/camera/2023_09_22_dark_room/keyboard/40lux/dolp/pol_2068.jpg',
        'data/camera/2023_09_22_dark_room/backpack/5lux/dolp/pol_4177.jpg',
        '--pretrained',
        '--conf_thresh', '0.2'
    ])
    return args


if __name__ == '__main__':
    args = parse_args()
    samples_pth = args.samples_pth
    weights = args.weights
    pretrained = args.pretrained
    polarized = args.polarized
    conf_thresh = args.conf_thresh
    iou_thresh = args.iou_thresh
    show_time = args.show_time
    main(samples_pth, weights, pretrained, polarized, conf_thresh, iou_thresh,
         show_time)
