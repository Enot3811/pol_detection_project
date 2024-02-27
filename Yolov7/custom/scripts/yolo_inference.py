"""Запустить yolo на указанных семплах.

Путь может указывать как на один .jpg, .png или .npy,
так и на директорию с несколькими.
"""


from pathlib import Path
import sys
import argparse
import json

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import cv2
import numpy as np

sys.path.append(str(Path(__file__).parents[3]))
from utils.image_utils.image_functions import read_image, collect_images_paths
from utils.torch_utils.torch_functions import draw_bounding_boxes
from Yolov7.yolov7.dataset import create_yolov7_transforms
from Yolov7.custom.model_utils import load_yolo_checkpoint, yolo_inference
from mako_camera.cameras_utils import split_raw_pol


def main(samples_pth: Path, config_pth: Path, conf_thresh: float,
         iou_thresh: float, show_time: bool):
    """Запустить yolo на указанных семплах.

    Путь может указывать как на один .jpg, .png или .npy,
    так и на директорию с несколькими.

    Parameters
    ----------
    samples_pth : Path
        Путь к семплу или директории.
    config_pth : Path
        Путь к конфигу модели.
    conf_thresh : float
        Порог уверенности модели.
        Пропускать только те предсказания, уверенность которых выше порога.
    iou_thresh : float
        Порог перекрытия рамок.
        Пропускать только те предсказания, чей коэффициент перекрытия с другим
        более уверенным предсказанием этого же класса меньше порога.
    show_time : bool
        Показывать время выполнения сети.
    """
    # Read config
    with open(config_pth, 'r') as f:
        config_str = f.read()
    config = json.loads(config_str)

    cls_id_to_name = {val: key for key, val in config['cls_to_id'].items()}
    num_classes = len(cls_id_to_name)
    num_channels = 4 if config['polarization'] else 3
    pad_colour = (114,) * num_channels

    # Получить все пути
    if samples_pth.is_dir():
        # Pol
        if config['polarization']:
            samples_pths = list(samples_pth.glob('*.npy'))
        # RGB
        else:
            samples_pths = collect_images_paths(samples_pth)
    elif samples_pth.is_file():
        samples_pths = [samples_pth]
    else:
        raise
    samples_pths.sort()

    # Загрузить модель
    weights_pth = Path(config['work_dir']) / 'ckpts' / 'best_checkpoint.pth'
    model = load_yolo_checkpoint(weights_pth, num_classes)

    # Обработка семплов
    process_transforms = create_yolov7_transforms(pad_colour=pad_colour)
    normalize_transforms = A.Compose(
        [ToTensorV2(transpose_mask=True)])

    for sample_pth in samples_pths:
        if config['polarization']:
            image = np.load(sample_pth)  # ndarray (h, w)
            image = split_raw_pol(image)  # ndarray (h//2, w//2, 4)
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
            image[..., :3], bboxes, class_labels=classes, confidences=confs)

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

    parser.add_argument('samples_pth', type=Path,
                        help='Путь к семплу или директории.')
    parser.add_argument('config_pth', type=Path,
                        help='Путь к конфигу модели.')
    parser.add_argument('--conf_thresh', type=float, default=0.3,
                        help='Порог уверенности модели.')
    parser.add_argument('--iou_thresh', type=float, default=0.2,
                        help='Порог перекрытия рамок.')
    parser.add_argument('--show_time', action='store_true',
                        help='Показывать время выполнения сети.')
    args = parser.parse_args()

    if not args.samples_pth.exists():
        raise FileExistsError('Samples file or dir does not exist.')
    if not args.config_pth.exists():
        raise FileExistsError('Config file does not exist.')
    return args


if __name__ == '__main__':
    args = parse_args()
    samples_pth = args.samples_pth
    config_pth = args.config_pth
    conf_thresh = args.conf_thresh
    iou_thresh = args.iou_thresh
    show_time = args.show_time
    main(samples_pth, config_pth, conf_thresh, iou_thresh, show_time)
