"""Скрипт для проверки работы RegionLocalizer с usb камерой.

Для работы скрипта необходимо передать конфиг, который использовался для
обучения нужной модели, и путь к изображению региона для тестирования.
"""


import argparse
from pathlib import Path
import json
import sys

import torch
import cv2
import numpy as np

sys.path.append(str(Path(__file__).parents[2]))
from region_localizer.models import RetinaRegionLocalizer, ModifiedRetinaV2
from utils.data_utils.data_functions import (
    read_image, resize_image, show_images_cv2)
from utils.torch_utils.torch_functions import (
    image_numpy_to_tensor, draw_bounding_boxes)


def main(config_pth: Path, image_pth: Path, device: str):
    # Read config
    with open(config_pth, 'r') as f:
        config_str = f.read()
    config = json.loads(config_str)

    # Prepare some stuff
    if device == 'auto':
        device: torch.device = (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu'))
    else:
        device: torch.device = torch.device(config['device'])
    input_size = config['val_dataset_params']['result_size']
    input_size = ((input_size,) * 2 if isinstance(input_size, int)
                  else input_size)

    # Load model
    if config['architecture'] == 'modified_retina':
        model = RetinaRegionLocalizer(
            img_min_size=config['train_dataset_params']['result_size'][0],
            img_max_size=config['train_dataset_params']['result_size'][0],
            pretrained=config['pretrained'],
            num_classes=config['num_classes'])
    elif config['architecture'] == 'modified_retina_v2':
        model = ModifiedRetinaV2.create_modified_retina(
            min_size=config['train_dataset_params']['result_size'][0],
            max_size=config['train_dataset_params']['result_size'][0],
            pretrained=config['pretrained'],
            num_classes=config['num_classes'])
    else:
        raise ValueError(f'Unrecognized architecture {config["architecture"]}')

    model.to(device=device)
    model.load_state_dict(torch.load(
        Path(config['work_dir']) / 'ckpts' / 'last_checkpoint.pth'
    )['model_state_dict'])
    model.eval()

    # Read region image
    region_img = read_image(image_pth)
    region_img = resize_image(region_img, input_size)

    # Read images from camera
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, piece_img = cap.read()
        piece_img = cv2.cvtColor(piece_img, cv2.COLOR_BGR2RGB)

        if success:
            # Prepare input for model
            piece_img = resize_image(piece_img, input_size)
            tensor_region_img = image_numpy_to_tensor(region_img, device)
            tensor_piece_img = image_numpy_to_tensor(piece_img, device)
            input_img = (torch.cat(
                [tensor_region_img, tensor_piece_img])[None, ...]
                .to(dtype=torch.float32) / 255)
            
            # Call the model
            _, predictions = model(input_img)

            # Prepare results
            if len(predictions[0]['boxes']) == 0:
                continue
            pred_bbox = predictions[0]['boxes'][0].tolist()
            bbox_region = draw_bounding_boxes(
                region_img, [pred_bbox], color=(255, 0, 0))
            show_image = np.hstack((piece_img, bbox_region))

            # Show results
            key = show_images_cv2(
                show_image, 'input/predict', destroy_windows=False, delay=1)
            
            if key == 27:
                return


def parse_args() -> argparse.Namespace:
    """Отпарсить передаваемые аргументы.

    Returns:
        argparse.Namespace: Полученные аргументы.
    """
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        'config_pth', type=Path,
        help='Путь к конфигу модели.')
    parser.add_argument(
        'image_pth', type=Path,
        help='Путь к изображения региона для теста.')
    parser.add_argument(
        '--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'],
        help=('Устройство, на котором проводить вычисления. '
              'auto выбирает cuda по возможности.'))
    args = parser.parse_args()

    if not args.config_pth.exists():
        raise FileNotFoundError(
            f'Указанный файл с конфигом "{str(args.config_pth)}" '
            'не существует.')
    if not args.image_pth.exists():
        raise FileNotFoundError(
            f'Указанная директория с изображениями "{str(args.image_pth)}" '
            'не существует.')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(config_pth=args.config_pth, image_pth=args.image_pth,
         device=args.device)
