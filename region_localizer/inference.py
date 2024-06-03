"""Скрипт для проверки работы RegionLocalizer на переданных изображениях.

Для работы скрипта необходимо передать конфиг, который использовался для
обучения нужной модели и путь к директории с проверочными изображениями.
Директория для проверки должна иметь следующую структуру
given_dir
├── regions
|   ├── 0.jpg
|   ├── 1.jpg
|   ├── ...
|   ├── {num_sample - 1}.jpg
├── pieces (optional)
|   ├── 0.jpg
|   ├── 0.txt (optional)
|   ├── 1.jpg
|   ├── 1.txt (optional)
|   ├── ...
|   ├── {num_sample - 1}.jpg
|   ├── {num_sample - 1}.txt (optional)
где в regions лежат изображения регионов, а в pieces изображения фрагментов
из соответствующих регионов и соответствующие текстовые файлы
с xyxy координатами фрагментов относительно региона.
Если директория pieces отсутствует, то фрагменты автоматически создадутся
из изображений регионов с помощью "generate_pieces" скрипта.
Файлы txt с координатами рамок используются для отображения рамки фрагмента
на изображении региона. Если файл отсутствует, то и отображение производиться
не будет.
"""


import argparse
from pathlib import Path
import json
import sys

import numpy as np
import torch
from torchmetrics.detection import IntersectionOverUnion

sys.path.append(str(Path(__file__).parents[1]))
from region_localizer.models import RetinaRegionLocalizer, ModifiedRetinaV2
from utils.data_utils.data_functions import resize_image, show_images_cv2
from utils.torch_utils.torch_functions import (
    image_numpy_to_tensor, draw_bounding_boxes)
from utils.data_utils.data_functions import collect_paths, read_image
from region_localizer.scripts.generate_pieces import main as generate_pieces


def main(
    config_pth: Path, image_dir: Path, device: str
):
    # Read config
    with open(config_pth, 'r') as f:
        config_str = f.read()
    config = json.loads(config_str)

    # Check passed images directories
    regions_dir = image_dir / 'regions'
    pieces_dir = image_dir / 'pieces'
    if not regions_dir.exists():
        raise FileNotFoundError(
            f'Директория с изображениями регионов {str(regions_dir)} '
            'не существует')
    if not pieces_dir.exists():
        print(f'Директория с фрагментами регионов {pieces_dir} не существует. '
              'При продолжении фрагменты будут созданы автоматически. '
              'Enter для продолжения.')
        
        generate_pieces(
            regions_dir, config['val_dataset_params']['min_crop_size'],
            config['val_dataset_params']['max_crop_size'],
            num_pieces=1, rotate_pieces=False)

    # Prepare some stuff
    if device == 'auto':
        device: torch.device = (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu'))
    else:
        device: torch.device = torch.device(config['device'])
    iou_metric = IntersectionOverUnion().to(device=device)
    input_size = config['val_dataset_params']['result_size']
    input_size = ((input_size,) * 2 if isinstance(input_size, int)
                  else input_size)
    if 'map_grayscale' in config['val_dataset_params']:
        map_grayscale = config['val_dataset_params']['map_grayscale']
    else:
        map_grayscale = False

    # Load model
    if config['architecture'] == 'modified_retina':
        model = RetinaRegionLocalizer(
            img_min_size=config['train_dataset_params']['result_size'][0],
            img_max_size=config['train_dataset_params']['result_size'][0],
            pretrained=config['pretrained'],
            num_classes=config['num_classes'])
    elif config['architecture'] == 'modified_retina_v2':
        model = ModifiedRetinaV2.create_modified_retina(
            input_channels=config['input_channels'],
            min_size=config['train_dataset_params']['result_size'][0],
            max_size=config['train_dataset_params']['result_size'][0],
            pretrained=config['pretrained'],
            num_classes=config['num_classes'],
            image_mean=config['image_mean'],
            image_std=config['image_std'])
    else:
        raise ValueError(f'Unrecognized architecture {config["architecture"]}')

    model.to(device=device)
    model.load_state_dict(torch.load(
        Path(config['work_dir']) / 'ckpts' / 'last_checkpoint.pth'
    )['model_state_dict'])
    model.eval()

    # Iterate over regions
    regions_pths = collect_paths(
        image_dir / 'regions', file_extensions=('jpg', 'npy'))
    regions_pths.sort()
    for region_pth in regions_pths:
        region_img = read_image(region_pth, map_grayscale)
        region_pieces_dir = pieces_dir / region_pth.stem

        # Prepare map image to pass to the model
        orig_size = region_img.shape[:2]
        dh = input_size[0] / orig_size[0]
        dw = input_size[1] / orig_size[1]
        region_img = resize_image(region_img, input_size)

        # Iterate over region's fragments
        pieces_bboxes_pths = list(region_pieces_dir.glob('*.txt'))
        pieces_bboxes_pths.sort()
        pieces_pths = collect_paths(
            region_pieces_dir, file_extensions=('jpg', 'npy'))
        pieces_pths.sort()
        for bbox_pth, piece_pth in zip(pieces_bboxes_pths, pieces_pths):
            orig_piece_img = read_image(piece_pth)
            with open(bbox_pth) as f:
                bbox = list(map(int, f.readline().split(' ')))
            
            # Prepare input for the model
            bbox[0] *= dw
            bbox[2] *= dw
            bbox[1] *= dh
            bbox[3] *= dh
            piece_img = resize_image(orig_piece_img, input_size)
            tensor_region_img = image_numpy_to_tensor(region_img, device)
            piece_img = image_numpy_to_tensor(piece_img, device)
            input_img = (torch.cat([tensor_region_img, piece_img])[None, ...]
                         .to(dtype=torch.float32) / 255)
            # Call the model
            _, predictions = model(input_img)

            # Prepare results

            bbox_region = region_img
            # Add 3-ch for colorful bboxes
            if bbox_region.shape[-1] == 1:
                bbox_region = np.repeat(bbox_region, 3, axis=2)

            pred_bbox = predictions[0]['boxes'][0].tolist()
            bbox_region = draw_bounding_boxes(
                bbox_region, [bbox], color=(0, 255, 0))
            bbox_region = draw_bounding_boxes(
                bbox_region, [pred_bbox], color=(255, 0, 0))
            
            # Resize for screen size
            if bbox_region.shape[0] > 1000:
                bbox_region = resize_image(bbox_region, (1000, 1000))
            if (orig_piece_img.shape[0] > 1000 and
                isinstance(config['train_dataset_params']['min_crop_size'],
                           int)):  # Проверка на квадрат
                orig_piece_img = resize_image(orig_piece_img, (1000, 1000))

            key = show_images_cv2(
                [bbox_region, orig_piece_img],
                ['Red - predict, green - gt', 'Input'],
                destroy_windows=False)
            if key == 27:
                return
            
            # Calculate metric
            pred = [{
                'boxes': predictions[0]['boxes'][0][None, ...],
                'scores': predictions[0]['scores'][0][None, ...],
                'labels': predictions[0]['labels'][0][None, ...]
            }]
            target = [{
                'boxes': torch.tensor(bbox, device=device)[None, ...],
                'labels': torch.tensor(1, device=device)[None, ...]
            }]
            iou_metric.update(pred, target)
    iou_val = iou_metric.compute()
    print('Overall IoU:', iou_val)


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
        'image_dir', type=Path,
        help='Путь к директории с изображениями.')
    parser.add_argument(
        '--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'],
        help=('Устройство, на котором проводить вычисления. '
              'auto выбирает cuda по возможности.'))
    args = parser.parse_args()

    if not args.config_pth.exists():
        raise FileNotFoundError(
            f'Указанный файл с конфигом "{str(args.config_pth)}" '
            'не существует.')
    if not args.image_dir.exists():
        raise FileNotFoundError(
            f'Указанная директория с изображениями "{str(args.image_dir)}" '
            'не существует.')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(config_pth=args.config_pth, image_dir=args.image_dir,
         device=args.device)
