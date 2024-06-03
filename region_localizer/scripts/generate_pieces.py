"""Сгенерировать директорию с вырезанными из региона фрагментами.

Каждому изображению региона из переданной директории в соответствие директория
с аналогичным названием, в которой будут находиться вырезанные .jpg изображения
фрагментов и соответствующие .txt файлы с их координатами.
pieces
├── 0
|   ├── 0_0.jpg
|   ├── 0_0.txt
|   ├── 0_1.jpg
|   ├── 0_1.txt
|   ├── ...
|   ├── 0_{required_pieces - 1}.jpg
|   ├── 0_{required_pieces - 1}.txt
├── 1
|   ├── 1_0.jpg
|   ├── 1_0.txt
|   ├── 1_1.jpg
|   ├── 1_1.txt
|   ├── ...
|   ├── 1_{required_pieces - 1}.jpg
|   ├── 1_{required_pieces - 1}.txt
...
├── {num_regions - 1}
|   ├── {num_regions - 1}_0.jpg
|   ├── {num_regions - 1}_0.txt
|   ├── {num_regions - 1}_1.jpg
|   ├── {num_regions - 1}_1.txt
|   ├── ...
|   ├── {num_regions - 1}_{required_pieces - 1}.jpg
|   ├── {num_regions - 1}_{required_pieces - 1}.txt
"""


import argparse
from pathlib import Path
import sys
from typing import Union, Tuple
import shutil

import numpy as np
from tqdm import tqdm
import albumentations as A

sys.path.append(str(Path(__file__).parents[2]))
from utils.argparse_utils import required_length, natural_int
from utils.torch_utils.torch_functions import random_crop
from utils.data_utils.data_functions import (
    collect_paths, read_volume, save_image)


def main(
    image_dir: Path, min_crop_size: Union[int, Tuple[int, int]],
    max_crop_size: Union[int, Tuple[int, int]],
    num_pieces: int, rotation: bool, color_jitter: bool, blur: bool
):
    # Prepare some stuff
    if rotation or color_jitter or color_jitter:
        transforms = []
        if rotation:
            transforms.append(A.RandomRotate90())
        if color_jitter:
            transforms.append(A.ColorJitter(
                brightness=(0.4, 1.3), contrast=(0.7, 1.2),
                saturation=(0.5, 1.4), hue=(-0.01, 0.01), p=0.5))
        if blur:
            transforms.append(A.Blur(blur_limit=3, p=0.5))
        transforms = A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='pascal_voc', label_fields=['labels']))
    else:
        transforms = None

    # Get images
    regions_pths = collect_paths(
        image_dir, file_extensions=('jpg', 'png', 'npy'))
    regions_pths.sort()

    # Check result dir
    result_dir = image_dir.parent / 'pieces'
    if result_dir.exists():
        input(f'Output directory "{result_dir}" already exists. '
              'If continue, this directory will be deleted. '
              'Press enter to continue.')
        shutil.rmtree(result_dir)

    # Generate pieces
    for reg_pth in tqdm(regions_pths, desc='Generate pieces'):
        reg_img = read_volume(reg_pth)
        pieces_dir = result_dir / reg_pth.stem

        for i in range(num_pieces):
            piece_img, bbox = random_crop(
                reg_img, min_size=min_crop_size, max_size=max_crop_size,
                return_position=True)
            if rotation:
                piece_img = transforms(
                    image=piece_img, bboxes=[], labels=[])['image']

            # Save pieces
            piece_name = f'{reg_pth.stem}_{i}'
            if reg_pth.suffix == '.npy':
                pieces_dir.mkdir(exist_ok=True, parents=True)
                np.save(pieces_dir / (piece_name + '.npy'), piece_img)
            else:
                save_image(piece_img, pieces_dir / (piece_name + '.jpg'))
            bbox = map(str, bbox)
            with open(pieces_dir / (piece_name + '.txt'), 'w') as f:
                f.write(' '.join(bbox))


def parse_args() -> argparse.Namespace:
    """Отпарсить передаваемые аргументы.

    Returns:
        argparse.Namespace: Полученные аргументы.
    """
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        'image_dir', type=Path,
        help='Путь к директории с изображениями регионов.')
    parser.add_argument(
        '--min_crop_size', type=natural_int, nargs='+',
        action=required_length(1, 2), required=True,
        help=('Минимальный размер вырезанного фрагмента. При передаче одной '
              'размерности фрагменты будут строго квадратными. '
              'При двух - прямоугольными.'))
    parser.add_argument(
        '--max_crop_size', type=natural_int, nargs='+',
        action=required_length(1, 2), required=True,
        help=('Максимальный размер вырезанного фрагмента. При передаче одной '
              'размерности фрагменты будут строго квадратными. '
              'При двух - прямоугольными.'))
    parser.add_argument(
        '--num_pieces', type=natural_int, default=1,
        help='Количество производимых фрагментов на один регион.')
    parser.add_argument(
        '--rotation', action='store_true',
        help='Следует ли крутить изображения фрагментов.')
    parser.add_argument(
        '--color_jitter', action='store_true',
        help='Применить color jitter на изображения фрагментов.')
    parser.add_argument(
        '--blur', action='store_true',
        help='Применить blur на изображения фрагментов.')
    
    args = parser.parse_args()

    if not args.image_dir.exists():
        raise FileNotFoundError(
            f'Указанная директория с изображениями "{args.image_dir}" '
            'не существует.')
    
    if len(args.min_crop_size) == 1:
        args.min_crop_size = args.min_crop_size[0]
    if len(args.max_crop_size) == 1:
        args.max_crop_size = args.max_crop_size[0]

    return args


if __name__ == '__main__':
    args = parse_args()
    image_dir = args.image_dir
    min_crop_size = args.min_crop_size
    max_crop_size = args.max_crop_size
    num_pieces = args.num_pieces
    rotation = args.rotation
    color_jitter = args.color_jitter
    blur = args.blur
    main(image_dir=image_dir, num_pieces=num_pieces,
         min_crop_size=min_crop_size, max_crop_size=max_crop_size,
         rotation=rotation, color_jitter=color_jitter, blur=blur)
