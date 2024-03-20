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

from tqdm import tqdm
from albumentations import RandomRotate90

sys.path.append(str(Path(__file__).parents[2]))
from utils.image_utils.image_functions import (
    collect_images_paths, read_image, save_image)
from utils.argparse_utils import required_length, natural_int
from utils.torch_utils.torch_functions import random_crop


def main(
    image_dir: Path, min_crop_size: Union[int, Tuple[int, int]],
    max_crop_size: Union[int, Tuple[int, int]],
    num_pieces: int, rotate_pieces: bool
):
    # Prepare some stuff
    if rotate_pieces:
        rotate = RandomRotate90()

    # Get images
    regions_pths = collect_images_paths(image_dir)
    regions_pths.sort()

    # Check result dir
    result_dir = image_dir.parent / 'pieces'
    if result_dir.exists():
        input(f'Output directory "{result_dir}" already exists. '
              'Сontinuing to work will delete the data located there. '
              'Press enter to continue.')
        shutil.rmtree(result_dir)

    # Generate pieces
    for reg_pth in tqdm(regions_pths, desc='Generate pieces'):
        reg_name = reg_pth.name.split('.')[0]
        reg_img = read_image(reg_pth)
        pieces_dir = result_dir / reg_name

        for i in range(num_pieces):
            piece_img, bbox = random_crop(
                reg_img, min_size=min_crop_size, max_size=max_crop_size,
                return_position=True)
            if rotate_pieces:
                piece_img = rotate(piece_img)

            # Save pieces
            piece_name = f'{reg_name}_{i}'
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
        help='Минимальный размер вырезанного фрагмента.')
    parser.add_argument(
        '--max_crop_size', type=natural_int, nargs='+',
        action=required_length(1, 2), required=True,
        help='Максимальный размер вырезанного фрагмента.')
    parser.add_argument(
        '--num_pieces', type=natural_int, default=1,
        help='Количество производимых фрагментов на один регион.')
    parser.add_argument(
        '--rotate_pieces', action='store_true',
        help='Следует ли крутить изображения фрагментов.')
    
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
    rotate_pieces = args.rotate_pieces
    main(image_dir=image_dir, num_pieces=num_pieces,
         min_crop_size=min_crop_size, max_crop_size=max_crop_size,
         rotate_pieces=rotate_pieces)
