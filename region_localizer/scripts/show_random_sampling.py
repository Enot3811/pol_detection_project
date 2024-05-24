"""Script to make some samples of random cropping from region map."""


import argparse
from pathlib import Path
import sys
from typing import Tuple, Optional
from random import randint

sys.path.append(str(Path(__file__).parents[2]))
from utils.data_utils.data_functions import (
    read_image, show_image_plt, save_image)
from utils.torch_utils.torch_functions import random_crop, draw_bounding_boxes
from utils.argparse_utils import natural_int, required_length


def main(
    src_image: Path, min_crop_size: Tuple[int, int],
    max_crop_size: Tuple[int, int], num_pieces: int, save_pth: Optional[Path],
    color: Optional[Tuple[int, int, int]]
):
    generate_color = color is None
    image = read_image(src_image)
    for _ in range(num_pieces):
        bbox = [
            random_crop(
                image, min_size=min_crop_size, max_size=max_crop_size,
                return_position=True
            )[1]]
        if generate_color:
            color = [randint(0, 255) for _ in range(3)]
        image = draw_bounding_boxes(image, bbox, color=color, line_width=2)
    show_image_plt(image, plt_show=True)
    if save_pth is not None:
        save_image(image, save_pth)


def parse_args() -> argparse.Namespace:
    """Отпарсить передаваемые аргументы.

    Returns:
        argparse.Namespace: Полученные аргументы.
    """
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        'src_image', type=Path,
        help='A path to an image of map.')
    parser.add_argument(
        '--min_crop_size', type=natural_int, nargs='+',
        action=required_length(1, 2), required=True,
        help=(
            'Minimum fragment size. When given only one dimension, '
            'the fragments will be strictly square. With two - rectangular.'))
    parser.add_argument(
        '--max_crop_size', type=natural_int, nargs='+',
        action=required_length(1, 2), required=True,
        help=(
            'Maximum fragment size. When given only one dimension, '
            'the fragments will be strictly square. With two - rectangular.'))
    parser.add_argument(
        '--num_pieces', type=natural_int, default=7,
        help='A number of random random crops to make.')
    parser.add_argument(
        '--save_pth', type=Path, default=None,
        help='Save path if saving is needed.')
    parser.add_argument(
        '--color', type=int, nargs=3, default=None, required=False,
        help='RGB color for bounding boxes.')

    args = parser.parse_args()

    if not args.src_image.exists():
        raise FileNotFoundError(
            f'Указанный файл с конфигом "{args.config_pth}" не существует.')
    
    if len(args.min_crop_size) == 1:
        args.min_crop_size = args.min_crop_size[0]
    else:
        args.min_crop_size = tuple(args.min_crop_size)
    if len(args.max_crop_size) == 1:
        args.max_crop_size = args.max_crop_size[0]
    else:
        args.max_crop_size = tuple(args.max_crop_size)

    return args


if __name__ == '__main__':
    args = parse_args()
    main(src_image=args.src_image, min_crop_size=args.min_crop_size,
         max_crop_size=args.max_crop_size, num_pieces=args.num_pieces,
         save_pth=args.save_pth, color=args.color)
