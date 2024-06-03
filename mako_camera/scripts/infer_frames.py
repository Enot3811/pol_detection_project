"""Обработка и просмотр поляризованных изображений по указанному пути.

Путь может указывать как на одно .npy, так и на директорию с несколькими.
Полученный pol изображения перегоняются в aolp, dolp, hsv и pseudo_rgb и
затем показываются или сохраняются в зависимости от заданного режима.
"""


import sys
import cv2
from pathlib import Path
from typing import Tuple, Optional
import argparse

import numpy as np

sys.path.append(str(Path(__file__).parents[2]))
from utils.data_utils.data_functions import (
    resize_image, normalize_to_image, save_image)
import mako_camera.cameras_utils as pol_func


def handle_frame(
    pth: Path, showing: bool, win_size: Tuple[int, int]
) -> Optional[int]:
    frame = np.load(pth)
    split_channels = pol_func.split_raw_pol(frame).astype(np.float32) / 255
    s0, s1, s2 = pol_func.calc_Stocks_param(
        split_channels[..., 0], split_channels[..., 1],
        split_channels[..., 2], split_channels[..., 3])
    pol_int = pol_func.pol_intensity(s1, s2)
    aolp = pol_func.calc_AoLP(s1, s2)
    dolp = pol_func.calc_DoLP(s0, pol_int)
    dolp_img = normalize_to_image(dolp)
    aolp_img = normalize_to_image(aolp)
    hsv = pol_func.hsv_pol(aolp, dolp, pol_int)
    pseudo_rgb = split_channels[..., :3]
    pseudo_rgb[..., 2] = (pseudo_rgb[..., 2] + split_channels[..., 3]) / 2

    key = None
    if showing:
        # cv2.imshow('Pseudo rgb (1, 2, 3)', resize_image(split_channels[..., :3], win_size))
        # cv2.imshow('Pseudo rgb (2, 3, 4)', resize_image(split_channels[..., 1:], win_size))
        # cv2.imshow('Pseudo rgb (1, 3, 4)', resize_image(split_channels[..., [0, 2, 3]], win_size))
        # cv2.imshow('Pseudo rgb (1, 2, 4)', resize_image(split_channels[..., [0, 1, 3]], win_size))
        cv2.imshow('Pseudo rgb', resize_image(pseudo_rgb, win_size))
        # cv2.imshow('Channel 0', resize_image(split_channels[..., 0], win_size))
        # cv2.imshow('Channel 45', resize_image(split_channels[..., 1], win_size))
        # cv2.imshow('Channel 90', resize_image(split_channels[..., 2], win_size))
        # cv2.imshow('Channel 135', resize_image(split_channels[..., 3], win_size))
        cv2.imshow('AoLP', resize_image(aolp_img, win_size))
        cv2.imshow('DoLP', resize_image(dolp_img, win_size))
        cv2.imshow('HSV', resize_image(hsv, win_size))
        print(pth.name)
        key = cv2.waitKey(0) & 0xFF
    else:
        name = pth.name.split('.')[0] + '.jpg'
        save_image((split_channels[..., :3] * 255).astype(np.uint8),
                   frame_pth.parent / 'pseudo_rgb' / name)
        save_image(hsv, frame_pth.parent / 'hsv' / name)
        save_image(aolp_img, frame_pth.parent / 'aolp' / name)
        save_image(dolp_img, frame_pth.parent / 'dolp' / name)
    return key


def main(frame_pth: Path, showing: bool, win_size: Tuple[int, int]):
    if frame_pth.is_dir():
        paths = list(frame_pth.glob('*.npy'))
        try:
            # Пробуем отсортировать по индексу
            paths = list(sorted(paths, key=lambda pth: int(pth.name[4:-4])))
        except:
            # Если названия фреймов не подходят и сортировка не удаётся,
            # то просто сортируем
            paths.sort()
        for path in paths:
            code = handle_frame(path, showing, win_size)
            if code == 27:
                break

    elif frame_pth.is_file():
        handle_frame(frame_pth, showing, win_size)

    else:
        raise

    if showing:
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('frame_pth',
                        help='Путь к фрейму или директории с фреймами.',
                        type=Path)
    parser.add_argument('--save_mode',
                        help='Включить режим сохранения. '
                        'Если указан, то вместо показа фреймы будут сохранены '
                        'в директории.',
                        action='store_true')
    parser.add_argument('--win_size',
                        help='Размер для ресайза перед показом.',
                        type=int, nargs=2,
                        default=(700, 700))

    args = parser.parse_args()

    if not args.frame_pth.exists():
        raise FileExistsError('Frame file or dir does not exist.')

    return args


if __name__ == '__main__':
    args = parse_args()
    frame_pth = args.frame_pth
    showing = not args.save_mode
    win_size = args.win_size
    main(frame_pth, showing, win_size)
