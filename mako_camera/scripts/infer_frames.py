"""Посмотреть на поляризованные изображения по указанному пути.

Путь может указывать как на одно .npy, так и на директорию с несколькими.
Применяется вся та же обработка, как и в скриптах работы с камерой.
"""


import sys
import cv2
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parents[2]))
from utils.image_utils.image_functions import (
    resize_image, normalize_to_image, save_image)
import mako_camera.cameras_utils as pol_func


def show_pol(pth: Path) -> int:
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
    
    cv2.imshow('Pseudo rgb', resize_image(split_channels[..., :3], WIN_SIZE))
    cv2.imshow('Channel 0', resize_image(split_channels[..., 0], WIN_SIZE))
    cv2.imshow('Channel 45', resize_image(split_channels[..., 1], WIN_SIZE))
    cv2.imshow('Channel 90', resize_image(split_channels[..., 2], WIN_SIZE))
    cv2.imshow('Channel 135', resize_image(split_channels[..., 3], WIN_SIZE))
    cv2.imshow('AoLP', resize_image(aolp_img, WIN_SIZE))
    cv2.imshow('DoLP', resize_image(dolp_img, WIN_SIZE))
    cv2.imshow('HSV', resize_image(hsv, WIN_SIZE))
    print(pth.name)
    if SAVE:
        name = pth.name.split('.')[0] + '.jpg'
        save_image((split_channels[..., :3] * 255).astype(np.uint8),
                   FRAME_PATH.parent / 'pseudo_rgb' / name)
        save_image(hsv, FRAME_PATH.parent / 'hsv' / name)
        save_image(aolp_img, FRAME_PATH.parent / 'aolp' / name)
        save_image(dolp_img, FRAME_PATH.parent / 'dolp' / name)
    key = cv2.waitKey(0) & 0xFF
    return key


def main():
    if FRAME_PATH.is_dir():
        paths = list(FRAME_PATH.glob('*.npy'))
        if SORT_BY_INDEX:
            paths = list(sorted(paths, key=lambda pth: int(pth.name[4:-4])))
        else:
            paths.sort()
        for path in paths:
            code = show_pol(path)
            if code == 27:
                break

    elif FRAME_PATH.is_file():
        show_pol(FRAME_PATH)

    else:
        raise


if __name__ == '__main__':
    FRAME_PATH = Path('data/tank_1set_pol/images')
    WIN_SIZE = (700, 700)
    SORT_BY_INDEX = True
    SAVE = False
    main()
