"""Посмотреть на поляризованные изображения по указанному пути.

Путь может указывать как на одно .npy, так и на директорию с несколькими.
Применяется вся та же обработка, как и в скриптах работы с камерой.
"""


import sys
import cv2
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parents[1]))
from utils.image_utils import resize_image
from utils.cameras_utils import (
    split_raw_pol, calc_AoLP, calc_DoLP, calc_Stocks_param, hsv_pol,
    pol_intensity, normalize_to_image)


def show_pol(pth: Path) -> int:
    frame = np.load(pth)
    split_channels = split_raw_pol(frame).astype(np.float32) / 255
    s0, s1, s2 = calc_Stocks_param(
        split_channels[..., 0], split_channels[..., 1],
        split_channels[..., 2], split_channels[..., 3])
    pol_int = pol_intensity(s1, s2)
    aolp = calc_AoLP(s1, s2)
    dolp = calc_DoLP(s0, pol_int)
    dolp_img = normalize_to_image(dolp)
    aolp_img = normalize_to_image(aolp)
    hsv = hsv_pol(aolp, dolp, pol_int)
    

    cv2.imshow('Original image', resize_image(split_channels[..., :3], WIN_SIZE))
    cv2.imshow('Channel 0', resize_image(split_channels[..., 0], WIN_SIZE))
    cv2.imshow('Channel 45', resize_image(split_channels[..., 1], WIN_SIZE))
    cv2.imshow('Channel 90', resize_image(split_channels[..., 2], WIN_SIZE))
    cv2.imshow('Channel 135', resize_image(split_channels[..., 3], WIN_SIZE))
    cv2.imshow('AoLP', resize_image(aolp_img, WIN_SIZE))
    cv2.imshow('DoLP', resize_image(dolp_img, WIN_SIZE))
    cv2.imshow('HSV', resize_image(hsv, WIN_SIZE))
    key = cv2.waitKey(0) & 0xFF
    return key


def main():
    if FRAME_PATH.is_dir():
        paths = list(FRAME_PATH.glob('*.npy'))
        paths = list(sorted(paths, key=lambda pth: int(pth.name[4:-4])))
        for path in paths:
            code = show_pol(path)
            if code == 27:
                break

    elif FRAME_PATH.is_file():
        show_pol(FRAME_PATH)

    else:
        raise


if __name__ == '__main__':
    FRAME_PATH = Path('/home/pc0/projects/mako_camera/data/2023_09_09/raw_pol/')
    WIN_SIZE = (500, 500)
    main()
