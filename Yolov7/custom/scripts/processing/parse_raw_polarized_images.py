"""Конвертировать сырое поляризованное изображение(-я) в 4-х канальное."""


from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).parents[3]))
from custom.utils.cameras_utils import split_raw_image_to_channels


def main():
    if SOURCE_PTH.is_file():
        image = np.load(SOURCE_PTH)
        image = split_raw_image_to_channels(image)
        np.save(DESTINATION_PTH, image)
    elif SOURCE_PTH.is_dir():
        img_pths = SOURCE_PTH.glob('*.npy')
        for img_pth in img_pths:
            image = np.load(img_pth)
            image = split_raw_image_to_channels(image)
            img_name = img_pth.name
            np.save(DESTINATION_PTH / img_name, image)
    else:
        raise ValueError('Incorrect paths.')


if __name__ == '__main__':
    SOURCE_PTH = Path('/home/pc0/projects/yolov7_training/data/cameras/30_08_23_1/raw_pol/')  # noqa
    DESTINATION_PTH = Path('/home/pc0/projects/yolov7_training/data/cameras/30_08_23_1/pol_4channels/')  # noqa
    main()
