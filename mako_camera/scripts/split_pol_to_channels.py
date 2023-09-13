"""Конвертировать сырое поляризованное изображение(-я) в 4-х канальное."""


from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).parents[2]))
from mako_camera.cameras_utils import split_raw_pol


def main():
    if SOURCE_PTH.is_file():
        DESTINATION_PTH.parent.mkdir(parents=True, exist_ok=True)
        image = np.load(SOURCE_PTH)
        image = split_raw_pol(image)
        np.save(DESTINATION_PTH, image)
    elif SOURCE_PTH.is_dir():
        DESTINATION_PTH.mkdir(parents=True, exist_ok=True)
        img_pths = SOURCE_PTH.glob('*.npy')
        for img_pth in img_pths:
            image = np.load(img_pth)
            image = split_raw_pol(image)
            img_name = img_pth.name
            np.save(DESTINATION_PTH / img_name, image)
    else:
        raise ValueError('Incorrect paths.')


if __name__ == '__main__':
    SOURCE_PTH = Path('path/to/raw/pol/npy/frames')
    DESTINATION_PTH = Path('path/to/save/split/frames')
    main()
