"""Конвертировать bayer rg8 изображение(-я) в rgb."""


from pathlib import Path
import sys

import numpy as np
import cv2

sys.path.append(str(Path(__file__).parents[1]))
from image_utils import save_image


def main():
    if SOURCE_PTH.is_file():
        image = np.load(SOURCE_PTH)
        image = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2BGR)
        save_image(image, DESTINATION_PTH)
    elif SOURCE_PTH.is_dir():
        img_pths = SOURCE_PTH.glob('*.npy')
        for img_pth in img_pths:
            image = np.load(img_pth)
            image = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2BGR)
            img_name = img_pth.name.split('.')[0] + '.jpg'
            save_image(image, DESTINATION_PTH / img_name)
    else:
        raise ValueError('Incorrect paths.')


if __name__ == '__main__':
    SOURCE_PTH = Path('path/to/source/dir')
    DESTINATION_PTH = Path('path/to/destination/dir')
    main()
