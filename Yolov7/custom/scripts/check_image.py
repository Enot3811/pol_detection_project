"""
Check a sample from polarized dataset.
"""


from pathlib import Path
from typing import Tuple

import numpy as np
import cv2 as cv
import pandas as pd


def split_RAW_image_to_channels(image: np.ndarray) -> np.ndarray:
    """
    Split polarized image with shape (h x w) to 4 channels image with shape (h/2 x w/2 x 4).

    Parameters
    ----------
    image : np.ndarray
        Raw polarized image.

    Returns
    -------
    np.ndarray
        Polarized image that is splitted to 4 channels.
    """
    ch_90 = image[::2, ::2][..., None]
    ch_45 = image[::2, 1::2][..., None]
    ch_135 = image[1::2, ::2][..., None]
    ch_0 = image[1::2, 1::2][..., None]
    return np.concatenate((ch_90, ch_45, ch_135, ch_0), 2)


def check_image(image: str, bbox: Tuple[int, int, int, int]):
    img = split_RAW_image_to_channels(np.load(image))

    cv.rectangle(img, bbox[:2], bbox[2:], (128, 0, 0), 2)

    cv.imshow('Image', img)
    cv.waitKey(1000)
    cv.destroyAllWindows()


def main():
    dset_path = Path('/home/enot/projects/data/polarized_dataset/')
    annotations = pd.read_csv(dset_path / 'annotations.csv')

    for _ in range(10):
        idx = np.random.randint(0, 100)
        image, xmin, ymin, xmax, ymax = annotations.iloc[idx]

        with open(dset_path / 'Labels' / (image + 'pseudo_RGB.txt')) as f:
            print(f.read())

        img_path = dset_path / 'Images' / image
        check_image(str(img_path), (xmin, ymin, xmax, ymax))



if __name__ == '__main__':
    main()
