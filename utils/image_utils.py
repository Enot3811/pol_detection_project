"""A module that contain functions for working with images."""


from pathlib import Path
from typing import Tuple, Union, Optional

import numpy as np
import cv2


def read_image(path: Union[Path, str], grayscale: bool = False) -> np.ndarray:
    """Read image to numpy array.

    Parameters
    ----------
    path : Union[Path, str]
        Path to image file
    grayscale : bool, optional
        Whether read image in grayscale, by default False

    Returns
    -------
    np.ndarray
        Array containing read image.

    Raises
    ------
    FileNotFoundError
        Did not find image.
    ValueError
        Image reading is not correct.
    """
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Did not find image {path}.')
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise ValueError('Image reading is not correct.')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def resize_image(image: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
    """Resize image to given size.

    Parameters
    ----------
    image : np.ndarray
        Image to resize.
    new_size : Tuple[int, int]
        Tuple containing new image size.

    Returns
    -------
    np.ndarray
        Resized image
    """
    return cv2.resize(
        image, new_size, None, None, None, interpolation=cv2.INTER_LINEAR)


def save_image(img: np.ndarray, path: Union[Path, str]) -> None:
    """Сохранить переданное изображение по указанному пути.

    Parameters
    ----------
    img : np.ndarray
        Сохраняемое изображение.
    path : Union[Path, str]
        Путь для сохранения изображения.
    """
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError('Не удалось сохранить изображение.')
