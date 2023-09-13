"""A module that contain functions for working with images."""


from pathlib import Path
from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray
import cv2


def read_image(path: Union[Path, str], grayscale: bool = False) -> NDArray:
    """Read an image to numpy array in rgb.

    Parameters
    ----------
    path : Union[Path, str]
        A path to the image file.
    grayscale : bool, optional
        Whether read the image in grayscale, by default False.

    Returns
    -------
    NDArray
        Array containing the read image.

    Raises
    ------
    FileNotFoundError
        Did not find the image.
    ValueError
        Could not read the image.
    """
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Did not find the image {path}.')
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise ValueError('Could not read the image.')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def resize_image(
    image: NDArray,
    new_size: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR
) -> NDArray:
    """Resize image to given size.

    Parameters
    ----------
    image : NDArray
        Image to resize with shape `(h, w, c)`.
    new_size : Tuple[int, int]
        Tuple containing new image size in format `(new_h, new_w)`.
    interpolation : int, optional
        Cv2 interpolation constant. By default is cv2.INTER_LINEAR.

    Returns
    -------
    NDArray
        Resized image with shape `(new_h, new_w, c)`.
    """
    return cv2.resize(image, new_size[::-1], interpolation=interpolation)


def save_image(img: NDArray, path: Union[Path, str]) -> None:
    """Сохранить переданное изображение по указанному пути.

    Parameters
    ----------
    img : NDArray
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


def normalize_to_image(values: NDArray) -> NDArray:
    """Convert array containing some float values to uint8 image.

    Parameters
    ----------
    values : NDArray
        Array with unnormalized values.

    Returns
    -------
    NDArray
        Normalized uint8 array.
    """
    min_val = values.min()
    max_val = values.max()
    return ((values - min_val) / (max_val - min_val) * 255).astype(np.uint8)
