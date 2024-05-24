"""A module that contain functions for working with images."""


from pathlib import Path
from typing import Tuple, Union, Optional, List

import numpy as np
from numpy.typing import NDArray
import cv2
import matplotlib.pyplot as plt

from utils.data_utils.data_functions import prepare_path


IMAGE_EXTENSIONS: List[str] = ['jpg', 'jpeg', 'JPG', 'png', 'PNG']


def read_image(path: Union[Path, str], grayscale: bool = False) -> NDArray:
    """Read the image to a numpy array.

    Parameters
    ----------
    path : Union[Path, str]
        A path to the image file.
    grayscale : bool, optional
        Whether read image in grayscale (1-ch image), by default is `False`.

    Returns
    -------
    NDArray
        The array containing the read image.

    Raises
    ------
    ValueError
        Raise when cv2 could not read the image.
    """
    path = prepare_path(path)
    color_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), color_flag)
    if img is None:
        raise ValueError('cv2 could not read the image.')
    if grayscale:
        img = img[..., None]  # Add chanel dimension
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    return img


def resize_image(
    image: NDArray, new_size: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR
) -> NDArray:
    """Resize image to a given size.

    Parameters
    ----------
    image : NDArray
        The image to resize.
    new_size : Tuple[int, int]
        The requested size in `(h, w)` format.
    interpolation : int, optional
        cv2 interpolation flag. By default equals `cv2.INTER_LINEAR`.

    Returns
    -------
    NDArray
        The resized image

    Raises
    ------
    ValueError
        Raise when got incorrect size.
    """
    if len(new_size) != 2 or new_size[0] <= 0 or new_size[1] <= 0:
        raise ValueError(
            f'New size is required to be "(h, w)" but got {new_size}.')
    # Reverse to (w, h) for cv2
    new_size = new_size[::-1]
    resized = cv2.resize(image, new_size, None, None, None,
                         interpolation=interpolation)
    # If resize 1-channel image, channel dimension will be lost
    if len(resized.shape) != len(image.shape) and image.shape[-1] == 1:
        resized = resized[..., None]
    return resized


def show_image_plt(
    img: NDArray,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (16, 8),
    plt_show: bool = False
) -> plt.Axes:
    """Display an image on a matplotlib figure.

    Parameters
    ----------
    img : NDArray
        An image to display with shape `(h, w, c)` in RGB.
    ax : Optional[plt.Axes], optional
        Axes for image showing. If not given then a new Figure and Axes
        will be created.
    figsize : Tuple[int, int], optional
        Figsize for pyplot figure. By default is `(16, 8)`.
    plt_show : bool, optional
        Whether to make `plt.show()` in this function's calling.
        By default is `False`.

    Returns
    -------
    plt.Axes
        Axes with showed image.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    if plt_show:
        plt.show()
    return ax


def show_images_cv2(
    images: Union[NDArray, List[NDArray]],
    window_title: Union[str, List[str]] = 'image',
    destroy_windows: bool = True,
    delay: int = 0
) -> int:
    """Display one or a few images by cv2.

    Press any key to return from function. Key's code will be returned.
    If `destroy_windows` is `True` then windows will be closed.

    Parameters
    ----------
    image : NDArray
        Image array or list of image arrays.
    window_title : Union[str, List[str]], optional
        Image window's title. If List is provided it must have the same length
        as the list of images.
    destroy_windows : bool, optional
        Whether to close windows after function's end.
    delay : int, optional
        Time in ms to wait before window closing. If `0` is passed then window
        won't be closed before any key is pressed. By default is `0`.

    Returns
    -------
    int
        Pressed key code.
    """
    key_code = -1
    try:
        if isinstance(images, (List, tuple)):
            if isinstance(window_title, str):
                one_title = True
            elif (isinstance(window_title, list) and
                  len(window_title) == len(images)):
                one_title = False
            else:
                raise TypeError(
                    '"window_title" must be str or List[str] with the same '
                    'length as the list of images.')
            for i, image in enumerate(images):
                if one_title:
                    title = f'{window_title}_{i}'
                else:
                    title = window_title[i]
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imshow(title, image)
        elif isinstance(images, np.ndarray):
            images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
            cv2.imshow(window_title, images)
        else:
            raise TypeError('"images" must be NDArray or List of NDArrays, '
                            f'but got {type(images)}')
        key_code = cv2.waitKey(delay)
        if destroy_windows:
            cv2.destroyAllWindows
    except KeyboardInterrupt:
        # Free cv2 windows if interrupted
        cv2.destroyAllWindows()
    return key_code


def normalize_to_image(values: NDArray) -> NDArray:
    """Convert an array containing some float values to a uint8 image.

    Parameters
    ----------
    values : NDArray
        The array with float values in range [0.0, 1.0].

    Returns
    -------
    NDArray
        The uint8 image array.
    """
    min_val = values.min()
    max_val = values.max()
    return ((values - min_val) / (max_val - min_val) * 255).astype(np.uint8)


def save_image(img: NDArray, path: Union[Path, str]) -> None:
    """Save a given image to a defined path.

    Parameters
    ----------
    img : NDArray
        The saving image.
    path : Union[Path, str]
        The save path.

    Raises
    ------
    RuntimeError
        Could not save image.
    """
    path = Path(path) if isinstance(path, str) else path
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError('Could not save image.')


def get_sliding_windows(
    source_image: np.ndarray,
    h_win: int,
    w_win: int,
    stride: int = None
) -> np.ndarray:
    """Cut a given image into windows with defined shapes and stride.

    Parameters
    ----------
    source_image : np.ndarray
        The original image.
    h_win : int
        Height of the windows.
    w_win : int
        Width of the windows.
    stride : int, optional
        The stride of the sliding windows.
        If not defined it will be set by `w_win` value.

    Returns
    -------
    np.ndarray
        The cut image with shape `(n_h_win, n_w_win, h_win, w_win, c)`.
    """
    h, w = source_image.shape[:2]

    if stride is None:
        stride = w_win

    x_indexer = (
        np.expand_dims(np.arange(w_win), 0) +
        np.expand_dims(np.arange(w - w_win, step=stride), 0).T
    )
    y_indexer = (
        np.expand_dims(np.arange(h_win), 0) +
        np.expand_dims(np.arange(h - h_win, step=stride), 0).T
    )
    windows = source_image[y_indexer][:, :, x_indexer].swapaxes(1, 2)
    return windows
