"""A module that contain some useful functions for work with data."""


from typing import Union, List, Iterable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from utils.image_utils.image_functions import read_image, IMAGE_EXTENSIONS


def prepare_path(path: Union[Path, str]) -> Path:
    """Check an existence of the given path and convert it to `Path`.

    Parameters
    ----------
    path : Union[Path, str]
        The given file path.

    Raises
    ------
    FileNotFoundError
        Raise when file was not found.
    """
    path = Path(path) if isinstance(path, str) else path
    if not path.exists():
        raise FileNotFoundError(f'Did not find the file "{str(path)}".')
    return path


def collect_paths(
    image_dir: Union[str, Path],
    file_extensions: Iterable[str]
) -> List[Path]:
    """Collect all paths with given extension from given directory.

    Parameters
    ----------
    image_dir : Union[str, Path]
        Directory from which image paths will be collected.
    file_extensions : Iterable[str]
        Extension of collecting files.

    Returns
    -------
    List[Path]
        Collected image paths.
    """
    paths: List[Path] = []
    for ext in file_extensions:
        paths.extend(image_dir.glob(f'*.{ext}'))
    return paths


def read_volume(path: Union[Path, str]) -> NDArray:
    """Read a volume from a npy or image file.

    Parameters
    ----------
    path : Union[Path, str]
        A path to the volume file.

    Returns
    -------
    NDArray
        The read volume.

    Raises
    ------
    ValueError
        Raise when given path does not have proper extension.
    """
    path = prepare_path(path)
    if path.suffix == '.npy':
        vol = np.load(path)
    elif path.suffix[1:] in IMAGE_EXTENSIONS:
        vol = read_image(path)
    else:
        raise ValueError(
            f'The file extension of the path "{str(path)}" is not proper.')
    return vol
