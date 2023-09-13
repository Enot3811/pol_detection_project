import numpy as np


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
