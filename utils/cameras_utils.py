"""Модуль с функциями для обработки кадров с камер."""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
import cv2


def split_raw_pol(image: NDArray) -> NDArray:
    """Split a polarized image to 4 channels.
    
    The given image with shape `(h, w)` splits to 4 channels image
    with shape `(h/2, w/2, 4)`.

    Parameters
    ----------
    image : NDArray
        Raw polarized image.

    Returns
    -------
    NDArray
        The polarized image that is splitted to 4 channels.
    """
    ch_90 = image[::2, ::2]
    ch_45 = image[::2, 1::2]
    ch_135 = image[1::2, ::2]
    ch_0 = image[1::2, 1::2]
    if len(image.shape) == 2:
        ch_90 = ch_90[..., None]
        ch_45 = ch_45[..., None]
        ch_135 = ch_135[..., None]
        ch_0 = ch_0[..., None]
    return np.concatenate((ch_0, ch_45, ch_90, ch_135), 2)


def calc_Stocks_param(
   ch_0: NDArray, ch_45: NDArray, ch_90: NDArray, ch_135: NDArray
) -> Tuple[NDArray, NDArray, NDArray]:
    """Calculate stokes parameters.

    S1 in range 0.0 <= S0 <= 1.0 for polarized light and
    in range 0.0 <= S0 <= 2.0 fpr unpolarized.
    S1 and S2 in range -1.0 <= S1, S2 <= 1.0
    All parameters avoid zeros by replacing them with 1e-7.

    Parameters
    ----------
    ch_0 : float
        0 angle channel.
    ch_45 : float
        45 angle channel.
    ch_90 : float
        90 angle channel.
    ch_135 : float
        135 angle channel.

    Returns
    -------
    Tuple[float, float, float]
        Return S1, S2, S3 parameters.
    """
    s_0 = ch_0 + ch_90 + 1
    # s_0 = (ch_0 + ch_90 + ch_45 + ch_135) / 2
    s_1 = ch_0 - ch_90
    s_2 = ch_45 - ch_135
    s_0[s_0 == 0.0] = 1e-7
    s_1[s_1 == 0.0] = 1e-7
    s_2[s_2 == 0.0] = 1e-7
    return s_0, s_1, s_2


def calc_AoLP(s_1: NDArray, s_2: NDArray) -> NDArray:
    """Calculate angle of polarization.

    Parameters
    ----------
    s_1 : NDArray
        S1 stoke parameter. Expected float array in range (-1, 1).
    s_2 : NDArray
        S2 stoke parameter. Expected float array in range (-1, 1).

    Returns
    -------
    NDArray
        Pixelwice angle of polarization in radians. Range is (0, 2*pi)
    """
    AoLP = 0.5 * np.arctan2(s_2, s_1)
    # mask = AoLP < 0
    # AoLP[mask] += np.pi
    # mask = s_1 >= 0
    # AoLP[mask] += np.pi / 2

    # mask = AoLP < 0
    # AoLP[mask] += 2 * np.pi
    return AoLP

def pol_intensity(s1: NDArray, s2: NDArray) -> NDArray:
    """Calculate polarization intensity.

    Parameters
    ----------
    s1 : NDArray
        S1 stock parameter.
    s2 : NDArray
        S2 stock parameter.

    Returns
    -------
    NDArray
        Calculated polarization intensity.
    """
    return np.sqrt(np.square(s1) + np.square(s2) + 1)
    # return np.sqrt(np.square(s1) + np.square(s2))


def calc_DoLP(s_0: NDArray, pol_int: NDArray) -> NDArray:
    DoLP = pol_int / s_0
    return DoLP


def hsv_pol(aolp: NDArray, dolp: NDArray, pol_int: NDArray) -> NDArray:
    h = ((aolp + np.pi / 2) * (180 / np.pi)).astype(np.uint8)
    s = (dolp / np.amax(dolp) * 255).astype(np.uint8)
    v = (pol_int / np.amax(pol_int) * 255).astype(np.uint8)
    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    
    hue = (aolp / (np.pi * 2) * 255).astype(np.uint8)
    saturation = dolp.astype(np.uint8)
    value = np.ones_like(saturation, dtype=np.uint8) * 10

    hsv = np.concatenate(
        (value[..., None],
         saturation[..., None],
         hue[..., None]
         ), axis=2)
    hsv = cv2.applyColorMap(hsv, cv2.COLORMAP_HSV)
    return hsv

def dolp_to_img(dolp):
    min_val = dolp.min()
    max_val = dolp.max()
    return ((dolp - min_val) / (max_val - min_val) * 255).astype(np.uint8)


def aolp_to_img(aolp):
    min_val = aolp.min()
    max_val = aolp.max()
    return ((aolp - min_val) / (max_val - min_val) * 255).astype(np.uint8)
