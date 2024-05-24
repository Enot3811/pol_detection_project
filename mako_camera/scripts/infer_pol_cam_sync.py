"""Synchronous pol camera inference."""

import sys
import cv2
from typing import Optional
from pathlib import Path

from vimba import *
import numpy as np

sys.path.append(str(Path(__file__).parents[2]))
from utils.data_utils.data_functions import resize_image, normalize_to_image
from mako_camera.cameras_utils import (
    split_raw_pol, calc_AoLP, calc_DoLP, calc_Stocks_param, hsv_pol,
    pol_intensity)


def abort(reason: str, return_code: int = 1, usage: bool = False):
    print(reason + '\n')
    sys.exit(return_code)


def get_camera(camera_id: Optional[str]) -> Camera:
    with Vimba.get_instance() as vimba:
        if camera_id:
            try:
                return vimba.get_camera_by_id(camera_id)

            except VimbaCameraError:
                abort('Failed to access Camera \'{}\'. Abort.'.format(camera_id))

        else:
            cams = vimba.get_all_cameras()
            if not cams:
                abort('No Cameras accessible. Abort.')

            return cams[0]


def setup_camera(cam: Camera):
    with cam:
        # Try to adjust GeV packet size. This Feature is only available for GigE - Cameras.
        try:
            cam.ExposureAuto.set('Off')
        except (AttributeError, VimbaFeatureError):
            pass
        try:
            cam.GVSPAdjustPacketSize.run()

            while not cam.GVSPAdjustPacketSize.is_done():
                pass

        except (AttributeError, VimbaFeatureError):
            pass


def main():
    cam_id = 'DEV_000F315D16C4'  # constant pol camera id

    with Vimba.get_instance():
        with get_camera(cam_id) as cam:
            setup_camera(cam)

            # Acquire 10 frame with a custom timeout (default is 2000ms) per frame acquisition.
            for frame in cam.get_frame_generator(limit=10, timeout_ms=3000):
                frame: Frame
                print('Got {}'.format(frame), flush=True)
                print('Exposure time',
                    cam.get_feature_by_name('ExposureTimeAbs').get())

                img = frame.as_opencv_image()
                img = np.squeeze(img)
                split_channels = split_raw_pol(img).astype(np.float32) / 255
                s0, s1, s2 = calc_Stocks_param(
                    split_channels[..., 0], split_channels[..., 1],
                    split_channels[..., 2], split_channels[..., 3])
                pol_int = pol_intensity(s1, s2)
                aolp = calc_AoLP(s1, s2)
                dolp = calc_DoLP(s0, pol_int)
                dolp_img = normalize_to_image(dolp)
                aolp_img = normalize_to_image(aolp)
                hsv = hsv_pol(aolp, dolp, pol_int)

                cv2.imshow('Original image', resize_image(img, (500, 500)))
                cv2.imshow('Channel 0', resize_image(split_channels[..., 0], (500, 500)))
                cv2.imshow('Channel 45', resize_image(split_channels[..., 1], (500, 500)))
                cv2.imshow('Channel 90', resize_image(split_channels[..., 2], (500, 500)))
                cv2.imshow('Channel 135', resize_image(split_channels[..., 3], (500, 500)))
                cv2.imshow('AoLP', resize_image(aolp_img, (500, 500)))
                cv2.imshow('DoLP', resize_image(dolp_img, (500, 500)))
                cv2.imshow('HSV', resize_image(hsv, (500, 500)))
                cv2.waitKey(0)


if __name__ == '__main__':
    main()
