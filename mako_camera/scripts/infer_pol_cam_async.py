"""POL camera inference script."""


import threading
import sys
import cv2
from typing import Optional
from pathlib import Path

from vimba import *
import numpy as np

sys.path.append(str(Path(__file__).parents[2]))
from utils.data_utils.data_functions import resize_image, normalize_to_image
import mako_camera.cameras_utils as cam_func


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
        # Enable auto exposure time setting if camera supports it
        try:
            cam.ExposureAuto.set('Continuous')

        except (AttributeError, VimbaFeatureError):
            pass

        # Try to adjust GeV packet size. This Feature is only available for GigE - Cameras.
        try:
            cam.GVSPAdjustPacketSize.run()

            while not cam.GVSPAdjustPacketSize.is_done():
                pass

        except (AttributeError, VimbaFeatureError):
            pass

        # Query available, open_cv compatible pixel formats
        # prefer color formats over monochrome formats
        cv_fmts = intersect_pixel_formats(cam.get_pixel_formats(), OPENCV_PIXEL_FORMATS)
        color_fmts = intersect_pixel_formats(cv_fmts, COLOR_PIXEL_FORMATS)

        if color_fmts:
            cam.set_pixel_format(color_fmts[0])

        else:
            mono_fmts = intersect_pixel_formats(cv_fmts, MONO_PIXEL_FORMATS)

            if mono_fmts:
                cam.set_pixel_format(mono_fmts[0])

            else:
                abort('Camera does not support a OpenCV compatible format natively. Abort.')


class Handler:
    def __init__(self):
        self.shutdown_event = threading.Event()

    def __call__(self, cam: Camera, frame: Frame):
        ESC_KEY_CODE = 27

        key = cv2.waitKey(1)
        if key == ESC_KEY_CODE:
            self.shutdown_event.set()
            return

        elif frame.get_status() == FrameStatus.Complete:
            print('{} acquired {}'.format(cam, frame), flush=True)
            print('Exposure time',
                  cam.get_feature_by_name('ExposureTimeAbs').get())

            img = frame.as_opencv_image()
            img = np.squeeze(img)
            split_channels = (
                cam_func.split_raw_pol(img).astype(np.float32) / 255)
            s0, s1, s2 = cam_func.calc_Stocks_param(
                split_channels[..., 0], split_channels[..., 1],
                split_channels[..., 2], split_channels[..., 3])
            pol_int = cam_func.pol_intensity(s1, s2)
            aolp = cam_func.calc_AoLP(s1, s2)
            dolp = cam_func.calc_DoLP(s0, pol_int)
            dolp_img = normalize_to_image(dolp)
            aolp_img = normalize_to_image(aolp)
            hsv = cam_func.hsv_pol(aolp, dolp, pol_int)

            cv2.imshow('Original image', resize_image(img, (500, 500)))
            cv2.imshow('Channel 0', resize_image(split_channels[..., 0], (500, 500)))
            cv2.imshow('Channel 45', resize_image(split_channels[..., 1], (500, 500)))
            cv2.imshow('Channel 90', resize_image(split_channels[..., 2], (500, 500)))
            cv2.imshow('Channel 135', resize_image(split_channels[..., 3], (500, 500)))
            cv2.imshow('AoLP', resize_image(aolp_img, (500, 500)))
            cv2.imshow('DoLP', resize_image(dolp_img, (500, 500)))
            cv2.imshow('HSV', resize_image(hsv, (500, 500)))
        cam.queue_frame(frame)


def main():
    cam_id = 'DEV_000F315D16C4'  # constant pol camera id

    with Vimba.get_instance():
        with get_camera(cam_id) as cam:

            # Start Streaming, wait for five seconds, stop streaming
            setup_camera(cam)
            handler = Handler()

            try:
                # Start Streaming with a custom a buffer of 10 Frames (defaults to 5)
                cam.start_streaming(handler=handler, buffer_count=10)
                handler.shutdown_event.wait()

            finally:
                cam.stop_streaming()


if __name__ == '__main__':
    main()
