"""Скрипт запуска синхронного стрима с указываемой mako камеры."""


import sys
from pathlib import Path

from vimba import Vimba, PixelFormat
import cv2

sys.path.append(str(Path(__file__).parents[3]))
from mako_camera.vimba_custom.vimba_tools import get_camera
from utils.data_utils.data_functions import resize_image


def main():
    esc_key_code = 27
    resize_shape = (700, 700)

    with Vimba.get_instance():
        with get_camera('DEV_000F315D630A') as camera:
            if camera.get_pixel_format() != PixelFormat.BayerRG8:
                raise ValueError(
                    'Camera should be in BayerRG8 pixel format '
                    f'but now it is in {camera.get_pixel_format()}.')
            
            for frame in camera.get_frame_generator(timeout_ms=3000):

                # Convert bayer to BGR
                image = frame.as_numpy_ndarray()
                image = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2BGR)
                image = resize_image(image, resize_shape)

                msg = 'Stream from \'{}\'. Press <Esc> to stop stream.'
                cv2.imshow(msg.format(camera.get_id()), image)
                key = cv2.waitKey(1)
                if key == esc_key_code:
                    break


if __name__ == '__main__':
    main()
