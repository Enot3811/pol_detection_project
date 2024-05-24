"""Скрипт запуска асинхронного стрима с указываемой mako камеры."""


import sys
from pathlib import Path
from typing import Tuple

from vimba import Vimba, Camera, Frame, FrameStatus, PixelFormat
import cv2

sys.path.append(str(Path(__file__).parents[3]))
from mako_camera.vimba_custom.vimba_tools import get_camera, BaseStreamHandler
from utils.data_utils.data_functions import resize_image


class StreamHandler(BaseStreamHandler):
    """Обработчик для стрима с изменением размера для изображения."""

    def __init__(self, resize_shape: Tuple[int, int]):
        self.resize_shape = resize_shape
        super().__init__()

    def __call__(self, cam: Camera, frame: Frame):
        esc_key_code = 27

        key = cv2.waitKey(1)
        if key == esc_key_code:
            self.shutdown_event.set()
            return

        elif frame.get_status() == FrameStatus.Complete:
            # Convert bayer to BGR
            image = frame.as_numpy_ndarray()
            image = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2BGR)
            image = resize_image(image, self.resize_shape)

            msg = 'Stream from \'{}\'. Press <Esc> to stop stream.'
            cv2.imshow(msg.format(cam.get_id()), image)

        cam.queue_frame(frame)


def main():
    with Vimba.get_instance():
        with get_camera('DEV_000F315D630A') as camera:
            if camera.get_pixel_format() != PixelFormat.BayerRG8:
                raise ValueError(
                    'Camera should be in BayerRG8 pixel format '
                    f'but now it is in {camera.get_pixel_format()}.')
            stream_handler = StreamHandler((700, 700))
            try:
                camera.start_streaming(handler=stream_handler)
                stream_handler.shutdown_event.wait()

            finally:
                camera.stop_streaming()


if __name__ == '__main__':
    main()
