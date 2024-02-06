"""Модуль с моими функциями для работы с vimba."""


import sys
from typing import Optional
import threading

from vimba import Vimba, Camera, VimbaCameraError, Frame, FrameStatus
import cv2


def get_camera(camera_id: Optional[str] = None) -> Camera:
    """Get vimba camera by id.

    Parameters
    ----------
    camera_id : Optional[str], optional
        Camera id. If not provided first camera from all will be gotten.

    Returns
    -------
    Camera
        vimba.Camera object.
    """
    with Vimba.get_instance() as vimba:
        if camera_id:
            try:
                return vimba.get_camera_by_id(camera_id)
            except VimbaCameraError:
                print(f'Failed to access Camera "{camera_id}".')
                sys.exit()

        else:
            cams = vimba.get_all_cameras()
            if not cams:
                print('No Cameras accessible.')
                sys.exit()
            return cams[0]
        

class BaseStreamHandler:
    """Класс функтор для обработки фреймов из стрима в асинхронном режиме.
    
    Передаётся в функцию camera.start_streaming.
    Метод __call__ переопределяется для любых нужд.
    """

    def __init__(self):
        self.shutdown_event = threading.Event()

    def __call__(self, cam: Camera, frame: Frame):
        esc_key_code = 27

        key = cv2.waitKey(1)
        if key == esc_key_code:
            self.shutdown_event.set()
            return

        elif frame.get_status() == FrameStatus.Complete:
            print('{} acquired {}'.format(cam, frame), flush=True)
            msg = 'Stream from \'{}\'. Press <Esc> to stop stream.'
            cv2.imshow(msg.format(cam.get_name()), frame.as_opencv_image())

        cam.queue_frame(frame)
