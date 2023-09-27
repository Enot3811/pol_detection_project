"""Начать съёмку."""


import time

import zmq
import pickle5 as pickle
import cv2
import numpy as np

import start_stz


def send_mess_cam_pol(data):
    """Отправка сообщения pol камере.
    
    Сообщение - кортеж, где:\n
    0-й элемент - флаг/команда для камеры\n
    1-й элемент - значение ExposureAuto\n
    2-й элемент - значение ExposureTimeAbs\n
    3-й элемент - значение Gain\n
    4-й элемент - значение ExposureAutoTarget\n
    """
    data = pickle.dumps(data)
    sock_cam_pol.send(data)


def recv_server_cam_pol() -> int:
    """Получение сообщения от поляризационной камеры.
    
    Сообщение - временная метка камеры в тиках.
    """
    recv_data = sock_cam_pol.recv()
    timestamp_pol = pickle.loads(recv_data)
    return timestamp_pol
    

def send_mess_cam_rgb(data):
    """Отправка сообщения rgb камере.
    
    Сообщение - кортеж, где:\n
    0-й элемент - флаг/команда для камеры\n
    1-й элемент - значение ExposureAuto\n
    2-й элемент - значение ExposureTimeAbs\n
    3-й элемент - значение Gain\n
    4-й элемент - значение ExposureAutoTarget\n
    """
    data = pickle.dumps(data)
    sock_cam_rgb.send(data)


def recv_server_cam_rgb():
    """Получение сообщения от rgb камеры.
    
    Сообщение - временная метка камеры в тиках.
    """
    recv_data = sock_cam_rgb.recv()
    timestamp_rgb = pickle.loads(recv_data)
    return timestamp_rgb


def get_user_input(user_input_ref):
    while True:
        input_str = input()
        if input_str == 's':
            user_input_ref[0] = input_str
        elif input_str == 'e':
            user_input_ref[0] = input_str
            break


def create_logo():
    cv_frame = np.zeros((50, 640, 1), np.uint8)
    cv_frame[:] = 0
    cv2.putText(cv_frame, 'ESC to exit, enter to save.', org=(30, 30),
                fontScale=1, color=255, thickness=1,
                fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)
    return cv_frame


def create_window():
    auto_exposure_val = 0
    exposure_val = 100000
    gain_val = 0
    exposure_target_val = 50
    def nothing(x): pass  # noqa
    cv2.namedWindow(win)
    cv2.createTrackbar(auto_exposure, win, 0, 2, nothing)
    cv2.createTrackbar(exposure, win, 0, 499993, nothing)
    cv2.setTrackbarMin(exposure, win, 32)
    cv2.createTrackbar(gain, win, 0, 40, nothing)
    cv2.createTrackbar(exposure_target, win, 0, 100, nothing)
    cv2.createTrackbar(video, win, 0, 1, nothing)
    cv2.setTrackbarPos(auto_exposure, win, auto_exposure_val)
    cv2.setTrackbarPos(exposure, win, exposure_val)
    cv2.setTrackbarPos(gain, win, gain_val)
    cv2.setTrackbarPos(exposure_target, win, exposure_target_val)
    cv2.setTrackbarPos(video, win, 0)


if __name__ == '__main__':
    # server cam pol
    context = zmq.Context()
    sock_cam_pol = context.socket(zmq.REQ)
    port = 'tcp://localhost:1234'
    sock_cam_pol.connect(port)
    print('Cam POL is ready')

    # server cam RGB
    context = zmq.Context()
    sock_cam_rgb = context.socket(zmq.REQ)
    port = 'tcp://localhost:12345'
    sock_cam_rgb.connect(port)
    print('Cam RGB is ready')

    start_stz.start_pol_cam()
    start_stz.start_pol_rgb()

    time.sleep(3)
    flag = 0

    # Работа окна
    win = 'main_window'
    auto_exposure = 'ExposureAuto'
    exposure = 'ExposureTimeAbs'
    gain = 'Gain'
    exposure_target = 'ExposureAutoTarget'
    auto_exposure_vals = {
        0: 'Off',
        1: 'Once',
        2: 'Continuous'
    }
    video = '0: photo / 1: video'
    logo = create_logo()
    create_window()
    while True:

        # Get a pressed key
        cv2.imshow('main_window', logo)

        k = cv2.waitKey(1) & 0xFF
        # Exit
        if k == 27:  # esc
            flag = 2
        # Save frames
        elif k == 13:  # enter
            flag = 1

        # Get parameters
        auto_exposure_val = auto_exposure_vals[
            cv2.getTrackbarPos(auto_exposure, win)]
        exposure_val = cv2.getTrackbarPos(exposure, win)
        gain_val = cv2.getTrackbarPos(gain, win)
        exposure_target_val = cv2.getTrackbarPos(exposure_target, win)

        data_to_pol = (flag, auto_exposure_val, exposure_val,
                       gain_val, exposure_target_val)
        send_mess_cam_pol(data_to_pol)
        timestamp_pol = recv_server_cam_pol()

        data_to_rgb = (flag, auto_exposure_val, exposure_val,
                       gain_val, exposure_target_val)
        send_mess_cam_rgb(data_to_rgb)
        timestamp_rgb = recv_server_cam_rgb()
        
        print(timestamp_pol - timestamp_rgb)

        if flag == 2:
            break
        else:
            if cv2.getTrackbarPos(video, win) == 0:
                flag = 0
            else:
                flag = 1
