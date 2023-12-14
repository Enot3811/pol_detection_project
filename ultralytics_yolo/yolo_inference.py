from pathlib import Path
import time

from ultralytics import YOLO
import cv2


if __name__ == '__main__':
    show_img = False
    show_time = True

    model = YOLO('runs/detect/train/weights/best.pt')
    img_pths = list(Path(
        'data/fire_smoke/train_fire_smoke_yolo/val/images/').glob('*jpg'))

    for img_pth in img_pths:
        if show_time:
            st_time = time.time()

        results = model(img_pth)

        bbox_img = results[0].plot()

        if show_img:
            cv2.imshow('Yolo inference (press any key)', bbox_img)
            key = cv2.waitKey(0)
            if key == 27:  # esc
                break

        if show_time:
            print('Затраченное время:', time.time() - st_time)
