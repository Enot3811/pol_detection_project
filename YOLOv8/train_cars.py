from pathlib import Path
from typing import Union

from ultralytics import YOLO


def main(model: Union[str, Path], dset_yaml: str):
    model = YOLO(model_name, task='detect')
    model.train(
        data=dset_yaml, device=0, project='YOLOv8/car_train', cos_lr=True,
        plots=True, mosaic=0.0, verbose=True, workers=1)


if __name__ == '__main__':
    model_name = Path('YOLOv8/model_ckpts/yolov8l.pt')
    dset_yaml = 'data/cars/traffic_yolo_small_dset/data.yaml'
    main(model_name, dset_yaml)
