from pathlib import Path
from typing import Union

from ultralytics import YOLO


def main(model_yaml: Union[str, Path], dset_yaml: Union[str, Path]):
    model = YOLO(model_yaml, task='detect').load('yolov8n.pt')
    model.train(
        data=dset_yaml, device=0, project='YOLOv8/thermal_train', cos_lr=True,
        plots=True, mosaic=0.7, verbose=True, workers=1, single_cls=True)


if __name__ == '__main__':
    model_yaml = Path('YOLOv8/model_cfgs/yolov8n.yaml')
    dset_yaml = Path('data/thermal_imager/plane/data.yaml')
    main(model_yaml, dset_yaml)
