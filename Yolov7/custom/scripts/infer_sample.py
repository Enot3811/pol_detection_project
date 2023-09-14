"""Запустить yolo на одном семпле."""


from pathlib import Path
from typing import Union
import sys

sys.path.append(str(Path(__file__).parents[2]))
from yolov7.plotting import show_image
from yolov7.models.yolo import Yolov7Model
from Yolov7.custom.model_utils import load_model, load_sample, inference



def main(sample_pth: Path, weights: Union[Path, bool], conf_thresh = 0.1, iou_thresh = 0.2):
    """Запустить yolo на одном семпле.

    Parameters
    ----------
    sample_pth : Path
        Path to image or npy
    weights : Union[Path, bool]
        Path to weights or bool for loading official pretrained weights
    conf_thresh : float, optional
        Confidence threshold, by default 0.1
    iou_thresh : float, optional
        IoU threshold, by default 0.2
    """    
    image = load_sample(sample_pth)
    c = image.shape[1]
    model: Yolov7Model = load_model(weights, c)
    boxes, class_ids, confidences = inference(
        model, image, conf_thresh, iou_thresh)

    show_image(image[0].permute(1, 2, 0), boxes.tolist()[:30],
               class_ids.tolist()[:30], confidences.tolist()[:30])


if __name__ == '__main__':
    # Path to image or npy
    sample_pth: Path = Path(
        '/home/pc0/projects/yolov7_training/data/cameras/14_08_23/rgb/'
        'rgb_65.jpg')
    # Path to weights or bool for loading official pretrained weights
    weights: Union[Path, bool] = True
    conf_thresh = 0.1
    iou_thresh = 0.2
    main()
