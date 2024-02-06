"""Функции для работы с yolov7."""


from pathlib import Path
from typing import Tuple, Dict

import torch
from torch import Tensor, FloatTensor

from yolov7 import create_yolov7_model
from yolov7.trainer import filter_eval_predictions
from yolov7.models.yolo import Yolov7Model


def intersect_dicts(da: Dict, db: Dict, exclude: Tuple = ()) -> Dict:
    """Пересечение словарей.

    Возвращает элементы, которые есть и в первом и во втором.
    Используется, чтобы выбрать только нужные веса модели при загрузке.

    Parameters
    ----------
    da : Dict
        Первый словарь.
    db : Dict
        Второй словарь.
    exclude : Tuple, optional
        Игнорируемые ключи (идут в ответ в любом случае).
        По умолчанию ().

    Returns
    -------
    Dict
        Пересечение.
    """
    return {
        k: v
        for k, v in da.items()
        if (k in db and not any(x in k for x in exclude) and
            v.shape == db[k].shape)
    }


def load_yolo_checkpoint(weights_pth: Path, num_classes: int) -> Yolov7Model:
    """Create yolo model and load given weights.

    The loaded weights are checked to determine a number of channels
    and a corresponding model is created.

    Parameters
    ----------
    weights : Path
        A path to pt file with model weights.
    num_classes : int
        A number of classes for predictions.

    Returns
    -------
    Yolov7Model
        The loaded yolo checkpoint.
    """
    state_dict = torch.load(weights_pth)['model_state_dict']
    # Get first conv layer and check its depth
    num_channels = state_dict['model.0.conv.weight'].shape[1]

    # Create empty model
    model = create_yolov7_model(
        'yolov7', num_classes=num_classes, pretrained=False,
        num_channels=num_channels)
    # Load weights
    state_dict = intersect_dicts(
        state_dict,
        model.state_dict(),
        exclude=['anchor'],
    )
    model.load_state_dict(state_dict, strict=False)
    print(
        f'Transferred {len(state_dict)} / {len(model.state_dict())} '
        f'items from {weights_pth}')
    model = model.eval()
    return model


def create_yolo(
    num_classes: int = 80,
    num_channels: int = 3,
    pretrained: bool = True,
    model_arch: str = 'yolov7'
) -> Yolov7Model:
    """Create yolo model.

    Parameters
    ----------
    num_classes : int, optional
        A number of classification classes. By default is 80.
    num_channels : int, optional
        A number of model's channels. By default is 3.
    pretrained : bool, optional
        Whether to load the official pretrained weights. By default is `True`.
    model_arch : str, optional
        Model's architecture.

    Returns
    -------
    Yolov7Model
        The created yolo model.
    """
    model = create_yolov7_model(
        model_arch, num_classes=num_classes, pretrained=pretrained,
        num_channels=num_channels)
    model = model.eval()
    return model


def yolo_inference(
    model: Yolov7Model,
    sample: Tensor,
    conf_thresh: float = 0.6,
    iou_thresh: float = 0.2
) -> Tuple[FloatTensor, FloatTensor, FloatTensor]:
    """Произвести вывод модели.

    Parameters
    ----------
    model : Yolov7Model
        Загруженная модель.
    sample : Tensor
        Тензор размером `(b, c, h, w)`.
    conf_thresh : float, optional
        A model's confidence threshold, by default 0.6.
    iou_thresh : float, optional
        A model's IoU threshold, by default 0.2.

    Returns
    -------
    Tuple[FloatTensor, FloatTensor, FloatTensor]
        Bounding boxes with shape `(n_boxes, 4)`,
        predicted classes with shape `(n_boxes,)`
        and predicted confidences with shape `(n_boxes,)`.
    """
    model.eval()
    with torch.no_grad():
        model_outputs = model(sample)
        preds = model.postprocess(model_outputs, multiple_labels_per_box=False)

    nms_predictions = filter_eval_predictions(
        preds, confidence_threshold=conf_thresh, nms_threshold=iou_thresh)

    boxes = nms_predictions[0][:, :4]
    class_ids = nms_predictions[0][:, -1]
    confidences = nms_predictions[0][:, -2]

    return boxes, class_ids, confidences


coco_labels = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']

coco_label2idx = {label: i for i, label in enumerate(coco_labels)}
coco_idx2label = {i: label for i, label in enumerate(coco_labels)}
