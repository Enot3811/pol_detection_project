"""Функции для работы с yolov7."""


from pathlib import Path
from typing import Tuple, Dict, Union, List, Any

import numpy as np
from numpy.typing import NDArray
import cv2
import torch
from torch import Tensor

from Yolov7.yolov7.dataset import create_yolov7_transforms
from yolov7 import create_yolov7_model
from yolov7.trainer import filter_eval_predictions
from yolov7.models.yolo import Yolov7Model
from utils.image_utils.image_functions import read_image
from mako_camera.cameras_utils import split_raw_pol


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


def load_rgb_sample(sample_pth: Path, input_size: int = 640) -> Tensor:
    """Load one RGB sample and prepare it to input to a yolo.

    Parameters
    ----------
    sample_pth : Path
        A path to the sample. It must have .npy or one of image extensions.
    input_size : int, optional
        An image size for resizing and padding.

    Returns
    -------
    Tensor
        Loaded and preprocessed image.

    Raises
    ------
    ValueError
        Incorrect shape of image.
    """
    # Numpy array
    if str(sample_pth).split('.')[-1] == 'npy':
        image = np.load(sample_pth)
        image = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2BGR)
    # Image
    else:
        image = read_image(sample_pth)
    if len(image.shape) != 3:
        raise ValueError('Incorrect shape of image.')

    preprocess_transforms = create_yolov7_transforms(
        image_size=(input_size, input_size))
    image = preprocess_transforms(image=image, bboxes=[], labels=[])['image']
    image = torch.as_tensor(image.transpose(2, 0, 1), dtype=torch.float32)
    image /= 255
    image = image[None, ...]  # Add batch dim
    return image


def load_pol_sample(sample_pth: Path, input_size: int = 640) -> Tensor:
    """Load one POL sample and prepare it to input to a yolo.

    Parameters
    ----------
    sample_pth : Path
        A path to the sample. It must have .npy or one of image extensions.
    input_size : int, optional
        An image size for resizing and padding.

    Returns
    -------
    Tensor
        Loaded and preprocessed image.

    Raises
    ------
    ValueError
        Incorrect shape of image.
    """
    # Numpy array
    image = np.load(sample_pth)
    image = split_raw_pol(image)
    if len(image.shape) != 3:
        raise ValueError('Incorrect shape of image.')

    preprocess_transforms = create_yolov7_transforms(
        image_size=(input_size, input_size))
    image = preprocess_transforms(image=image, bboxes=[], labels=[])['image']
    image = torch.as_tensor(image.transpose(2, 0, 1), dtype=torch.float32)
    image /= 255
    image = image[None, ...]  # Add batch dim
    return image


def load_yolo_checkpoint(weights_pth: Path) -> Yolov7Model:
    """Create yolo model and load given weights.

    The loaded weights are checked to determine a number of channels
    and a corresponding model is created.

    Parameters
    ----------
    weights : Path
        A path to pt file with model weights.

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
        'yolov7', pretrained=False, num_channels=num_channels)
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
    

def create_yolo(num_channels: int = 3, pretrained: bool = True) -> Yolov7Model:
    """Create yolo model.

    Parameters
    ----------
    num_channels : int, optional
        A number of model's channels. By default is 3.
    pretrained : bool, optional
        Whether to load the official pretrained weights. By default is `True`.

    Returns
    -------
    Yolov7Model
        The created yolo model.

    Raises
    ------
    ValueError
        There are not pretrained weights for 4 channels model.
    """
    if num_channels == 4 and pretrained:
        raise ValueError('There are not pretrained weights '
                         'for 4 channels model.')
    model = create_yolov7_model(
        'yolov7', pretrained=pretrained, num_channels=num_channels)
    model = model.eval()
    return model


def inference(
    model: Yolov7Model,
    sample: Tensor,
    conf_thresh: float = 0.8,
    iou_thresh: float = 0.5
):
    """Произвести вывод модели.

    Parameters
    ----------
    model : Yolov7Model
        Загруженная модель.
    sample : Tensor
        Тензор размером 
    conf_thresh : float, optional
        _description_, by default 0.8
    iou_thresh : float, optional
        _description_, by default 0.5

    Returns
    -------
    _type_
        _description_
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


def draw_bboxes_cv2(
    image: NDArray, bboxes: List[Tuple], class_labels: List[Any] = None,
    confidences: List = None, bbox_format: str = 'xyxy'
) -> NDArray:
    """Нарисовать рамки предсказанным объектам с помощью cv2

    Parameters
    ----------
    image : NDArray
        Исходное изображение.
    bboxes : List[Tuple]
        Рамки в виде списка кортежей по 4 элемента для xyxy.
    class_labels : List, optional
        Метки для рамок, by default None
    confidences : List, optional
        Уверенность для рамок, by default None
    bbox_format : str, optional
        Формат рамок (доступен только xyxy), by default 'xyxy'

    Returns
    -------
    NDArray
        Изображение с нарисованными рамками.

    Raises
    ------
    NotImplementedError
        Доступно только для xyxy.
    """    
    image = image.copy()
    if bbox_format != 'xyxy':
        raise NotImplementedError('Доступно только для xyxy.')
    
    for i, bbox in enumerate(bboxes):
        bbox = list(map(int, bbox))
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=1)
        if class_labels is not None:
            put_text = f'cls: {class_labels[i]} '
        else:
            put_text = ''
        if confidences is not None:
            put_text += 'conf: {:.2f}'.format(confidences[i])
        if put_text != '':
            cv2.putText(image, put_text, (x1, y1 - 2), 0, 0.3, (255, 255, 255), 1)
    
    return image


coco_labels = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']

label2idx = {label: i for i, label in enumerate(coco_labels)}
idx2label = {i: label for i, label in enumerate(coco_labels)}
