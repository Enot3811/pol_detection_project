"""A module containing some helpful functions working with Torch."""


from typing import List, Tuple, Union

import numpy as np
from numpy.typing import NDArray
import cv2
import torch
from torch import Tensor, tensor
from torchvision.ops import box_convert


IntBbox = Tuple[int, int, int, int]
FloatBbox = Tuple[float, float, float, float]
Bbox = Union[IntBbox, FloatBbox]


def image_tensor_to_numpy(tensor: Tensor) -> NDArray:
    """Convert an image or a batch of images from tensor to ndarray.

    Parameters
    ----------
    tensor : Tensor
        The tensor with shape `(h, w)`, `(c, h, w)` or `(b, c, h, w)`.

    Returns
    -------
    NDArray
        The array with shape `(h, w)`, `(h, w, c)` or `(b, h, w, c)`.
    """
    if len(tensor.shape) == 3:
        return tensor.detach().cpu().permute(1, 2, 0).numpy()
    elif len(tensor.shape) == 4:
        return tensor.detach().cpu().permute(0, 2, 3, 1).numpy()
    elif len(tensor.shape) == 2:
        return tensor.detach().cpu().numpy()
    

def image_numpy_to_tensor(array: NDArray) -> Tensor:
    """Convert an image or a batch of images from ndarray to tensor.

    Parameters
    ----------
    array : NDArray
        The array with shape `(h, w)`, `(h, w, c)` or `(b, h, w, c)`.

    Returns
    -------
    Tensor
        The tensor with shape `(h, w)`, `(c, h, w)` or `(b, c, h, w)`.
    """
    if len(array.shape) == 3:
        return torch.tensor(array.transpose(2, 0, 1))
    elif len(array.shape) == 4:
        return torch.tensor(array.transpose(0, 3, 1, 2))
    elif len(array.shape) == 2:
        return torch.tensor(array)


def draw_bounding_boxes(
    image: NDArray,
    bboxes: List[Bbox],
    class_labels: List[Union[str, int, float]] = None,
    confidences: List[float] = None,
    bbox_format: str = 'xyxy',
    line_width: int = 1,
    color: Tuple[int, int, int] = (255, 255, 255),
    exclude_classes: List[Union[str, int, float]] = None
) -> NDArray:
    """Draw bounding boxes and corresponding labels on a given image.

    Parameters
    ----------
    image : NDArray
        The given image with shape `(h, w, c)`.
    bboxes : List[Bbox]
        The bounding boxes with shape `(n_boxes, 4)`.
    class_labels : List, optional
        Bounding boxes' labels. By default is None.
    exclude_classes : List[str, int, float]
        Classes which bounding boxes won't be showed. By default is None.
    confidences : List, optional
        Bounding boxes' confidences. By default is None.
    bbox_format : str, optional
        A bounding boxes' format. It should be one of "xyxy", "xywh" or
        "cxcywh". By default is 'xyxy'.
    line_width : int, optional
        A width of the bounding boxes' lines. By default is 1.
    color : Tuple[int, int, int], optional
        A color of the bounding boxes' lines in RGB.
        By default is `(255, 255, 255)`.

    Returns
    -------
    NDArray
        The image with drawn bounding boxes.

    Raises
    ------
    NotImplementedError
        Implemented only for "xyxy", "xywh" and "cxcywh"
        bounding boxes formats.
    """
    image = image.copy()
    if exclude_classes is None:
        exclude_classes = []

    # Convert to "xyxy"
    if bbox_format != 'xyxy':
        if bbox_format in ('xywh', 'cxcywh'):
            bboxes = box_convert(tensor(bboxes), bbox_format, 'xyxy').tolist()
        else:
            raise NotImplementedError(
                'Implemented only for "xyxy", "xywh" and "cxcywh"'
                'bounding boxes formats.')
    
    for i, bbox in enumerate(bboxes):
        # Check if exclude
        if class_labels is not None and class_labels[i] in exclude_classes:
            continue

        # Draw bbox
        bbox = list(map(int, bbox))  # convert float bbox to int if needed
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2),
                      color=color, thickness=line_width)
        
        # Put text if needed
        if class_labels is not None:
            put_text = f'cls: {class_labels[i]} '
        else:
            put_text = ''
        if confidences is not None:
            put_text += 'conf: {:.2f}'.format(confidences[i])
        if put_text != '':
            cv2.putText(image, put_text, (x1, y1 - 2), 0, 0.3,
                        color, 1)
    return image


def random_crop(
    image: Union[Tensor, NDArray],
    min_crop_x: int, max_crop_x: int,
    min_crop_y: int, max_crop_y: int,
    return_position: bool = False
) -> Union[Tensor, NDArray, Tuple[Tensor, IntBbox], Tuple[NDArray, IntBbox]]:
    """Make random crop from a given image.

    The size of the crop region size is in ranges `(min_crop_x, max_crop_x)`
    and `(min_crop_y, max_crop_y)`.

    Parameters
    ----------
    image : Union[Tensor, NDArray]
        The given image for crop with shape `(c, h w)` for `Tensor` type and
        `(h, w, c)` for `NDArray`. `Image` can be a batch of images with shape
        `(b, c, h, w)` or `(b, h, w, c)` depending on its type.
    min_crop_x : int
        Minimum width bound for crop.
    max_crop_x : int
        Maximum width bound for crop.
    min_crop_y : int
        Minimum height bound for crop.
    max_crop_y : int
        Maximum height bound for crop.
    return_position : bool, optional
        Whether to return bounding box of made crop. By default is `False`.

    Returns
    -------
    Union[Tensor, NDArray, Tuple[Tensor, Bbox], Tuple[NDArray, Bbox]]
        The crop region in the same type as the original image and it's
        bounding box if `return_position` is `True`.
    """
    if len(image.shape) not in {3, 4}:
        raise ValueError(
            'Image must be 3-d for one instance and 4-d for a batch,'
            f'but its shape is {image.shape}.')
    if isinstance(image, Tensor):
        randint = torch.randint
        crop_dims = (-2, -1)
    elif isinstance(image, np.ndarray):
        randint = np.random.randint
        crop_dims = (-3, -2)
    else:
        raise ValueError(
            'Image must be either "torch.Tensor" or "numpy.ndarray"'
            f'but it is {type(image)}.')
    h = image.shape[crop_dims[0]]
    w = image.shape[crop_dims[1]]
    # Get random size of crop
    x_size = randint(min_crop_x, max_crop_x + 1, ())
    y_size = randint(min_crop_y, max_crop_y + 1, ())
    # Get random window
    x_min = randint(0, w - x_size + 1, ())
    y_min = randint(0, h - y_size + 1, ())
    x_max = x_min + x_size
    y_max = y_min + y_size
    # Crop the window
    crop_indexes = [slice(None)] * len(image.shape)
    crop_indexes[crop_dims[0]] = slice(y_min, y_max)
    crop_indexes[crop_dims[1]] = slice(x_min, x_max)
    cropped = image[tuple(crop_indexes)]
    if return_position:
        return cropped, (x_min, y_min, x_max, y_max)
    else:
        return cropped
