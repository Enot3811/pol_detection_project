"""A module containing some helpful functions working with Torch."""


from typing import List, Tuple, Union, Dict

import numpy as np
from numpy.typing import NDArray
import cv2
import torch
from torch import Tensor, tensor
from torch.nn import Module
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
    min_size: Union[int, Tuple[int, int]],
    max_size: Union[int, Tuple[int, int]],
    return_position: bool = False
) -> Union[Tensor, NDArray, Tuple[Tensor, IntBbox], Tuple[NDArray, IntBbox]]:
    """Make random crop from a given image.

    If `min_size` is `int` then the crop will be a square with shape belonging
    to the range from `(min_size, min_size)` to `(max_size, max_size)`.
    If `min_size` is `tuple` then the crop will be a rectangle with shape
    belonging to the range from `min_size` to `max_size`.

    Parameters
    ----------
    image : Union[Tensor, NDArray]
        The given image for crop with shape `(c, h w)` for `Tensor` type and
        `(h, w, c)` for `NDArray`. `Image` can be a batch of images with shape
        `(b, c, h, w)` or `(b, h, w, c)` depending on its type.
    min_size : Union[int, Tuple[int, int]]
        Minimum size of crop. It should be either min size of square as `int`
        or min size of rectangle as `tuple` in format `(h, w)`.
    max_size : Union[int, Tuple[int, int]]
        Maximum size of crop. It should be either max size of square as `int`
        or max size of rectangle as `tuple` in format `(h, w)`.
    return_position : bool, optional
        Whether to return bounding box of made crop. By default is `False`.

    Returns
    -------
    Union[Tensor, NDArray, Tuple[Tensor, IntBbox], Tuple[NDArray, IntBbox]]
        The crop region in the same type as the original image and it's
        bounding box if `return_position` is `True`.

    Raises
    ------
    ValueError
        "image" must be 3-d for one instance and 4-d for a batch.
    ValueError
        "image" must be either "torch.Tensor" or "numpy.ndarray".
    TypeError
        "min_size" must be int or Tuple[int, int].
    """
    if len(image.shape) not in {3, 4}:
        raise ValueError(
            '"image" must be 3-d for one instance and 4-d for a batch,'
            f'but its shape is {image.shape}.')
    if isinstance(image, Tensor):
        randint = torch.randint
        crop_dims = (-2, -1)
    elif isinstance(image, np.ndarray):
        randint = np.random.randint
        crop_dims = (-3, -2)
    else:
        raise ValueError(
            '"image" must be either "torch.Tensor" or "numpy.ndarray"'
            f'but it is {type(image)}.')
    h = image.shape[crop_dims[0]]
    w = image.shape[crop_dims[1]]
    # Get random size of crop
    if isinstance(min_size, int):
        x_size = y_size = randint(min_size, max_size + 1, ())
    elif isinstance(min_size, tuple) and len(min_size) == 2:
        y_size = randint(min_size[0], max_size[0] + 1, ())
        x_size = randint(min_size[1], max_size[1] + 1, ())
    else:
        raise TypeError('"min_size" must be int or Tuple[int, int] '
                        f'but it is {type(min_size)}.')
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
    

def make_compatible_state_dict(
    model: Module, state_dict: Dict[str, Tensor],
    return_discarded: bool = False
) -> Union[Dict[str, Tensor], Tuple[Dict[str, Tensor], Dict[str, Tensor]]]:
    """Discard model-incompatible weights from `state_dict`.

    Weights that are not represented in the model
    or that are not size-compatible to the model's parameters will be
    discarded from the `state_dict`.

    Parameters
    ----------
    model : Module
        The model for which to combine `state_dict`.
    state_dict : Dict[str, Tensor]
        The dict of parameters to make compatible.
    return_discarded : bool, optional
        Whether to return discarded parameters. By default is `False`.

    Returns
    -------
    Union[Dict[str, Tensor], Tuple[Dict[str, Tensor], Dict[str, Tensor]]]
        Compatible state dict and dict containing discarded parameters
        if `return_discarded` is `True`.
    """
    model_state_dict = model.state_dict()
    if return_discarded:
        discarded_parameters = {}
    keys = list(state_dict.keys())
    for k in keys:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                discarded_param = state_dict.pop(k)
                if return_discarded:
                    discarded_parameters[k] = discarded_param
        else:
            discarded_param = state_dict.pop(k)
            if return_discarded:
                discarded_parameters[k] = discarded_param

    if return_discarded:
        return state_dict, discarded_parameters
    else:
        return state_dict
