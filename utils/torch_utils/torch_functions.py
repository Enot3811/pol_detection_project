"""A module containing some helpful functions working with Torch."""


from typing import List, Tuple, Union

from numpy.typing import NDArray
from torch import Tensor
import cv2


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


def draw_bboxes(
    image: NDArray,
    bboxes: List[Tuple],
    class_labels: List[Union[str, int, float]] = None,
    confidences: List[float] = None,
    bbox_format: str = 'xyxy'
) -> NDArray:
    """Draw bounding boxes on a given image.

    Parameters
    ----------
    image : NDArray
        The given image.
    bboxes : List[Tuple]
        The bounding boxes with shape `(n_boxes, 4)`.
    class_labels : List, optional
        Bounding boxes' labels. By default is None.
    confidences : List, optional
        Bounding boxes' confidences. By default is None.
    bbox_format : str, optional
        A bounding boxes' format (only xyxy is available).
        By default is 'xyxy'.

    Returns
    -------
    NDArray
        The image with drawn bounding boxes.

    Raises
    ------
    NotImplementedError
        Only xyxy bounding boxes format is available.
    """
    image = image.copy()
    if bbox_format != 'xyxy':
        # TODO
        raise NotImplementedError(
            'Only xyxy bounding boxes format is available.')
    
    for i, bbox in enumerate(bboxes):
        bbox = list(map(int, bbox))
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2),
                      color=(255, 255, 255), thickness=1)
        if class_labels is not None:
            put_text = f'cls: {class_labels[i]} '
        else:
            put_text = ''
        if confidences is not None:
            put_text += 'conf: {:.2f}'.format(confidences[i])
        if put_text != '':
            cv2.putText(image, put_text, (x1, y1 - 2), 0, 0.3,
                        (255, 255, 255), 1)
    return image
