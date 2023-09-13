from pathlib import Path
from typing import Union
import sys

import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt  # noqa

sys.path.append(str(Path(__file__).parents[2]))
from yolov7.dataset import create_yolov7_transforms
from yolov7 import create_yolov7_model
from yolov7.trainer import filter_eval_predictions
from yolov7.plotting import show_image
from yolov7.models.yolo import Yolov7Model
from custom.utils.image_utils import read_image


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes,
    # omitting 'exclude' keys, using da values
    return {
        k: v
        for k, v in da.items()
        if (k in db and not any(x in k for x in exclude) and
            v.shape == db[k].shape)
    }


def load_sample(sample_pth: Path, input_size: int = 640) -> Tensor:
    """Load one sample and prepare it to input to a yolo.

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
    ValueError
        Number of channels must be 3 or 4.
    """
    # Numpy array
    if str(sample_pth).split('.')[-1] == 'npy':
        image = np.load(sample_pth)
    # Image
    else:
        image = read_image(sample_pth)

    # TODO Убрать
    # image = image[1200:1700, 1700:2200, 0:3]
    # plt.imshow(image)
    # plt.show()
    # sys.exit()

    if len(image.shape) != 3:
        raise ValueError('Incorrect shape of image.'
                         f'It must be (h, w, c) but it is {image.shape}.')
    c = image.shape[-1]
    if c not in {3, 4}:
        raise ValueError(f'Number of channels must be 3 or 4, but it is {c}.')

    preprocess_transforms = create_yolov7_transforms(
        image_size=(input_size, input_size))
    image = preprocess_transforms(image=image, bboxes=[], labels=[])['image']
    image = torch.as_tensor(image.transpose(2, 0, 1), dtype=torch.float32)
    image /= 255
    image = image[None, ...]  # Add batch dim
    return image


def load_model(pretrained: Union[Path, bool] = False, num_channels: int = 3):
    # If there is a pt weights
    if isinstance(pretrained, Path):
        model = create_yolov7_model(
            'yolov7', pretrained=False, num_channels=num_channels)
        state_dict = intersect_dicts(
            torch.load(pretrained)['model_state_dict'],
            model.state_dict(),
            exclude=['anchor'],
        )
        model.load_state_dict(state_dict, strict=False)
        print(
            f'Transferred {len(state_dict)} / {len(model.state_dict())} '
            f'items from {pretrained}'
        )
    # If need to load pretrained or clean model
    elif isinstance(pretrained, bool):
        if pretrained and num_channels == 4:
            raise ValueError('There are not pretrained weights'
                             'for 4 channels model.')

        model = create_yolov7_model(
            'yolov7', pretrained=pretrained, num_channels=num_channels)
    else:
        raise

    model = model.eval()
    return model


def inference(
    model: Yolov7Model,
    sample: Tensor,
    conf_thresh: float = 0.8,
    iou_thresh: float = 0.5
):
    with torch.no_grad():
        model_outputs = model(sample)
        preds = model.postprocess(model_outputs, multiple_labels_per_box=False)

    nms_predictions = filter_eval_predictions(
        preds, confidence_threshold=conf_thresh, nms_threshold=iou_thresh)

    boxes = nms_predictions[0][:, :4]
    class_ids = nms_predictions[0][:, -1]
    confidences = nms_predictions[0][:, -2]

    return boxes, class_ids, confidences


def main():
    image = load_sample(SAMPLE_PTH)
    c = image.shape[1]
    # TODO Убрать потом
    c = 3
    # plt.imshow(image[0].permute(1, 2, 0).numpy())
    # plt.show()
    # return
    #
    model = load_model(WEIGHTS, c)
    boxes, class_ids, confidences = inference(
        model, image, CONF_THRESH, IOU_THRESH)

    show_image(image[0].permute(1, 2, 0), boxes.tolist()[:30],
               class_ids.tolist()[:30], confidences.tolist()[:30])


if __name__ == '__main__':
    # Path to image or npy
    SAMPLE_PTH: Path = Path(
        '/home/pc0/projects/yolov7_training/data/cameras/14_08_23/rgb/'
        'rgb_65.jpg')
    # Path to weights or bool for loading official pretrained weights
    WEIGHTS: Union[Path, bool] = True
    CONF_THRESH = 0.1
    IOU_THRESH = 0.2
    main()
