from pathlib import Path
import random

import torch

import sys
sys.path.append(str(Path(__file__).parents[2]))

from custom.polarized_dataset import PolarizedDatasetAdaptor, load_df
from yolov7.mosaic import MosaicMixupDataset
from yolov7.dataset import (
    Yolov7Dataset, create_base_transforms, create_yolov7_transforms)
from yolov7 import create_yolov7_model
from yolov7.trainer import filter_eval_predictions
from yolov7.plotting import show_image


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes,
    # omitting 'exclude' keys, using da values
    return {
        k: v
        for k, v in da.items()
        if (k in db and not any(x in k for x in exclude) and
            v.shape == db[k].shape)
    }


def get_dataset():
    image_size = 640
    data_path = Path(__file__).absolute().parents[3] / 'data/polarized_dataset'

    images_path = data_path / 'Images'
    annotations_file_path = data_path / 'annotations.csv'
    df, _, _ = load_df(annotations_file_path, images_path)

    dset = PolarizedDatasetAdaptor(
        images_path, df, transforms=create_base_transforms(image_size)
    )

    dset = MosaicMixupDataset(
        dset,
        # post_mosaic_transforms=create_post_mosaic_transform(
        #     output_height=image_size, output_width=image_size
        # ),
        pad_colour=(114,) * CHANNELS
    )

    dset.disable()

    yds = Yolov7Dataset(
        dset,
        create_yolov7_transforms(
            training=False, image_size=(image_size, image_size)),
    )

    return yds


def load_model(pretrained: bool = False, num_channels: int = 3):
    state_dict_path = 'best_model.pt'
    model = create_yolov7_model(
        'yolov7', num_classes=1, pretrained=pretrained,
        num_channels=num_channels)

    state_dict = intersect_dicts(
        torch.load(state_dict_path)['model_state_dict'],
        model.state_dict(),
        exclude=["anchor"],
    )
    model.load_state_dict(state_dict, strict=False)
    print(
        f"Transferred {len(state_dict)}/{len(model.state_dict())} "
        f"items from {state_dict_path}"
    )
    model = model.eval()
    return model


def inference(model, dataset, idx=0):
    image_tensor, labels, image_id, image_size = dataset[idx]
    with torch.no_grad():
        model_outputs = model(image_tensor[None])
        preds = model.postprocess(
            model_outputs, conf_thres=0., multiple_labels_per_box=False)

    nms_predictions = filter_eval_predictions(
        preds, confidence_threshold=0.1, nms_threshold=0.4)

    boxes = nms_predictions[0][:, :4]
    class_ids = nms_predictions[0][:, -1]
    confidences = nms_predictions[0][:, -2]

    show_image(image_tensor.permute(1, 2, 0), boxes.tolist()[:30],
               class_ids.tolist()[:30], confidences.tolist()[:30])
    print(f'Image id: {image_id}')
    print(f'Original Image size: {image_size}')
    print(f'Resized Image size: {image_tensor.shape}')


def main():
    dataset = get_dataset()
    model = load_model()
    for i in range(10):
        inference(model, dataset, random.randint(0, len(dataset)))


if __name__ == '__main__':
    CHANNELS = 4
    main()
