from pathlib import Path
import random

import torch

import sys
sys.path.append(str(Path(__file__).parents[1]))

from post_examples.train_cars import load_cars_df, CarsDatasetAdaptor
from yolov7.mosaic import MosaicMixupDataset, create_post_mosaic_transform
from yolov7.dataset import (
    Yolov7Dataset,
    create_base_transforms,
    create_yolov7_transforms)
from yolov7 import create_yolov7_model
from yolov7.trainer import filter_eval_predictions
from yolov7.plotting import show_image
from yolov7.models.yolo import Yolov7Model


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
    data_path = Path(__file__).absolute().parents[2] / 'data/cars'

    images_path = data_path / 'testing_images'
    annotations_file_path = data_path / 'annotations.csv'
    df, _, _ = load_cars_df(annotations_file_path, images_path)

    cars_dset = CarsDatasetAdaptor(
        images_path, df, transforms=create_base_transforms(image_size)
    )

    cars_dset = MosaicMixupDataset(
        cars_dset,
        # post_mosaic_transforms=create_post_mosaic_transform(
        #     output_height=image_size, output_width=image_size
        # ),
        pad_colour=(114,) * CHANNELS
    )

    cars_dset.disable()

    yds = Yolov7Dataset(
        cars_dset,
        create_yolov7_transforms(
            training=False, image_size=(image_size, image_size)),
    )

    return yds


def load_model():
    model = create_yolov7_model(
        'yolov7', num_classes=1, pretrained=False, num_channels=CHANNELS)

    state_dict = intersect_dicts(
        torch.load(MODEL)['model_state_dict'],
        model.state_dict(),
        exclude=["anchor"],
    )
    model.load_state_dict(state_dict, strict=False)
    print(
        f"Transferred {len(state_dict)}/{len(model.state_dict())} "
        f"items from {MODEL}"
    )
    model = model.eval()
    return model


def inference(model: Yolov7Model, dataset, idx=0):
    image_tensor, labels, image_id, image_size = dataset[idx]
    with torch.no_grad():
        model_outputs = model(image_tensor[None])
        preds = model.postprocess(
            model_outputs, conf_thres=0., multiple_labels_per_box=False)

    nms_predictions = filter_eval_predictions(
        preds, confidence_threshold=0.5, nms_threshold=0.7)

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
    CHANNELS = 3
    MODEL = 'cars_3ch_best.pt'
    main()
