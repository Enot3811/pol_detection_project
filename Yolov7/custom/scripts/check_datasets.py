"""
Check polarized, mosaic and yolo datasets with 4 channels.
"""


from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from torchvision.ops import box_convert

import sys
sys.path.append(str(Path(__file__).parents[2]))

from custom.polarized_dataset import PolarizedDatasetAdaptor, load_df
from yolov7.mosaic import MosaicMixupDataset, create_post_mosaic_transform
from yolov7.dataset import (
    Yolov7Dataset,
    create_base_transforms,
    create_yolov7_transforms)


def main():
    image_size = 640
    data_path = Path(__file__).absolute().parents[3] / 'data/polarized_dataset'
    idxs = range(40, 50)

    images_path = data_path / "Images"
    annotations_file_path = data_path / "annotations.csv"
    train_df, valid_df, lookups = load_df(annotations_file_path, images_path)

    train_ds = PolarizedDatasetAdaptor(
        images_path, train_df, transforms=create_base_transforms(image_size)
    )

    # for i in idxs:
    #     img, bbox, _, _, _ = train_ds[i]
    #     bbox = tuple(bbox.astype(np.int16).reshape(4))
    #     cv.rectangle(img, bbox[:2], bbox[2:], (128, 0, 0), 2)

    #     cv.imshow('Polarized dset', img)
    #     cv.waitKey(1000)
    #     cv.destroyAllWindows()

    mds = MosaicMixupDataset(
        train_ds,
        apply_mixup_probability=0.15,
        post_mosaic_transforms=create_post_mosaic_transform(
            output_height=image_size, output_width=image_size
        ),
        pad_colour=(114,) * 4
    )

    # for i in idxs:
    #     img, bboxes, _, _, _ = mds[i]
    #     bboxes = bboxes.astype(np.int16)

    #     for bbox in bboxes:
    #         cv.rectangle(img, bbox[:2], bbox[2:], (128, 0, 0), 2)

    #     cv.imshow('Mosaic dset', img)
    #     cv.waitKey(1000)
    #     cv.destroyAllWindows()

    train_yds = Yolov7Dataset(
        mds,
        create_yolov7_transforms(training=True, image_size=(image_size, image_size)),
    )

    for i in idxs:
        img, labels, _, _ = train_yds[i]
        img = img.numpy().transpose(1, 2, 0)

        bboxes = box_convert(
            torch.as_tensor(labels[:, 2:], dtype=torch.float32), "cxcywh", "xyxy"
        ).numpy()
        h, w = img.shape[:2]
        bboxes[:, ::2] *= w
        bboxes[:, 1::2] *= h
        bboxes = bboxes.astype(np.int16)

        for bbox in bboxes:
            cv.rectangle(img, bbox[:2], bbox[2:], (128, 0, 0), 2)

        cv.imshow('Yolo dset', img)
        cv.waitKey(1000)
        cv.destroyAllWindows()

    # return (
    #         torch.as_tensor(image.transpose(2, 0, 1), dtype=torch.float32),
    #         labels_out,
    #         image_id_tensor,
    #         torch.as_tensor(original_image_size),
    #     )


if __name__ == '__main__':
    main()