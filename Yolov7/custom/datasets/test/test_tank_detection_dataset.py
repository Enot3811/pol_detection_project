"""Скрипт для обучения yolov7."""


import sys
from pathlib import Path

from torchvision.ops import box_convert
import albumentations as A

sys.path.append(str(Path(__file__).parents[4]))
from Yolov7.yolov7.dataset import Yolov7Dataset, create_yolov7_transforms
from Yolov7.yolov7.mosaic import (
    MosaicMixupDataset, create_post_mosaic_transform)
from Yolov7.custom.datasets import TankDetectionDataset
from utils.image_utils.image_functions import show_images_cv2
from utils.torch_utils.torch_functions import (
    image_tensor_to_numpy, draw_bounding_boxes)


def main(
    dset_dir: Path, input_size: int, polarization: bool, random_crop: bool,
    horizontal_flip: bool, mosaic_prob: float, mixup_prob: float
):
    # Polarization difference
    if polarization:
        num_channels = 4
    else:
        num_channels = 3
    pad_colour = (114,) * num_channels

    # Get transforms
    post_mosaic_transforms = create_post_mosaic_transform(
        input_size, input_size, pad_colour=pad_colour)
    
    training_transforms = []
    if random_crop:
        training_transforms.append(
            A.RandomCropFromBorders(crop_left=0.4, crop_right=0.4,
                                    crop_top=0.4, crop_bottom=0.4))
    if horizontal_flip:
        training_transforms.append(A.HorizontalFlip())

    yolo_transforms = create_yolov7_transforms(
        (input_size, input_size), training=True,
        pad_colour=pad_colour, training_transforms=training_transforms)
    
    resize_func = A.Compose(
        [A.Resize(910, 1620)],
        bbox_params=A.BboxParams(format='pascal_voc',
                                 label_fields=['classes']))

    # Get datasets and loaders
    dset = TankDetectionDataset(
        dset_dir, polarization=polarization)

    mosaic_mixup_dset = MosaicMixupDataset(
        dset,
        apply_mosaic_probability=mosaic_prob,
        apply_mixup_probability=mixup_prob,
        pad_colour=pad_colour,
        post_mosaic_transforms=post_mosaic_transforms)
    
    yolo_dset = Yolov7Dataset(mosaic_mixup_dset, yolo_transforms)

    # Check dataset
    for orig_sample, mosaic_sample, yolo_sample in zip(dset,
                                                       mosaic_mixup_dset,
                                                       yolo_dset):
        orig_img, orig_bboxes, orig_cls, orig_id, orig_shape = orig_sample
        mosaic_img, mosaic_bboxes, mosaic_cls, mosaic_id, mosaic_shape = (
            mosaic_sample)
        yolo_img, yolo_labels, yolo_id, yolo_shape = yolo_sample

        np_yolo_img = image_tensor_to_numpy(yolo_img)
        titles = [
            'orig_img ' + str(orig_shape),
            'mosaic_img ' + str(mosaic_shape),
            'yolo_img ' + str(yolo_shape)
        ]

        res_orig = resize_func(
            image=orig_img, bboxes=orig_bboxes, classes=orig_cls)
        res_mosaic = resize_func(
            image=mosaic_img, bboxes=mosaic_bboxes, classes=mosaic_cls)
        res_orig_img = res_orig['image']
        res_mosaic_img = res_mosaic['image']
        res_orig_bboxes = res_orig['bboxes']
        res_mosaic_bboxes = res_mosaic['bboxes']

        yolo_bboxes = yolo_labels[:, 2:]
        yolo_bboxes[:, 0] *= np_yolo_img.shape[1]
        yolo_bboxes[:, 2] *= np_yolo_img.shape[1]
        yolo_bboxes[:, 1] *= np_yolo_img.shape[0]
        yolo_bboxes[:, 3] *= np_yolo_img.shape[0]
        yolo_bboxes = box_convert(yolo_bboxes, 'cxcywh', 'xyxy')
        yolo_bboxes = yolo_bboxes.tolist()

        res_orig_img = draw_bounding_boxes(res_orig_img, res_orig_bboxes)
        res_mosaic_img = draw_bounding_boxes(res_mosaic_img, res_mosaic_bboxes)
        np_yolo_img = draw_bounding_boxes(np_yolo_img, yolo_bboxes)

        key_code = show_images_cv2(
            [res_orig_img, res_mosaic_img, np_yolo_img], titles)
        if key_code == 27:
            break


if __name__ == "__main__":
    # Configs
    dset_dir = Path('data/tank_5set_rgb')
    input_size = 640
    polarization = False
    random_crop = True
    horizontal_flip = False
    mosaic_prob = 0.0
    mixup_prob = 0.0
    main(dset_dir=dset_dir, input_size=input_size, polarization=polarization,
         random_crop=random_crop, mosaic_prob=mosaic_prob,
         mixup_prob=mixup_prob, horizontal_flip=horizontal_flip)
