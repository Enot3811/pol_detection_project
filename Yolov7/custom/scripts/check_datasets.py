"""Скрипт для итерации по датасетам перед обучениям и просмотра семплов.

Позволяет покрутить параметры датасетов и посмотреть на результаты аугментаций.
"""


import sys
from pathlib import Path
import argparse

from torchvision.ops import box_convert
import cv2

sys.path.append(str(Path(__file__).parents[3]))
from Yolov7.yolov7.dataset import (
    Yolov7Dataset, create_base_transforms, create_yolov7_transforms)
from Yolov7.yolov7.mosaic import (
    MosaicMixupDataset, create_post_mosaic_transform)
from Yolov7.custom.datasets import TankDetectionDataset
from utils.torch_utils.torch_functions import (
    draw_bounding_boxes, image_tensor_to_numpy)


def main(**kwargs):
    # Parse args
    dset_dir = kwargs['dset_dir']
    dset_type = kwargs['dset_type']

    # Get dataset parameters
    input_size = 640
    mosaic_prob = 1.0
    mixup_prob = 0.15
    training = dset_type == 'train'
    
    # Get transforms
    base_transforms = create_base_transforms(input_size)
    post_mosaic_transforms = create_post_mosaic_transform(
        input_size, input_size)
    # post_mosaic_transforms = None
    yolo_transforms = create_yolov7_transforms(
        (input_size, input_size), training)

    # Get dataset
    obj_dset = TankDetectionDataset(dset_dir, transforms=base_transforms)
    index_to_cls = obj_dset.index_to_class

    mosaic_mixup_dset = MosaicMixupDataset(
        obj_dset,
        apply_mosaic_probability=mosaic_prob,
        apply_mixup_probability=mixup_prob,
        post_mosaic_transforms=post_mosaic_transforms)
    
    yolo_dset = Yolov7Dataset(mosaic_mixup_dset, yolo_transforms)

    # iterate over datasets
    for obj_sample, mos_sample, yolo_sample in zip(obj_dset,
                                                   mosaic_mixup_dset,
                                                   yolo_dset):
        orig_img, orig_bboxes, orig_cls, img_name, orig_shape = obj_sample
        mos_img, mos_bboxes, mos_cls, _, mos_shape = mos_sample
        yolo_img, yolo_labels, _, yolo_shape = yolo_sample

        yolo_cls = yolo_labels[:, 1]
        yolo_bboxes = yolo_labels[:, 2:]

        # Convert yolo sample to normal format
        np_yolo_img = image_tensor_to_numpy(yolo_img)
        ls_yolo_cls = yolo_cls.tolist()
        
        yolo_bboxes[:, 0] *= np_yolo_img.shape[1]
        yolo_bboxes[:, 2] *= np_yolo_img.shape[1]
        yolo_bboxes[:, 1] *= np_yolo_img.shape[0]
        yolo_bboxes[:, 3] *= np_yolo_img.shape[0]
        yolo_bboxes = box_convert(yolo_bboxes, 'cxcywh', 'xyxy')
        ls_yolo_bboxes = yolo_bboxes.tolist()

        str_orig_cls = list(map(lambda idx: index_to_cls[idx], orig_cls))
        str_mos_cls = list(map(lambda idx: index_to_cls[idx], mos_cls))
        str_ls_yolo_cls = list(map(lambda idx: index_to_cls[idx], ls_yolo_cls))

        orig_img_bboxes = draw_bounding_boxes(
            orig_img, orig_bboxes, str_orig_cls)
        mos_img_bboxes = draw_bounding_boxes(
            mos_img, mos_bboxes, str_mos_cls)
        yolo_img_bboxes = draw_bounding_boxes(
            np_yolo_img, ls_yolo_bboxes, str_ls_yolo_cls)

        print('Original shape:', orig_shape,
              'Augmented base shape:', orig_img.shape)
        print('Mosaic original shape:', mos_shape,
              'Augmented mosaic shape:', mos_img.shape)
        print('Mosaic original shape:', yolo_shape,
              'Augmented mosaic shape:', yolo_img.shape, '\n')

        cv2.imshow('orig', cv2.cvtColor(orig_img_bboxes, cv2.COLOR_RGB2BGR))
        cv2.imshow('mos_mix', cv2.cvtColor(mos_img_bboxes, cv2.COLOR_RGB2BGR))
        cv2.imshow('yolo', cv2.cvtColor(yolo_img_bboxes, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(0)
        if key == 27:
            break


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'dset_dir', type=Path,
        help='Путь к директории с датасетом. Подразумевается CVAT формат.')
    parser.add_argument(
        '--dset_type', type=str, default='train', choices=['train', 'val'],
        help='Для "train" применяются аугментации, а для "val" нет.')

    args = parser.parse_args([
        'data/debug_dset/train'
    ])
    return args


if __name__ == "__main__":
    args = parse_args()
    dset_dir = args.dset_dir
    dset_type = args.dset_type
    main(dset_dir=dset_dir, dset_type=dset_type)
