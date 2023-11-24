"""Тест для модуля `tank_detection_additional`.

Требуется конфиг обучения, где планируется использовать данный модуль.
"""

import sys
from pathlib import Path
import argparse
import json

from torchvision.ops import box_convert
import cv2
import albumentations as A

sys.path.append(str(Path(__file__).parents[4]))
from Yolov7.yolov7.dataset import (
    Yolov7Dataset, create_yolov7_transforms)
from Yolov7.yolov7.mosaic import (
    MosaicMixupDataset, create_post_mosaic_transform)
from Yolov7.custom.datasets import TankDetectionDataset
from utils.torch_utils.torch_functions import (
    draw_bounding_boxes, image_tensor_to_numpy)


def main(config_pth: str, dset_type: str):
    # Read config
    with open(config_pth, 'r') as f:
        config_str = f.read()
    config = json.loads(config_str)

    dset_pth = Path(config['dataset']) / dset_type

    # Get transforms
    post_mosaic_transforms = create_post_mosaic_transform(
        config['input_size'], config['input_size'])
    resize_to_show = A.Compose(
        [A.LongestMaxSize(config['input_size'])],
        bbox_params=A.BboxParams(format='pascal_voc',
                                 label_fields=['classes']))
    # post_mosaic_transforms = None
    training_transforms = [
        A.RandomCropFromBorders(crop_left=0.4, crop_right=0.4,
                                crop_top=0.4, crop_bottom=0.4),
        A.HorizontalFlip()
    ]
    yolo_transforms = create_yolov7_transforms(
        (config['input_size'], config['input_size']), dset_type,
        training_transforms=training_transforms)

    # Get dataset
    obj_dset = TankDetectionDataset(
        dset_pth, name2index=config['cls_to_id'],
        polarization=config['polarization'])
    index_to_cls = obj_dset.index_to_class

    mosaic_mixup_dset = MosaicMixupDataset(
        obj_dset,
        apply_mosaic_probability=config['mosaic_prob'],
        apply_mixup_probability=config['mixup_prob'],
        post_mosaic_transforms=post_mosaic_transforms)
    if dset_type != 'train':
        mosaic_mixup_dset.disable()
    
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

        str_orig_cls = list(map(lambda idx: index_to_cls[int(idx)],
                                orig_cls))
        str_mos_cls = list(map(lambda idx: index_to_cls[int(idx)],
                               mos_cls))
        str_ls_yolo_cls = list(map(lambda idx: index_to_cls[int(idx)],
                                   ls_yolo_cls))

        resized = resize_to_show(
            image=orig_img, bboxes=orig_bboxes, classes=orig_cls)
        orig_img = resized['image']
        orig_bboxes = resized['bboxes']
        resized = resize_to_show(
            image=mos_img, bboxes=mos_bboxes, classes=mos_cls)
        mos_img = resized['image']
        mos_bboxes = resized['bboxes']

        orig_img_bboxes = draw_bounding_boxes(
            orig_img, orig_bboxes, str_orig_cls, line_width=1)
        mos_img_bboxes = draw_bounding_boxes(
            mos_img, mos_bboxes, str_mos_cls, line_width=1)
        yolo_img_bboxes = draw_bounding_boxes(
            np_yolo_img, ls_yolo_bboxes, str_ls_yolo_cls)

        print('Original shape:', orig_shape,
              'Augmented base shape:', orig_img.shape)
        print('Mosaic original shape:', mos_shape,
              'Augmented mosaic shape:', mos_img.shape)
        print('Yolo original shape:', yolo_shape,
              'Augmented yolo shape:', yolo_img.shape, '\n')

        cv2.imshow('orig', cv2.cvtColor(orig_img_bboxes,
                   cv2.COLOR_RGB2BGR))
        cv2.imshow('mos_mix', cv2.cvtColor(mos_img_bboxes,
                   cv2.COLOR_RGB2BGR))
        cv2.imshow('yolo', cv2.cvtColor(yolo_img_bboxes,
                   cv2.COLOR_RGB2BGR))
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
        'config_pth', type=Path,
        help='Путь к конфигу обучения.')
    parser.add_argument(
        '--dset_type', type=str, default='train', choices=['train', 'val'],
        help='Для "train" применяются аугментации, а для "val" нет.')

    args = parser.parse_args()

    if not args.config_pth.exists():
        raise FileExistsError('Config file does not exist.')
    return args


if __name__ == "__main__":
    args = parse_args()
    config_pth = args.config_pth
    dset_type = args.dset_type
    main(config_pth=config_pth, dset_type=dset_type)
