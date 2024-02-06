"""Синхронный стрим с камеры на YOLOv7."""


import sys
from pathlib import Path
import json

from vimba import Vimba, PixelFormat
import cv2
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

sys.path.append(str(Path(__file__).parents[3]))
from mako_camera.vimba_custom.vimba_tools import get_camera
from Yolov7.custom.model_utils import (
    create_yolo, load_yolo_checkpoint, yolo_inference, coco_idx2label)
from Yolov7.yolov7.dataset import create_yolov7_transforms
from utils.torch_utils.torch_functions import (
    image_tensor_to_numpy, draw_bounding_boxes)
# from mako_camera.cameras_utils import split_raw_pol


def main(
    camera_id: str, config_pth: Path, conf_thresh: float, iou_thresh: float
):
    # Prepare YOLOv7
    if config_pth:
        # Read config
        with open(config_pth, 'r') as f:
            config_str = f.read()
        config = json.loads(config_str)
        
        cls_id_to_name = {val: key for key, val in config['cls_to_id'].items()}
        num_classes = len(cls_id_to_name)
        polarized = config['polarization']
        num_channels = 4 if polarized else 3
        weights_pth = Path(
            config['work_dir']) / 'ckpts' / 'best_checkpoint.pth'
    else:
        # Set COCO parameters
        cls_id_to_name = coco_idx2label
        num_classes = len(coco_idx2label)
        num_channels = 3
    
    pad_colour = (114,) * num_channels

    # Загрузка модели
    if config_pth:
        model = load_yolo_checkpoint(weights_pth, num_classes)
    else:
        # Создать COCO модель
        model = create_yolo(num_classes)

    # Обработка семплов
    process_transforms = create_yolov7_transforms(pad_colour=pad_colour)
    normalize_transforms = A.Compose([ToTensorV2(transpose_mask=True)])

    # Start streaming
    esc_key_code = 27
    with Vimba.get_instance():
        with get_camera('DEV_000F315D630A') as camera:
            if camera.get_pixel_format() != PixelFormat.BayerRG8:
                raise ValueError(
                    'Camera should be in BayerRG8 pixel format '
                    f'but now it is in {camera.get_pixel_format()}.')
            
            for frame in camera.get_frame_generator(timeout_ms=3000):

                # Подготавливаем картинку
                image = frame.as_numpy_ndarray()

                if polarized:
                    # TODO доделать для pol камеры
                    raise NotImplementedError()
                    # image = split_raw_pol(raw_pol)  # ndarray (h, w, 4)
                else:
                    # Raw rgb
                    image = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2BGR)

                # Предобработка данных
                image = process_transforms(
                    image=image, bboxes=[], labels=[])['image']
                tensor_image = normalize_transforms(image=image)['image']
                tensor_image = tensor_image.to(torch.float32) / 255
                tensor_image = tensor_image[None, ...]  # Add batch dim

                # Запуск модели
                boxes, class_ids, confidences = yolo_inference(
                    model, tensor_image, conf_thresh, iou_thresh)

                image = image_tensor_to_numpy(tensor_image)
                image = image[..., :3]

                labels = list(map(lambda idx: cls_id_to_name[idx],
                                  class_ids.tolist()[:30]))
                image = draw_bounding_boxes(
                    image[0],
                    boxes.tolist()[:30],
                    labels,
                    confidences.tolist()[:30])
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                msg = 'YOLOv7 predict. Press <Esc> to stop stream.'
                cv2.imshow(msg, image)
                key = cv2.waitKey(1)
                if key == esc_key_code:
                    break


if __name__ == '__main__':
    camera_id = 'DEV_000F315D630A'  # RGB
    # camera_id = 'DEV_000F315D16C4'  # POL
    config_pth = Path('Yolov7/custom/configs/tank_10.json')
    conf_thresh = 0.7
    iou_thresh = 0.2
    main(camera_id=camera_id, config_pth=config_pth,
         conf_thresh=conf_thresh, iou_thresh=iou_thresh)
