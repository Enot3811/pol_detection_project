"""Скрипт для обучения yolov7."""
# TODO дописать

import sys
from functools import partial
from pathlib import Path
import argparse

import torch
from pytorch_accelerated.callbacks import (
    ModelEmaCallback,
    ProgressBarCallback,
    SaveBestModelCallback,
    get_default_callbacks)
from pytorch_accelerated.schedulers import CosineLrScheduler
import cv2

sys.path.append(str(Path(__file__).parents[3]))
from Yolov7.yolov7 import create_yolov7_model
from Yolov7.yolov7.dataset import (
    Yolov7Dataset, create_base_transforms, create_yolov7_transforms,
    yolov7_collate_fn)
from Yolov7.yolov7.evaluation import CalculateMeanAveragePrecisionCallback
from Yolov7.yolov7.loss_factory import create_yolov7_loss
from Yolov7.yolov7.mosaic import (
    MosaicMixupDataset, create_post_mosaic_transform)
from Yolov7.yolov7.trainer import Yolov7Trainer, filter_eval_predictions
from Yolov7.yolov7.utils import SaveBatchesCallback, Yolov7ModelEma


from Yolov7.custom.datasets import TankDetectionDataset
from utils.torch_utils.torch_functions import draw_bounding_boxes
from utils.image_utils.image_functions import read_image, resize_image


DATA_PATH = Path(__file__).absolute().parents[2] / 'data/polarized_dataset'


def main(**kwargs):
    # Parse args
    dset_dir = kwargs['dset_dir']
    continue_training = kwargs['continue_training']
    device = kwargs['device']

    # Get dataset parameters
    train_dir = dset_dir / 'train'
    val_dir = dset_dir / 'val'
    name2index = {'Tank': 0}
    index2name = {0: 'Tank'}
    input_size = 640
    mosaic_prob = 1.0
    mixup_prob = 0.15

    # Get model parameters
    model_arch = 'yolov7'
    conf_thresh = 0.8
    iou_thresh = 0.2
    pretrained = False  # TODO сделать fine tunning

    # Get training parameters
    lr = 0.01
    batch_size = 16
    base_weight_decay = 0.0005
    num_epochs = 20

    # Prepare some stuff
    if device == 'auto':
        device: torch.device = (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu'))
    else:
        device: torch.device = torch.device(device)
    work_dir = Path(__file__).parents[2] / 'work_dir' / 'train_1'
    work_dir.mkdir(parents=True, exist_ok=True)
    if continue_training:
        checkpoint = torch.load(work_dir / 'last_checkpoint.pt')
        model_params = checkpoint['model']
        optim_params = checkpoint['optimizer']
        start_ep = checkpoint['epoch'] + 1
    else:
        model_params = None
        optim_params = None
        start_ep = 0
    
    # Get transforms
    base_transforms = create_base_transforms(input_size)
    post_mosaic_transforms = create_post_mosaic_transform(
        input_size, input_size)
    yolo_train_transforms = create_yolov7_transforms(
        (input_size, input_size), training=True)
    yolo_val_transforms = create_yolov7_transforms(
        (input_size, input_size), training=False)

    # Get dataset
    train_dset = TankDetectionDataset(
        train_dir, name2index=name2index, transforms=base_transforms)
    val_dset = TankDetectionDataset(val_dir, name2index=name2index)

    num_classes = len(train_dset.labels)

    mosaic_mixup_dset = MosaicMixupDataset(
        train_dset,
        apply_mosaic_probability=mosaic_prob,
        apply_mixup_probability=mixup_prob,
        post_mosaic_transforms=post_mosaic_transforms)
    
    # if pretrained:
    #     # disable mosaic if finetuning
    #     mds.disable()
    
    train_yolo_dset = Yolov7Dataset(mosaic_mixup_dset, yolo_train_transforms)
    val_yolo_dset = Yolov7Dataset(val_dset, yolo_val_transforms)

    # Get the model, loss amd optimizer
    model = create_yolov7_model(
        architecture=model_arch,
        num_classes=num_classes,
        pretrained=pretrained)
    param_groups = model.get_parameter_groups()

    loss_func = create_yolov7_loss(model, image_size=input_size)

    optimizer = torch.optim.SGD(
        param_groups["other_params"], lr=lr, momentum=0.937, nesterov=True)
    
    # Get the trainer
    trainer = Yolov7Trainer(
        model=model,
        optimizer=optimizer,
        loss_func=loss_func,
        filter_eval_predictions_fn=partial(
            filter_eval_predictions,
            confidence_threshold=conf_thresh,
            nms_threshold=iou_thresh),
        callbacks=[
            ModelEmaCallback(
                decay=0.9999,
                model_ema=Yolov7ModelEma,
                callbacks=[ProgressBarCallback, calculate_map_callback],
            ),
            SaveBestModelCallback(save_path='cars_3ch_best.pt',
                                  watch_metric="map",
                                  greater_is_better=True),
            SaveBatchesCallback("./batches", num_images_per_batch=3),
            *get_default_callbacks(progress_bar=True),
        ],
    )


    # # calculate scaled weight decay and gradient accumulation steps
    # total_batch_size = (
    #     batch_size * trainer._accelerator.num_processes
    # )  # batch size across all processes

    # nominal_batch_size = 64
    # num_accumulate_steps = max(round(nominal_batch_size / total_batch_size), 1)
    # base_weight_decay = 0.0005
    # scaled_weight_decay = (
    #     base_weight_decay * total_batch_size * num_accumulate_steps /
    #     nominal_batch_size
    # )

    # optimizer.add_param_group(
    #     {"params": param_groups["conv_weights"],
    #      "weight_decay": scaled_weight_decay}
    # )

    # # run training
    # trainer.train(
    #     num_epochs=num_epochs,
    #     train_dataset=train_yds,
    #     eval_dataset=eval_yds,
    #     per_device_batch_size=batch_size,
    #     create_scheduler_fn=CosineLrScheduler.create_scheduler_fn(
    #         num_warmup_epochs=5,
    #         num_cooldown_epochs=5,
    #         k_decay=2,
    #     ),
    #     collate_fn=yolov7_collate_fn,
    #     gradient_accumulation_steps=num_accumulate_steps,
    # )


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'dset_dir', type=Path,
        help='Путь к директории с датасетом. '
        'Подразумевается директория с 3 подкаталогами "train", "val" и "test",'
        ' в каждом из которых находится датасет в формате CVAT.')
    parser.add_argument(
        '--continue_training', action='store_true',
        help='Продолжить ли обучение с последнего чекпоинта '
        '"last_checkpoint.pt" в "work_dir/train_n".')
    # parser.add_argument(
    #     '--pretrained', action='store_true',
    #     help='Загрузить ли официальные предобученные веса. '
    #     'Игнорируется, если указан аргумент "--weights".')
    # parser.add_argument(
    #     '--polarized', action='store_true',
    #     help='Поляризационная ли съёмка. Если не указан, то RGB.')
    # parser.add_argument(
    #     '--conf_thresh', type=float, default=0.6,
    #     help='Порог уверенности модели.')
    # parser.add_argument(
    #     '--iou_thresh', type=float, default=0.2,
    #     help='Порог перекрытия рамок.')
    # parser.add_argument(
    #     '--show_time', action='store_true',
    #     help='Показывать время выполнения.')
    
    parser.add_argument(
        '--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'],
        help='The device on which calculations are performed. '
        '"auto" selects "cuda" when it is possible.')
    args = parser.parse_args([
        'data/debug_dset/',
        '--device', 'cpu'
    ])
    return args


if __name__ == "__main__":
    args = parse_args()
    dset_dir = args.dset_dir
    continue_training = args.continue_training
    device = args.device
    main(dset_dir=dset_dir, continue_training=continue_training, device=device)
