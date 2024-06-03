"""Train yolov7 with polarization dataset."""


import sys
from pathlib import Path
import shutil
import json
import argparse

import torch
import torch.optim as optim
from torch import FloatTensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import box_convert
from torchmetrics import Metric
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
import albumentations as A

sys.path.append(str(Path(__file__).parents[3]))
from Yolov7.yolov7.dataset import (
    Yolov7Dataset, create_yolov7_transforms, yolov7_collate_fn)
from Yolov7.yolov7.loss_factory import create_yolov7_loss
from Yolov7.yolov7.mosaic import (
    MosaicMixupDataset, create_post_mosaic_transform)
from Yolov7.yolov7.trainer import filter_eval_predictions
from Yolov7.custom.datasets import (
    PolarizationObjectDetectionDataset, PolarizationObjectDetectionDataset2ch)
from Yolov7.custom.model_utils import create_yolo


class YoloLossMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('loss',
                       default=torch.tensor(0, dtype=torch.float64),
                       dist_reduce_fx='sum')
        self.add_state('n_total',
                       default=torch.tensor(0, dtype=torch.float64),
                       dist_reduce_fx='sum')

    def update(self, batch_loss: FloatTensor):
        self.loss += batch_loss
        self.n_total += 1

    def compute(self):
        return self.loss / self.n_total


def main(**kwargs):
    # Read config
    config_pth = kwargs['config_pth']
    with open(config_pth, 'r') as f:
        config_str = f.read()
    config = json.loads(config_str)

    # Prepare some stuff
    torch.manual_seed(config['random_seed'])

    if config['device'] == 'auto':
        device: torch.device = (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu'))
    else:
        device: torch.device = torch.device(config['device'])

    work_dir = Path(config['work_dir'])
    tensorboard_dir = work_dir / 'tensorboard'
    ckpt_dir = work_dir / 'ckpts'
    if not config['continue_training']:
        if work_dir.exists():
            input(f'Specified directory "{str(work_dir)}" already exists. '
                  'If continue, this directory will be deleted. '
                  'Press enter to continue.')
            shutil.rmtree(work_dir)
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    if config['polarization']:
        if 'active_ch' in config:
            num_channels = 2
            active_ch = config['active_ch']
        else:
            num_channels = 4
            active_ch = None
    else:
        num_channels = 3
        active_ch = None
    pad_colour = (114,) * num_channels

    # Check and load checkpoint
    if config['continue_training']:
        checkpoint = torch.load(ckpt_dir / 'last_checkpoint.pth')
        model_params = checkpoint['model_state_dict']
        optim_params = checkpoint['optimizer_state_dict']
        lr_params = checkpoint['scheduler_state_dict']
        start_ep = checkpoint['epoch']
    else:
        model_params = None
        optim_params = None
        lr_params = None
        start_ep = 0

    # Get tensorboard
    log_writer = SummaryWriter(str(tensorboard_dir))

    # Get transforms
    post_mosaic_transforms = create_post_mosaic_transform(
        config['input_size'], config['input_size'], pad_colour=pad_colour)

    if config['random_crop']:
        training_transforms = [
            A.RandomCropFromBorders(crop_left=0.4, crop_right=0.4,
                                    crop_top=0.4, crop_bottom=0.4),
            A.HorizontalFlip()
        ]
    else:
        training_transforms = [A.HorizontalFlip()]
    yolo_train_transforms = create_yolov7_transforms(
        (config['input_size'], config['input_size']), training=True,
        pad_colour=pad_colour, training_transforms=training_transforms)
    yolo_val_transforms = create_yolov7_transforms(
        (config['input_size'], config['input_size']), training=False,
        pad_colour=pad_colour)

    # Get datasets and loaders
    if active_ch is None:
        train_dset = PolarizationObjectDetectionDataset(
            config['train_dset'], class_to_index=config['cls_to_id'],
            polarization=config['polarization'])
        val_dset = PolarizationObjectDetectionDataset(
            config['val_dset'], class_to_index=config['cls_to_id'],
            polarization=config['polarization'])
    else:
        train_dset = PolarizationObjectDetectionDataset2ch(
            config['train_dset'], class_to_index=config['cls_to_id'],
            polarization=config['polarization'],
            active_ch=active_ch)
        val_dset = PolarizationObjectDetectionDataset2ch(
            config['val_dset'], class_to_index=config['cls_to_id'],
            polarization=config['polarization'],
            active_ch=active_ch)
    num_classes = len(config['cls_to_id'])

    mosaic_mixup_dset = MosaicMixupDataset(
        train_dset,
        apply_mosaic_probability=config['mosaic_prob'],
        apply_mixup_probability=config['mixup_prob'],
        pad_colour=pad_colour,
        post_mosaic_transforms=post_mosaic_transforms)

    train_yolo_dset = Yolov7Dataset(mosaic_mixup_dset, yolo_train_transforms)
    val_yolo_dset = Yolov7Dataset(val_dset, yolo_val_transforms)

    train_loader = DataLoader(train_yolo_dset,
                              batch_size=config['batch_size'],
                              shuffle=config['shuffle_train'],
                              num_workers=config['num_workers'],
                              collate_fn=yolov7_collate_fn)
    val_loader = DataLoader(val_yolo_dset,
                            batch_size=config['batch_size'],
                            shuffle=config['shuffle_val'],
                            num_workers=config['num_workers'],
                            collate_fn=yolov7_collate_fn)

    # Get the model and loss
    model = create_yolo(num_classes=num_classes,
                        num_channels=num_channels,
                        pretrained=config['pretrained'],
                        model_arch=config['model_arch'])
    model.to(device=device)
    if model_params:
        model.load_state_dict(model_params)

    loss_func = create_yolov7_loss(model, image_size=config['input_size'])
    loss_func.to(device=device)

    # Get the optimizer
    if config['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=config['lr'],
                              momentum=0.937,
                              nesterov=True)
    elif config['optimizer'] == 'SGD_groups':
        param_groups = model.get_parameter_groups()
        optimizer = optim.SGD(param_groups['other_params'],
                              lr=config['lr'],
                              momentum=0.937,
                              nesterov=True)
        optimizer.add_param_group({'params': param_groups['conv_weights'],
                                   'weight_decay': config['weight_decay']})
    elif config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=config['lr'],
                               weight_decay=config['weight_decay'])
    else:
        raise ValueError(
            f'Got unsupported optimizer type {str(config["optimizer"])}')
    if optim_params:
        optimizer.load_state_dict(optim_params)
    optimizer.state_dict()

    # Get the scheduler
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['n_epoch'], eta_min=1e-6,
        last_epoch=start_ep - 1)
    if lr_params:
        lr_scheduler.load_state_dict(lr_params)

    # Get metrics
    train_map_metric = MeanAveragePrecision(
        iou_thresholds=[config['iou_thresh']], compute_on_cpu=True)
    train_loss_metric = YoloLossMetric()
    val_map_metric = MeanAveragePrecision(
        iou_thresholds=[config['iou_thresh']], compute_on_cpu=True)
    val_loss_metric = YoloLossMetric()
    train_loss_metric.to(device=device)
    train_map_metric.to(device=device)
    val_map_metric.to(device=device)
    val_loss_metric.to(device=device)

    # Do training
    best_metric = None
    for epoch in range(start_ep, config['n_epoch']):

        print(f'Epoch {epoch + 1}')

        # Train epoch
        model.train()
        loss_func.train()
        step = 0
        for batch in tqdm(train_loader, 'Train step'):
            images, labels, img_names, img_sizes = batch
            images = images.to(device=device)
            labels = labels.to(device=device)
            fpn_heads_outputs = model(images)
            loss, _ = loss_func(
                fpn_heads_outputs=fpn_heads_outputs,
                targets=labels, images=images)
            loss = loss[0] / config['accumulate_steps']
            loss.backward()

            # Whether to update weights
            if ((step + 1) % config['accumulate_steps'] == 0 or
                    (step + 1 == len(train_loader))):
                # TODO при продолжении обучения ломается на step из-за
                # размерностей
                optimizer.step()
                optimizer.zero_grad()

            train_loss_metric.update(loss)

            # Translate predicts to mAP compatible format
            preds = model.postprocess(
                fpn_heads_outputs, multiple_labels_per_box=False)

            nms_predictions = filter_eval_predictions(
                preds,
                confidence_threshold=config['conf_thresh'],
                nms_threshold=config['iou_thresh'])

            map_preds = [
                {'boxes': image_preds[:, :4],
                 'labels': image_preds[:, -1].long(),
                 'scores': image_preds[:, -2]}
                for image_preds in nms_predictions]

            map_targets = []
            for i in range(images.shape[0]):
                img_labels = labels[labels[:, 0] == i]
                img_boxes = img_labels[:, 2:] * images.shape[2]
                img_boxes = box_convert(img_boxes, 'cxcywh', 'xyxy')
                img_classes = img_labels[:, 1].long()
                map_targets.append({
                    'boxes': img_boxes,
                    'labels': img_classes
                })

            train_map_metric.update(map_preds, map_targets)

            step += 1
        
        # Val epoch
        with torch.no_grad():
            model.eval()
            loss_func.eval()
            for batch in tqdm(val_loader, 'Val step'):
                images, labels, img_names, img_sizes = batch
                images = images.to(device=device)
                labels = labels.to(device=device)
                fpn_heads_outputs = model(images)
                loss, _ = loss_func(
                    fpn_heads_outputs=fpn_heads_outputs,
                    targets=labels, images=images)
                loss = loss[0]
                val_loss_metric.update(loss)
                
                # Translate predicts to mAP compatible format
                preds = model.postprocess(
                    fpn_heads_outputs, multiple_labels_per_box=False)

                nms_predictions = filter_eval_predictions(
                    preds,
                    confidence_threshold=config['conf_thresh'],
                    nms_threshold=config['iou_thresh'])

                map_preds = [
                    {'boxes': image_preds[:, :4],
                     'labels': image_preds[:, -1].long(),
                     'scores': image_preds[:, -2]}
                    for image_preds in nms_predictions]

                map_targets = []
                for i in range(images.shape[0]):
                    img_labels = labels[labels[:, 0] == i]
                    img_boxes = img_labels[:, 2:] * images.shape[2]
                    img_boxes = box_convert(img_boxes, 'cxcywh', 'xyxy')
                    img_classes = img_labels[:, 1].long()
                    map_targets.append({
                        'boxes': img_boxes,
                        'labels': img_classes
                    })
                    
                val_map_metric.update(map_preds, map_targets)

        # Lr scheduler
        lr_scheduler.step()
        lr = lr_scheduler.get_last_lr()[0]

        # Log epoch metrics
        train_loss = train_loss_metric.compute()
        val_loss = val_loss_metric.compute()
        train_loss_metric.reset()
        val_loss_metric.reset()
        log_writer.add_scalars('loss', {
            'train': train_loss,
            'val': val_loss
        }, epoch)

        train_map_dict = train_map_metric.compute()
        val_map_dict = val_map_metric.compute()
        train_map_metric.reset()
        val_map_metric.reset()
        log_writer.add_scalars('map', {
            'train': train_map_dict['map'],
            'val': val_map_dict['map']
        }, epoch)
        log_writer.add_scalars('map_small', {
            'train': train_map_dict['map_small'],
            'val': val_map_dict['map_small']
        }, epoch)
        log_writer.add_scalars('map_medium', {
            'train': train_map_dict['map_medium'],
            'val': val_map_dict['map_medium']
        }, epoch)
        log_writer.add_scalars('map_large', {
            'train': train_map_dict['map_large'],
            'val': val_map_dict['map_large']
        }, epoch)

        log_writer.add_scalar('Lr', lr, epoch)

        print('TrainLoss:', train_loss.item())
        print('ValLoss:', val_loss.item())
        print('TrainMap:', train_map_dict['map'].item())
        print('ValMap:', val_map_dict['map'].item())
        print('Lr:', lr)

        # Save model
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'epoch': epoch + 1
        }
        torch.save(checkpoint, ckpt_dir / 'last_checkpoint.pth')

        if best_metric is None or best_metric < val_map_dict['map'].item():
            torch.save(checkpoint, ckpt_dir / 'best_checkpoint.pth')
            best_metric = val_map_dict['map'].item()

    log_writer.close()


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'config_pth', type=Path,
        help='Path to train config.')
    args = parser.parse_args()

    if not args.config_pth.exists():
        raise FileNotFoundError(
            f'Config file "{str(args.config_pth)}" is not found.')
    return args


if __name__ == "__main__":
    args = parse_args()
    config_pth = args.config_pth
    main(config_pth=config_pth)
