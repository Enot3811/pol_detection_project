"""Скрипт для обучения yolov7."""


import sys
from pathlib import Path
import json
import argparse
from math import ceil

import torch
import torch.optim as optim
from torch import FloatTensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import box_convert
from torchmetrics import Metric
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[3]))
from Yolov7.yolov7.dataset import (
    Yolov7Dataset, create_yolov7_transforms, yolov7_collate_fn)
from Yolov7.yolov7.loss_factory import create_yolov7_loss
from Yolov7.yolov7.mosaic import (
    MosaicMixupDataset, create_post_mosaic_transform)
from Yolov7.yolov7.trainer import filter_eval_predictions
from Yolov7.custom.datasets import TankDetectionDataset
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
    cpu_device = torch.device('cpu')
    
    work_dir = Path(config['work_dir'])
    tensorboard_dir = work_dir / 'tensorboard'
    ckpt_dir = work_dir / 'ckpts'
    if not config['continue_training']:
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    n_accumulate_steps = ceil(
        config['nominal_batch_size'] / config['batch_size'])

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
        config['input_size'], config['input_size'])
    yolo_train_transforms = create_yolov7_transforms(
        (config['input_size'], config['input_size']), training=True)
    yolo_val_transforms = create_yolov7_transforms(
        (config['input_size'], config['input_size']), training=False)

    # Get datasets and loaders
    train_dir = Path(config['dataset']) / 'train'
    val_dir = Path(config['dataset']) / 'val'

    train_dset = TankDetectionDataset(
        train_dir, name2index=config['cls_to_id'])
    val_dset = TankDetectionDataset(
        val_dir, name2index=config['cls_to_id'])
    num_classes = len(config['cls_to_id'])

    mosaic_mixup_dset = MosaicMixupDataset(
        train_dset,
        apply_mosaic_probability=config['mosaic_prob'],
        apply_mixup_probability=config['mixup_prob'],
        post_mosaic_transforms=post_mosaic_transforms)
    
    if config['pretrained']:
        # disable mosaic if finetuning
        mosaic_mixup_dset.disable()
    
    train_yolo_dset = Yolov7Dataset(mosaic_mixup_dset, yolo_train_transforms)
    val_yolo_dset = Yolov7Dataset(val_dset, yolo_val_transforms)

    train_loader = DataLoader(train_yolo_dset,
                              batch_size=config['batch_size'],
                              shuffle=config['shuffle_train'],
                              collate_fn=yolov7_collate_fn)
    val_loader = DataLoader(val_yolo_dset,
                            batch_size=config['batch_size'],
                            shuffle=config['shuffle_val'],
                            collate_fn=yolov7_collate_fn)

    # Get the model and loss
    model = create_yolo(num_classes=num_classes,
                        pretrained=config['pretrained'],
                        model_arch=config['model_arch'])
    model.to(device=device)
    if model_params:
        model.load_state_dict(model_params)

    loss_func = create_yolov7_loss(model, image_size=config['input_size'])
    loss_func.to(device=device)

    # Get the optimizer
    param_groups = model.get_parameter_groups()
    optimizer = optim.SGD(param_groups['other_params'],
                          lr=config['lr'],
                          momentum=0.937,
                          nesterov=True)
    optimizer.add_param_group({'params': param_groups['conv_weights'],
                               'weight_decay': config['weight_decay']})
    # optimizer = optim.Adam(model.parameters(),
    #                        lr=config['lr'],
    #                        weight_decay=config['weight_decay'])
    if optim_params:
        optimizer.load_state_dict(optim_params)

    # Get the scheduler
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['n_epoch'], eta_min=1e-6,
        last_epoch=start_ep - 1)
    if lr_params:
        lr_scheduler.load_state_dict(lr_params)
    
    # Get metrics
    val_map_metric = MeanAveragePrecision(extended_summary=False)
    train_loss_metric = YoloLossMetric()
    val_loss_metric = YoloLossMetric()
    val_map_metric.to(device=device)
    train_loss_metric.to(device=device)
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
            loss = loss[0] / n_accumulate_steps
            loss.backward()
            
            perform_gradient_update = (
                (step + 1) % n_accumulate_steps == 0
            ) or (step + 1 == len(train_loader))
            if perform_gradient_update:
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss_metric.update(loss)
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
        train_loss_metric.reset()
        log_writer.add_scalar('loss/train', train_loss, epoch)
        val_loss = val_loss_metric.compute()
        val_loss_metric.reset()
        log_writer.add_scalar('loss/val', val_loss, epoch)

        val_map_dict = val_map_metric.compute()
        val_map_metric.reset()
        log_writer.add_scalar(
            'map/val',
            val_map_dict['map'].to(device=cpu_device),
            epoch)
        log_writer.add_scalar(
            'map_small/val',
            val_map_dict['map_small'].to(device=cpu_device),
            epoch)
        log_writer.add_scalar(
            'map_medium/val',
            val_map_dict['map_medium'].to(device=cpu_device),
            epoch)
        log_writer.add_scalar(
            'map_large/val',
            val_map_dict['map_large'].to(device=cpu_device),
            epoch)

        log_writer.add_scalar('Lr', lr, epoch)

        print('TrainLoss:', train_loss.item())
        print('ValLoss:', val_loss.item())
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
        help='Путь к конфигу обучения.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config_pth = args.config_pth
    main(config_pth=config_pth)
