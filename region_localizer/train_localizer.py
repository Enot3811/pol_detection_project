"""Скрипт для обучения локализатора участка местности."""

from pathlib import Path
import shutil
import json
import argparse
import sys

import torch
import torch.optim as optim
from torch import FloatTensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Metric
from torchmetrics.detection import IntersectionOverUnion
from tqdm import tqdm
import albumentations as A

sys.path.append(str(Path(__file__).parents[1]))
from region_localizer.models import RetinaRegionLocalizer, ModifiedRetinaV2
from region_localizer.datasets import RegionDataset, RegionDatasetV2


class LossMetric(Metric):
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


def main(config_pth: Path):
    # Read config
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
            input('Specified directory already exists. '
                  'Сontinuing to work will delete the data located there. '
                  'Press enter to continue.')
            shutil.rmtree(work_dir)
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

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

    # Get datasets and loaders
    train_dir = Path(config['dataset']) / 'train'
    val_dir = Path(config['dataset']) / 'val'

    if config['rotation']:
        transforms = A.Compose([
            A.RandomRotate90(always_apply=True)
        ], bbox_params=A.BboxParams(format='pascal_voc',
                                    label_fields=['labels']))
    else:
        transforms = None

    if config['dataset_type'] == 'region_dataset':
        dataset_cls = RegionDataset
    elif config['dataset_type'] == 'region_dataset_v2':
        dataset_cls = RegionDatasetV2
    else:
        raise ValueError(f'Unrecognized dataset {config["dataset_type"]}')

    # TODO добавить аргумент piece_transforms
    train_dset = dataset_cls(
        train_dir, **config['train_dataset_params'], device=device,
        transforms=transforms)
    val_dset = dataset_cls(
        val_dir, **config['val_dataset_params'], device=device)

    train_loader = DataLoader(train_dset,
                              batch_size=config['batch_size'],
                              shuffle=config['shuffle_train'],
                              collate_fn=dataset_cls.collate_func)
    val_loader = DataLoader(val_dset,
                            batch_size=config['batch_size'],
                            shuffle=config['shuffle_val'],
                            collate_fn=dataset_cls.collate_func)

    # Get the model
    if config['architecture'] == 'modified_retina':
        model = RetinaRegionLocalizer(
            img_min_size=config['train_dataset_params']['result_size'][0],
            img_max_size=config['train_dataset_params']['result_size'][0],
            pretrained=config['pretrained'],
            num_classes=config['num_classes'])
    elif config['architecture'] == 'modified_retina_v2':
        model = ModifiedRetinaV2.create_modified_retina(
            min_size=config['train_dataset_params']['result_size'][0],
            max_size=config['train_dataset_params']['result_size'][0],
            pretrained=config['pretrained'],
            num_classes=config['num_classes'])
    else:
        raise ValueError(f'Unrecognized architecture {config["architecture"]}')

    model.to(device=device)
    if model_params:
        model.load_state_dict(model_params)

    # Get the optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'],
                           weight_decay=config['weight_decay'])
    if optim_params:
        optimizer.load_state_dict(optim_params)

    # Get the scheduler
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['n_epoch'], eta_min=1e-6,
        last_epoch=start_ep - 1)
    if lr_params:
        lr_scheduler.load_state_dict(lr_params)
    
    # Get metrics
    train_iou_metric = IntersectionOverUnion().to(device=device)
    val_iou_metric = IntersectionOverUnion().to(device=device)
    train_cls_loss_metric = LossMetric().to(device=device)
    train_reg_loss_metric = LossMetric().to(device=device)
    val_cls_loss_metric = LossMetric().to(device=device)
    val_reg_loss_metric = LossMetric().to(device=device)

    # Do training
    best_metric = None
    for epoch in range(start_ep, config['n_epoch']):

        print(f'Epoch {epoch + 1}')

        # Train epoch
        model.train()
        for batch in tqdm(train_loader, 'Train step'):
            losses, predicts = model(*batch)
            loss = losses['classification'] + losses['bbox_regression']
            loss.backward()
    
            optimizer.step()
            optimizer.zero_grad()

            targets = batch[-1]
            # Collect only most confident bbox
            if sum(map(lambda x: x['scores'].numel(), predicts)) != 0:
                most_confident = []
                for i in range(len(predicts)):
                    if predicts[i]['boxes'].numel() != 0:
                        most_confident.append({
                            'boxes': predicts[i]['boxes'][0][None, ...],
                            'scores': predicts[i]['scores'][0][None, ...],
                            'labels': predicts[i]['labels'][0][None, ...]
                        })
                    else:
                        most_confident.append(predicts[i])
                train_iou_metric.update(most_confident, targets)
            else:
                train_iou_metric.update(predicts, targets)

            train_cls_loss_metric.update(losses['classification'])
            train_reg_loss_metric.update(losses['bbox_regression'])
        
        # Val epoch
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, 'Val step'):
                losses, predicts = model(*batch)
                targets = batch[-1]

                # Collect only most confident bbox
                if sum(map(lambda x: x['scores'].numel(), predicts)) != 0:
                    most_confident = []
                    for i in range(len(predicts)):
                        if predicts[i]['boxes'].numel() != 0:
                            most_confident.append({
                                'boxes': predicts[i]['boxes'][0][None, ...],
                                'scores': predicts[i]['scores'][0][None, ...],
                                'labels': predicts[i]['labels'][0][None, ...]
                            })
                        else:
                            most_confident.append(predicts[i])
                    val_iou_metric.update(most_confident, targets)
                else:
                    val_iou_metric.update(predicts, targets)

                val_cls_loss_metric.update(losses['classification'])
                val_reg_loss_metric.update(losses['bbox_regression'])

        # Lr scheduler
        lr_scheduler.step()
        lr = lr_scheduler.get_last_lr()[0]

        # Log epoch metrics
        train_cls_loss = train_cls_loss_metric.compute()
        train_reg_loss = train_reg_loss_metric.compute()
        train_iou = train_iou_metric.compute()
        val_cls_loss = val_cls_loss_metric.compute()
        val_reg_loss = val_reg_loss_metric.compute()
        val_iou = val_iou_metric.compute()

        train_cls_loss_metric.reset()
        train_reg_loss_metric.reset()
        train_iou_metric.reset()
        val_cls_loss_metric.reset()
        val_reg_loss_metric.reset()
        val_iou_metric.reset()

        log_writer.add_scalars('cls_loss', {
            'train': train_cls_loss,
            'val': val_cls_loss
        }, epoch)
        log_writer.add_scalars('reg_loss', {
            'train': train_reg_loss,
            'val': val_reg_loss
        }, epoch)
        log_writer.add_scalars('iou', {
            'train': train_iou['iou'],
            'val': val_iou['iou']
        }, epoch)
        log_writer.add_scalar('lr', lr, epoch)

        print('Train metrics:')
        print('ClsLoss:', train_cls_loss.item())
        print('RegLoss:', train_reg_loss.item())
        print('IoU:', train_iou['iou'].item())
        print('Val metrics:')
        print('ClsLoss:', val_cls_loss.item())
        print('RegLoss:', val_reg_loss.item())
        print('IoU:', val_iou['iou'].item())
        print('Lr:', lr)

        # Save model
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'epoch': epoch + 1
        }
        torch.save(checkpoint, ckpt_dir / 'last_checkpoint.pth')

        sum_loss = val_cls_loss.item() + val_reg_loss.item()
        if (best_metric is None or best_metric > sum_loss):
            torch.save(checkpoint, ckpt_dir / 'best_checkpoint.pth')
            best_metric = sum_loss

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

    if not args.config_pth.exists():
        raise FileNotFoundError('Specified config file does not exists.')
    return args


if __name__ == "__main__":
    args = parse_args()
    config_pth = args.config_pth
    main(config_pth=config_pth)
