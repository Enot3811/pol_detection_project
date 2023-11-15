"""Запустить yolo на тестовой выборке, посчитать метрики."""


from pathlib import Path
import sys
import argparse
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_convert
from torchmetrics.detection import MeanAveragePrecision

sys.path.append(str(Path(__file__).parents[3]))
from Yolov7.custom.model_utils import load_yolo_checkpoint
from Yolov7.custom.datasets import TankDetectionDataset
from Yolov7.yolov7.trainer import filter_eval_predictions
from Yolov7.yolov7.dataset import (
    Yolov7Dataset, create_yolov7_transforms, yolov7_collate_fn)


def main(test_dset_pth: Path, config_pth: Path, conf_thresh: float,
         iou_thresh: float):
    """Запустить yolo на тестовой выборке, посчитать метрики.

    Parameters
    ----------
    test_dset_pth : Path
        Путь к CVAT директории тестового датасета.
    config_pth : Path
        Путь к конфигу модели.
    conf_thresh : float
        Порог уверенности модели.
        Пропускать только те предсказания, уверенность которых выше порога.
    iou_thresh : float
        Порог перекрытия рамок.
        Пропускать только те предсказания, чей коэффициент перекрытия с другим
        более уверенным предсказанием этого же класса меньше порога.
    """
    # Read the config
    with open(config_pth, 'r') as f:
        config_str = f.read()
    config = json.loads(config_str)

    cls_id_to_name = {val: key for key, val in config['cls_to_id'].items()}
    num_classes = len(cls_id_to_name)

    # Prepare some stuff
    torch.manual_seed(config['random_seed'])

    if config['device'] == 'auto':
        device: torch.device = (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu'))
    else:
        device: torch.device = torch.device(config['device'])

    num_channels = 4 if config['polarization'] else 3
    pad_colour = (114,) * num_channels
    
    # Get the dataset and loader
    yolo_transforms = create_yolov7_transforms(
        (config['input_size'], config['input_size']), training=False,
        pad_colour=pad_colour)

    dset = TankDetectionDataset(
        test_dset_pth, name2index=config['cls_to_id'],
        polarization=config['polarization'])
    
    yolo_dset = Yolov7Dataset(dset, yolo_transforms)
    dloader = DataLoader(yolo_dset, collate_fn=yolov7_collate_fn)

    # Load the model
    weights_pth = Path(config['work_dir']) / 'ckpts' / 'best_checkpoint.pth'
    model = load_yolo_checkpoint(weights_pth, num_classes)
    model.to(device=device)
    model.eval()

    # Get the metrics
    map_metric = MeanAveragePrecision(iou_thresholds=[iou_thresh])

    # Evaluate model
    with torch.no_grad():
        for batch in tqdm(dloader, 'Evaluate model'):
            images, labels, img_names, img_sizes = batch
            images = images.to(device=device)
            labels = labels.to(device=device)
            fpn_heads_outputs = model(images)

            preds = model.postprocess(
                fpn_heads_outputs, multiple_labels_per_box=False)
            nms_predictions = filter_eval_predictions(
                preds,
                confidence_threshold=conf_thresh,
                nms_threshold=iou_thresh)

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
                
            map_metric.update(map_preds, map_targets)

        map_dict = map_metric.compute()
        map_metric.reset()
        print(*map_dict.items(), sep='\n')


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('test_dset_pth', type=Path,
                        help='Путь к тестовой CVAT выборке.')
    parser.add_argument('config_pth', type=Path,
                        help='Путь к конфигу модели.')
    parser.add_argument('--conf_thresh', type=float, default=0.3,
                        help='Порог уверенности модели.')
    parser.add_argument('--iou_thresh', type=float, default=0.2,
                        help='Порог перекрытия рамок.')
    args = parser.parse_args()

    if not args.test_dset_pth.exists():
        raise FileExistsError('Dataset dir does not exist.')
    if not args.config_pth.exists():
        raise FileExistsError('Config file does not exist.')
    return args


if __name__ == '__main__':
    args = parse_args()
    test_dset_pth = args.test_dset_pth
    config_pth = args.config_pth
    conf_thresh = args.conf_thresh
    iou_thresh = args.iou_thresh
    main(test_dset_pth, config_pth, conf_thresh, iou_thresh)
