"""Convert YOLO dataset to CVAT format.

YOLO format is:
dataset_dir
├── data.yaml
├── train
|   ├── images
|   |   ├── 0.jpg
|   |   ├── 1.jpg
|   |   ├── ...
|   |   ├── {num_sample - 1}.jpg
|   ├── labels
|   |   ├── 0.txt
|   |   ├── 1.txt
|   |   ├── ...
|   |   ├── {num_sample - 1}.txt
├── val (optional)
|   ├── images
|   |   ├── 0.jpg
|   |   ├── 1.jpg
|   |   ├── ...
|   |   ├── {num_sample - 1}.jpg
|   ├── labels
|   |   ├── 0.txt
|   |   ├── 1.txt
|   |   ├── ...
|   |   ├── {num_sample - 1}.txt
├── test (optional)
|   ├── images
|   |   ├── 0.jpg
|   |   ├── 1.jpg
|   |   ├── ...
|   |   ├── {num_sample - 1}.jpg
|   ├── labels
|   |   ├── 0.txt
|   |   ├── 1.txt
|   |   ├── ...
|   |   ├── {num_sample - 1}.txt

Where each txt corresponds to a jpg with the same name.
Txt file consists of lines like: `"cls_id cx cy h w"`, where `cls_id` is an int
that corresponds to some class and `"cxcywh"` is a bounding box of object.
Every value of bounding box is normalized relative to image shape.
"""


import argparse
from pathlib import Path
import sys
from typing import List, Tuple, Dict, Any
import shutil
import yaml

from tqdm import tqdm
from torch import tensor
from torchvision.ops import box_convert

sys.path.append(str(Path(__file__).parents[3]))
from utils.data_utils.cvat_functions import create_cvat_xml_from_dataset
from utils.data_utils.data_functions import read_volume


def main(yolo_pth: Path, save_dir: Path, copy_images: bool, verbose: bool):
    
    # Check existing
    if save_dir.exists():
        input(f'Specified directory "{str(save_dir)}" already exists. '
              'If continue, this directory will be deleted. '
              'Press enter to continue.')
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True)

    # Prepare helper paths
    yolo_images_pth = yolo_pth / 'images'
    yolo_labels_pth = yolo_pth / 'labels'
    yolo_yaml = yolo_pth / 'data.yaml'
    cvat_images_pth = save_dir / 'images'
    cvat_xml = save_dir / 'annotations.xml'
    if copy_images:
        cvat_images_pth.mkdir()

    # Read id to class dict
    with open(yolo_yaml) as f:
        id_to_cls = yaml.safe_load(f)['names']
    cls_names = list(id_to_cls.values())

    # Read yolo samples
    samples: List[Dict[str, Any]] = []
    iterator = list(yolo_labels_pth.glob('*.txt'))
    if verbose:
        iterator = tqdm(iterator, 'Convert annotations')
    for txt_pth in iterator:

        # Path, shape, image
        img_pth = yolo_images_pth / txt_pth.with_suffix('.jpg').name
        sample_img = read_volume(img_pth)
        shape = sample_img.shape[:2]
        if copy_images:
            # TODO проверить быстродействие
            dst_pth = cvat_images_pth / img_pth.name
            shutil.copy2(img_pth, dst_pth)

        # Labels, bboxes
        with open(txt_pth) as f:
            sample_targets = f.readlines()

            # Iterate over sample annotations
            labels: List[str] = []
            bboxes: List[Tuple[float, float, float, float]] = []
            for sample_target in sample_targets:
                sample_target = sample_target.split(' ')
                label = id_to_cls[int(sample_target[0])]
                bbox = tensor(list(map(float, sample_target[1:])))
                bbox[[1, 3]] *= shape[0]
                bbox[[0, 2]] *= shape[1]
                bbox = box_convert(bbox, 'cxcywh', 'xyxy').tolist()
                labels.append(label)
                bboxes.append(bbox)
        
        samples.append({
            'img_pth': img_pth,
            'shape': shape,
            'bboxes': bboxes,
            'labels': labels
        })
        
    create_cvat_xml_from_dataset(
        cvat_xml, cls_names, samples, 'train', verbose)


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'yolo_pth', type=Path,
        help='Path to YOLO dataset to convert.')
    parser.add_argument(
        'save_dir', type=Path,
        help='A path to save the converted CVAT dataset.')
    parser.add_argument(
        '--copy_images', action='store_true',
        help='Whether to copy dataset images.')
    parser.add_argument(
        '--verbose', action='store_true',
        help='Whether to show progress of converting.')
    args = parser.parse_args([
        'data/cars/VisDrone/VisDrone2019-DET-train',
        'data/cars/VisDrone/cvat/train',
        '--copy_images',
        '--verbose'
    ])
    
    if not args.yolo_pth.exists():
        raise FileNotFoundError(
            f'Dataset "{str(args.yolo_pth)}" does not exist.')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(yolo_pth=args.yolo_pth, save_dir=args.save_dir,
         copy_images=args.copy_images, verbose=args.verbose)
