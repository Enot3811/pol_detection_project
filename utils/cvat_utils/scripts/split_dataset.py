"""Merge several object detection CVAT datasets into one."""


import argparse
from pathlib import Path
import sys
from typing import List, Set
import shutil
import random
from math import ceil

sys.path.append(str(Path(__file__).parents[3]))
from utils.cvat_utils.cvat_datasets import CvatObjectDetectionDataset
from utils.data_utils.datasets import (
    BaseObjectDetectionAnnotation, BaseObjectDetectionSample)
from utils.cvat_utils.cvat_functions import create_cvat_object_detection_xml


def main(**kwargs):
    dset_dir = kwargs['dataset_dir']
    save_dir = kwargs['save_dir']
    proportions = kwargs['proportions']
    random_seed = kwargs['random_seed']

    images_dir = dset_dir / 'images'

    # Collect all samples and class labels
    samples: List[BaseObjectDetectionSample] = []
    set_classes: Set[str] = set()

    dset = CvatObjectDetectionDataset(dset_dir)
    for sample in dset:
        img_name = sample['name']
        classes = sample['labels']
        bboxes = sample['bboxes']

        img_annots: List[BaseObjectDetectionAnnotation] = []
        for cls, bbox in zip(classes, bboxes):
            img_annots.append(BaseObjectDetectionAnnotation(
                bbox[0], bbox[1], bbox[2], bbox[3], cls))
            set_classes.add(cls)

        img_pth = images_dir / img_name
        samples.append(BaseObjectDetectionSample(img_pth, img_annots))
    set_classes = list(set_classes)

    # Shuffle and split
    random.seed(random_seed)
    random.shuffle(samples)

    st_idx = 0
    for i, proportion in enumerate(proportions):
        n_samples = ceil(len(samples) * proportion)
        subset_samples = samples[st_idx:st_idx + n_samples]
        st_idx += n_samples
        subset_name = f'{save_dir.name}_{i}'
        subset_dir = save_dir / subset_name
        subset_images_dir = subset_dir / 'images'
        subset_annots_pth = subset_dir / 'annotations.xml'
        subset_images_dir.mkdir(parents=True, exist_ok=True)

        # Create CVAT xml
        create_cvat_object_detection_xml(
            subset_annots_pth, subset_samples, subset_name,
            set_classes, verbose=True)

        # Copy images
        for sample in subset_samples:
            src_pth = sample.get_image_path()
            dst_pth = subset_images_dir / src_pth.name
            shutil.copy2(src_pth, dst_pth)


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'dataset_dir', type=Path,
        help='Paths to CVAT dataset to split.')
    parser.add_argument(
        'save_dir', type=Path,
        help='A path to save the split CVAT datasets.')
    parser.add_argument(
        'proportions', type=float, nargs='+',
        help='Float proportions for split.')
    parser.add_argument(
        '--random_seed', type=int, default=42,
        help='A random seed for split.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    dataset_dir = args.dataset_dir
    save_dir = args.save_dir
    proportions = args.proportions
    random_seed = args.random_seed
    main(dataset_dir=dataset_dir,
         save_dir=save_dir,
         proportions=proportions,
         random_seed=random_seed)
