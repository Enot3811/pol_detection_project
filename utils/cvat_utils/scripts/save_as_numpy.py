"""Save all images as .npy files and change CVAT annotations."""


import argparse
from pathlib import Path
import sys
from typing import List

import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[3]))
from utils.cvat_utils.cvat_datasets import CvatObjectDetectionDataset
from utils.data_utils.datasets import (
    BaseObjectDetectionAnnotation, BaseObjectDetectionSample)
from utils.cvat_utils.cvat_functions import create_cvat_object_detection_xml
from utils.image_utils.image_functions import read_image


class TempSample(BaseObjectDetectionSample):
    """Temporary sample class with additional path to save image."""
    def __init__(
        self,
        img_pth: Path,
        img_annots: List[BaseObjectDetectionAnnotation],
        new_pth: Path
    ):
        super().__init__(img_pth, img_annots)
        self.new_pth = new_pth


def main(**kwargs):
    dset_dir = kwargs['dataset_dir']
    save_dir = kwargs['save_dir']

    new_images_pth = save_dir / 'images'
    new_annots_pth = save_dir / 'annotations.xml'

    # Recollect all samples
    samples: List[TempSample] = []

    img_dir = dset_dir / 'images'
    dset = CvatObjectDetectionDataset(dset_dir)
    for sample in dset:
        img_name = sample['name']
        classes = sample['labels']
        bboxes = sample['bboxes']

        img_annots: List[BaseObjectDetectionAnnotation] = []
        for cls, bbox in zip(classes, bboxes):
            img_annots.append(BaseObjectDetectionAnnotation(
                bbox[0], bbox[1], bbox[2], bbox[3], cls))

        img_pth = img_dir / img_name
        img_name_without_ext = img_name.split('.')[0]
        new_img_pth = new_images_pth / f'{img_name_without_ext}.npy'
        samples.append(TempSample(img_pth, img_annots, new_img_pth))

    # Create CVAT xml
    new_images_pth.mkdir(parents=True, exist_ok=True)
    create_cvat_object_detection_xml(
        new_annots_pth, samples, 'train', dset.get_labels(), verbose=True)
    # Convert images
    for sample in tqdm(samples, 'Images converting'):
        src_pth = sample.get_image_path()
        dst_pth = sample.new_pth
        np.save(dst_pth, read_image(src_pth))


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'dataset_dir', type=Path,
        help='Paths to CVAT dataset to convert.')
    parser.add_argument(
        'save_dir', type=Path,
        help='Path to save converted CVAT dataset.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    dataset_dir = args.dataset_dir
    save_dir = args.save_dir
    main(dataset_dir=dataset_dir,
         save_dir=save_dir)
