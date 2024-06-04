"""Merge several object detection CVAT datasets into one."""


import argparse
from pathlib import Path
import sys
from typing import List
import shutil

from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[3]))
from utils.torch_utils.datasets import CvatDetectionDataset
from utils.data_utils.cvat_functions import create_cvat_xml_from_dataset


def main(datasets_paths: List[Path], save_dir: Path, verbose: bool):
    
    union_images_pth = save_dir / 'images'
    union_annots_pth = save_dir / 'annotations.xml'
    if save_dir.exists():
        input(f'Specified directory "{str(save_dir)}" already exists. '
              'If continue, this directory will be deleted. '
              'Press enter to continue.')
        shutil.rmtree(save_dir)
    union_images_pth.mkdir(parents=True)

    union_cls_names = set()
    union_dset_len = 0
    union_dset_samples = []
    for dset_pth in datasets_paths:
        dset = CvatDetectionDataset(dset_pth)
        # Get cls names and len for meta
        union_cls_names.union(dset.get_class_to_index().keys())
        union_dset_len += len(dset)

        # Get samples and:
        # 1) Copy images to new union directory
        # 2) Change "img_pth" attribute
        # 3) Concatenate changed samples for new union dataset
        samples = dset._samples
        iterator = range(len(samples))
        if verbose:
            iterator = tqdm(
                iterator, f'Iterate over "{dset_pth.name}" dataset')
        for i in iterator:
            old_pth = samples[i]['img_pth']
            # New name is source dset name + img name
            new_name = '_'.join((old_pth.parents[1].name, old_pth.name))
            new_pth = union_images_pth / new_name
            shutil.copy2(old_pth, new_pth)
            samples[i]['img_pth'] = new_pth
        union_dset_samples += samples

    # And then put it to xml creator function
    create_cvat_xml_from_dataset(
        union_annots_pth, list(union_cls_names), union_dset_samples, 'merged',
        verbose)


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'datasets_paths', type=Path, nargs='+',
        help='Paths to CVAT datasets to merge.')
    parser.add_argument(
        'save_dir', type=Path,
        help='A path to save the merged CVAT dataset.')
    parser.add_argument(
        '--verbose', action='store_true',
        help='Whether to show progress of merging.')
    args = parser.parse_args()
    
    for pth in args.datasets_paths:
        if not pth.exists():
            raise FileNotFoundError(f'Dataset "{str(pth)}" does not exist.')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(datasets_paths=args.datasets_paths, save_dir=args.save_dir,
         verbose=args.verbose)
