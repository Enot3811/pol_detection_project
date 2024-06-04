"""Split CVAT object detection dataset into several subsets."""


import argparse
from pathlib import Path
import sys
from typing import List
import shutil
import random
from math import ceil

from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[3]))
from utils.torch_utils.datasets import CvatDetectionDataset
from utils.data_utils.cvat_functions import create_cvat_xml_from_dataset


def main(
    dataset_path: Path, save_dir: Path, proportions: List[float],
    random_seed: int, verbose: bool
):
    if save_dir.exists():
        input(f'Specified directory "{str(save_dir)}" already exists. '
              'If continue, this directory will be deleted. '
              'Press enter to continue.')
        shutil.rmtree(save_dir)
    save_dir.mkdir()

    # Get samples and classes
    dset = CvatDetectionDataset(dataset_path)
    samples = dset.get_samples_annotations()
    cls_names = list(dset.get_class_to_index().keys())

    # Shuffle samples
    random.seed(random_seed)
    random.shuffle(samples)

    # Iterate over proportions to cut dataset
    st_idx = 0
    for i, proportion in enumerate(proportions):
        n_samples = ceil(len(samples) * proportion)
        subset_samples = samples[st_idx:st_idx + n_samples]
        st_idx += n_samples
        subset_name = f'{save_dir.name}_{i}'
        subset_dir = save_dir / subset_name
        subset_images_dir = subset_dir / 'images'
        subset_annots_pth = subset_dir / 'annotations.xml'
        subset_images_dir.mkdir(parents=True)

        # Create CVAT xml
        create_cvat_xml_from_dataset(
            subset_annots_pth, cls_names, subset_samples, subset_name,
            verbose=verbose)

        # Copy images
        if verbose:
            subset_samples = tqdm(
                subset_samples, f'Copy images for {subset_name}')
        for sample in subset_samples:
            src_pth = sample['img_pth']
            dst_pth = subset_images_dir / sample['img_pth'].name
            shutil.copy2(src_pth, dst_pth)


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'dataset_path', type=Path,
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
    parser.add_argument(
        '--verbose', action='store_true',
        help='Whether to show progress of splitting.')
    args = parser.parse_args()
    if not args.dataset_path.exists():
        raise FileNotFoundError(
            f'Given dataset "{str(args.dataset_path)}" does not exist.')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(dataset_path=args.dataset_path,
         save_dir=args.save_dir,
         proportions=args.proportions,
         random_seed=args.random_seed,
         verbose=args.verbose)
