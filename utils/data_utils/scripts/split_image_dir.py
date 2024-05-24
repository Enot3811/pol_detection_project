"""Split images set from given dir into several subsets."""


import argparse
from pathlib import Path
import sys
from typing import List
import shutil
import random
from math import ceil

sys.path.append(str(Path(__file__).parents[3]))
from utils.data_utils.data_functions import IMAGE_EXTENSIONS
from utils.data_utils.data_functions import collect_paths


def main(
    images_dir: Path, save_dir: Path, proportions: List[float],
    random_seed: int
):
    img_pths = collect_paths(images_dir, IMAGE_EXTENSIONS)

    # Shuffle and split
    random.seed(random_seed)
    random.shuffle(img_pths)

    st_idx = 0
    for i, proportion in enumerate(proportions):
        n_samples = ceil(len(img_pths) * proportion)
        subset_pths = img_pths[st_idx:st_idx + n_samples]
        st_idx += n_samples
        subset_name = f'{save_dir.name}_{i}'
        subset_dir = save_dir / subset_name
        subset_dir.mkdir(parents=True, exist_ok=True)

        # Copy images
        for src_pth in subset_pths:
            dst_pth = subset_dir / src_pth.name
            shutil.copy2(src_pth, dst_pth)


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'images_dir', type=Path,
        help='Paths to images dir to split.')
    parser.add_argument(
        'save_dir', type=Path,
        help='A path to save the split images subdirectories.')
    parser.add_argument(
        'proportions', type=float, nargs='+',
        help='Float proportions for split.')
    parser.add_argument(
        '--random_seed', type=int, default=42,
        help='A random seed for split.')
    args = parser.parse_args([
        'data/satellite_dataset/source_images',
        'data/satellite_dataset/dataset',
        '0.85', '0.15'])
    return args


if __name__ == '__main__':
    args = parse_args()
    images_dir = args.images_dir
    save_dir = args.save_dir
    proportions = args.proportions
    random_seed = args.random_seed
    main(images_dir=images_dir,
         save_dir=save_dir,
         proportions=proportions,
         random_seed=random_seed)
