"""Script that merge several datasets in YOLO format into one.

YOLO format represented as:
dset_dir
├── 0.jpg
├── 0.txt
├── 1.jpg
├── 1.txt
├── ...
├── {num_sample - 1}.jpg
├── {num_sample - 1}.txt

when txt contain objects int labels and bboxes in cxcywh format in 0-1 range.
"""

from typing import List
from pathlib import Path
import argparse
import sys
import shutil

sys.path.append(str(Path(__file__).parents[2]))
from utils.data_utils.data_functions import collect_paths
from utils.data_utils.data_functions import IMAGE_EXTENSIONS


def main(dsets_pths: List[Path], dst_pth: Path):
    
    for dset_pth in dsets_pths:
        img_pths = collect_paths(dset_pth, IMAGE_EXTENSIONS)
        try:
            # Try to sort by index in name
            img_pths = sorted(img_pths, key=lambda pth: int(pth.name[:-4]))
        finally:
            dst_pth.mkdir(parents=True, exist_ok=True)
            for img_pth in img_pths:
                txt_pth = img_pth.with_name(img_pth.name[:-4] + '.txt')

                img_name = dset_pth.name + '_' + img_pth.name
                txt_name = dset_pth.name + '_' + txt_pth.name
                shutil.copyfile(img_pth, dst_pth / img_name)
                shutil.copyfile(txt_pth, dst_pth / txt_name)


def parse_args() -> argparse.Namespace:
    """Parse arguments.

    Returns
    -------
    argparse.Namespace
        Passed arguments.

    Raises
    ------
    FileNotFoundError
        Raise when any given dataset dir does not exist.
    """
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        'dsets_pths', type=Path, nargs='+',
        help='Paths to YOLO datasets to merge.')
    parser.add_argument(
        'dst_pth', type=Path,
        help='Path to save merged dataset.')
    args = parser.parse_args()

    for pth in args.dsets_pths:
        if not pth.exists():
            raise FileNotFoundError(
                f'Given dataset dir "{str(pth)}" does not exist.')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(dsets_pths=args.dsets_pths, dst_pth=args.dst_pth)
