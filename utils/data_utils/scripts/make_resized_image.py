"""Load passed image, resize it to defined shape and save it."""


from typing import Optional, Tuple
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[3]))
from utils.data_utils.data_functions import (
    read_image, save_image, resize_image)
from utils.argparse_utils import natural_int


def main(
    image_pth: Path,
    new_size: Tuple[int, int],
    save_pth: Optional[Path] = None
):
    image = read_image(image_pth)
    image = resize_image(image, new_size)
    if save_pth is None:
        file_name, file_ext = image_pth.name.split('.')
        file_name += '_resized.' + file_ext
        save_pth = image_pth.parent / file_name
    save_image(image, save_pth)


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'image_pth', type=Path,
        help='A path to an image to process.')
    parser.add_argument(
        'new_size', type=natural_int, nargs=2,
        help='New image size.')
    parser.add_argument(
        '--save_pth', type=Path, default=None,
        help=('A path to save the resized image. '
              'If not given then file will be saved nearby the original one.'))
    
    args = parser.parse_args()

    args.new_size = tuple(args.new_size)
    if not args.image_pth.exists():
        raise FileNotFoundError(
            f'Given image "{str(args.src_json)}" does not exist.')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(image_pth=args.image_pth,
         new_size=args.new_size,
         save_pth=args.save_pth)
