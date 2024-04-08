"""Add indents to json file.

Often json files are written in one line without indents. And those file
are hardly readable. This script rewrite passed file with normal json indents.
"""

from typing import Optional
import sys
from pathlib import Path
import json
import argparse

sys.path.append(str(Path(__file__).parents[2]))
from utils.argparse_utils import natural_int


def main(
    src_json: Path, dst_json: Optional[Path] = None, indent_size: int = 4
):
    with open(src_json) as f:
        raw_json = f.read()
    json_dict = json.loads(raw_json)
    raw_json = json.dumps(json_dict, indent=indent_size)
    if dst_json is None:
        file_name = src_json.name.split('.')[0] + '_indented.json'
        dst_json = src_json.parent / file_name
    if dst_json.exists():
        input(f'File at the saving path "{str(dst_json)}" already exists.'
              'If continue then it will be rewritten. '
              'Press "enter" to continue.')
    with open(dst_json, 'w') as f:
        f.write(raw_json)


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'src_json', type=Path, help='A path to an original json file.')
    parser.add_argument(
        '--dst_json', type=Path, default=None,
        help=('A path to save new json file. '
              'If not given then file will be saved nearby the original one.'))
    parser.add_argument(
        '--indent_size', type=natural_int, default=4,
        help='A number of spaces in one indent.')
    args = parser.parse_args()

    if not args.src_json.exists():
        raise FileNotFoundError(
            f'Given json file "{str(args.src_json)}" does not exist.')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(src_json=args.src_json, dst_json=args.dst_json,
         indent_size=args.indent_size)
