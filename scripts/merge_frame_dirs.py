"""Объединить несколько съёмок в одну.

При нескольких съёмках создаётся ситуация, когда фреймы имеют
одинаковые названия. В этом случае просто слитие директорий не сработает.
"""

from pathlib import Path
import shutil
from typing import List


def main(source_dirs: List[str], dest_dir: str):
    source_dirs = list(map(lambda pth: Path(pth), source_dirs))
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    for source_dir in source_dirs:
        pths = list(sorted(source_dir.glob('*.npy'),
                           key=lambda pth: int(pth.name[4:-4])))
        subset_name = pths[0].name[:3]
        for pth in pths:
            shutil.copyfile(pth, dest_dir / f'{subset_name}_{idx}.npy')
            idx += 1


if __name__ == '__main__':
    # Пути к директориям с фреймами съёмок
    source_dirs = [
        '/home/pc0/projects/mako_camera/data/images_rgb_2023-09-09 15_07_13.973400',
        '/home/pc0/projects/mako_camera/data/images_rgb_2023-09-09 14_30_18.380607'
    ]
    # Путь к папке слияния.
    dest_dir = '/home/pc0/projects/mako_camera/data/2023_09_09/bayer_rg8'
    main(source_dirs, dest_dir)
