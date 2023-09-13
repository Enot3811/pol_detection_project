"""Объединить несколько съёмок в одну.

При нескольких съёмках создаётся ситуация, когда фреймы имеют
одинаковые названия. В этом случае просто слитие директорий не сработает.
"""

from pathlib import Path
import shutil


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    idx = 0
    for source_dir in SOURCE_DIRS:
        pths = list(sorted(source_dir.glob('*.npy'),
                           key=lambda pth: int(pth.name[4:-4])))
        subset_name = pths[0].name[:3]
        for pth in pths:
            shutil.copyfile(pth, DEST_DIR / f'{subset_name}_{idx}.npy')
            idx += 1


if __name__ == '__main__':
    # Пути к директориям с фреймами съёмок
    SOURCE_DIRS = [
        Path('path/to/stream/dir1'),
        Path('path/to/stream/dir2')
    ]
    # Путь к папке слияния.
    DEST_DIR = Path('pth/to/save/dir')
    main()
