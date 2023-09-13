"""
Get only images that have labels.
"""


from pathlib import Path
import shutil

from tqdm import tqdm


def main():
    dset_path = Path('/home/enot/projects/data/polarized_dataset/')

    # Get images glob
    images_path = dset_path / 'Raw_images'
    images = list(images_path.rglob('*.npy'))
    images.sort()
    print(len(images))

    # Get labels glob
    labels_path = dset_path / 'Labels'
    labels = list(labels_path.rglob('*.txt'))
    labels.sort()
    print(len(labels))

    names = list(map(lambda path: path.name[:17], labels))
    parsed_images_dir = dset_path / 'Images'
    for image_path in tqdm(images, 'Parse images'):
        if image_path.name in names:
            shutil.copy(str(image_path), str(parsed_images_dir / image_path.name))


if __name__ == '__main__':
    main()