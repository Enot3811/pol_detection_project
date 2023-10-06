"""Скрипт для создания csv аннотаций для temporary train."""

from pathlib import Path
import sys
import csv

import tqdm

sys.path.append(str(Path(__file__).parents[3]))
from Yolov7.custom.datasets import TankDetectionDataset


def write_csv_from_dataset(dset_pth: Path):
    """Транслировать аннотации TankDetectionDataset в csv формат из блога."""
    dset = TankDetectionDataset(dset_pth)
    csv_frame = [['image', 'xmin', 'ymin', 'xmax', 'ymax']]
    for sample in tqdm(dset):
        image, bboxes, classes, img_id, shape = sample
        img_name = dset.img_id_to_name[img_id]
        for i in range(bboxes.shape[0]):
            csv_frame.append(
                [img_name, bboxes[i, 0], bboxes[i, 1],
                 bboxes[i, 2], bboxes[i, 3]])
        
    csv_pth = dset_pth / (dset_pth.name + '.csv')
    with open(csv_pth, 'w') as f:
        writer = csv.writer(f)
        for row in csv_frame:
            writer.writerow(row)


def main(**kwargs):
    dset_dir = kwargs['dset_dir']
    write_csv_from_dataset(dset_dir)


if __name__ == '__main__':
    dset_dir = Path('data/union_tank_dataset')
    main(dset_dir=dset_dir)
