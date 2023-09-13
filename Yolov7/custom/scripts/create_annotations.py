"""
Create csv annotation from txt dir.
"""


from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision.ops import box_convert
import torch


def main():
    dset_path = Path('/home/enot/projects/data/polarized_dataset')
    labels_dir = dset_path / 'Labels'
    images_dir = dset_path / 'Images'
    labels_files = list(labels_dir.glob('*.txt'))

    frame_dict = {
        'image': [],
        'xmin': [],
        'ymin': [],
        'xmax': [],
        'ymax': []
    }

    for label_file in tqdm(labels_files, 'Parse labels'):
        # Get bboxes description
        with open(label_file, 'r') as f:
            bboxes_dsc = f.readlines()

        # Get height and width of corresponding image
        image_name = label_file.name[:17]
        sh, sw = np.load(images_dir / image_name).shape
        h = sh // 2
        w = sw // 2

        for bbox_dsc in bboxes_dsc:
            relative_bbox = list(map(float, bbox_dsc.split()[1:]))

            relative_bbox = list(box_convert(
                torch.as_tensor(relative_bbox, dtype=torch.float32), "cxcywh", "xyxy"
            ).numpy())

            abs_bbox = [w * coord if i % 2 == 0 else h * coord
                        for i, coord in enumerate(relative_bbox)]
            
            frame_dict['image'].append(image_name)

            frame_dict['xmin'].append(abs_bbox[0])
            frame_dict['ymin'].append(abs_bbox[1])
            frame_dict['xmax'].append(abs_bbox[2])
            frame_dict['ymax'].append(abs_bbox[3])
            
    df = pd.DataFrame(frame_dict)
    df.to_csv(dset_path / 'annotations.csv', index=False)



if __name__ == '__main__':
    main()