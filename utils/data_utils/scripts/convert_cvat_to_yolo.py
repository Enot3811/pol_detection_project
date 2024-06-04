"""Convert a CVAT dataset to the YOLO format.

The original dataset must have structure:
cvat_contain_dir
├── train
|   ├── images
|   |   ├── 0.jpg
|   |   ├── 1.jpg
|   |   ├── ...
|   |   ├── {num_sample - 1}.jpg
|   ├── annotations.xml
├── val (optional)
|   ├── images
|   |   ├── 0.jpg
|   |   ├── 1.jpg
|   |   ├── ...
|   |   ├── {num_sample - 1}.jpg
|   ├── annotations.xml
├── test (optional)
|   ├── images
|   |   ├── 0.jpg
|   |   ├── 1.jpg
|   |   ├── ...
|   |   ├── {num_sample - 1}.jpg
|   ├── annotations.xml

and it is converted to:
save_dir
├── data.yaml
├── train
|   ├── images
|   |   ├── 0.jpg
|   |   ├── 1.jpg
|   |   ├── ...
|   |   ├── {num_sample - 1}.jpg
|   ├── labels
|   |   ├── 0.txt
|   |   ├── 1.txt
|   |   ├── ...
|   |   ├── {num_sample - 1}.txt
├── val (optional)
|   ├── images
|   |   ├── 0.jpg
|   |   ├── 1.jpg
|   |   ├── ...
|   |   ├── {num_sample - 1}.jpg
|   ├── labels
|   |   ├── 0.txt
|   |   ├── 1.txt
|   |   ├── ...
|   |   ├── {num_sample - 1}.txt
├── test (optional)
|   ├── images
|   |   ├── 0.jpg
|   |   ├── 1.jpg
|   |   ├── ...
|   |   ├── {num_sample - 1}.jpg
|   ├── labels
|   |   ├── 0.txt
|   |   ├── 1.txt
|   |   ├── ...
|   |   ├── {num_sample - 1}.txt

Where each txt corresponds to a jpg with the same name.
Txt file consists of lines like: `"cls_id cx cy h w"`, where `cls_id` is an int
that corresponds to some class and `"cxcywh"` is a bounding box of object.
Every value of bounding box is normalized relative to image shape.
"""


import argparse
from pathlib import Path
import sys
import shutil
import yaml

from tqdm import tqdm
import torch
from torchvision.ops import box_convert

sys.path.append(str(Path(__file__).parents[3]))
from utils.torch_utils.datasets import CvatDetectionDataset


def main(cvat_contain_dir: Path, save_dir: Path, copy_images: bool = False):
    if save_dir.exists():
        input(f'Given save_dir "{str(save_dir)}" already exists. '
              'All data will be deleted if continue. Press enter to continue.')
        shutil.rmtree(save_dir)
    save_dir.mkdir()

    train_dset = cvat_contain_dir / 'train'
    val_dset = cvat_contain_dir / 'val'
    test_dset = cvat_contain_dir / 'test'

    if not train_dset.exists():
        raise FileNotFoundError(
            'There is no a train set in the defined directory.')
    dsets = [train_dset]
    if val_dset.exists():
        dsets.append(val_dset)
    if test_dset.exists():
        dsets.append(test_dset)

    # Iterate over all subsets
    for i, orig_dset_pth in enumerate(dsets):
        save_dset_pth = save_dir / orig_dset_pth.name
        labels_dir = save_dset_pth / 'labels'
        labels_dir.mkdir(parents=True, exist_ok=True)
        if copy_images:
            images_dir = save_dset_pth / 'images'
            images_dir.mkdir(parents=True, exist_ok=True)

        dset = CvatDetectionDataset(orig_dset_pth)
        if i == 0:
            cls_to_id = dset.get_class_to_index()

        # Create txt for every sample and copy image if needed
        samples = dset.get_samples_annotations()
        desc = f'Convert {orig_dset_pth.name} set'
        for sample in tqdm(samples, desc=desc):
            txt_pth = labels_dir / sample['img_pth'].with_suffix('.txt').name
            # If there are some objects
            if len(sample['labels']) > 0:
                # Read classes of sample
                classes = torch.tensor(
                    list(map(lambda cls_name: cls_to_id[cls_name],
                             sample['labels']))
                )[:, None]
                # Read bboxes of sample
                xyxy = torch.tensor(sample['bboxes'])
                cxcywh = box_convert(xyxy, 'xyxy', 'cxcywh')
                # normalized height and width 0-1
                cxcywh[:, [1, 3]] /= sample['shape'][0]
                cxcywh[:, [0, 2]] /= sample['shape'][1]
                img_labels = torch.hstack((classes, cxcywh))
                img_labels = img_labels.tolist()
                str_labels = []
                for img_label in img_labels:
                    img_label[0] = int(img_label[0])
                    str_labels.append(' '.join(map(str, img_label)) + '\n')
                with open(txt_pth, 'w') as f:
                    f.writelines(str_labels)
            # If there are no objects on image
            else:
                # Make an empty file
                with open(txt_pth, 'w') as f:
                    pass

            if copy_images:
                orig_img_pth = sample['img_pth']
                dst_img_pth = images_dir / sample['img_pth'].name
                shutil.copy2(orig_img_pth, dst_img_pth)

    # Create yaml
    yaml_list = [('train', '../train/images')]
    if val_dset.exists():
        yaml_list.append(('val', '../val/images'))
    if test_dset.exists():
        yaml_list.append(('test', '../test/images'))

    yaml_list.append(('nc', len(cls_to_id)))
    yaml_list.append(('names', list(cls_to_id.keys())))
    
    yaml_dict = dict(yaml_list)

    with open(save_dir / 'data.yaml', 'w') as f:
        yaml.dump(yaml_dict, f)


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'cvat_contain_dir', type=Path,
        help='A path to a dir that contain source CVAT datasets.')
    parser.add_argument(
        'save_dir', type=Path,
        help='A path to save the converted datasets.')
    parser.add_argument(
        '--copy_images', action='store_true',
        help='Whether to copy dataset images.')
    args = parser.parse_args()

    if not args.cvat_contain_dir.exists():
        raise FileNotFoundError(
            f'Given "{str(args.cvat_contain_dir)}" dir does not exist.')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(cvat_contain_dir=args.cvat_contain_dir,
         save_dir=args.save_dir,
         copy_images=args.copy_images)
