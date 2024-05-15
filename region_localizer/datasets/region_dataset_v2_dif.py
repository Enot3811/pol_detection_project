from pathlib import Path
from typing import Dict, List, Any
import sys

import torch
from torch import Tensor
from albumentations import BboxParams
import numpy as np

sys.path.append(str(Path(__file__).parents[2]))
from utils.torch_utils.torch_functions import random_crop
from region_localizer.datasets.region_dataset_v2 import RegionDatasetV2
from utils.image_utils.image_functions import read_image, collect_images_paths


class RegionDatasetV2Dif(RegionDatasetV2):
    """The same `RegionDatasetV2` but with two different images for map
    and for crops.

    First image must be named as "{sample_name}_map.{ext}"
    and second as "{sample_name}_crop.{ext}."
    """

    def _collect_samples(self, dset_pth: Path) -> List[Path]:
        """Collect sample dirs.

        Each dir contain a pair of images: first as a map and second for
        crops making.

        Parameters
        ----------
        dset_pth : Path
            Dataset path.

        Returns
        -------
        List[Path]
            List of sample directories.
        """
        return [sample_dir for sample_dir in dset_pth.iterdir()]
    
    def get_sample(self, index: int) -> Dict[str, Any]:
        """Get sample from region dataset.

        Get map image and cropping image, make `num_crops` crops
        from second one and pack it to `dict`.

        Parameters
        ----------
        index : int
            An index of sample.

        Returns
        -------
        Dict[str, Any]
            Sample dict with keys:
            - "map_img" - `uint8 NDArray` with shape `(*result_size, 3)`
            - "pieces_imgs" - `list of uint8 NDArray` with shapes
            `(*result_size, 3)`
            - "bboxes" - `list of FloatBbox`.
        """
        pths = collect_images_paths(
            self.samples[index],
            image_extensions=('jpg', 'JPG', 'png', 'PNG', 'bmp', 'npy'))
        # If name of image contain _reg suffix
        map_pth = next(filter(
            lambda pth: pth.name.split('.')[-2][-4::1] == '_map', pths))
        # If name of image contain _crop suffix
        crop_pth = next(filter(
            lambda pth: pth.name.split('.')[-2][-5::1] == '_crop', pths))

        if map_pth.name.split('.')[-1] == 'npy':
            map_img = np.load(map_pth)
        else:
            map_img = read_image(map_pth)
        if crop_pth.name.split('.')[-1] == 'npy':
            crop_img = np.load(crop_pth)
        else:
            crop_img = read_image(crop_pth)

        bboxes = []
        pieces_imgs = []
        for _ in range(self.num_crops):
            piece_img, bbox = random_crop(
                crop_img, self.min_crop_size, self.max_crop_size,
                return_position=True)
            bboxes.append(bbox)
            piece_img = self.resize_transf(
                image=piece_img, bboxes=[], labels=[])['image']
            pieces_imgs.append(piece_img)

        resized = self.resize_transf(
            image=map_img, bboxes=bboxes, labels=[0] * self.num_crops)
        map_img = resized['image']
        bboxes = resized['bboxes']
        return {
            'map_img': map_img, 'pieces_imgs': pieces_imgs, 'bboxes': bboxes}
    

if __name__ == '__main__':
    import albumentations as A
    from torch.utils.data import DataLoader
    from utils.image_utils.image_functions import show_images_cv2
    from utils.torch_utils.torch_functions import (
        image_tensor_to_numpy, draw_bounding_boxes)

    # Overall params
    image_dir = ('data/rastr_osm_dataset/train')
    b_size = 2
    num_crops = 4
    device = torch.device('cuda')
    # Dataset sizes params
    img_size = (2464, 2464)
    result_size = (900, 900)
    max_crop_size = int(img_size[0] * 0.8)
    min_crop_size = int(img_size[0] * 0.5)
    rectangle = False
    # Transforms params
    rotate = False
    blur = False
    # Piece transforms
    piece_blur = True
    piece_color_jitter = True

    if rectangle:
        min_crop_size = (min_crop_size, min_crop_size)
        max_crop_size = (max_crop_size, max_crop_size)

    if rotate or blur:
        transforms = []
        if rotate:
            transforms.append(A.RandomRotate90(always_apply=True))
        if blur:
            transforms.append(A.Blur(blur_limit=7, p=1.0))
        transforms = A.Compose(
            transforms,
            bbox_params=BboxParams(
                format='pascal_voc', label_fields=['labels']))
    else:
        transforms = None

    if piece_blur or piece_color_jitter:
        piece_transforms = []
        if piece_color_jitter:
            piece_transforms.append(A.ColorJitter(
                brightness=(0.4, 1.3), contrast=(0.7, 1.2),
                saturation=(0.5, 1.4), hue=(-0.01, 0.01), p=1.0))
        if piece_blur:
            piece_transforms.append(A.Blur(blur_limit=3, p=1.0))
        piece_transforms = A.Compose(
            piece_transforms,
            bbox_params=BboxParams(
                format='pascal_voc', label_fields=['labels']))
    else:
        piece_transforms = None

    # Get dataset and dloader
    dset = RegionDatasetV2Dif(
        image_dir, min_crop_size, max_crop_size, result_size, num_crops,
        device, transforms, piece_transforms)
    dloader = DataLoader(
        dset, batch_size=b_size, collate_fn=RegionDatasetV2Dif.collate_func)
    
    # Iterate over dloader
    for batch in dloader:
        for i in range(num_crops):
            stacked_img = batch[0][i]
            target = batch[1][i]

            if isinstance(stacked_img, Tensor):
                stacked_img = image_tensor_to_numpy(
                    (stacked_img * 255).to(dtype=torch.uint8))
                bbox = target['boxes'].squeeze().tolist()
            else:
                bbox = target['boxes']

            map_img = stacked_img[..., :3]
            region_img = stacked_img[..., 3:]
            boxes_img = draw_bounding_boxes(map_img, [bbox])

            print('Bbox:', bbox, 'h_size:', bbox[3] - bbox[1],
                  'w_size:', bbox[2] - bbox[0])

            key = show_images_cv2(
                [boxes_img, region_img],
                ['region', 'piece'],
                destroy_windows=False)
            if key == 27:
                exit()