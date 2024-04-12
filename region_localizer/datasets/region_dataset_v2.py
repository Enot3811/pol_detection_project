from pathlib import Path
from typing import Tuple, Dict, List, Any
import sys
from functools import reduce

import numpy as np
import torch
from torch import Tensor, FloatTensor
from albumentations import BboxParams

sys.path.append(str(Path(__file__).parents[2]))
from utils.torch_utils.torch_functions import image_numpy_to_tensor
from region_localizer.datasets.region_dataset import RegionDataset


class RegionDatasetV2(RegionDataset):
    """Dataset for region localizer v2.
    
    Each sample represented as a map image stacked with cropped piece
    from this map and bounding box of this piece.
    The map image and piece image may be processed with some augmentations.
    Pieces can be processed separately with `piece_transforms` argument.
    At the end both these images will be resized to the same `result_size`.
    """

    def postprocess_sample(
        self, sample: Dict[str, Any]
    ) -> Tuple[List[FloatTensor], List[Dict[str, Tensor]]]:
        """Convert to tensors and stack map images with cropped pieces.

        Parameters
        ----------
        sample : Dict[str, Any]
            Sample dict with keys:
            - "map_img" - `uint8 NDArray` with shape `(*result_size, 3)`
            - "pieces_imgs" - `list of uint8 NDArray` with shapes
            `(*result_size, 3)`
            - "bboxes" - `list of FloatBbox`.

        Returns
        -------
        Tuple[List[FloatTensor], List[Dict[str, Tensor]]]
            - Maps stacked with pieces with shapes `(6, *result_size)`
            - Targets dict with "boxes" and "labels" tensors.
        """
        map_img = sample['map_img']
        pieces_imgs = sample['pieces_imgs']
        bboxes = sample['bboxes']
        result_vols: List[FloatTensor] = []

        # Convert to tensors
        map_img = image_numpy_to_tensor(
            map_img.astype(np.float32) / 255, device=self.device)
        
        targets = []
        for i in range(self.num_crops):
            # Iterate over List[ndarray], convert, normalize and concatenate
            # each piece with map
            result_vols.append(torch.cat((map_img, image_numpy_to_tensor(
                pieces_imgs[i].astype(np.float32) / 255, device=self.device))
            ))
            # Convert and add dimension to the boxes
            targets.append({
                'boxes': torch.tensor(
                    bboxes[i], dtype=torch.float32)[None, ...],
                'labels': torch.tensor([1], dtype=torch.int64)
            })  # background - 0, target - 1
        return result_vols, targets

    @staticmethod
    def collate_func(
        batch: List[Tuple[List[Tensor], List[Dict[str, Tensor]]]],
    ) -> Tuple[List[Tensor], List[Dict[str, Tensor]]]:
        """Prepare batch to `RetinaRegionLocalizerV2` input requirements.

        Stacked maps and pieces images must be `list[FloatTensors]`
        and targets should be `list[dict]` when each dict have "boxes"
        and "labels" tensors.

        Parameters
        ----------
        batch : List[Tuple[List[Tensor], List[Dict[str, Tensor]]]]
            Batch as a list of samples.

        Returns
        -------
        Tuple[List[FloatTensor], List[FloatTensor], List[Dict[str, Tensor]]]
            Batch in required format.
        """
        return reduce(
            lambda stacked, sample: (
                stacked[0] + sample[0],
                stacked[1] + sample[1]),
            batch)
    

if __name__ == '__main__':
    import albumentations as A
    from torch.utils.data import DataLoader
    from utils.image_utils.image_functions import show_images_cv2
    from utils.torch_utils.torch_functions import (
        image_tensor_to_numpy, draw_bounding_boxes)

    # Overall params
    image_dir = ('data/satellite_dataset/dataset/train')
    b_size = 2
    num_crops = 4
    device = torch.device('cuda')
    # Dataset sizes params
    img_size = (2464, 2464)
    result_size = (900, 900)
    max_crop_size = img_size[0] * 0.8
    min_crop_size = img_size[0] * 0.5
    rectangle = True
    # Transforms params
    rotate = False
    blur = False
    # Piece transforms
    piece_blur = True

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

    # TODO добавить оттенки
    if piece_blur:
        piece_transforms = []
        if piece_blur:
            piece_transforms.append(A.Blur(blur_limit=7, p=1.0))
    else:
        piece_transforms = None

    # Get dataset and dloader
    dset = RegionDatasetV2(
        image_dir, min_crop_size, max_crop_size, result_size, num_crops,
        device, transforms, piece_transforms)
    dloader = DataLoader(
        dset, batch_size=b_size, collate_fn=RegionDatasetV2.collate_func)
    
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
