from pathlib import Path
from typing import Callable, Union, Tuple, Dict, List, Optional, Any
import sys
from functools import reduce

import numpy as np
import torch
from torch import Tensor, FloatTensor
from albumentations import Resize, Compose, BboxParams

sys.path.append(str(Path(__file__).parents[2]))
from utils.torch_utils.torch_functions import (
    random_crop, image_numpy_to_tensor)
from utils.torch_utils.datasets import AbstractTorchDataset
from utils.data_utils.data_functions import (
    read_image, IMAGE_EXTENSIONS, collect_paths)


class RegionDataset(AbstractTorchDataset):
    """Dataset for region localizer v1.

    Each sample represented as a map image, cropped piece from this map
    and bounding box of this piece.
    The map image and piece image may be processed with some augmentations.
    Pieces can be processed separately with `piece_transforms` argument.
    At the end both these images will be resized to the same `result_size`.
    """

    def __init__(
        self,
        dset_pth: Union[str, Path],
        min_crop_size: Union[int, Tuple[int, int]],
        max_crop_size: Union[int, Tuple[int, int]],
        result_size: Union[int, Tuple[int, int]],
        num_crops: int = 1,
        transforms: Optional[Callable] = None,
        piece_transforms: Optional[Callable] = None,
    ):
        """Initialize dataset.

        Parameters
        ----------
        dset_pth : Union[str, Path]
            A directory containing images of maps.
        min_crop_size : Tuple[int, int]
            Minimum size for piece random crop.
            It should be either min size of square as `int`
            or min size of rectangle as `tuple` in format `(h, w)`.
            Consistency in format is required with `max_crop_size`.
        max_crop_size : Tuple[int, int]
            Maximum size for piece random crop.
            It should be either max size of square as `int`
            or max size of rectangle as `tuple` in format `(h, w)`.
            Consistency in format is required with `min_crop_size`.
        result_size : Union[int, Tuple[int, int]]
            Result size that is used in resize of result images
            in `int` or `(h, w)` format.
        num_crops : int, optional
            How many crops to make and return for one map image.
            By default is `1`.
        transforms : Optional[Callable], optional
            Dataset transforms. Performs on both maps and pieces images.
            By default is `None`.
        piece_transforms : Optional[Callable], optional
            Pieces transforms. performs only on pieces images.
            By default is `None`.
        """
        super().__init__(dset_pth, transforms)
        # Save other parameters
        self.min_crop_size = min_crop_size
        self.max_crop_size = max_crop_size
        self.result_size = result_size
        self.num_crops = num_crops
        self.piece_transforms = piece_transforms
        # Resize to be compatible with model input size
        if isinstance(result_size, int):
            result_size = (result_size, result_size)
        self.resize_transf = Compose(
            [Resize(*result_size)],
            bbox_params=BboxParams(
                format='pascal_voc', label_fields=['labels']))
    
    def _parse_dataset_pth(self, dset_pth: Union[Path, str]) -> Path:
        return super()._parse_dataset_pth(dset_pth)
        
    def _collect_samples(self, dset_pth: Path) -> List[Path]:
        return collect_paths(dset_pth, IMAGE_EXTENSIONS)
    
    def get_sample(self, index: int) -> Dict[str, Any]:
        """Get sample from region dataset.

        Get map image, make `num_crops` crops from it and pack it to `dict`.

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
        map_img = read_image(self._samples[index])
        bboxes = []
        pieces_imgs = []
        for _ in range(self.num_crops):
            piece_img, bbox = random_crop(
                map_img, self.min_crop_size, self.max_crop_size,
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
    
    def apply_transforms(
        self, sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply transforms on map image and cropped pieces.

        Both map image and pieces will transformed with `transforms` and then
        pieces will be transformed with `piece_transforms` separately.

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
        Dict[str, Any]
            Transformed sample.
        """
        if self.transforms:
            transformed = self.transforms(
                image=sample['map_img'], bboxes=sample['bboxes'],
                labels=[0] * self.num_crops)
            sample['map_img'] = transformed['image']
            sample['bboxes'] = transformed['bboxes']
            # Iterate over pieces list and transform each of them
            sample['pieces_imgs'] = list(map(
                lambda region: self.transforms(
                    image=region, bboxes=[], labels=[])['image'],
                sample['pieces_imgs']))
        if self.piece_transforms:
            sample['pieces_imgs'] = list(map(
                lambda region: self.piece_transforms(
                    image=region, bboxes=[], labels=[])['image'],
                sample['pieces_imgs']))
        return sample
    
    def postprocess_sample(
        self, sample: Dict[str, Any]
    ) -> Tuple[List[FloatTensor], List[FloatTensor], List[Dict[str, Tensor]]]:
        """Convert to tensors and make tuple with maps, pieces and bboxes.

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
        Tuple[List[FloatTensor], List[FloatTensor], List[Dict[str, Tensor]]]
            - Maps tensors with shapes `(3, *result_size)`
            - Pieces tensors with shapes `(3, *result_size)`
            - Targets dict with "boxes" and "labels" tensors.
        """
        map_img = sample['map_img']
        pieces_imgs = sample['pieces_imgs']
        bboxes = sample['bboxes']

        # Convert to tensors and copy for each piece
        map_img = image_numpy_to_tensor(map_img.astype(np.float32) / 255)
        map_imgs = list(torch.unbind(
            map_img[None, ...].repeat(self.num_crops, 1, 1, 1)))
        
        targets = []
        for i in range(self.num_crops):
            # Iterate over List[ndarray], convert, normalize and concatenate
            # each piece with map
            pieces_imgs[i] = image_numpy_to_tensor(
                pieces_imgs[i].astype(np.float32) / 255)
            
            # Convert and add dimension to the boxes
            targets.append({
                'boxes': torch.tensor(
                    bboxes[i], dtype=torch.float32)[None, ...],
                'labels': torch.tensor([1], dtype=torch.int64)
            })  # background - 0, target - 1

        return map_imgs, pieces_imgs, targets

    def __getitem__(
        self, index: int
    ) -> Tuple[List[FloatTensor], List[FloatTensor], List[Dict[str, Tensor]]]:
        return super().__getitem__(index)
    
    @staticmethod
    def collate_func(
        batch: List[
            Tuple[List[Tensor], List[Tensor], List[Dict[str, Tensor]]]],
    ) -> Tuple[List[Tensor], List[Tensor], List[Dict[str, Tensor]]]:
        """Prepare batch to `RetinaRegionLocalizer` input requirements.

        Maps and pieces images must be `list[FloatTensors]` and targets should
        be `list[dict]` when each dict have "boxes" and "labels" tensors.

        Parameters
        ----------
        batch : Tuple[List[Tensor], List[Tensor], List[Dict[str, Tensor]]]
            Batch as a list of samples.

        Returns
        -------
        Tuple[List[FloatTensor], List[FloatTensor], List[Dict[str, Tensor]]]
            Batch in required format.
        """
        return reduce(
            lambda stacked, sample: (
                stacked[0] + sample[0],
                stacked[1] + sample[1],
                stacked[2] + sample[2]),
            batch)
    

if __name__ == '__main__':
    import albumentations as A
    from torch.utils.data import DataLoader
    from utils.data_utils.data_functions import show_images_cv2
    from utils.torch_utils.torch_functions import (
        image_tensor_to_numpy, draw_bounding_boxes)

    # Overall params
    image_dir = ('data/satellite_dataset/dataset/train')
    b_size = 1
    num_crops = 1
    # Dataset sizes params
    img_size = (2464, 2464)
    result_size = (900, 900)
    max_crop_size = img_size[0] * 0.8
    min_crop_size = img_size[0] * 0.5
    rectangle = True
    # Transforms params
    rotate = True
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
    dset = RegionDataset(
        image_dir, min_crop_size, max_crop_size, result_size, num_crops,
        transforms, piece_transforms)
    dloader = DataLoader(
        dset, batch_size=b_size, collate_fn=RegionDataset.collate_func)
    
    # Iterate over dloader
    for batch in dloader:
        for i in range(b_size * num_crops):
            map_img = batch[0][i]
            region_img = batch[1][i]
            target = batch[2][i]

            if isinstance(map_img, Tensor):
                map_img = image_tensor_to_numpy(
                    (map_img * 255).to(dtype=torch.uint8))
                region_img = image_tensor_to_numpy(
                    (region_img * 255).to(dtype=torch.uint8))
                bbox = target['boxes'].squeeze().tolist()
            else:
                bbox = target['boxes']

            boxes_img = draw_bounding_boxes(map_img, [bbox])

            print('Bbox:', bbox, 'h_size:', bbox[3] - bbox[1],
                  'w_size:', bbox[2] - bbox[0])

            key = show_images_cv2(
                [boxes_img, region_img],
                ['region', 'piece'],
                destroy_windows=False)
            if key == 27:
                exit()
