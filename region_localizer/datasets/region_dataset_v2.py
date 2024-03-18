from pathlib import Path
from typing import Callable, Union, Tuple, Dict, List
import sys

from numpy.typing import NDArray
from torch import Tensor, FloatTensor
import torch
from albumentations import BboxParams

sys.path.append(str(Path(__file__).parents[2]))
from utils.torch_utils.torch_functions import (
    random_crop, image_numpy_to_tensor, FloatBbox)
from utils.image_utils.image_functions import read_image
from region_localizer.datasets import RegionDataset


class RegionDatasetV2(RegionDataset):
    def __init__(
        self, image_dir: Union[str, Path],
        min_crop_size: Union[int, Tuple[int, int]],
        max_crop_size: Union[int, Tuple[int, int]],
        result_size: Tuple[int, int], transforms: Callable = None,
        num_classes: int = 2,
        num_crops: int = 1
    ):
        """Initialize `RegionDatasetV2` object.

        Parameters
        ----------
        image_dir : Union[str, Path]
            An image directory of dataset.
        min_crop_size : Tuple[int, int]
            Minimum crop size in `int` or `(h, w)` format.
        max_crop_size : Tuple[int, int]
            Maximum crop size in `int` or `(h, w)` format.
        result_size : Tuple[int, int]
            Result size that will be used in resize of result images
            in (h, w) format.
        transforms : Callable, optional
            Dataset transforms. Its activation can be tune by overriding
            `apply_transforms` method. By default is None.
        num_crops : int, optional
            How many crops to make and return for one source image.
            By default is `1`.
        num_classes : int, optional
            Temporary parameter to let to work with old one class models.
            By default is `2`.
        """
        super().__init__(
            image_dir, min_crop_size, max_crop_size, result_size, transforms,
            num_classes)
        self.num_crops = num_crops

    def __getitem__(
        self, idx: int
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """Get sample with a map, regions and regions bboxes with dummy labels.

        The map and the region are `FloatTensor` images and target is a dict
        with "boxes" and "labels" that represented as `FloatTensor` with shape
        `(num_crops, 4)` and `IntTensor` with shape `(num_crops,)`.

        Parameters
        ----------
        idx : int
            An index of sample.

        Returns
        -------
        Tuple[Tensor, Tensor, Dict[str, Tensor]]
            The map, the region and the targets dict.
        """
        map_img = read_image(self.img_pths[idx])
        bboxes = []
        result_regions = []
        for _ in range(self.num_crops):
            region, bbox = random_crop(
                map_img, self.min_crop_size, self.max_crop_size,
                return_position=True)
            bboxes.append(bbox)
            result_region = self.resize_transf(
                image=region, bboxes=[], labels=[])['image']
            result_regions.append(result_region)
        
        resized = self.resize_transf(
            image=map_img, bboxes=bboxes, labels=[0] * self.num_crops)
        result_map = resized['image']
        result_bboxes = resized['bboxes']

        if self.transforms:
            result_map, result_regions, result_bboxes = self.apply_transforms(
                result_map, result_regions, result_bboxes)

        # Convert to tensors
        result_map = (image_numpy_to_tensor(result_map)
                      .to(dtype=torch.float32) / 255)

        targets = []
        label = 0 if self.num_classes == 1 else 1  # temporary label assignment
        for i in range(self.num_crops):
            # Iterate over List[ndarray], convert, normalize and concatenate
            # each region with map
            result_regions[i] = torch.cat((
                result_map,
                (image_numpy_to_tensor(result_regions[i])
                 .to(dtype=torch.float32) / 255)))
            
            # Convert and add dimension
            targets.append({
                'boxes': torch.tensor(result_bboxes[i],
                                      dtype=torch.float32)[None, ...],
                'labels': torch.tensor([label], dtype=torch.int64)
            })  # background - 0, target - 1

        return result_regions, targets
    
    def apply_transforms(
        self, map_image: NDArray, region_images: List[NDArray],
        bboxes: List[FloatBbox]
    ) -> Tuple[Union[NDArray, Tensor], Union[NDArray, Tensor], Tensor]:
        """Apply transforms.

        Parameters
        ----------
        map_image : NDArray
            Map image in uint8.
        region_images : NDArray
            Region image in uint8.
        bboxes : FloatBbox
            List of `FloatBbox`.

        Returns
        -------
        Tuple[Union[NDArray, Tensor], Union[NDArray, Tensor], Tensor]
            Transformed data.
        """
        transformed = self.transforms(
            image=map_image, bboxes=bboxes, labels=[0] * self.num_crops)
        map_image = transformed['image']
        bboxes = transformed['bboxes']
        # Iterate over regions list and transform each of them
        region_images = list(map(
            lambda region: (self.transforms(image=region, bboxes=[], labels=[])
                            ['image']),
            region_images))
    
        return map_image, region_images, bboxes
    
    @staticmethod
    def collate_func(
        batch: Tuple[Tuple[List[FloatTensor], List[Dict[str, FloatTensor]]]],
        device: torch.device
    ) -> Tuple[List[FloatTensor], List[FloatTensor], List[Dict[str, Tensor]]]:
        """Collate function that prepare batch to Retina input requirements.

        RetinaNet input images must be list of FloatTensors that may have
        different shapes.

        Parameters
        ----------
        batch : Tuple[Tuple[List[FloatTensor], List[Dict[str, FloatTensor]]]]
            Batched data in retina format, tensors `(b, c, h, w)`.

        Returns
        -------
        Tuple[List[FloatTensor], List[FloatTensor], List[Dict[str, Tensor]]]
            Data in RetinaNet input format.
        """
        map_images, targets = batch[0]
        for i in range(len(map_images)):
            map_images[i] = map_images[i].to(device=device)
            targets[i]['boxes'] = targets[i]['boxes'].to(device=device)
            targets[i]['labels'] = targets[i]['labels'].to(device=device)
        return map_images, targets
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import albumentations as A
    from torch.utils.data import DataLoader
    from utils.image_utils.image_functions import show_image_plt
    from utils.torch_utils.torch_functions import (
        image_tensor_to_numpy, draw_bounding_boxes)
    from functools import partial

    image_dir = ('data/satellite_dataset/dataset/train')
    num_crops = 3
    b_size = 1
    img_size = (2464, 2464)
    result_size = (1024, 1024)
    max_ratio = 0.8
    min_ratio = 0.5
    max_size = int(img_size[0] * max_ratio)
    min_size = int(img_size[0] * min_ratio)
    transf = False

    if transf:
        transforms = A.Compose([
            A.RandomRotate90(always_apply=True)
        ], bbox_params=BboxParams(format='pascal_voc',
                                  label_fields=['labels']))
    else:
        transforms = None

    collate_func = partial(
        RegionDatasetV2.collate_func, device=torch.device('cpu'))
    dset = RegionDatasetV2(
        image_dir, min_size, max_size,
        result_size=result_size, transforms=transforms, num_crops=num_crops)
    dloader = DataLoader(dset, batch_size=b_size, collate_fn=collate_func)
    
    for batch in dloader:
        for i in range(num_crops):
            stacked_img = batch[0][i]
            target = batch[1][i]
            fig, axes = plt.subplots(1, 2)

            if isinstance(stacked_img, Tensor):
                stacked_img = image_tensor_to_numpy(
                    (stacked_img * 255).to(dtype=torch.uint8))
                bbox = target['boxes'].squeeze().tolist()
            else:
                bbox = target['boxes']

            map_img = stacked_img[..., :3]
            region_img = stacked_img[..., 3:]
            boxes_img = draw_bounding_boxes(map_img, [bbox])

            show_image_plt(boxes_img, axes[0])
            show_image_plt(region_img, axes[1])
            
            fig.set_size_inches((30, 15))
            print('Bbox:', bbox, 'h_size:', bbox[3] - bbox[1],
                  'w_size:', bbox[2] - bbox[0])
            plt.show()
