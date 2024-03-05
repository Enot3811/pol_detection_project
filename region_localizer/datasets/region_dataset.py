from pathlib import Path
from typing import Callable, Union, Tuple, Dict, List
import sys

from numpy.typing import NDArray
from torch import Tensor, FloatTensor
import torch
from albumentations import Resize, Compose, BboxParams

sys.path.append(str(Path(__file__).parents[2]))
from utils.torch_utils.datasets import TorchImageDataset
from utils.torch_utils.torch_functions import (
    random_crop, image_numpy_to_tensor, FloatBbox)
from utils.image_utils.image_functions import read_image


class RegionDataset(TorchImageDataset):

    def __init__(
        self, image_dir: Union[str, Path],
        min_crop_size: Union[int, Tuple[int, int]],
        max_crop_size: Union[int, Tuple[int, int]],
        result_size: Tuple[int, int], transforms: Callable = None
    ):
        """Initialize `RegionDataset` object.

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
            in `(h, w)` format.
        transforms : Callable, optional
            Dataset transforms. Its activation can be tune by overriding
            `apply_transforms` method. By default is None.
        """
        super().__init__(image_dir, transforms)
        self.min_crop_size = min_crop_size
        self.max_crop_size = max_crop_size
        self.result_size = result_size
        self.resize_transf = Compose(
            [Resize(*result_size)],
            bbox_params=BboxParams(
                format='pascal_voc', label_fields=['labels']))

    def __getitem__(
        self, idx: int
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """Get sample with a map, a region and a region bbox with dummy label.

        The map and the region are `FloatTensor` images and target is a dict
        with "boxes" and "labels" that represented as `FloatTensor` with shape
        `(1, 4)` and `IntTensor` with shape `(1,)`.

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
        region, bbox = random_crop(
            map_img, self.min_crop_size, self.max_crop_size,
            return_position=True)
        
        resized = self.resize_transf(image=map_img, bboxes=[bbox], labels=[0])
        result_map = resized['image']
        result_bbox = resized['bboxes'][0]
        result_region = self.resize_transf(
            image=region, bboxes=[], labels=[])['image']

        if self.transforms:
            result_map, result_region, result_bbox = self.apply_transforms(
                result_map, result_region, result_bbox)

        result_map = image_numpy_to_tensor(result_map)
        result_region = image_numpy_to_tensor(result_region)
        result_map = result_map.to(dtype=torch.float32) / 255
        result_region = result_region.to(dtype=torch.float32) / 255
        result_bbox = torch.tensor(
            list(map(int, result_bbox)), dtype=torch.float32)
            
        targets = {
            'boxes': result_bbox[None, ...],
            'labels': torch.tensor([1])}  # background - 0, target - 1

        return result_map, result_region, targets
    
    def apply_transforms(
        self, map_image: NDArray, region_image: NDArray, bbox: FloatBbox
    ) -> Tuple[Union[NDArray, Tensor], Union[NDArray, Tensor], Tensor]:
        """Apply transforms and normalize data.

        After transforms images will be dived by 255 and converted to float32
        and bbox will be converted to float32 tensor.

        Parameters
        ----------
        map_image : NDArray
            Map image in uint8.
        region_image : NDArray
            Region image in uint8.
        bbox : FloatBbox
            Bounding box float list.

        Returns
        -------
        Tuple[Union[NDArray, Tensor], Union[NDArray, Tensor], Tensor]
            Transformed and normalized data.
        """
        transformed = self.transforms(
            image=map_image, bboxes=[bbox], labels=[0])
        map_image = transformed['image']
        bbox = transformed['bboxes'][0]
        region_image = self.transforms(
            image=region_image, bboxes=[], labels=[])['image']

        return map_image, region_image, bbox
    
    @staticmethod
    def collate_func(
        batch: List[Tuple[FloatTensor, FloatTensor, Dict[str, Tensor]]],
        device: torch.device
    ) -> Tuple[List[FloatTensor], List[FloatTensor], List[Dict[str, Tensor]]]:
        """Collate function that prepare batch to Retina input requirements.

        RetinaNet input images must be list of FloatTensors that may have
        different shapes.

        Parameters
        ----------
        batch : Tuple[FloatTensor, FloatTensor, Dict[str, Tensor]]
            Batched data in original format, tensors `(b, c, h, w)`.

        Returns
        -------
        Tuple[List[FloatTensor], List[FloatTensor], List[Dict[str, Tensor]]]
            Data in RetinaNet input format.
        """
        map_images, region_images, targets = tuple(map(list, zip(*batch)))
        for i in range(len(map_images)):
            map_images[i] = map_images[i].to(device=device)
            region_images[i] = region_images[i].to(device=device)
            targets[i]['boxes'] = targets[i]['boxes'].to(device=device)
            targets[i]['labels'] = targets[i]['labels'].to(device=device)
        return map_images, region_images, targets
    

if __name__ == '__main__':
    from functools import partial
    import matplotlib.pyplot as plt
    import albumentations as A
    from torch.utils.data import DataLoader
    from utils.image_utils.image_functions import show_image_plt
    from utils.torch_utils.torch_functions import (
        image_tensor_to_numpy, draw_bounding_boxes)

    image_dir = ('data/satellite_dataset/dataset/train')
    b_size = 4
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
        RegionDataset.collate_func, device=torch.device('cpu'))
    dset = RegionDataset(
        image_dir, min_size, max_size,
        result_size=result_size, transforms=transforms)
    dloader = DataLoader(
        dset, batch_size=b_size, collate_fn=collate_func)
    
    for batch in dloader:
        for i in range(b_size):
            map_img = batch[0][i]
            region_img = batch[1][i]
            target = batch[2][i]
            fig, axes = plt.subplots(1, 2)

            if isinstance(map_img, Tensor):
                map_img = image_tensor_to_numpy(
                    (map_img * 255).to(dtype=torch.uint8))
                region_img = image_tensor_to_numpy(
                    (region_img * 255).to(dtype=torch.uint8))
                bbox = target['boxes'].squeeze().tolist()
            else:
                bbox = target['boxes']

            boxes_img = draw_bounding_boxes(map_img, [bbox])

            show_image_plt(boxes_img, axes[0])
            show_image_plt(region_img, axes[1])

            fig.set_size_inches((30, 15))
            print('Bbox:', bbox, 'h_size:', bbox[3] - bbox[1],
                  'w_size:', bbox[2] - bbox[0])
            plt.show()
