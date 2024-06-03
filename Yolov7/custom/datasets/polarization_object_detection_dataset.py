"""Polarization object detection dataset in CVAT format for yolo project.

Samples format of this dataset is compatible to use in yolo mosaic dataset.
It's required that this dataset returns an image, a bounding boxes,
classes ids, an image id, original hw of the image.
"""


from pathlib import Path
from typing import Dict, Union, Callable, Any, Tuple

import numpy as np
from numpy.typing import NDArray

from utils.torch_utils.datasets import CvatDetectionDataset
from mako_camera.cameras_utils import split_raw_pol


class PolarizationObjectDetectionDataset(CvatDetectionDataset):
    """Polarization object detection dataset in CVAT format for yolo project.

    Sample format of this dataset is compatible to use in `MosaicMixupDataset`.
    It's required that this dataset returns an image, a bounding boxes,
    classes ids, an image id, original hw of the image.
    """
    
    def __init__(
        self,
        dset_pth: Union[Path, str],
        transforms: Callable = None,
        class_to_index: Dict[str, int] = None,
        polarization: bool = False
    ) -> None:
        """Initialize `PolarizationObjectDetectionDataset`.

       Parameters
        ----------
        dset_pth : Union[Path, str]
            Path to CVAT dataset directory.
        transforms : Callable, optional
            Transforms that performs on sample.
            Required that it has `albumentations.Compose` like structure.
            By default is `None`.
        class_to_index : Dict[str, int], optional
            User-defined class to index mapping. It required that this dict
            contains all classes represented in the dataset.
            By default is `None`.
        polarization : bool, optional
            Is this a polarization dataset or RGB. By default is False (RGB).
        """
        super().__init__(dset_pth, transforms, class_to_index)
        # Save polarization mode
        self.polarization = polarization
        # Create img to id maps (required by mosaic and YOLOv7 datasets)
        self.img_name_to_id = {
            cvat_sample['img_pth'].name: i
            for i, cvat_sample in enumerate(self.get_samples_annotations())}
        self.img_id_to_name = {
            id: name
            for name, id, in self.img_name_to_id.items()}

    def get_sample(self, index: int) -> Dict[str, Any]:
        """Get sample.

        Sample represented as a dict that contains "image" `ndarray`,
        "bboxes" `list[list[float]]`, "classes" `list[int]`, "img_pth" `Path`
        and "shape" `tuple[int, int]`.

        If `self.polarization` is `True` then image will be parsed from
        `(h, w)` shape to `(h / 2, w / 2, 4)`.

        Parameters
        ----------
        index : int
            Index of sample.

        Returns
        -------
        Dict[str, Any]
            CVAT tank detection sample by index.
        """
        sample = super().get_sample(index)
        sample['img_id'] = self.img_name_to_id[sample['img_pth'].name]
        # Process if polarization
        if self.polarization:
            sample['image'] = split_raw_pol(sample['image'])
        return sample
    
    def postprocess_sample(
        self, sample: Dict[str, Any]
    ) -> Tuple[NDArray, NDArray, NDArray, int, Tuple[int, int]]:
        """Process the sample to make it suitable for `MosaicMixupDataset`.

        Unpack CVAT dictionary sample, convert bboxes and classes to `NDArray`
        and pack it to a `tuple`.

        Parameters
        ----------
        sample : Dict[str, Any]
            CVAT sample with additional "img_id" `int`.

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray, int, Tuple[int, int]]
            Image, bboxes, classes, image id, original shape.
        """
        # Convert bboxes and classes from lists to arrays as required
        sample['bboxes'] = np.array(sample['bboxes'])
        sample['classes'] = np.array(sample['labels'])
        image = sample['image']
        bboxes = sample['bboxes']
        classes = sample['classes']
        shape = sample['shape']
        img_id = sample['img_id']
        return image, bboxes, classes, img_id, shape
