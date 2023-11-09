"""Датасет для обнаружения танков в rgb и pol."""


from pathlib import Path
from typing import Dict, Union, Callable, Tuple, List

import numpy as np
from numpy.typing import NDArray

from utils.torch_utils.datasets import ObjectDetectionDataset
from mako_camera.cameras_utils import split_raw_pol


class TankDetectionDataset(ObjectDetectionDataset):
    """Tanks object detection dataset for yolo project.

    This dataset is used as a connection between base torch object detection
    class and yolo mosaic dataset class.
    It's required that this dataset returns an image, a bounding boxes,
    classes ids, an image id, original hw of the image.
    """
    def __init__(
        self,
        cvat_dset_dir: Union[str, Path],
        name2index: Dict[str, int] = None,
        transforms: Callable = None,
        polarization: bool = False
    ):
        """Initialize TankDetectionDataset object.

        Parameters
        ----------
        cvat_dset_dir : Union[str, Path]
            A path to a cvat dataset directory.
        name2index : Dict[str, int], optional
            Name to index dict converter.
            If not provided then will be generated automatically.
        transforms : Callable, optional
            Dataset transforms.
            It's expected that "Albumentations" lib will be used.
            By default is None.
        pol_sample : bool, optional
            Is this a polarization dataset or RGB. By default is False (RGB).
        """
        super().__init__(cvat_dset_dir, name2index, transforms)
        self.polarization = polarization
        self.img_name_to_id = {
            cvat_sample['name']: i
            for i, cvat_sample in enumerate(self.cvat_dset)}
        self.img_id_to_name = {
            id: name
            for name, id, in self.img_name_to_id.items()}

    def __getitem__(
        self, idx: int
    ) -> Tuple[NDArray, NDArray, NDArray, int, Tuple[int, int]]:
        """
        Get a sample from dataset by index.

        Parameters
        ----------
        idx : int
            An index of sample.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, int, Tuple[int, int]]
            An image, a bounding box, classes ids, an image id,
            original hw of the image.
        """
        # Polarization read
        if self.polarization:
            sample = self.cvat_dset[idx]
            img_name = sample['name']
            classes = sample['labels']
            classes = list(map(lambda label: float(self.class_to_index[label]),
                               classes))
            bboxes = sample['bboxes']
            shape = sample['shape']
            image = np.load(self.image_dir / img_name)
            image = split_raw_pol(image)
            shape = image.shape[:2]
            if self.transforms is not None:
                self.apply_transforms(image, bboxes, classes)
                transformed = self.transforms(
                    image=image, bboxes=bboxes, labels=classes
                )
                image = transformed['image']
                bboxes = transformed['bboxes']
                classes = transformed['classes']
        # RGB read
        else:
            image, bboxes, classes, img_name, shape = super().__getitem__(idx)
        bboxes = np.array(bboxes)
        classes = np.array(classes)
        img_id = self.img_name_to_id[img_name]
        return image, bboxes, classes, img_id, shape

    def apply_transforms(
        self, image: NDArray, bboxes: List[List[float]], classes: List[float]
    ) -> Tuple[NDArray, List[List[float]], List[float]]:
        # Labels argument's name was changed
        transformed = self.transforms(
            image=image, bboxes=bboxes, labels=classes)
        image = transformed['image']
        bboxes = transformed['bboxes']
        classes = transformed['labels']
        return image, bboxes, classes
