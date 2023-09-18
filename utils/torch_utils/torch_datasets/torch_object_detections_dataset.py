"""A torch general object detection dataset."""


from pathlib import Path
from typing import Dict, Union, Callable, Any

from torch.utils.data import Dataset

from utils.image_utils.image_functions import read_image
from utils.cvat_utils.cvat_datasets import CvatObjectDetectionDataset


class ObjectDetectionDataset(Dataset):
    """A torch general object detection dataset."""

    def __init__(
        self,
        cvat_dset_dir: Union[str, Path],
        name2index: Dict[str, int] = None,
        transforms: Callable = None
    ):
        """Initialize object.

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
        """
        if isinstance(cvat_dset_dir, str):
            cvat_dset_dir = Path(cvat_dset_dir)
        self.dset_dir = cvat_dset_dir
        self.image_dir = cvat_dset_dir / 'images'
        self.cvat_dset = CvatObjectDetectionDataset(cvat_dset_dir)
        self.labels = self.cvat_dset.get_labels()
        if name2index is None:
            self.name2index = {label: i for i, label in enumerate(self.labels)}
        else:
            self.name2index = name2index
        self.transforms = transforms

    def __len__(self):
        return len(len(self.cvat_dset))

    def __getitem__(self, idx: int) -> Any:
        """Return a sample by its index.

        Parameters
        ----------
        idx : int
            The index of the sample.

        Returns
        -------
        Any
            By default sample is a tuple that contains
            image ndarray with shape `(h, w, c)`,
            bounding boxes float list with shape `(n_obj, 4)`
            and classes float list with shape `(n_obj,)`.
        """
        img_name, classes, bboxes, shape = self.cvat_dset[idx]
        classes = map(lambda label: float(self.name2index[label]), classes)
        img_pth = self.image_dir / img_name
        image = read_image(img_pth)

        if self.transforms:
            transformed = self.transforms(
                image=image, bboxes=bboxes, classes=classes)
            image = transformed['image']  # tensor
            bboxes = transformed['bboxes']  # list[list[float]]
            classes = transformed['classes']  # list[float]

        return image, bboxes, classes
