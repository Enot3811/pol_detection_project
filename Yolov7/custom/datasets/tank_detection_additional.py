"""Tanks object detection dataset with additional pol features.

This dataset is able to return pol samples as 2-channel ndarray
that consists of 0-90 or 45-135 channels.
"""


from pathlib import Path
from typing import Dict, Union, Callable, Tuple

from numpy.typing import NDArray

from Yolov7.custom.datasets import TankDetectionDataset


class TankDetectionDatasetAdditional(TankDetectionDataset):
    """Tanks object detection dataset with additional pol features.

    This dataset is able to return pol samples as 2-channel ndarray
    that consists of 0-90 or 45-135 channels.
    """
    def __init__(
        self,
        cvat_dset_dir: Union[str, Path],
        name2index: Dict[str, int] = None,
        transforms: Callable = None,
        polarization: bool = False,
        active_ch: str = '0_90'
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
        active_ch : str, optional
            A mode of 2ch sampling. It can be "0_90" or "45_135".
            By default is "0_90".
        """
        super().__init__(cvat_dset_dir, name2index, transforms, polarization)
        if active_ch not in ('0_90', '45_135'):
            raise ValueError('active_ch must be equal "0_90" or "45_135".')
        self.active_ch = active_ch

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
        image, bboxes, classes, img_id, shape = super().__getitem__(idx)
        if self.active_ch == '0_90':
            image = image[:, :, 0::2]
        else:
            image = image[:, :, 1::2]
        return image, bboxes, classes, img_id, shape
