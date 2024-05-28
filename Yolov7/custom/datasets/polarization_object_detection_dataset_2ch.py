"""Polarization object detection dataset with 2-channel pol sampling.

This dataset is able to return pol samples as 2-channel ndarray
that consists of 0 and 90 or 45 and 135 channels.
"""


from pathlib import Path
from typing import Any, Dict, Union, Callable

from .polarization_object_detection_dataset import (
    PolarizationObjectDetectionDataset)


class PolarizationObjectDetectionDataset2ch(
        PolarizationObjectDetectionDataset):
    """Polarization object detection dataset with 2-channel pol sampling.

    This dataset returns only "pol" samples as 2-channel ndarray
    that consists of 0 and 90 or 45 and 135 channels.
    """
    def __init__(
        self,
        cvat_dset_dir: Union[str, Path],
        name2index: Dict[str, int] = None,
        transforms: Callable = None,
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
        active_ch : str, optional
            A mode of 2ch sampling. It can be "0_90" or "45_135".
            By default is "0_90".
        """
        super().__init__(
            cvat_dset_dir, name2index, transforms, polarization=True)
        if active_ch not in ('0_90', '45_135'):
            raise ValueError('"active_ch" must be equal "0_90" or "45_135".')
        self.active_ch = active_ch

    def get_sample(self, index: int) -> Dict[str, Any]:
        sample = super().get_sample(index)
        if self.active_ch == '0_90':
            sample['image'] = sample['image'][:, :, 0::2]
        else:
            sample['image'] = sample['image'][:, :, 1::2]
        return sample
