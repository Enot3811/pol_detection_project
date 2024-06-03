from abc import ABC, abstractmethod
from typing import Union, Sequence, Any, List, Callable, Optional, Collection
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ...data_utils.data_functions import prepare_path


class AbstractTorchDataset(ABC, Dataset):
    """Abstract class for any custom torch dataset.

    Parameters
    ----------
    dset_pth : Union[Path, str]
        Path to dataset directory or some file.
    transforms : Optional[Callable], optional
        Transforms that performs on sample. By default is `None`.
    """
    def __init__(
        self,
        dset_pth: Union[Path, str],
        transforms: Optional[Callable] = None
    ) -> None:
        super().__init__()
        self.dset_pth = self._parse_dataset_pth(dset_pth)
        self._samples = self._collect_samples(self.dset_pth)
        self.transforms = transforms

    @abstractmethod
    def _parse_dataset_pth(self, dset_pth: Union[Path, str]) -> Path:
        """Parse and check dataset path according to its realization.

        Parameters
        ----------
        dset_pth : Union[Path, str]
            Path to dataset directory.

        Returns
        -------
        Path
            Parsed and checked dataset path.

        Raises
        ------
        ValueError
            Raise when `dset_pth` has wrong type.
        FileNotFoundError
            Raise when `dset_pth` does not exists.
        """
        return prepare_path(dset_pth)
    
    @abstractmethod
    def _collect_samples(self, dset_pth: Path) -> Sequence[Any]:
        """Collect samples according to the dataset signature.

        Parameters
        ----------
        dset_pth : Path
            Dataset path by which samples may be collected.

        Returns
        -------
        Sequence[Any]
            Sequence of dataset's samples.
        """
        pass

    @abstractmethod
    def get_sample(self, index: Any) -> Any:
        """Get sample according to the dataset format.

        Parameters
        ----------
        index : Any
            Index of sample.

        Returns
        -------
        Any
            Prepared sample.
        """
        return self._samples[index]
    
    def get_samples_annotations(self) -> Collection:
        """Get dataset's samples annotations.

        Returns
        -------
        Collection
            Samples annotations collection.
        """
        return self._samples

    @abstractmethod
    def postprocess_sample(self, sample: Any) -> Any:
        """Make postprocessing for sample after getting and augmentations.
        For example, convert sample to torch compatible formats.

        Parameters
        ----------
        sample : Any
            Sample in original view.

        Returns
        -------
        Any
            Sample in torch compatible view.
        """
        return torch.tensor(sample, dtype=torch.float32) / 255

    @abstractmethod
    def apply_transforms(self, sample: Any) -> Any:
        """Apply passed transforms on the sample.

        Parameters
        ----------
        sample : Any
            Sample to transform.

        Returns
        -------
        Any
            Transformed sample.
        """
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    @staticmethod
    @abstractmethod
    def collate_func(batch: List[Any]) -> Any:
        """Dataset's collate function for `DataLoader`.

        Parameters
        ----------
        batch : List[Any]
            Samples to make batch.

        Returns
        -------
        Any
            Batched samples.
        """
        pass

    def __len__(self) -> int:
        """Get length of the dataset.

        Returns
        -------
        int
            length of the dataset.
        """
        return len(self._samples)
    
    def __getitem__(self, index: Any) -> Any:
        sample = self.get_sample(index)
        sample = self.apply_transforms(sample)
        sample = self.postprocess_sample(sample)
        return sample
