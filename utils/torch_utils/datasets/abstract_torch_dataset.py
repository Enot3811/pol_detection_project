from abc import ABC, abstractmethod
from typing import Union, Sequence, Any, List, Callable, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset


class AbstractTorchDataset(ABC, Dataset):
    """Abstract class for any custom torch dataset.

    Parameters
    ----------
    dset_pth : Union[Path, str]
        Path to dataset directory or some file.
    device : torch.device, optional
        Device for dataset samples. By default is `torch.device('cpu')`.
    transforms : Optional[Callable], optional
        Transforms that performs on sample. By default is `None`.
    """
    def __init__(
        self,
        dset_pth: Union[Path, str],
        device: torch.device = torch.device('cpu'),
        transforms: Optional[Callable] = None
    ) -> None:
        super().__init__()
        self.dset_pth = self._parse_dataset_pth(dset_pth)
        self.samples = self._collect_samples(self.dset_pth)
        self.device = device
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
        if isinstance(dset_pth, str):
            dset_pth = Path(dset_pth)
        elif not isinstance(dset_pth, Path):
            raise ValueError(
                'Dataset path required to be str or Path '
                f'but got {type(dset_pth)}.')
        if not dset_pth.exists():
            raise FileNotFoundError(
                f'Given path "{dset_pth}" does not exist.')
        return dset_pth
    
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
        return self.samples[index]

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
        return torch.tensor(
            sample, dtype=torch.float32, device=self.device) / 255

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
        return len(self.samples)
    
    def __getitem__(self, index: Any) -> Any:
        sample = self.get_sample(index)
        sample = self.apply_transforms(sample)
        sample = self.postprocess_sample(sample)
        return sample
