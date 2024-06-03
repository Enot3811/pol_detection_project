"""Module contains abstract torch classification dataset."""

from abc import abstractmethod
from pathlib import Path
from typing import Sequence, Union, Any, Tuple, Dict, Callable, Optional

from .abstract_torch_dataset import AbstractTorchDataset


class AbstractClassificationDataset(AbstractTorchDataset):
    """Abstract torch dataset with classification labels.

    Parameters
    ----------
    dset_pth : Union[Path, str]
        Path to dataset directory or some file.
    transforms : Callable, optional
        Transforms that performs on sample. By default is `None`.
    class_to_index : Dict[str, int], optional
        User-defined class to index mapping. It required that this dict
        contains all classes represented in the dataset. By default is `None`.
    """

    def __init__(
        self,
        dset_pth: Union[Path, str],
        transforms: Callable = None,
        class_to_index: Dict[str, int] = None
    ) -> None:
        super().__init__(dset_pth, transforms)
        self._class_to_index, self._index_to_class = (
            self._collect_class_labels(self._samples, class_to_index))

    @abstractmethod
    def _collect_class_labels(
        self,
        samples: Sequence[Any],
        class_to_index: Optional[Dict[str, int]] = None
    ) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Collect all class labels and set them corresponding indexes.

        Iterate over all samples and collect unique classes collection.
        Make `self.class_to_index` and `self.index_to_class` dicts
        to convert labels.
        If `class_to_index` parameter was passed to initialization then
        check that it contain all classes represented in the dataset.

        Parameters
        ----------
        samples : Sequence[Any]
            Dataset samples.
        class_to_index : Optional[Dict[str, int]]
            Used defined `class_to_index` mapping. By default is `None`.

        Returns
        -------
        Tuple[Dict[str, int], Dict[int, str]]
            Collected mappings for class label to index and index
            to class label.

        Raises
        ------
        ValueError
            Raise when user-passed `class_to_index` does not contain
            all classes represented in the dataset.
        """
        labels = set()
        for sample in samples:
            labels.add(sample['label'])
        if class_to_index is not None:
            if not labels.issubset(class_to_index.keys()):
                raise ValueError(
                    'Passed "class_to_index" does not contain all classes '
                    'represented in the dataset.')
        else:
            class_to_index = {label: i for i, label in enumerate(labels)}
        index_to_class = {
            idx: label for label, idx in class_to_index.items()}
        return class_to_index, index_to_class
    
    @abstractmethod
    def get_sample(self, index: Any) -> Dict[str, Any]:
        """Get sample according to the dataset format.

        Parameters
        ----------
        index : Any
            Index of sample.

        Returns
        -------
        Dict[str, Any]
            Sample in dict format.
        """
        return super().get_sample(index)
    
    def get_class_to_index(self) -> Dict[str, int]:
        """Get dataset's class to index mapping.

        Returns
        -------
        Dict[str, int]
            The class to index mapping.
        """
        return self._class_to_index
    
    def get_index_to_class(self) -> Dict[str, int]:
        """Get dataset's index to class mapping.

        Returns
        -------
        Dict[str, int]
            The index to class mapping.
        """
        return self._index_to_class
