"""Module contains torch object detection dataset in CVAT format.

CVAT format represented as:

{dset_pth}
├── annotations.xml
├── images
|   ├── {sample_name_0}.jpg
|   ├── {sample_name_1}.jpg
|   ├── {sample_name_2}.jpg
|   ├── ...
|   ├── {sample_name_n-1}.jpg
|   ├── {sample_name_n}.jpg
"""

from pathlib import Path
from typing import Union, Any, List, Tuple, Dict, Optional
import xml.etree.ElementTree as ET

import torch

from .abstract_detection_dataset import AbstractDetectionDataset
from ..torch_functions import FloatBbox, image_numpy_to_tensor
from ...data_utils.data_functions import read_volume


class CvatDetectionDataset(AbstractDetectionDataset):
    """CVAT compatible object detection dataset.

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
        contains all classes represented in the dataset. By default is `None`.
    """

    def _parse_dataset_pth(self, dset_pth: Union[Path, str]) -> Path:
        """Check passed CVAT dataset path and save path to image directory.

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
        dset_pth = super()._parse_dataset_pth(dset_pth)
        self.image_dir = dset_pth / 'images'
        return dset_pth
    
    def _collect_samples(self, dset_pth: Path) -> List[Dict[str, Any]]:
        """Parse CVAT annotations.xml and make `list` of samples.

        Each sample is a `dict` that contain: "img_pth", "labels", "bboxes",
        "shape".

        Parameters
        ----------
        dset_pth : Path
            CVAT dataset directory.

        Returns
        -------
        List[Dict[str, Any]]
            Collected samples from the CVAT dataset.
        """
        annots = ET.parse(dset_pth / 'annotations.xml').getroot()
        
        # Get all class labels and its CVAT colors
        self._label_to_color = {}
        self._class_labels = []
        labels_annots = annots.findall('meta/job/labels/label')
        for label_annot in labels_annots:
            class_name = label_annot.find('name').text
            hex_color = label_annot.find('color').text
            color = [int(hex_color[j:j + 2], 16)
                     for j in range(1, len(hex_color), 2)]
            self._label_to_color[class_name] = color
            self._class_labels.append(class_name)

        # Get image annotations and paths
        samples: List[Dict[str, Any]] = []
        imgs_annots = annots.findall('image')
        for img_annots in imgs_annots:
            img_pth = self.image_dir / img_annots.get('name')
            shape = (int(img_annots.get('height')),
                     int(img_annots.get('width')))
            img_bboxes = img_annots.findall('box')
            img_labels: List[str] = []
            img_bboxes_pts: List[FloatBbox] = []
            for bbox in img_bboxes:
                label = bbox.get('label')
                img_labels.append(label)
                x1 = float(bbox.get('xtl'))
                y1 = float(bbox.get('ytl'))
                x2 = float(bbox.get('xbr'))
                y2 = float(bbox.get('ybr'))
                img_bboxes_pts.append((x1, y1, x2, y2))
            samples.append({
                'img_pth': img_pth,
                'labels': img_labels,
                'bboxes': img_bboxes_pts,
                'shape': shape
            })
        return samples
    
    def _collect_class_labels(
        self,
        samples: List[Dict[str, Any]],
        class_to_index: Optional[Dict[str, int]] = None
    ) -> Tuple[Dict[str, int], Dict[int, str]]:
        labels = set()
        for sample in samples:
            labels.update(sample['labels'])
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
    
    def get_sample(
        self, index: int
    ) -> Dict[str, Any]:
        """Get CVAT object detection sample.

        Sample represented as a dict that contains "image" `ndarray`,
        "bboxes" `list[list[float]]`, "labels" `list[int]`, "img_pth" `Path`
        and "shape" `tuple[int, int]`.

        Parameters
        ----------
        index : int
            Index of sample.

        Returns
        -------
        Dict[str, Any]
            CVAT object detection sample by index.
        """
        sample_annots = self.get_samples_annotations()[index]
        img_pth = sample_annots['img_pth']
        labels = sample_annots['labels']
        bboxes = sample_annots['bboxes']
        shape = sample_annots['shape']
        # Convert str labels to indexes
        labels = list(map(lambda label: self._class_to_index[label], labels))
        # Check extension of image and load it
        image = read_volume(img_pth)
        sample = {
            'image': image,
            'bboxes': bboxes,
            'labels': labels,
            'img_pth': img_pth,
            'shape': shape}
        return sample
    
    def postprocess_sample(
        self, sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert CVAT object detection sample to tensors.

        Image `uint8 ndarray` will be converted to `FloatTensor` in 0...1
        interval. Bboxes to FloatTensor and classes to IntTensor.

        Parameters
        ----------
        sample : Dict[str, Any]
            Sample in original format.

        Returns
        -------
        Dict[str, Any]
            Converted sample.
        """
        # Convert image, bboxes and classes to tensor
        image_numpy_to_tensor(sample['image'])
        sample['image'] = torch.tensor(
            sample['image'], dtype=torch.float32) / 255
        sample['bboxes'] = torch.tensor(sample['bboxes'], dtype=torch.float32)
        sample['classes'] = torch.tensor(sample['classes'], dtype=torch.int64)
        return sample
    
    def apply_transforms(
        self, sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply passed transforms on the sample.

        Parameters
        ----------
        sample : Dict[str, Any]
            Sample to transform.

        Returns
        -------
        Dict[str, Any]
            Transformed sample.
        """
        # Using albumentations transforms
        if self.transforms:
            transformed = self.transforms(image=sample['image'],
                                          bboxes=sample['bboxes'],
                                          classes=sample['labels'])
            sample['image'] = transformed['image']  # ArrayLike
            sample['bboxes'] = transformed['bboxes']  # list[list[float]]
            sample['labels'] = transformed['labels']  # list[int]
        return sample
    
    def get_labels_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Get labels with corresponding colors.

        Returns
        -------
        Dict[str, Tuple[int, int, int]]
            Dict that contains labels as keys and "color" as values.
        """
        return self._label_to_color
    
    @staticmethod
    def collate_func(batch: List[Dict[str, Any]]) -> Any:
        # TODO check and implement
        raise NotImplementedError()
        return batch
