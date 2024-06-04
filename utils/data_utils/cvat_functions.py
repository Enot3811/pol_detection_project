"""The module with helper functions for CVAT annotating."""


from xml.dom.minidom import Document
import datetime
from typing import Union, List, Dict, Any
from pathlib import Path

from tqdm import tqdm


def create_cvat_meta(
    xml_doc: Document, set_size: int, labels_names: List[str], subset_name: str
) -> None:
    """Create "meta" tag of cvat annotations.

    Take meta information and write meta heder in CVAT format
    to the given xml document.

    Parameters
    ----------
    xml_doc : Document
        An xml document for cvat annotations.
    set_size : int
        Length of the dataset to annotate.
    labels_names : List[str]
        Class names of the dataset ot annotate.
    set_name : str
        A name of the annotated set.
        For example "train", "val" or something else.
    """
    date = datetime.datetime.now(datetime.timezone(datetime.timedelta()))
    annotations = xml_doc.createElement('annotations')
    xml_doc.appendChild(annotations)

    version = xml_doc.createElement('version')
    version.appendChild(xml_doc.createTextNode('1.1'))
    annotations.appendChild(version)

    meta = xml_doc.createElement('meta')
    job = xml_doc.createElement('job')
    meta.appendChild(job)
    annotations.appendChild(meta)

    id = xml_doc.createElement('id')
    id.appendChild(xml_doc.createTextNode('1'))
    size = xml_doc.createElement('size')
    size.appendChild(xml_doc.createTextNode(str(set_size)))
    mode = xml_doc.createElement('mode')
    mode.appendChild(xml_doc.createTextNode('annotation'))
    overlap = xml_doc.createElement('overlap')
    overlap.appendChild(xml_doc.createTextNode('0'))
    bugtracker = xml_doc.createElement('bugtracker')
    created = xml_doc.createElement('created')
    created.appendChild(xml_doc.createTextNode(str(date)))
    updated = xml_doc.createElement('updated')
    updated.appendChild(xml_doc.createTextNode(str(date)))
    subset = xml_doc.createElement('subset')
    subset.appendChild(xml_doc.createTextNode(subset_name))
    start_frame = xml_doc.createElement('start_frame')
    start_frame.appendChild(xml_doc.createTextNode('0'))
    stop_frame = xml_doc.createElement('stop_frame')
    stop_frame.appendChild(xml_doc.createTextNode(str(set_size)))
    frame_filter = xml_doc.createElement('frame_filter')
    job.appendChild(id)
    job.appendChild(size)
    job.appendChild(mode)
    job.appendChild(overlap)
    job.appendChild(bugtracker)
    job.appendChild(created)
    job.appendChild(updated)
    job.appendChild(subset)
    job.appendChild(start_frame)
    job.appendChild(stop_frame)
    job.appendChild(frame_filter)

    segments = xml_doc.createElement('segments')
    job.appendChild(segments)

    segment = xml_doc.createElement('segment')
    segments.appendChild(segment)

    id = xml_doc.createElement('id')
    id.appendChild(xml_doc.createTextNode('1'))
    start = xml_doc.createElement('start')
    start.appendChild(xml_doc.createTextNode('0'))
    stop = xml_doc.createElement('stop')
    stop.appendChild(xml_doc.createTextNode(str(set_size)))
    url = xml_doc.createElement('url')
    url.appendChild(xml_doc.createTextNode('http://localhost:8080/api/jobs/1'))
    segment.appendChild(id)
    segment.appendChild(start)
    segment.appendChild(stop)
    segment.appendChild(url)

    owner = xml_doc.createElement('owner')
    job.appendChild(owner)

    username = xml_doc.createElement('username')
    username.appendChild(xml_doc.createTextNode('enot'))
    email = xml_doc.createElement('email')
    owner.appendChild(username)
    owner.appendChild(email)

    job.appendChild(xml_doc.createElement('assignee'))

    labels = xml_doc.createElement('labels')
    job.appendChild(labels)

    for label_name in labels_names:
        label = xml_doc.createElement('label')
        name = xml_doc.createElement('name')
        name.appendChild(xml_doc.createTextNode(label_name))
        color = xml_doc.createElement('color')
        color.appendChild(xml_doc.createTextNode('#000000'))
        type = xml_doc.createElement('type')
        type.appendChild(xml_doc.createTextNode('any'))
        attributes = xml_doc.createElement('attributes')
        label.appendChild(name)
        label.appendChild(color)
        label.appendChild(type)
        label.appendChild(attributes)
        labels.appendChild(label)

    dumped = xml_doc.createElement('dumped')
    dumped.appendChild(xml_doc.createTextNode(str(date)))
    meta.appendChild(dumped)


def create_cvat_detection_annotations(
    xml_doc: Document,
    samples: List[Dict[str, Any]],
    verbose: bool = False
) -> None:
    """Create CVAT object detection annotations.

    Take a detection dataset and write images' annotations in CVAT format
    to the given xml document.

    Parameters
    ----------
    xml_doc : Document
        An xml document for cvat annotations.
    samples : List[Dict[str, Any]]
        Samples of the dataset. Each sample is required to be a `Dict`
        that contains "img_pth": `Path`, "shape": `Tuple[int, int]`,
        "bboxes": `List[Tuple[float, float, float, float]]`
        and "labels": `List[str]`.
    verbose : bool, optional
        Whether to show progress of converting. By default is `False`.
    """
    annots_tag = xml_doc.getElementsByTagName("annotations")[0]
    iterator = range(len(samples))
    if verbose:
        iterator = tqdm(iterator, 'Form cvat images annotations')
    for i in iterator:
        sample = samples[i]
        pth = sample['img_pth']
        shape = sample['shape']
        bboxes = sample['bboxes']
        labels = sample['labels']
        image = xml_doc.createElement('image')
        image.setAttribute('id', str(i))
        image.setAttribute('name', pth.name)
        image.setAttribute('width', str(shape[1]))
        image.setAttribute('height', str(shape[0]))
        for bbox, label in zip(bboxes, labels):
            box = xml_doc.createElement('box')
            box.setAttribute('label', label)
            box.setAttribute('occluded', '0')
            box.setAttribute('source', 'manual')
            box.setAttribute('xtl', str(bbox[0]))
            box.setAttribute('ytl', str(bbox[1]))
            box.setAttribute('xbr', str(bbox[2]))
            box.setAttribute('ybr', str(bbox[3]))
            box.setAttribute('z_order', '0')
            image.appendChild(box)
        annots_tag.appendChild(image)


def create_cvat_xml_from_dataset(
    save_pth: Union[str, Path],
    labels_names: List[str],
    samples: List[Dict[str, Any]],
    set_name: str,
    verbose: bool = False
):
    """Save annotations as a CVAT xml document.

    Parameters
    ----------
    save_pth : Union[str, Path]
        A path to save created xml file.
    labels_names : List[str]
        A list containing names of the dataset's classes.
    samples : List[Dict[str, Any]]
        Samples of the dataset. Each sample is required to be a `Dict`
        that contains "img_pth": `Path`, "shape": `Tuple[int, int]`,
        "bboxes": `List[Tuple[float, float, float, float]]`
        and "labels": `List[str]`.
    set_name : str
        A name of saving set ("train", "val", etc).
    verbose : bool, optional
        Whether to show progress of converting. By default is `False`.
    """
    if isinstance(save_pth, str):
        save_pth = Path(str)
    xml_doc = Document()
    # Get len and classes names and create meta
    set_size = len(samples)
    create_cvat_meta(xml_doc, set_size, labels_names, set_name)

    # Iterate over samples to create annotations
    create_cvat_detection_annotations(xml_doc, samples, verbose)
    
    save_pth.parent.mkdir(parents=True, exist_ok=True)
    xml_doc.writexml(open(save_pth, 'w'), indent="  ", addindent="  ",
                     newl='\n', encoding='utf-8')
    xml_doc.unlink()
