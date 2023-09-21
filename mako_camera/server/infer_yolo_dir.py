"""Запустить yolo на сканирование папки и обработку всех новых кадров."""

import sys
import time
from pathlib import Path
from typing import Union
import argparse

import numpy as np
import cv2

sys.path.append(str(Path(__file__).parents[2]))
sys.path.append(str(Path(__file__).parents[2] / 'Yolov7'))  # TODO сделать что-нибудь с этим
from Yolov7.custom.model_utils import load_model, load_sample, inference, draw_bboxes_cv2
from utils.torch_utils import image_tensor_to_numpy


def main(frames_dir: Path, weights: Union[Path, bool],
         conf_thresh = 0.1, iou_thresh = 0.2, show_time: bool = False
) -> None:
    """Запустить yolo на сканирование папки и обработку всех новых кадров.

    Parameters
    ----------
    frames_dir : Path
        Папка сканирования.
    file_ext : str
        Расширения файлов, которые следует загружать.
    weights : Union[Path, bool]
        Путь к pt файлу с весами модели или bool для загрузки официальных весов.
    conf_thresh : float, optional
        Confidence threshold, by default 0.1
    iou_thresh : float, optional
        IoU threshold, by default 0.2
    show_time : bool, optional
        Показывать время выполнения, by default False
    """
    img_paths = set(frames_dir.glob('*.*'))
    model = None
    image = np.zeros((500, 500, 3), np.uint8)
    while True:
        # Читаем все пути
        updated_paths = set(frames_dir.glob('*.*'))
        # Отсеиваем старые для быстродействия
        new_paths = updated_paths - img_paths
        img_paths = updated_paths

        new_paths = list(new_paths)
        
        if len(new_paths) != 0:
            # Из оставшихся новых берём 1 самый последний
            new_paths.sort()
            pth = new_paths[-1]
            time.sleep(0.1)

            if show_time:
                start = time.time()

            image = load_sample(pth)
            c = image.shape[1]

            # Если прочитали pol
            if c == 4:
                image = image[:, :3]
                c = 3

            # Первая загрузка модели
            if model is None:
                model = load_model(weights, c)
            boxes, class_ids, confidences = inference(
                model, image, conf_thresh, iou_thresh)

            if show_time:
                print('Время обработки:', time.time() - start)

            image = image_tensor_to_numpy(image)
            image = draw_bboxes_cv2(image[0], boxes.tolist()[:30],
                                    class_ids.tolist()[:30], confidences.tolist()[:30])
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
        cv2.imshow('yolo', image)
        k = cv2.waitKey(1) & 0xFF
        # Exit
        if k == 27:  # esc
            cv2.destroyAllWindows()
            break
        # # Save frames
        # elif k == 13:  # enter
        #     flag = 1


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--frames_dir', help='Директория для сканирования.', type=Path)
    parser.add_argument('--weights',
                        help='Веса модели. Путь к pt файлу или bool для загрузки официальных весов',
                        type=str, default='True')
    parser.add_argument('--conf_thresh', help='Порог уверенности модели.', type=float, default=0.8)
    parser.add_argument('--iou_thresh', help='Порог перекрытия рамок.', type=float, default=0.2)
    parser.add_argument('--show_time', help='Показывать время выполнения.', action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    frames_dir = args.frames_dir
    conf_thresh = args.conf_thresh
    iou_thresh = args.iou_thresh
    show_time = args.show_time
    weights = args.weights
    if weights == 'True':
        weights = True
    elif weights == 'False':
        weights = False
    else:
        weights = Path(weights)
    main(frames_dir, weights, conf_thresh, iou_thresh, show_time)