import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.append(str(Path(__file__).parents[1]))
from utils.torch_utils.torch_functions import draw_bounding_boxes
from utils.image_utils.image_functions import show_images_cv2


if __name__ == '__main__':
    # Get the model
    model = 'yolov8l.pt'
    model = YOLO(model)
    model.fuse()
    # Configure classes
    # car, motorcycle, plane, bus and truck
    selected_classes = [2, 3, 4, 5, 7]
    id_to_class = model.model.names
    exclude_classes = [cls_label
                       for cls_id, cls_label in id_to_class.items()
                       if cls_id not in selected_classes]
    # Get the video
    pth = 'data/camera/2024_03_28_zoom/точка1/zoom36_1.mp4'
    vidcap = cv2.VideoCapture(pth)

    # Iterate over frames
    success = True
    while success:
        success, frame = vidcap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = model(frame, verbose=False)[0]
        boxes = result.boxes
        if len(boxes.cls) != 0:
            bboxes = boxes.xyxy.tolist()
            labels = list(map(lambda cls_id: id_to_class[int(cls_id)],
                              boxes.cls.tolist()))
            frame = draw_bounding_boxes(
                frame, bboxes, labels, exclude_classes=exclude_classes)
        show_images_cv2(frame, delay=1)
