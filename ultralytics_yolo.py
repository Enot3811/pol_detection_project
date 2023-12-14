from pathlib import Path

from ultralytics import YOLO
from ultralytics import settings
import matplotlib.pyplot as plt
import cv2

# Train
model = YOLO('runs/detect/train3/weights/last.pt')  # load a pretrained model (recommended for training)
dset_yaml = 'data/fire_smoke/train_fire_smoke_yolo/data.yaml'
# results = model.train(
#     data=dset_yaml,
#     epochs=100,
#     imgsz=640,
#     verbose=True,
#     batch=32,
#     single_cls=True)
results = model.train(resume=True)

model = YOLO('yolov8l.pt')
results = model.train(
    data=dset_yaml,
    epochs=100,
    imgsz=640,
    verbose=True,
    batch=32,
    single_cls=True)

model = YOLO('yolov8x.pt')
results = model.train(
    data=dset_yaml,
    epochs=100,
    imgsz=640,
    verbose=True,
    batch=32,
    single_cls=True)
##################################33

# model = YOLO('runs/detect/train2/weights/last.pt')  # load a pretrained model (recommended for training)
# dset_yaml = 'data/fire_smoke/train_fire_smoke_yolo/data.yaml'
# results = model.train(resume=True)

# settings.update(datasets_dir='',
#                 weights_dir='')
# print(settings)


# model = YOLO('runs/detect/train_small/weights/best.pt')
# # model.predict('rtsp://192.168.1.18:554/StreamId=1', save=True)

# video_path = "test.mp4"
# cap = cv2.VideoCapture(0)

# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()

#     if success:
#         # Run YOLOv8 tracking on the frame, persisting tracks between frames
#         results = model(frame, conf=0.65)
#         bbox_img = results[0].plot()
#         cv2.imshow('Yolo inference (press any key)', bbox_img)
#         key = cv2.waitKey(1)
#         if key == 27:  # esc
#             break

# img_pths = list(Path('data/fire_smoke/train_fire_smoke_yolo/val/images/').glob('*jpg'))

# for img_pth in img_pths:
#     results = model(img_pth)

#     bbox_img = results[0].plot()
#     cv2.imshow('Yolo inference (press any key)', bbox_img)
#     key = cv2.waitKey(0)
#     if key == 27:  # esc
#         break
# plt.imshow(res_img[..., ::-1])
# plt.show()
# print(results)
