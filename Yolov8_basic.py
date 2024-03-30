
from ultralytics import YOLO
from PIL import Image


video_path = "pics/sheep.mp4"

model=YOLO("Models/yolov8l.pt")

results = model(video_path, save=True)



class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    probs = result.probs  # Class probabilities for classification outputs
    cls = boxes.cls.tolist()  # Convert tensor to list
    xyxy = boxes.xyxy
    xywh = boxes.xywh  # box with xywh format, (N, 4)
    conf = boxes.conf
    print(cls)
    for class_index in cls:
        class_name = class_names[int(class_index)]
        print("Class:", class_name)
