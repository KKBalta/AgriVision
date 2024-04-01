
from ultralytics import YOLO
from PIL import Image


video_path = "pics/cars.mp4"

model=YOLO("Models/yolov8l.pt")

results = model(video_path, save=True)
