import cv2
import threading
from ultralytics import YOLO
import supervision as sv
import numpy as np


rtsp_url = 'rtsp://192.168.1.244:554/ch01.264?dev=1'

# Use OpenCV to capture the video stream
def process_camera_stream(rtsp_url):
    model = YOLO("yolov8l.pt")
    # Pass the RTSP URL directly instead of the cv2.VideoCapture object
    for result in model.track(source=rtsp_url, show=True, stream=True, agnostic_nms=True):
        frame = result.orig_img
        detections = sv.Detections.from_ultralytics(result)

        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        
        detections = detections[(detections.class_id != 60) & (detections.class_id != 0)]

        if (cv2.waitKey(30) == 27):  # Wait for ESC key to exit
            break
def main():
    # List of your camera IPs
    camera_ips = [
        "192.168.1.246",
        "192.168.1.242",
        
        # Add more IP addresses as needed
    ]

    # Constructing RTSP URLs from the IP addresses
    rtsp_urls = [f"rtsp://{ip}:554/ch01.264?dev=1" for ip in camera_ips]

    threads = []

    for url in rtsp_urls:
        thread = threading.Thread(target=process_camera_stream, args=(url,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()
