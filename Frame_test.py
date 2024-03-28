import cv2
import threading
from ultralytics import YOLO
import supervision as sv
import numpy as np


rtsp_url = 'rtsp://192.168.1.244:554/ch01.264?dev=1'

# Use OpenCV to capture the video stream
def process_camera_stream(rtsp_url):
    model = YOLO("yolov8l.pt")  # Use CUDA if available
    for result in model.track(source=rtsp_url, show=False, stream=True, agnostic_nms=True):
        frame = result.orig_img
        detections = sv.Detections.from_ultralytics(result)

        # Resize the frame before drawing bounding boxes
        resized_frame = cv2.resize(frame, (640, 360))  # Adjust the size as needed

        # Calculate scale ratios
        scale_width, scale_height = 640 / frame.shape[1], 360 / frame.shape[0]

        for det in detections:
            if len(det) >= 6 and isinstance(det[0], (list, np.ndarray)) and len(det[0]) == 4:
                # Unpack the bounding box coordinates from the first element assuming it's an array or list
                x1, y1, x2, y2 = det[0]
                confidence, class_id = det[4], int(det[5])  # Adjust these indices if your format is different

                # Debugging: Print out the corrected values
                print(f'Corrected values: x1={x1}, y1={y1}, x2={x2}, y2={y2}, frame width={frame.shape[1]}, scale_width={scale_width}')

                # Scale the bounding box coordinates to the resized frame size
                x1, y1, x2, y2 = [int(x1 * scale_width), int(y1 * scale_height), int(x2 * scale_width), int(y2 * scale_height)]

                # Draw the bounding boxes and labels on the resized frame
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if class_id < len(detections.names):  # Ensure class_id is within the range of detections.names
                    label = f'{detections.names[class_id]} {confidence:.2f}'
                    cv2.putText(resized_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Display the resized frame with bounding boxes
        window_name = rtsp_url.split('/')[-1]  # Create a unique window name based on the RTSP URL
        cv2.imshow(window_name, resized_frame)

        if (cv2.waitKey(30) & 0xFF) == 27:  # Exit if the ESC key is pressed
            break

    cv2.destroyWindow(window_name)  # Ensure the window is closed when exiting the loop

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
