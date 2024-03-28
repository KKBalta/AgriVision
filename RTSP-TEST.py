import cv2

# Replace the below URL with your camera's RTSP stream URL
rtsp_url = 'rtsp://192.168.1.251:554/ch01.264?dev=1'

# Use OpenCV to capture the video stream
cap = cv2.VideoCapture(rtsp_url)

while True:
    # Read a frame from the stream
    ret, frame = cap.read()

    # If a frame was successfully read...
    if ret:
        # Display the frame
        cv2.imshow('IP Camera Stream', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Unable to read the stream")
        break

# Release the VideoCapture object and close the window
cap.release()
cv2.destroyAllWindows()