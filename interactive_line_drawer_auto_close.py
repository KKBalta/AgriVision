
import cv2

# Initialize list to store line end points
line_points = []

def draw_line(event, x, y, flags, param):
    global line_points, img

    if event == cv2.EVENT_LBUTTONDOWN:
        line_points = [(x, y)]  # First click - add start point

    elif event == cv2.EVENT_LBUTTONUP:
        line_points.append((x, y))  # Second click - add end point

        # Draw the line on the image
        cv2.line(img, line_points[0], line_points[1], (0, 255, 0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(500)  # Wait for 500 ms to show the line
        cv2.destroyAllWindows()  # Close the window

# Load an image or a frame from your video
img = cv2.imread('pics/stabley.jpg')

cv2.imshow("Image", img)
cv2.setMouseCallback("Image", draw_line)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Use line_points for your counting logic
