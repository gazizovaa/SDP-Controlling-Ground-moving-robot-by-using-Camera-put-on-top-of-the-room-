import cv2
import numpy as np

# Open the camera (use 0 for the default camera)
cap = cv2.VideoCapture(0)


def draw_curve_with_mouse_events(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        circles_arr.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        print(x, y)
        circles_arr.append((x, y))
    elif event == cv2.EVENT_MOUSEMOVE:
        print(x, y)


# Create a window
cv2.namedWindow("Original Frame")

# Set the callback function
cv2.setMouseCallback("Original Frame", draw_curve_with_mouse_events)

circles_arr = []


# Function to detect a specific color range and return coordinates
def detect_color_and_get_coordinates(frame, lower_blue_color, upper_blue_color):
    # Convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the blue color range
    blue_mask = cv2.inRange(hsv_frame, lower_blue_color, upper_blue_color)
    blue_color = cv2.bitwise_and(frame, frame, mask=blue_mask)

    # Create a mask for the red color range
    # red_mask = cv2.inRange(hsv_frame, lower_red_color, upper_red_color)
    # red_color = cv2.bitwise_and(frame, frame, mask=red_mask)

    # Find contours in the mask
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a list to store the coordinates
    coordinates = []

    # Iterate through the detected contours and get their coordinates
    for contour in contours:
        M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        coordinates.append((cX, cY))

    # Draw the contours on the frame
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    return blue_color, coordinates


# Blue color range
lower_blue_color = np.array([101, 50, 38])
upper_blue_color = np.array([110, 255, 255])

# Red color range
# lower_red_color = np.array([160, 20, 70])
# upper_red_color = np.array([190, 255, 255])

try:
    while True:
        # Load the frame
        ret, frame = cap.read()

        # Draw the circles
        for center_pos in circles_arr:
            cv2.circle(frame, center_pos, 10, (255, 255, 255), -1)

        # Detect the specified color in the frame and get coordinates
        detected_frame, coordinates = detect_color_and_get_coordinates(frame, lower_blue_color, upper_blue_color)

        # Display the original frame, the detected frame with contours, and the coordinates
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Detected Frame", detected_frame)
        print("Coordinates:", coordinates)

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord("D"):  # Deleting circles
            circles_arr = []
except KeyboardInterrupt:
    print("Program is interrupted!")

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
