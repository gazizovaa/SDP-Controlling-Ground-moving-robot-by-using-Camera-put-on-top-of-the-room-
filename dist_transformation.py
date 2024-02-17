import cv2
import numpy as np

# Open the camera (use 0 for the default camera)
cap = cv2.VideoCapture(0)


# Function to implement distance transformation for a captured frame
def distance_transform(frame):
    # Convert the image color to gray scale
    gray_scale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create a binary image
    ret, threshold = cv2.threshold(gray_scale_image, 130, 255, cv2.THRESH_BINARY)

    # Calculate the distance transformation
    dist_transformation = cv2.distanceTransform(threshold, cv2.DIST_L2, 3)

    cv2.imshow('Transformed Distance Image', dist_transformation)


def draw_curve_with_mouse_events(event, x, y, flags, params):
    global lines_arr
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        lines_arr.append((x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        lines_arr.append((x, y))

    elif event == cv2.EVENT_MOUSEMOVE and (flags and cv2.EVENT_FLAG_LBUTTON):
        lines_arr.append((x, y))


# Create a window
cv2.namedWindow("Original Frame")

# Set the callback function
cv2.setMouseCallback("Original Frame", draw_curve_with_mouse_events)

lines_arr = []


# Function to detect a specific color range and return coordinates
def detect_color_and_get_coordinates(frame, lower_blue_color, upper_blue_color, lower_red_color, upper_red_color):
    # Convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create the mask for both colors
    blue_mask = cv2.inRange(hsv_frame, lower_blue_color, upper_blue_color)
    blue_color = cv2.bitwise_and(frame, frame, mask=blue_mask)

    # red_mask = cv2.inRange(hsv_frame, lower_red_color, upper_red_color)
    # red_color = cv2.bitwise_and(frame, frame, mask=red_mask)

    # Combine two masks to get both color regions
    # combined_mask = cv2.bitwise_or(blue_mask, red_mask)
    # combined_color = cv2.bitwise_and(frame, frame, mask=combined_mask)

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


# Function to calculate distance, angle, and midpoint of two coordinates
def calculate_distance_angle_and_midpoint(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2

    # Check if the coordinates are distinct
    if (x1, y1) != (x2, y2):
        # Calculate distance using the Euclidean distance formula
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Calculate angle using arctangent
        angle_rad = np.arctan2(y2 - y1, x2 - x1)
        angle_deg = np.degrees(angle_rad)

        # Calculate midpoint
        midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)

        return distance, angle_deg, midpoint
    else:
        # Coordinates are the same, return placeholder values
        return float('nan'), float('nan'), (0, 0)


# Blue color range
lower_blue_color = np.array([101, 50, 38])
upper_blue_color = np.array([110, 255, 255])

# Red color range
lower_red_color = np.array([160, 20, 70])
upper_red_color = np.array([190, 255, 255])

try:
    while True:
        # Load the frame
        ret, frame = cap.read()

        # Draw the path
        for start_point in range(1, len(lines_arr)):
            cv2.line(frame, lines_arr[start_point - 1], lines_arr[start_point], (0, 0, 0), 6)

        # Detect the specified color in the frame and get coordinates
        detected_frame, coordinates = detect_color_and_get_coordinates(frame, lower_blue_color, upper_blue_color,
                                                                       lower_red_color, upper_red_color)

        if len(coordinates) >= 2:
            # Calculate the distance, angle, and midpoint between the two detected points
            distance, angle, mid_point = calculate_distance_angle_and_midpoint(coordinates[0], coordinates[1])

            # Draw a line between the two detected points
            cv2.line(frame, coordinates[0], coordinates[1], (0, 255, 0), 2)

            # Draw a circle at the midpoint
            cv2.circle(frame, mid_point, 5, (0, 255, 0), -1)

            # Display distance transformation on a frame
            distance_transform(frame)

            print("Distance:", distance)
            print("Angle:", angle)
            print("Midpoint Coordinates:", mid_point)

        # Display the original frame, the detected frame with contours, and the coordinates
        cv2.imshow("Original Frame", frame)
        # cv2.imshow("Detected Frame", detected_frame)
        print("Coordinates:", coordinates)

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord("D"):  # Deleting circles
            lines_arr = []
except KeyboardInterrupt:
    print("Program is interrupted!")

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
