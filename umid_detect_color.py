import cv2
import numpy as np


def detect_color_and_get_coordinates(frame, lower_color1, upper_color1, lower_color2, upper_color2):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for both colors
    color_mask1 = cv2.inRange(hsv_frame, lower_color1, upper_color1)
    color_mask2 = cv2.inRange(hsv_frame, lower_color2, upper_color2)

    # Combine the masks to get both color regions
    combined_mask = cv2.bitwise_or(color_mask1, color_mask2)

    # Find contours in the combined mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    coordinates = []

    for contour in contours:
        # Calculate moments for each contour
        M = cv2.moments(contour)

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            coordinates.append((cX, cY))

    return combined_mask, coordinates


def calculate_distance_angle_and_midpoint(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2

    # Check if the coordinates are distinct
    if (x1, y1) != (x2, y2):
        # Calculate distance using the Euclidean distance formula
        distance = np.sqrt((x2 - x1) * 2 + (y2 - y1) * 2)

        # Calculate angle using arctangent
        angle_rad = np.arctan2(y2 - y1, x2 - x1)
        angle_deg = np.degrees(angle_rad)

        # Calculate midpoint
        midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)

        return distance, angle_deg, midpoint
    else:
        # Coordinates are the same, return placeholder values
        return float('nan'), float('nan'), (0, 0)


lower_color1 = np.array([30, 100, 100])
upper_color1 = np.array([60, 255, 255])

lower_color2 = np.array([90, 100, 100])
upper_color2 = np.array([120, 255, 255])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    detected_frame, coordinates = detect_color_and_get_coordinates(frame, lower_color1, upper_color1, lower_color2,
                                                                   upper_color2)

    if len(coordinates) >= 2:
        # Calculate the distance, angle, and midpoint between the two detected points
        distance, angle, mid_point = calculate_distance_angle_and_midpoint(coordinates[0], coordinates[1])

        # Draw a line between the two detected points
        cv2.line(frame, coordinates[0], coordinates[1], (0, 255, 0), 2)

        # Draw a circle at the midpoint
        cv2.circle(frame, mid_point, 5, (0, 255, 0), -1)

        print("Distance:", distance)
        print("Angle:", angle)
        print("Midpoint Coordinates:", mid_point)

    cv2.imshow("Original Frame", frame)
    cv2.imshow("Detected Color", detected_frame)

    cv2.waitKey(200)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
