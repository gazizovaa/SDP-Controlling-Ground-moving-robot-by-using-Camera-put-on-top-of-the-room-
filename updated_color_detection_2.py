import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Open the camera (use 0 for the default camera)
cap = cv2.VideoCapture(0)

# Calculate the robot kinematics
radius_of_robots = 1.0
radius_of_motors = 1.0
duration_of_time_steps = 0.1

robot_pose = np.array([0, 0, np.pi / 2])
motor_velocities = np.array([1.0, 1.0])


def calculate_robot_kinematics():
    global robot_pose
    left_velocity, right_velocity = motor_velocities

    # Calculate the linear and angular velocities of motors
    linear_velocity = radius_of_motors * (left_velocity + right_velocity) / 2
    angular_velocity = radius_of_motors * (right_velocity - left_velocity) / (radius_of_robots * 2)

    # Duration in robot's pose
    x_delta = linear_velocity * duration_of_time_steps * np.cos(robot_pose[2])
    y_delta = linear_velocity * duration_of_time_steps * np.sin(robot_pose[2])
    alpha_delta = angular_velocity * duration_of_time_steps

    # Update the robot's pose
    robot_pose = robot_pose + np.array([x_delta, y_delta, alpha_delta])


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
    # blue_mask = cv2.inRange(hsv_frame, lower_blue_color, upper_blue_color)
    # blue_color = cv2.bitwise_and(frame, frame, mask=blue_mask)

    red_mask = cv2.inRange(hsv_frame, lower_red_color, upper_red_color)
    red_color = cv2.bitwise_and(frame, frame, mask=red_mask)

    # Combine two masks to get both color regions
    # combined_mask = cv2.bitwise_or(blue_mask, red_mask)
    # combined_color = cv2.bitwise_and(frame, frame, mask=combined_mask)

    # Find contours in the mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

    return red_color, coordinates


# Function to calculate distance, angle, and midpoint of two coordinates
def calculate_distance_angle_and_midpoint(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2

    # Check if the coordinates are distinct
    if (x1, y1) != (x2, y2):
        # Calculate distance using the Euclidean distance formula
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Calculate angle using arc tangent
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

        # Invoke the function related to the robot kinematics
        calculate_robot_kinematics()

        # Draw the path
        for line_start_point in range(1, len(lines_arr)):
            cv2.line(frame, lines_arr[line_start_point - 1], lines_arr[line_start_point], (0, 0, 0), 6)

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

            print("Distance:", distance)
            print("Angle:", angle)
            print("Midpoint Coordinates:", mid_point)

        # Display the original frame, the detected frame with contours, and the coordinates
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Detected Frame", detected_frame)
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

# Converting the defined list into a NumPy array
lines_arr_np = np.array(lines_arr)


# Declare curve fitting function
def fit_curve(x, a, b, c):
    return a * ((x + b) ** 2) + c


# Identify x and y data points
x_data_points = lines_arr_np[:, 0]
y_data_points = lines_arr_np[:, 1]

# Plot determined x and y data points
plt.plot(x_data_points, y_data_points, 'bo')

# Implement curve fitting
popt, pcov = curve_fit(fit_curve, x_data_points, y_data_points)
print(popt)

# Plot the fitted curve function
plt.plot(x_data_points, fit_curve(x_data_points, *popt))

# Display the shape of data points
print(lines_arr_np.shape)

# Display the plot
plt.scatter(x_data_points, y_data_points)
plt.show()
