#!/usr/bin/env python3

import numpy as np
import cv2
import math
import csv
import time

# Constants
FIXED_HEIGHT_Z = 1300.75  # Camera height above ground (mm)
IMAGE_WIDTH_PIXELS = 160  # Camera resolution width in pixels
IMAGE_HEIGHT_PIXELS = 120  # Camera resolution height in pixels
FOV_HORIZONTAL = np.pi * 57.0 / 180  # Horizontal Field of View (radians)
FOV_DIAGONAL = np.pi * 71.0 / 180  # Diagonal Field of View (radians)

# Calculate Vertical Field of View
FOV_VERTICAL = 2 * np.arctan2(
    np.sqrt(np.square(np.tan(FOV_DIAGONAL / 2)) - np.square(np.tan(FOV_HORIZONTAL / 2))), 
    1
)

# Global variables for initial position
INITIAL_X = None
INITIAL_Y = None

# CSV file setup for logging data
csv_file = open("uav_distances.csv", mode="w", newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp (ms)", "UAV_Global_X (mm)", "UAV_Global_Y (mm)", "UAV_Z (mm)"])


def calculate_position(center):
    """
    Calculate the global X and Y positions of the UAV marker based on the camera's frame center.

    Args:
        center (list): Coordinates of the detected marker center in the frame.

    Returns:
        tuple: Global X and Y positions of the UAV marker.
    """
    global INITIAL_X, INITIAL_Y

    # Angle per pixel for horizontal and vertical directions
    apx = FOV_HORIZONTAL / IMAGE_WIDTH_PIXELS
    apy = FOV_VERTICAL / IMAGE_HEIGHT_PIXELS

    # Marker height
    marker_alt = FIXED_HEIGHT_Z

    # Calculate pixel deltas from image center
    delta_pixel_x = center[0] - IMAGE_WIDTH_PIXELS / 2
    delta_pixel_y = center[1] - IMAGE_HEIGHT_PIXELS / 2

    # Convert pixel deltas to angular deltas
    alpha = np.abs(delta_pixel_x) * apx
    beta = np.abs(delta_pixel_y) * apy

    # Calculate physical distances in mm
    deltax_img = np.sign(delta_pixel_x) * np.tan(alpha) * marker_alt
    deltay_img = np.sign(delta_pixel_y) * np.tan(beta) * marker_alt
    deltaS_img = np.sqrt(np.square(deltax_img) + np.square(deltay_img))

    # Angles for position calculation
    theta_img = np.arctan2(deltay_img, deltax_img)
    theta_horizontal = theta_img - np.pi / 2  # Angle in x-y plane

    # Initialize the global coordinates on the first detection
    if INITIAL_X is None and INITIAL_Y is None:
        INITIAL_X = deltaS_img * np.cos(theta_horizontal)
        INITIAL_Y = deltaS_img * np.sin(theta_horizontal)

    # Compute global coordinates
    global_x = INITIAL_X - deltaS_img * np.cos(theta_horizontal)
    global_y = INITIAL_Y - deltaS_img * np.sin(theta_horizontal)

    return global_x, global_y


# Initialize the camera
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH_PIXELS)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT_PIXELS)

while True:
    # Capture a frame from the video
    ret, cv2_img = cam.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        break

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HSV)

    # Define thermal marker detection range
    lower_thermal_limit = np.array([16, 24, 246])  # Lower bound for H, S, V
    upper_thermal_limit = np.array([36, 255, 255])  # Upper bound for H, S, V

    # Create a binary mask for detected thermal areas
    mask = cv2.inRange(hsv, lower_thermal_limit, upper_thermal_limit)
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # Process detected contours
    if len(contours) > 0:
        # Use the largest contour
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contour = contours[0]

        # Draw bounding box and find the center
        (xg, yg, wg, hg) = cv2.boundingRect(contour)
        cv2.rectangle(cv2_img, (xg, yg), (xg + wg, yg + hg), (0, 255, 0), 2)
        center = [xg + wg / 2, yg + hg / 2]

        # Calculate global coordinates
        global_x, global_y = calculate_position(center)
        print(f"X Position: {global_x:.2f}, Y Position: {global_y:.2f}")

        # Log data to the CSV file
        timestamp = int(time.time() * 1000)  # Current time in milliseconds
        csv_writer.writerow([timestamp, global_x, global_y, FIXED_HEIGHT_Z])

        # Overlay coordinates on the frame
        cv2.putText(cv2_img, f"X: {np.round(global_x)} mm", 
                    (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        cv2.putText(cv2_img, f"Y: {np.round(global_y)} mm", 
                    (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    # Display the frames
    resized_frame = cv2.resize(cv2_img, None, fx=3, fy=3)
    resized_mask = cv2.resize(mask, None, fx=3, fy=3)
    cv2.imshow("Image with Bounding Box", resized_frame)
    cv2.imshow("Mask", resized_mask)

    # Exit on pressing "q"
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cam.release()
csv_file.close()
cv2.destroyAllWindows()
