import cv2
import numpy as np
import subprocess



subprocess.run(["libcamera-still", "-o", "Photos/test.jpg", "-t", "1", "--nopreview", "--hdr"], check=True)

# Load the image
image = cv2.imread('Photos/test.jpg')

# Convert the image to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the color range for black in HSV: hue, saturation, value(lightness)
# For black, the range can be quite narrow
lower_black = np.array([0, 0, 0])
upper_black = np.array([255, 255, 75])
mask = cv2.inRange(hsv, lower_black, upper_black)

# Apply the mask to get only the black parts
masked_image = cv2.bitwise_and(image, image, mask=mask)

# Convert masked image to grayscale
gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 1, 2, apertureSize=3)

# Detect lines using Hough Transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

# Draw the lines on the original image
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 10)

# Show the result
image = cv2.resize(image, (854, 480))
edges = cv2.resize(edges, (854, 480))
cv2.imshow('Detected Lines', image)
cv2.imshow('Detected Edges', edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
