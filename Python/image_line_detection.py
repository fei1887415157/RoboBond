import cv2
import numpy as np
import math



# Load the image
image_path = '/home/robobond/Desktop/OpenCV_Test/Photos/test.jpg'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection
edges = cv2.Canny(gray, 100, 300, 3)

# Detect lines using the Hough Line Transform
lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)

# Draw the lines on the original image
if lines is not None:
	for rho, theta in lines[:, 0]:
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a * rho
		y0 = b * rho
		x1 = int(x0 + 1000 * (-b))
		y1 = int(y0 + 1000 * (a))
		x2 = int(x0 - 1000 * (-b))
		y2 = int(y0 - 1000 * (a))
		cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

resized_image = cv2.resize(image, (math.floor(3280/5), math.floor(2464/5)))

# Display the result
cv2.imshow('Detected Lines', resized_image)
cv2.waitKey(0)      # wait for any key press
cv2.destroyAllWindows()
