import cv2
import numpy as np
import subprocess
import math
import time



image = cv2.imread("Photos/test.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred_gray = cv2.GaussianBlur(gray, (51, 51), 0)

resized_blur = cv2.resize(blurred_gray, (854, 480))

cv2.imshow("Blurred Gray", resized_blur)

key = cv2.waitKey(0)