#libcamera-vid -t 0 --inline --width 820 --height 662 --framerate 30 --hdr




import cv2
import subprocess
import numpy as np
import time

global proc, stdout

# Command to start libcamera-vid and output to stdout
#cmd = "libcamera-vid -t 3000 --inline --width 1640 --height 1232 --framerate 30 --hdr -n"
cmd = "libcamera-raw -t 1000 --inline --width 1640 --height 1232 --framerate 30"
proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1 * 1024 * 1024 * 1024)
time.sleep(2)
stdout_data, stderr_data = proc.communicate()

if stderr_data:
    print(f"Error output: {stderr_data.decode()}")

if stdout_data:
    print(f"Standard output: {stdout_data.decode()}")

# Initialize OpenCV window for displaying processed frames
cv2.namedWindow("Line Detection", cv2.WINDOW_NORMAL)



while False:
	
	print("running")

	proc.wait()

	# Read frames from libcamera-vid with pipe
	raw_frame = proc.stdout.read(1232 * 1640 * 3)

	if len(raw_frame) == 0:
		print("no frame, exiting")
		break

	#frame = np.frombuffer(raw_frame, dtype=np.uint8)
	frame = np.array(raw_frame)
	#frame = frame.reshape((1232, 1640, 3)) 		# notice they flips

	print("0")

	



	print("1")

	# Perform line detection (example: using Canny edge detection)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	print("2")
	
	# Use Canny edge detection
	edges = cv2.Canny(gray, threshold1=100, threshold2=300, apertureSize=3)

	print("3")

	# Use Probabilistic Hough Line Transform to detect lines
	lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

	print("4")

	# Draw the lines on the image
	for line in lines:
		x1, y1, x2, y2 = line[0]
		cv2.line(frame, (x1,y1), (x2,y2), (0,0,255), 20)     # overlay lines color and thickness

	print("5")

	resized_canny = cv2.resize(edges, (665, 500))
	resized_frame = cv2.resize(frame, (665, 500))

	print("6")

	# Display the resulting image
	#cv2.imshow("Canny Edges", resized_canny)
	#cv2.imshow("Frame", resized_frame)

	# Exit loop on 'q' key press
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break



# Release resources
proc.kill()
cv2.destroyAllWindows()
