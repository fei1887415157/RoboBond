import cv2
import numpy as np
import multiprocessing
import subprocess



# Tested on Pi 5, default variable fan speed
# Camera: 	IMX219, 1/4'', 8MP, Arducam, libcamera-vid,
# 			FOV 62.2 H 48.8 V, focus 20cm - inf, rolling shutter, no IR
	
# Resolutions (max FOV)      FPS			CPU/GPU Temp (C)
# max  	3280 * 2464   	   10 - 21 		    80, thermal throttle !
# half 	1640 * 1323			 30					70
# min 	820  * 662			 30					67

def start_video_stream():
	subprocess.run(["libcamera-vid", "-t", "0", "--inline", "--width", "820", "--height", "662", "--framerate", "30", "|", "nc", "-l", "-p", "5000"], check=True)



def main():
	
	# Open a connection to the video stream using GStreamer pipeline
	#capture = cv2.VideoCapture("udpsrc port=5000 ! application/x-rtp, encoding-name=JPEG,payload=26 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
	capture = cv2.VideoCapture("udp://127.0.0.1:5000")
	capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)
	capture.set(cv2.CAP_PROP_FRAME_WIDTH, 820)
	capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 662)

	if not capture.isOpened():
		print("Error: Could not open video UDP stream.")
		return



	while True:
		# Capture frame-by-frame
		ret, frame = capture.read()
		
		if not ret:
			print("Error: Could not read frame.")
			break
		
		# Convert to grayscale
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		# Use Canny edge detection
		edges = cv2.Canny(gray, threshold1=100, threshold2=300, apertureSize=3)
		
		# Use Probabilistic Hough Line Transform to detect lines
		lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
		
		# Draw the lines on the image
		for line in lines:
			x1, y1, x2, y2 = line[0]
			cv2.line(frame, (x1,y1), (x2,y2), (0,0,255), 20)     # overlay lines color and thickness
		
		#resized_frame = cv2.resize(frame, (820, 662))

		# Check if the frame is empty
		if frame is None or frame.size == 0:
			print("Received empty frame.")
		else:
			print("frame ok")

		# Display the resulting frame
		cv2.imshow('UDP Video Stream', frame)
		
		# Break the loop on any key press
		if cv2.waitKey(0):
			break
	


	# When everything is done, release the capture and close windows
	capture.release()
	cv2.destroyAllWindows()





if __name__ == "__main__":
	p1 = multiprocessing.Process(target=start_video_stream)
	p2 = multiprocessing.Process(target=main)

	# Start the processes
	p1.start()
	p2.start()

	# Wait for processes to complete
	p1.join()
	p2.join()

	print("Both processes have finished")
