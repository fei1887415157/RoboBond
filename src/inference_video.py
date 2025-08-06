"""
This script performs real-time object detection on a live video stream from a USB camera
using a YOLOv12 model optimized with TensorRT (.engine).

It continuously captures frames, runs inference, draws bounding boxes for a specific
class, and displays the output with an FPS counter.

Press 'q' to quit the application.
PyTorch 2.6.0 + CUDA 12.6
"""



import cv2
import torch
import os
import time
from ultralytics import YOLO



# --- CONFIGURATION ---
# IMPORTANT: Update the model path before running the script
MODEL_PATH = "model/Finetuned Models/mAP 0.62/FP16.engine"  # Path to your TensorRT engine file
CAMERA_INDEX = 0  # 0 for default USB camera, or the specific index of your camera
CONFIDENCE_THRESHOLD = 0.5
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SCALE = 1
FONT_THICKNESS = 1
TARGET_CLASS_ID = 0     # 0: Intersection, 1: Spacing (Rebar)
TARGET_CLASS_COLOR = (0, 255, 0)  # Green
TEXT_COLOR = (0, 0, 0)  # Black



def run_live_inference():
	"""
	Initializes a YOLOv12 model and runs real-time inference on a USB camera feed.
	"""
	# --- 1. Setup and Pre-run Checks ---

	# Check for CUDA availability for GPU acceleration
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print(f"Using device: {device}")

	if device == 'cpu':
		print("Warning: A TensorRT .engine model requires a CUDA-enabled GPU. This script will likely fail.")
		return

	# Check if the model file exists
	if not os.path.exists(MODEL_PATH):
		print(f"Error: Model file not found at '{MODEL_PATH}'")
		return

	# --- 2. Load Model ---
	try:
		# For exported models like .engine or .onnx, you MUST specify the task.
		print(f"Loading TensorRT model from {MODEL_PATH} for detection task...")
		model = YOLO(MODEL_PATH, task='detect')
		print("Model loaded successfully.")
	except Exception as e:
		print(f"Error loading model: {e}")
		return

	# --- 3. Initialize Video Capture ---
	print(f"Opening camera at index {CAMERA_INDEX}...")
	cap = cv2.VideoCapture(CAMERA_INDEX)

	if not cap.isOpened():
		print(f"Error: Could not open camera at index {CAMERA_INDEX}.")
		print("Please check if the camera is connected and the index is correct.")
		return

	# Optional: Set camera properties like resolution. Note that the model will resize
	# the input anyway, but this can help stabilize the input stream.
	# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
	# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
	print("Camera opened successfully. Starting inference loop...")

	# --- 4. Real-time Inference Loop ---
	prev_time = 0
	while True:
		# Read a frame from the camera
		success, frame = cap.read()
		if not success:
			print("Failed to grab frame. End of stream or camera disconnected.")
			break

		# --- Perform Optimized Inference ---
		# The model will handle resizing the frame to its expected input size (e.g., 1280x736)
		results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, device=device, classes=[TARGET_CLASS_ID],
		                        verbose=False)

		# Get the result object for the current frame
		result = results[0]

		# Create a copy of the frame to draw on
		annotated_frame = frame.copy()

		# --- Process Detections and Draw Custom Annotations ---
		for box in result.boxes:
			# Extract bounding box coordinates and confidence
			x1, y1, x2, y2 = map(int, box.xyxy[0])
			confidence = float(box.conf[0])
			conf_text = f'{confidence:.2f}'

			# Calculate text size for the background rectangle
			(text_width, text_height), baseline = cv2.getTextSize(conf_text, FONT, FONT_SCALE, FONT_THICKNESS)

			# Draw the bounding box rectangle on the frame
			cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), TARGET_CLASS_COLOR, 2)

			# --- Draw a filled rectangle as a background for the confidence text ---
			# Position the background rectangle just above the main bounding box
			text_bg_y1 = y1 - text_height - baseline - 4
			text_bg_y2 = y1  # End at the top of the bounding box

			# Draw the filled rectangle
			cv2.rectangle(annotated_frame, (x1, text_bg_y1), (x1 + text_width + 4, text_bg_y2), TARGET_CLASS_COLOR, -1)

			# Put the confidence text on the background
			cv2.putText(annotated_frame, conf_text, (x1 + 2, y1 - baseline), FONT, FONT_SCALE, TEXT_COLOR,
			            FONT_THICKNESS, cv2.LINE_AA)

		# --- Calculate and Display FPS ---
		curr_time = time.time()
		# Avoid division by zero on the first frame
		if prev_time > 0:
			fps = 1 / (curr_time - prev_time)
			cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), FONT, 1, (0, 0, 255), 2)
		prev_time = curr_time

		# --- Display the Result ---
		cv2.imshow('YOLOv12 Live Inference', annotated_frame)

		# --- Exit Condition ---
		# Wait for 1ms and check if the 'q' key was pressed
		if cv2.waitKey(1) & 0xFF == ord('q'):
			print("'q' pressed. Exiting...")
			break

	# --- 5. Cleanup ---
	print("Releasing resources.")
	cap.release()
	cv2.destroyAllWindows()



if __name__ == '__main__':
	run_live_inference()
