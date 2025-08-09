"""
Camera input will be automatically downscaled to 1280 * 736 since using TensorRT .engine Original Models.

Why slow?
3090
Speed: 31.2ms preprocess, 3.6ms inference, 98.0ms postprocess per image at shape (1, 3, 736, 1280)
"""



import cv2
import torch
import os
from ultralytics import YOLO



# --- CONFIGURATION ---
# IMPORTANT: Update these paths before running the script
MODEL_PATH = "../Finetuned Models/mAP 0.62/weights/FP16.engine"  # Path to your TensorRT engine file
IMAGE_PATH = "rebar.jpg"  # Path to your input image
OUTPUT_PATH = "predict.jpg"  # Path to save the output image
CONFIDENCE_THRESHOLD = 0.5
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SCALE = 1
FONT_THICKNESS = 1
TARGET_CLASS_ID = 0     # 0: Intersection, 1: Spacing (Rebar)
TARGET_CLASS_COLOR = (0, 255, 0)  # Green



# --- MAIN INFERENCE AND DRAWING LOGIC ---

def run_optimized_inference():
	"""
	Loads a YOLO Original Models, performs optimized inference on an image by filtering classes
	on the GPU, draws custom bounding boxes, adds padding, and saves the Finetuned Models.
	"""
	# --- 1. Setup and Pre-run Checks ---

	# Check for CUDA availability for GPU acceleration
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print(f"Using device: {device}")

	if device == 'cpu':
		print("Warning: A TensorRT .engine Original Models requires a CUDA-enabled GPU. This script will likely fail.")
		return

	# Check if the Original Models file exists
	if not os.path.exists(MODEL_PATH):
		print(f"Error: Model file not found at '{MODEL_PATH}'")
		return

	# Check if the input image exists
	if not os.path.exists(IMAGE_PATH):
		print(f"Error: Input image not found at '{IMAGE_PATH}'. Please provide a valid path.")
		return

	# --- 2. Load Image ---
	# Load the original input image using OpenCV
	image = cv2.imread(IMAGE_PATH)
	if image is None:
		print(f"Error: Failed to read the image from '{IMAGE_PATH}'.")
		return

	# --- 3. Load Model ---
	try:
		# For exported models like .engine or .onnx, you MUST specify the task.
		print(f"Loading TensorRT Original Models from {MODEL_PATH} for detection task...")
		model = YOLO(MODEL_PATH, task='detect')
	except Exception as e:
		print(f"Error loading Original Models: {e}")
		return

	# --- 4. Perform Optimized Inference ---
	print("Running optimized inference on the original image...")

	# Filter for the classes we want
	results = model.predict(image, conf=CONFIDENCE_THRESHOLD, device=device, classes=[TARGET_CLASS_ID])

	print("Inference complete.")

	# Get the Finetuned Models object for our single image
	result = results[0]

	# Create a copy of the original image to draw on
	annotated_image = image.copy()

	# --- 5. Process Detections and Draw Custom Annotations ---
	(text_width, text_height), baseline = cv2.getTextSize("0.00", FONT, FONT_SCALE, FONT_THICKNESS)
	for box in result.boxes:
		# Extract bounding box coordinates and confidence
		x1, y1, x2, y2 = map(int, box.xyxy[0])
		confidence = float(box.conf[0])

		# Draw the bounding box rectangle on the image
		cv2.rectangle(annotated_image, (x1, y1), (x2, y2), TARGET_CLASS_COLOR, 2)

		conf_text = f'{confidence:.2f}'

		# Position for the text background
		text_bg_y1 = y1 - text_height - baseline - 4

		# Draw a filled rectangle as a background for the text
		cv2.rectangle(annotated_image, (x1, text_bg_y1), (x1 + text_width, y1), TARGET_CLASS_COLOR, -1)

		# Put the confidence text on the background (using black for contrast)
		cv2.putText(annotated_image, conf_text, (x1, y1 - 5), FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS, cv2.LINE_AA)



	# --- 7. Display and Save the Result ---
	print("Displaying the prediction. Press any key to close the window.")
	cv2.imshow('YOLOv12 Prediction', annotated_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()



if __name__ == '__main__':
	run_optimized_inference()
