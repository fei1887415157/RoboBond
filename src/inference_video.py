"""
This script performs real-time object detection on a live video stream from a USB camera
using a YOLOv12 Original Models optimized with TensorRT (.engine).

It continuously captures frames, applies a 2x digital zoom, runs inference,
draws bounding boxes for a specific class, and displays the output with an FPS counter.

Press 'q' to quit the application.
PyTorch 2.6.0 + CUDA 12.6
"""

import cv2
import torch
import os
import time
from ultralytics import YOLO
from arduino import ArduinoCommunicator  # Import the new class

# --- CONFIGURATION ---
MODEL_PATH = "../Finetuned Models/mAP 0.63/weights/FP32.engine"  # Path to your TensorRT engine file
CAMERA_INDEX = 1  # 0 for default USB camera, or the specific index of your camera
ARDUINO_PORT = "/dev/ttyACM0"     # Commonly "/dev/ttyACM0" or "/dev/ttyUSB0" on Jetson, or "COM5" on Windows
ARDUINO_COOLDOWN = 2  # Seconds to wait between sending signals to the Arduino
CONFIDENCE_THRESHOLD = 0.5
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SCALE = 1
FONT_THICKNESS = 1
TARGET_CLASS_ID = 0  # 0: Intersection, 1: Spacing (Rebar)
TARGET_CLASS_COLOR = (0, 255, 0)  # Green
TEXT_COLOR = (0, 0, 0)  # Black
ZOOM_FACTOR = 1.0 # The zoom level to apply

# --- New Display Window Size Configuration ---
DISPLAY_WIDTH = 1000
DISPLAY_HEIGHT = int(DISPLAY_WIDTH / (1920 / 1080))


def run_live_inference():
    """
    Initializes a YOLOv12 Original Models and runs real-time inference on a USB camera feed.
    """
    # --- 1. Setup and Pre-run Checks ---

    # Check for CUDA availability for GPU acceleration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if device == 'cpu' and MODEL_PATH.endswith('.engine'):
        print("Error: A TensorRT .engine model requires a CUDA-enabled GPU and cannot run on CPU. Exiting.")
        return
    elif device == 'cpu':
        print("Warning: No CUDA-enabled GPU found. Inference will run on the CPU, which may be slow.")

    # Check if the Original Models file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        return

    # --- 2. Load Model ---
    try:
        # For exported models like .engine or .onnx, you MUST specify the task.
        print(f"Loading model from {MODEL_PATH} for detection task...")
        model = YOLO(MODEL_PATH, task='detect')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading Original Models: {e}")
        return

    # --- 3. Initialize Video Capture ---
    print(f"Opening camera at index {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"Error: Could not open camera at index {CAMERA_INDEX}.")
        print("Please check if the camera is connected and the index is correct.")
        return

    # --- Set desired camera properties ---
    # Note: This is a request. The camera will use the closest supported resolution.
    print("Requesting 1920 * 1080 @ 30fps...")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # --- Get and print actual camera properties ---
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Actual camera resolution: {frame_width}x{frame_height}")
    print(f"Actual camera FPS: {fps}")

    # --- Initialize Arduino Communication ---
    arduino = ArduinoCommunicator(port=ARDUINO_PORT)
    arduino_connected = arduino.connect()
    if not arduino_connected:
        # The script can still run the camera feed without the Arduino
        print("Warning: Could not connect to Arduino. Video inference will continue without it.")


    print("Camera opened successfully. Starting inference loop...")

    # --- 4. Real-time Inference Loop ---
    prev_time = 0
    last_signal_time = 0  # Track when the last signal was sent to enforce a cooldown
    while True:
        # Read a frame from the camera
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame. End of stream or camera disconnected.")
            break

        # --- START: ZOOM LOGIC ---
        # Get original frame dimensions
        h, w, _ = frame.shape

        # Calculate the dimensions of the cropped area
        crop_w = int(w / ZOOM_FACTOR)
        crop_h = int(h / ZOOM_FACTOR)

        # Calculate the top-left corner of the crop to keep it centered
        crop_x = (w - crop_w) // 2
        crop_y = (h - crop_h) // 2

        # Crop the frame to the center to create the zoomed-in view
        zoomed_frame = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
        # --- END: ZOOM LOGIC ---

        # --- Perform Optimized Inference on the ZOOMED frame ---
        # The Original Models will handle resizing the frame to its expected input size
        results = model.predict(zoomed_frame, conf=CONFIDENCE_THRESHOLD, device=device, classes=[TARGET_CLASS_ID],
                                verbose=False)

        # Get the Finetuned Models object for the current frame
        result = results[0]

        # Create a copy of the zoomed frame to draw on.
        # We draw on the smaller cropped frame for efficiency before resizing.
        annotated_frame = zoomed_frame.copy()

        # --- Process Detections and Draw Custom Annotations ---
        # Bounding box coordinates are relative to the `zoomed_frame`
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

        # --- Send Signal to Arduino with Cooldown ---
        # If an object is detected, send a signal, but only if the cooldown period has passed.
        if arduino_connected and len(result.boxes) > 0 and (time.time() - last_signal_time > ARDUINO_COOLDOWN):
            print(f"Object detected. Sending 'DETECT' signal to Arduino on {ARDUINO_PORT}.")
            arduino.send("DETECT\n")
            last_signal_time = time.time()  # Reset the cooldown timer

        # --- Resize for Display ---
        # Resize the annotated frame to the desired display size
        display_frame = cv2.resize(annotated_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_LINEAR)

        # --- Calculate and Display FPS ---
        curr_time = time.time()
        # Avoid division by zero on the first frame
        if prev_time > 0:
            fps = 1 / (curr_time - prev_time)
            # Draw FPS on the final, resized frame
            cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 30), FONT, 2, (0, 0, 255), 2)
        prev_time = curr_time

        # --- Display the Result ---
        cv2.imshow('YOLOv12 Live Inference', display_frame)

        # --- Exit Condition ---
        # Wait for 1ms and check if the 'q' key was pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("'q' pressed. Exiting...")
            break

    # --- 5. Cleanup ---
    print("Releasing resources.")
    # Close the Arduino connection if it was successfully opened
    if arduino_connected:
        arduino.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_live_inference()
