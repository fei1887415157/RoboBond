import cv2
import time  # Import the time library to calculate FPS
from ultralytics import YOLO
from arduino import ArduinoCommunicator


# --- Configuration ---
ARDUINO_PORT = "COM4"     # Commonly "/dev/ttyACM0" or "/dev/ttyUSB0" on Jetson, or "COM5" on Windows
ARDUINO_COOLDOWN = 5  # Seconds to wait between sending signals to the Arduino
# Path to your fine-tuned PyTorch model (.pt file)
MODEL_PATH = "../model/Finetuned Models/mAP 0.63/weights/best_fp32_openvino_model"
# Confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.6
# Index of the camera to use (0 is usually the default webcam)
CAMERA_INDEX = 1
# Target class ID for inference
TARGET_CLASS_ID = 0
# Bounding box color (Green in BGR format)
BOX_COLOR = (0, 255, 0)
# Bounding box thickness
BOX_THICKNESS = 5
# --- MODIFICATION: Center detection parameters ---
CENTER_TOLERANCE_PX = 100
CENTER_BOX_COLOR = (255, 255, 0)  # Cyan for centered objects
# FPS counter text color (Red in BGR format)
FPS_COLOR = (0, 0, 255)
# --- MODIFICATION: Define display window size ---
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = int (DISPLAY_WIDTH / 16 * 9)


# --- Load the exported OpenVINO model ---
print(f"Loading OpenVINO model from '{MODEL_PATH}'...")
try:
    # The YOLO class automatically finds the correct model file within the directory
    ov_model = YOLO(MODEL_PATH, task="detect")
    print("OpenVINO model loaded successfully.")
except Exception as e:
    print(f"Error loading OpenVINO model: {e}")
    print("Please ensure the model path is correct and the model has been exported properly.")
    exit()

# --- Initialize Video Capture ---
# We use the global CAMERA_INDEX variable to select the camera.
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"Error: Could not open camera at index {CAMERA_INDEX}.")
    exit()

# --- Set camera capture resolution and FPS ---
print("Attempting to set camera resolution to 1920x1080 @ 30 FPS...")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

# --- Initialize Arduino Communication ---
arduino = ArduinoCommunicator(port=ARDUINO_PORT)
arduino_connected = arduino.connect()
if not arduino_connected:
    # The script can still run the camera feed without the Arduino
    print("Warning: Could not connect to Arduino. Video inference will continue without it.")

# --- Verify the settings ---
# It's good practice to check what settings the camera actually accepted.
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Actual camera settings: {actual_width}x{actual_height} @ {actual_fps:.2f} FPS")
if actual_width != 1920 or actual_height != 1080:
    print("Warning: Camera did not accept the requested resolution and may be using a default.")


print("Camera opened. Starting live inference...")
print(f"Detecting only class ID: {TARGET_CLASS_ID}")
print("Press 'q' to quit.")

# --- FPS Counter Variables ---
# We need to store the time of the previous frame to calculate the time difference.
prev_time = 0
fps = 0

# --- Step 4: Main Loop for Live Inference ---
last_signal_time = 0  # Track when the last signal was sent to enforce a cooldown
while True:
    # --- FPS Calculation (Start) ---
    # Get the current time.
    current_time = time.time()

    # Capture frame-by-frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # Run inference on the current frame using the OpenVINO model.
    try:
        # Added verbose=False to hide terminal output
        results = ov_model(frame, device="intel:gpu", conf=CONFIDENCE_THRESHOLD, classes=TARGET_CLASS_ID, verbose=False)
    except Exception as e:
        # Fallback to CPU if GPU fails or is not available
        print(f"Inference on GPU failed: {e}. Falling back to CPU.")
        # Added verbose=False to hide terminal output
        results = ov_model(frame, device="intel:cpu", conf=CONFIDENCE_THRESHOLD, classes=TARGET_CLASS_ID, verbose=False)

    annotated_frame = frame.copy()

    # Get the bounding box results
    boxes = results[0].boxes

    # --- MODIFICATION: Logic to detect objects in the center of the frame ---
    frame_height, frame_width, _ = frame.shape
    image_center_x = frame_width // 2
    image_center_y = frame_height // 2

    centered_object_detected = False

    # For visualization, draw the center tolerance zone
    zone_x1 = image_center_x - CENTER_TOLERANCE_PX
    zone_y1 = image_center_y - CENTER_TOLERANCE_PX
    zone_x2 = image_center_x + CENTER_TOLERANCE_PX
    zone_y2 = image_center_y + CENTER_TOLERANCE_PX
    cv2.rectangle(annotated_frame, (zone_x1, zone_y1), (zone_x2, zone_y2), CENTER_BOX_COLOR, 2)

    # Iterate over each detected box
    for box in boxes:
        # Get the coordinates of the bounding box (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Calculate the center of the bounding box
        box_center_x = (x1 + x2) // 2
        box_center_y = (y1 + y2) // 2

        # Check if the box center is within the tolerance of the image center
        if (image_center_x - CENTER_TOLERANCE_PX <= box_center_x <= image_center_x + CENTER_TOLERANCE_PX) and \
           (image_center_y - CENTER_TOLERANCE_PX <= box_center_y <= image_center_y + CENTER_TOLERANCE_PX):
            centered_object_detected = True
            # Draw the centered box with a special color
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), CENTER_BOX_COLOR, BOX_THICKNESS)
        else:
            # Draw other boxes with the default color
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

    # --- FPS Calculation & Display ---
    # Calculate the frames per second (FPS).
    # FPS is the reciprocal of the time difference between frames.
    if prev_time > 0:
        fps = 1 / (current_time - prev_time)

    # Update the previous time for the next frame's calculation.
    prev_time = current_time

    # Prepare the FPS text to be displayed.
    fps_text = f"FPS: {int(fps)}"

    # Use cv2.putText to draw the FPS counter on the frame.
    # Arguments: frame, text, position (top-left), font, font scale, color, thickness
    cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, FPS_COLOR, 2)

    # --- Send Signal to Arduino with Cooldown ---
    # If a centered object is detected, send a signal, but only if the cooldown period has passed.
    if arduino_connected and centered_object_detected and (time.time() - last_signal_time > ARDUINO_COOLDOWN):
        print(f"Centered object detected. Sending 'DETECT' signal to Arduino on {ARDUINO_PORT}.")
        arduino.send("DETECT\n")
        last_signal_time = time.time()  # Reset the cooldown timer

    # --- MODIFICATION: Resize the frame for display ---
    # Downscale the annotated frame to the desired display size.
    display_frame = cv2.resize(annotated_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    # Display the resulting frame in a window
    cv2.imshow('YOLO Live Inference (OpenVINO)', display_frame)

    # Check for the 'q' key to be pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Step 5: Clean Up ---
# When the loop is exited, release the camera and destroy all OpenCV windows.
print("Clean up...")
# Close the Arduino connection if it was successfully opened
if arduino_connected:
    arduino.close()
cap.release()
cv2.destroyAllWindows()
