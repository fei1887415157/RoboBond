#
# Description:
# PyTorch .pt -> ONNX -> TensorRT .engine
# FP32 / FP16 / INT8
#
# Prerequisites:
# 1. An NVIDIA GPU with CUDA, cuDNN, and TensorRT installed.
# 2. The 'ultralytics' Python package installed (`pip install ultralytics`).
# 3. A trained model file (e.g., 'best.pt').
# 4. For INT8 conversion, you need your dataset's YAML file (e.g., 'rebar_dataset.yaml').
#    Ultralytics uses the validation set specified in this YAML for calibration.
#
# How to Run:
# 1. Place this script in your YOLOv8 project directory.
# 2. Make sure 'best.pt' and your dataset YAML file are in the expected paths.
# 3. Run the script from your terminal: `python convert_model.py`
#

from ultralytics import YOLO
import torch



# --- Configuration ---
model_path = "../result/mAP 0.62/weights/best.pt"
# Only convert one at a time to prevent overwriting
print("1: FP32 \t 2: FP16 \t 3: INT8")
print("Input: ")
i = input()
match i:
	case "1":   precision = "FP32"
	case "2":   precision = "FP16"
	case "3":   precision = "INT8"
	case _:     print("Wrong input. Exiting."); exit()



# Input image size the model was trained on
# [height, width]
image_size = [720, 1280]

# GPU device to use for conversion (0 for the first GPU)
device_id = 0

# Check if a GPU is available
if not torch.cuda.is_available():
	print("CUDA is not available. TensorRT conversion requires an NVIDIA GPU.")

print(f"Using GPU device: {torch.cuda.get_device_name(device_id)}")
print("-" * 30)

# --- Step 1: Load the YOLO model ---
# This loads your trained weights and model architecture.
try:
	model = YOLO(model_path)
	print(f"Successfully loaded model from '{model_path}'")
except Exception as e:
	print(f"Error loading model: {e}")
	print("Please ensure 'best.pt' is in the same directory as this script.")


def convert_model_FP32():
	# --- Step 2: Convert to FP32 TensorRT Engine ---
	# FP32 (single-precision) offers a great balance of speed and accuracy.
	# It's usually the recommended first step for optimization.
	print("\nStarting FP32 conversion...")
	try:
		model.export(
			format='engine',  # Specify the TensorRT format
			imgsz=image_size,  # Define the input image size
			half=False,  # Disable FP16 quantization
			device=device_id,  # Specify the GPU device
			verbose=False  # Set to True for more detailed export logs
		)
		print(f"✅ FP32 conversion successful!")
		print(f"   Engine saved as: {model_path.replace('.pt', '.engine')}")
	except Exception as e:
		print(f"❌ FP32 conversion failed: {e}")


def convert_model_FP16():
	# --- Step 3: Convert to FP16 TensorRT Engine ---
	# FP16 (half-precision) offers a great balance of speed and accuracy.
	# It's usually the recommended first step for optimization.
	print("\nStarting FP16 conversion...")
	try:
		model.export(
			format='engine',  # Specify the TensorRT format
			imgsz=image_size,  # Define the input image size
			half=True,  # Enable FP16 quantization
			device=device_id,  # Specify the GPU device
			verbose=False  # Set to True for more detailed export logs
		)
		print(f"✅ FP16 conversion successful!")
		print(f"   Engine saved as: {model_path.replace('.pt', '.engine')}")
	except Exception as e:
		print(f"❌ FP16 conversion failed: {e}")


def convert_model_INT8():
	# --- Step 4: Convert to INT8 TensorRT Engine ---
	# INT8 provides the highest performance but requires a 'calibration' step.
	# Ultralytics automatically uses your validation dataset for calibration.
	# Make sure your dataset's YAML file is correctly referenced by your model
	# or specify it with the `data='path/to/your/dataset.yaml'` argument in export().
	print("\nStarting INT8 conversion...")
	print("Note: INT8 conversion requires your dataset's YAML file for calibration.")
	try:
		# We re-export from the original .pt model
		model.export(
			format='engine',  # Specify the TensorRT format
			imgsz=image_size,  # Define the input image size
			int8=True,  # Enable INT8 quantization
			device=device_id,  # Specify the GPU device
			verbose=False,  # Set to True for more detailed export logs
			data="../dataset/data.yaml"
		)
		print(f"✅ INT8 conversion successful!")
		print(f"   Engine saved as: {model_path.replace('.pt', '.engine')}")

	except Exception as e:
		print(f"❌ INT8 conversion failed: {e}")
		print("   Common issues include:")
		print("   - The dataset YAML file could not be found.")
		print("   - The calibration data (validation set images) is missing or corrupt.")



if __name__ == '__main__':
	print("-" * 30)
	match precision:
		case "FP32":    convert_model_FP32()
		case "FP16":    convert_model_FP16()
		case "INT8":    convert_model_INT8()
	print("-" * 30)