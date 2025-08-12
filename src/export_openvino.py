import os
from pathlib import Path
from ultralytics import YOLO

# --- Configuration ---
# Set the path to your trained model weights.
MODEL_PATH = "../model/Finetuned Models/mAP 0.63/weights/best.pt"
# Choose the quantization type: "FP32", "FP16", or "INT8".
QUANTIZATION = "INT8"


# --- 1. Set Up Paths ---
# Use pathlib for robust path manipulation.
model_path = Path(MODEL_PATH)
model_name_stem = model_path.stem  # Gets the model's base name, e.g., "best"
weights_dir = model_path.parent    # Gets the directory containing the weights.

# Define the final output directory name based on the quantization setting.
# Example: ".../weights/best_int8_openvino_model"
quantized_export_dir = weights_dir / f"{model_name_stem}_{QUANTIZATION.lower()}_openvino_model"


# --- 2. Export Model (Only if it doesn't already exist) ---
# Check if the target directory already exists to avoid overwriting.
if quantized_export_dir.exists():
    print(f"✅ Model directory '{quantized_export_dir.name}' already exists. Skipping export.")
else:
    print(f"Directory '{quantized_export_dir.name}' not found. Starting new export...")
    # Load the base PyTorch model.
    model = YOLO(MODEL_PATH)

    # Define the default directory name that Ultralytics will create during export.
    default_export_dir = weights_dir / f"{model_name_stem}_openvino_model"

    print(f"Starting export with {QUANTIZATION} quantization...")

    # Export the model to OpenVINO format with the specified quantization.
    if QUANTIZATION == "FP32":
        model.export(format="openvino", half=False, int8=False)
    elif QUANTIZATION == "FP16":
        model.export(format="openvino", half=True, int8=False)
    elif QUANTIZATION == "INT8":
        model.export(format="openvino", half=False, int8=True, data="../dataset/data.yaml")
    else:
        # Raise an error for invalid quantization types.
        raise ValueError("Invalid Quantization type. Please choose from 'FP32', 'FP16', or 'INT8'.")

    print(f"Model successfully exported to the default directory: {default_export_dir}")

    # --- 3. Rename the Output Directory ---
    # Rename the newly created default export directory to our custom, quantization-specific name.
    print(f"Renaming '{default_export_dir}' to '{quantized_export_dir}'")
    os.rename(default_export_dir, quantized_export_dir)
    print("✅ Export and rename complete.")

print("\nScript finished.")