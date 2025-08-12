from ultralytics import YOLO

MODEL_PATH = "../model/Finetuned Models/mAP 0.63/weights/best.pt"



model = YOLO(MODEL_PATH)

# Export the model
model.export(format="openvino")  # creates 'yolo11n_openvino_model/'

# Load the exported OpenVINO model
ov_model = YOLO("../model/Finetuned Models/mAP 0.63/weights/best_openvino_model/best.bin")

# Run inference with specified device, available devices: ["intel:gpu", "intel:npu", "intel:cpu"]
results = ov_model("https://ultralytics.com/images/bus.jpg", device="intel:npu")