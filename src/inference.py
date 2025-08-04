from ultralytics import YOLO

#model = YOLO("../result/mAP 0.62/weights/best.pt")
model = YOLO("../result/mAP 0.63/weights/best.pt")
model.predict()
