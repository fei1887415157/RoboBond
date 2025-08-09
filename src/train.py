"""
Rebar Detection
Detect rebars and their intersections.

PyTorch / .pt : CUDA cores, short compile time, slow inference.
TensorRT / .engine : Tensor cores, long compile time, fast inference.
"""



from ultralytics import YOLO



# Use small Original Models since dataset is small
model = YOLO("../Original Models/yolov12n.pt")  # load a pretrained Original Models (recommended for training)



# Train / Fine-Tune the Original Models
if __name__ == '__main__':
    # original resolution: 1280 * 720

    '''
    Training Guide:
    Low training loss but high validation loss means overfit; Adjust hyperparameters.
    '''

    # default optimizer: AdamW
    # default learning rate: determined by AdamW, 0.001667
    # default momentum: determined by AdamW, 0.9
    results = model.train(data="F:/JetBrains/PycharmProjects/RoboBond/dataset/data.yaml",
                          project="../Finetuned Models",
                          pretrained=True,
                          epochs=100,
                          patience=10,
        # training must use square image size, stride of 32
        # rect=True enables auto padding
        # inference can use other aspect ratio
                          imgsz=1280,       # Must be square, will auto pad black
                          rect=True,
                          cos_lr=True,        # Cosine Annealing, learning rate schedule
                          workers=7,          # CPU intensive, number of cores
                          # Overfit increases when number of batch is too small.
                          batch=16,            # GPU VRAM / RAM intensive, number of images in a batch;

                          # Augmentation
                          degrees=10,
                          shear=5,
                          perspective=0.000,
                          flipud=0.1,
                          mosaic=0.5,

                          # Overfit Suppression
                          weight_decay=0.0,  # penalty for large weights
                          dropout=0       # 0 to 1, randomly drop neurons
                          )





