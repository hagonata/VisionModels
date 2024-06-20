import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
import coremltools as ct

yolo_model = YOLO('yolov8s.pt')

class YOLOv8DetectionAndFeatureExtractorModel(DetectionModel):
    def __init__(self, cfg, ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)

    def custom_forward(self, x):
        y = []
        features = None
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if torch.is_tensor(x):
                features = x  # keep the last tensor as features
            x = m(x)  # run
            if torch.is_tensor(x):
                features = x
            y.append(x if m.i in self.save else None)  # save output
        if torch.is_tensor(x):
            features = x
        return features, x  # return features and detection output

# Get the configuration of the loaded model
model_cfg = yolo_model.model.yaml

# Create the modified YOLOv8 model
yolov8_model = YOLOv8DetectionAndFeatureExtractorModel(cfg=model_cfg)

# Test the model with a dummy input
dummy_input = torch.randn(1, 3, 640, 640)
features, detections = yolov8_model.custom_forward(dummy_input)

print(f"Features shape: {features.shape}")
print(f"Detections type: {type(detections)}")

for idx, det in enumerate(detections):
    if torch.is_tensor(det):
        print(f"Detection {idx} shape: {det.shape}")
    else:
        print(f"Detection {idx} is of type {type(det)}")

# Optionally, print the detection tensor if they are not too large
if len(detections) > 0 and torch.is_tensor(detections[0]):
    print(detections[0])

# Wrapper class for tracing
class TracedYOLOv8(torch.nn.Module):
    def __init__(self, model):
        super(TracedYOLOv8, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model.custom_forward(x)

traced_model = torch.jit.trace(TracedYOLOv8(yolov8_model), dummy_input)

# Convert to Core ML
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(name="input_image", shape=dummy_input.shape, scale=1/255.0, bias=[0, 0, 0])]
)

mlmodel.save("YOLOv8sFeatureExtractor.mlpackage")
print("Model conversion completed successfully")
