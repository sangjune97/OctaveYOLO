from ultralytics import YOLOv10

# Load the YOLOv8 model
model = YOLOv10("yolov10n.pt")

# Export the model to TensorRT format
model.export(format="engine")  # creates 'yolov8n.engine'

# Load the exported TensorRT model
tensorrt_model = YOLOv10("yolov10n.engine")

# Run inference
results = tensorrt_model("https://ultralytics.com/images/bus.jpg")