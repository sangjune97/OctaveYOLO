from ultralytics import YOLOv10
from ultralytics import YOLO
# Load a model
#model = YOLOv10("/home/sangjun/yolov10/runs/detect/train13/weights/best.pt")  # load an official model
#
## Validate the model
#metrics = model.val(device=[3], batch=64, imgsz=640, rect = False)  # no arguments needed, dataset and settings remembered

model = YOLOv10("/home/sangjun/yolov10/runs/detect/train19/weights/best.pt")  # load an official model
metrics = model.val(device=[3], batch=64, imgsz=640, rect = False)  # no arguments needed, dataset and settings remembered

#model = YOLO("/home/sangjun/yolov10/yolov8n.pt")  # load an official model
#metrics = model.val(device=[3], batch=64, imgsz=640, rect = False)  # no arguments needed, dataset and settings remembered

