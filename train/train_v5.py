from ultralytics import YOLOv10

model = YOLOv10("v5_oyolov10n.yaml")

# model = YOLOv10('yolov10{n/s/m/b/l/x}.pt')

model.train(data='coco.yaml', device=[0,1,2,3,4,5,6,7], epochs=500, batch=256, val=False)