from ultralytics import YOLOv10

model = YOLOv10("yolov10m.yaml")

# model = YOLOv10('yolov10{n/s/m/b/l/x}.pt')

model.train(data='coco.yaml', device=[0,1,2,3,4,5,6,7], epochs=50, batch=64, teacher_model="/home/sangjun/yolov10/checkpoint/yolov10l.pt")