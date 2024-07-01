from ultralytics import YOLOv10
#model = YOLO("octave_yolov8n.yaml")
#
#model.train(data="VOC.yaml", epochs=300, device=[0, 1], batch=256, seed=0)  # train the model

model = YOLOv10("yolov10n.yaml")
model.model.model[-1].export = True
model.model.model[-1].format = 'onnx'
del model.model.model[-1].cv2
del model.model.model[-1].cv3
model.info()

#model = YOLOv10("oyolov10n.yaml")
#model.model.model[-1].export = True
#model.model.model[-1].format = 'onnx'
#del model.model.model[-1].cv2
#del model.model.model[-1].cv3
#model.fuse()
#model.info()
#
#model = YOLOv10("v3_oyolov10n.yaml")
#model.model.model[-1].export = True
#model.model.model[-1].format = 'onnx'
#del model.model.model[-1].cv2
#del model.model.model[-1].cv3
#model.fuse()
#model.info()
#
#model = YOLOv10("v4_oyolov10n.yaml")
#model.model.model[-1].export = True
#model.model.model[-1].format = 'onnx'
#del model.model.model[-1].cv2
#del model.model.model[-1].cv3
#model.fuse()
#model.info()
