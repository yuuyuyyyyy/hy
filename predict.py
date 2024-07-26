from ultralytics import YOLOv10

model = YOLOv10("runs/detect/train_v1010/weights/best.pt")
model.predict(source="data-2/test/images",imgsz=640,conf=0.05,save=True)