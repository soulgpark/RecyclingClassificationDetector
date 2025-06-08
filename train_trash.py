from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

model.train(
    data="trash_dataset/train", 
    epochs=50,
    imgsz=256,
    batch=32,
    lr0=0.01 
)

print("모델이 저장되었습니다:", model.ckpt_path)