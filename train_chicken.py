
from ultralytics import YOLO

# 1. Nạp mô hình YOLO đã huấn luyện sẵn
model = YOLO("yolo26n.pt")

# 2. Huấn luyện thêm bằng ảnh gà của bạn
model.train(
    data="chicken_dataset/data.yaml",
    epochs=10,
    imgsz=512,
    batch=1,
    workers=0,
    device="cpu",
    project="runs/chicken",
    name="demo",
    exist_ok=True
)

print("HUAN LUYEN HOAN THANH!")
