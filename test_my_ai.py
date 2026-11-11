
from ultralytics import YOLO
import cv2

# Nạp AI bạn vừa huấn luyện
model = YOLO(
    "runs/detect/runs/chicken/demo/weights/best.pt"
)

# Nhận diện ảnh gà
results = model.predict(
    source="ga.png",
    conf=0.01,
    imgsz=640
)

# Đếm số gà
count = len(results[0].boxes)

# Vẽ khung
image = results[0].plot()

cv2.putText(
    image,
    f"Chicken: {count}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

# Lưu kết quả
cv2.imwrite("ket_qua_ai_tu_huan_luyen.jpg", image)

print("So ga AI phat hien:", count)
