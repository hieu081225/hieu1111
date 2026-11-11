
from ultralytics import YOLO
import cv2

# 1. Tải mô hình đã huấn luyện nhận diện gà
model = YOLO("best.pt")

# 2. Hiển thị các lớp mô hình nhận diện được
print("Cac lop:", model.names)

# 3. Tìm ID của lớp chicken
chicken_id = next(
    (i for i, name in model.names.items()
     if name.lower() == "chicken"),
    None
)

if chicken_id is None:
    raise ValueError("Mo hinh khong co lop chicken")

# 4. Nhận diện gà trong ảnh
results = model.predict(
    source="ga.png",
    classes=[chicken_id],
    conf=0.10,
    imgsz=1280
)

# 5. Lấy kết quả
result = results[0]

# 6. Đếm số gà AI phát hiện
count = len(result.boxes)

# 7. Vẽ khung nhận diện
image = result.plot()

# 8. Hiển thị số lượng gà
cv2.putText(
    image,
    f"Chicken count: {count}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

# 9. Lưu ảnh kết quả
cv2.imwrite("ket_qua_chicken.jpg", image)

print(f"So ga phat hien: {count}")
print("Da luu ket_qua_chicken.jpg")
