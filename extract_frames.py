
import cv2
import os

# Tạo thư mục lưu ảnh
os.makedirs("raw_images", exist_ok=True)

# Mở video
cap = cv2.VideoCapture("ga.mp4")

if not cap.isOpened():
    raise RuntimeError("Khong mo duoc ga.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 25

# Lấy khoảng 1 ảnh mỗi giây
interval = max(1, round(fps))

frame_number = 0
saved = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_number % interval == 0:
        filename = f"raw_images/ga_{saved:04d}.jpg"
        cv2.imwrite(filename, frame)
        saved += 1

    frame_number += 1

cap.release()

print(f"Da luu {saved} anh vao raw_images")
