
from ultralytics import YOLO
import cv2

# 1. Tải mô hình AI đã huấn luyện
model = YOLO("best.pt")
print("Cac lop AI nhan dien:", model.names)
# 2. Mở video đầu vào
cap = cv2.VideoCapture("ga.mp4")

if not cap.isOpened():
    print("Loi: Khong tim thay video ga.mp4")
    exit()

# 3. Lấy thông tin video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 25

# 4. Tạo file video kết quả
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    "ket_qua_video.mp4",
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    cap.release()
    raise RuntimeError("Khong tao duoc video ket qua")

# 5. Nhận diện từng khung hình
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # AI nhận diện lớp bird (14)
    results = model.predict(
        frame,
        classes=[14],
        conf=0.25,
        verbose=False
    )

    # Lấy kết quả
    result = results[0]

    # Đếm số đối tượng được phát hiện
    count = len(result.boxes)

    # Vẽ khung nhận diện
    image = result.plot()

    # Hiển thị số lượng
    cv2.putText(
        image,
        f"Bird count: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Hiển thị video trực tiếp
    cv2.imshow("Chicken AI", image)

    # Ghi vào video kết quả
    out.write(image)

    # Nhấn Q để dừng
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 6. Giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()

print("Da tao file ket_qua_video.mp4")
