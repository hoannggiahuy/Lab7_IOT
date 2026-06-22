# Lab 7 - Object Detection / Image AI Integration

Lab 7 phát triển trực tiếp từ Lab 6. Lab 6 đã có camera stream, snapshot, video, motion capture, metadata và image event. Lab 7 dùng luồng camera đó để chạy object detection, tạo bounding box, confidence, detection log và vision event.

## Chạy nhanh

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux/WSL
source .venv/bin/activate
pip install -r requirements.txt
python run_lab7_demo.py
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Mở trình duyệt: http://127.0.0.1:8000/

## Quan sát chính

- Bật stream nhận diện từ camera laptop (`source=0`).
- Đưa vật thể như chai nước, điện thoại, sách, laptop vào trước camera.
- Quan sát bounding box, class, confidence và latency.
- Thay threshold 0.25, 0.50, 0.70 để xem số lượng bbox thay đổi.
- Kiểm tra `outputs/detection_log.csv` và `outputs/vision_event_log.csv`.

## Ghi chú model

Lab ưu tiên YOLO nano pretrained. Lần chạy đầu có thể cần Internet để tải weights. Nếu chưa tải được hoặc chưa có ultralytics, app tự dùng fallback contour detector để vẫn chạy được pipeline log và dashboard; tuy nhiên để trải nghiệm object detection thật, cần chạy với YOLO.
