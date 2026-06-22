[README.md](https://github.com/user-attachments/files/29192413/README.md)
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
[README.md](https://github.com/user-attachments/files/29192418/README.md)
# Lab 7 mở rộng — Trạm 2: Tracking & Counting

Project này nâng cấp trực tiếp từ Lab 7 Object Detection:

```text
Lab 7 cơ bản
Camera → YOLO → class + confidence + bounding box

Lab 7 mở rộng, Trạm 2
Camera → YOLO → centroid tracker → object ID + trajectory
       → kiểm tra cắt vạch → count IN/OUT → snapshot + CSV event
```

## Vì sao chọn Trạm 2?

Tracking & Counting dùng lại toàn bộ nền tảng YOLO của Lab 7 nhưng bổ sung yếu tố **thời gian**. Detection một frame chỉ biết có vật gì; tracking biết vật đó có còn là cùng một đối tượng ở frame sau hay không. Nhờ object ID và trajectory, hệ thống mới có thể đếm người/xe đi qua vạch.

## Chức năng đã nâng cấp

- YOLOv8n detection thật khi có `ultralytics`.
- Fallback contour trung thực khi chưa có YOLO.
- Centroid tracking giữ `track_id` qua nhiều frame.
- Vẽ quỹ đạo di chuyển.
- Vạch đếm ngang hoặc dọc, tùy chỉnh vị trí.
- Đếm `count_in` và `count_out` có hysteresis/cooldown chống rung.
- Class filter, confidence, max distance, max missed, detect every N frame.
- Dashboard live, API stats, reset bộ đếm.
- Lưu snapshot khi có đối tượng cắt vạch.
- Sinh `tracking_log.csv`, `counting_event_log.csv`, `vision_event_log.csv`.

## Cài đặt Windows PowerShell

Mỗi lệnh phải chạy trên một dòng riêng:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Có thể chỉ cài bản nhẹ để chạy fallback:

```powershell
python -m pip install -r requirements_core.txt
```

## Kiểm tra không cần camera

```powershell
python run_tracking_demo.py
```

Kết quả đúng:

```text
LOCAL_TRACKING_PIPELINE_TEST_PASS
```

Kiểm tra thêm:

```text
RUN_TEST_LOG.txt
outputs/tracking_demo_report.json
outputs/tracking_demo_events.csv
data/snapshots/demo_crossing_*.jpg
```

## Chạy dashboard

```powershell
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Mở: `http://127.0.0.1:8000/`

## Thao tác đề xuất

1. Giữ `source=0`, `confidence=0.35`, `class_filter=person`.
2. Chọn vạch ngang, vị trí `0.55`.
3. Bấm **Bắt đầu Tracking**.
4. Đi từ phía trên xuống dưới vạch, sau đó đi ngược lại.
5. Quan sát object ID, trajectory, COUNT IN và COUNT OUT.
6. Thử confidence `0.25 → 0.50 → 0.70`.
7. Thử `max_distance=40`, `90`, `160` để quan sát ID bị đổi hoặc ghép sai.
8. Mở ba file log trên dashboard và chụp ảnh minh chứng.

## Ý nghĩa tham số

- `confidence`: thấp thì nhạy nhưng dễ nhận sai; cao thì dễ bỏ sót.
- `class_filter`: chỉ theo dõi class phục vụ use-case, ví dụ `person` hoặc `car`.
- `max_distance`: khoảng cách centroid tối đa để ghép detection mới vào track cũ.
- `max_missed`: số frame cho phép mất detection trước khi xóa track.
- `detect_every`: chạy detection mỗi N frame; N lớn nhẹ máy hơn nhưng tracking kém mượt.
- `line_position`: vị trí vạch theo tỷ lệ 0–1 của chiều cao/chiều rộng ảnh.

## Sản phẩm nộp bài

- Ảnh dashboard live tracking.
- Ảnh có bbox, class, confidence, object ID và trajectory.
- Ảnh khi count IN tăng và ảnh khi count OUT tăng.
- Ít nhất một snapshot trong `data/snapshots/`.
- `outputs/tracking_log.csv`.
- `outputs/counting_event_log.csv`.
- `outputs/vision_event_log.csv`.
- Báo cáo thí nghiệm tham số theo mẫu trong `docs/BAO_CAO_MAU.md`.
