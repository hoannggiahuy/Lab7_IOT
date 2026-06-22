# Hướng dẫn chạy và quan sát Lab 7

## 1. Mục tiêu thao tác

Lab 7 tiếp nối Lab 6. Ở Lab 6, camera laptop hoặc IP camera đã tạo ra stream, snapshot, video, motion event, metadata và dashboard. Ở Lab 7, luồng chính vẫn là camera laptop, nhưng mỗi frame được đưa qua model nhận diện để tạo ra:

- class: đối tượng được nhận diện;
- confidence: độ tin cậy của dự đoán;
- bounding box: vị trí đối tượng trong ảnh;
- detection log: bản ghi kết quả nhận diện;
- vision event: sự kiện thị giác cho dashboard và lớp quyết định.

## 2. Cài đặt

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux/WSL
source .venv/bin/activate
pip install -r requirements.txt
```

Quan sát: nếu cài đặt đúng, không có lỗi khi import `fastapi`, `cv2`, `PIL` và `ultralytics`.

## 3. Chạy thử không cần camera

```bash
python run_lab7_demo.py
```

Cần quan sát:

- `RUN_TEST_LOG.txt` có `LOCAL_PIPELINE_TEST_PASS`;
- thư mục `data/annotated_images/` có ảnh kết quả;
- `outputs/vision_event_log.csv` có event mới;
- nếu YOLO chưa sẵn sàng, hệ thống có thể dùng fallback detector để kiểm tra pipeline.

## 4. Chạy service và mở dashboard

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Mở trình duyệt:

```text
http://127.0.0.1:8000/
```

## 5. Thao tác chính trên dashboard

| Thao tác | Cần làm | Kết quả cần nhìn thấy |
|---|---|---|
| Bật stream nhận diện | Giữ `source=0`, bấm Bật stream | Camera laptop hiển thị cùng bbox/class nếu model nhận diện được |
| Đưa vật thể lên camera | Đưa chai nước, điện thoại, sách, laptop hoặc người vào khung hình | Class/confidence/bbox thay đổi theo vật thể |
| Chụp từ camera và Detect | Bấm Chụp từ camera và Detect | Ảnh annotated xuất hiện và log được ghi |
| Upload ảnh | Chọn ảnh từ máy và bấm Upload và Detect | API trả JSON, dashboard hiển thị ảnh có bbox |
| Đổi threshold | Thử 0.25, 0.50, 0.70 | Số bbox thường giảm khi threshold tăng |
| Lọc class | Nhập `person` hoặc `bottle` | Chỉ giữ class liên quan đến use-case |

## 6. Nếu chưa từng chạy YOLO

Lần đầu chạy YOLO có thể cần Internet để tải model nhẹ. Nếu backend hiển thị là `fallback`, pipeline vẫn chạy được nhưng chưa phải object detection thật. Khi có Internet, chạy lại:

```bash
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

Sau đó khởi động lại service.

## 7. Cần hiểu sau khi chạy

- Bbox cho biết vị trí đối tượng trong ảnh.
- Confidence không phải chân lý tuyệt đối; nó là điểm tin cậy của model.
- Threshold càng cao thì hệ thống càng thận trọng, nhưng có thể bỏ sót.
- Model output chưa phải quyết định điều khiển; cần event rule và safety rule.
- Log giúp truy vết: model nào, threshold nào, ảnh nào, thời điểm nào, latency bao nhiêu.
