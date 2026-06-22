# Phân tích code Lab 7

Lab 7 cố tình giữ ít file Python để dễ đọc luồng chính. File cần đọc chính là `app.py`.

## 1. `app.py` xử lý những gì?

| Thành phần | Vai trò |
|---|---|
| `load_detector()` | Tải YOLO nếu có; nếu chưa có thì dùng fallback để pipeline vẫn chạy |
| `run_detection()` | Nhận một frame và trả danh sách detection |
| `detect_and_log()` | Hàm lõi: lưu ảnh, chạy detect, vẽ bbox, ghi detection log và sinh event |
| `draw_detections()` | Vẽ bounding box, class và confidence lên ảnh |
| `severity_from_detections()` | Chuyển output của model thành visual event |
| `/video_feed` | Stream camera laptop/IP camera kèm kết quả nhận diện |
| `/snapshot-detect` | Chụp một frame từ camera rồi chạy detection |
| `/upload-detect` | Upload ảnh rồi chạy detection |
| `/detections` | Đọc lại `detection_log.csv` |
| `/vision-events` | Đọc lại `vision_event_log.csv` |

## 2. Đường đi quan trọng nhất

```text
Frame từ camera hoặc ảnh upload
→ run_detection()
→ detections = class + confidence + bbox
→ draw_detections()
→ annotated image
→ detection_log.csv
→ vision_event_log.csv
→ index.html hiển thị
```

## 3. Cần đọc kỹ phần nào?

Đọc kỹ `detect_and_log()` vì đây là nơi nối tất cả thành phần:

- ảnh được lưu vào `data/input_images/`;
- model chạy inference;
- ảnh kết quả được lưu vào `data/annotated_images/`;
- mỗi bbox được ghi vào `outputs/detection_log.csv`;
- kết quả được chuyển thành event trong `outputs/vision_event_log.csv`.

## 4. Vì sao cần `threshold_used`?

Cùng một ảnh nhưng threshold khác nhau có thể tạo kết quả khác nhau. Nếu không ghi threshold vào log, về sau không biết hệ thống đã chạy với cấu hình nào.

## 5. Vì sao cần `inference_time_ms`?

Trong AIoT, model không chỉ cần đúng mà còn cần đủ nhanh. Camera giám sát hoặc hệ thống cảnh báo không thể chờ quá lâu mới có kết quả.
