# Rubric Lab 7

| Tiêu chí | Mô tả | Điểm |
|---|---|---:|
| Dashboard và camera | Mở được dashboard, chạy được live detection hoặc fallback stream | 1.5 |
| Object detection | Upload/chụp ảnh và nhận được class, confidence, bbox | 1.5 |
| Annotated image | Có ảnh kết quả được vẽ bounding box | 1.0 |
| Detection log | `detection_log.csv` có model, threshold, class, confidence, bbox, latency | 1.5 |
| Vision event | `vision_event_log.csv` có event_type, severity, explanation/action_hint | 1.0 |
| Thay đổi tham số | Thử threshold/class filter và phân tích kết quả | 1.0 |
| Phân tích code | Giải thích được `app.py`, `index.html`, `detect_and_log()` | 1.0 |
| Phân tích lỗi | Nêu được ít nhất 3 trường hợp nhận diện đúng/sai/confidence thấp | 1.0 |
| Liên hệ hệ thống AIoT | Giải thích được vì sao model output cần event/rule/dashboard | 0.5 |
| Tổng |  | 10 |
