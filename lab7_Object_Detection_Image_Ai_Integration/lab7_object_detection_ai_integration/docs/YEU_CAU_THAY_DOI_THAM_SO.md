# Yêu cầu thay đổi tham số để hiểu bản chất

Hoàn thành tối thiểu 4 yêu cầu dưới đây.

| Mã | Yêu cầu | Cần quan sát | Cần giải thích |
|---|---|---|---|
| L7-S1 | Đổi threshold 0.25 → 0.50 → 0.70 | Số bbox thay đổi | Threshold thấp/cao ảnh hưởng false positive và missed detection thế nào? |
| L7-S2 | Nhập class filter `person` | Chỉ giữ detection class person | Vì sao một hệ thống thực tế thường chỉ quan tâm một số class? |
| L7-S3 | Nhập class filter `bottle` hoặc `cell phone` rồi đưa vật thể tương ứng lên camera | Detection có thay đổi không | Model pretrained có nhận tốt đồ vật nhỏ không? |
| L7-S4 | Đưa vật thể ra xa camera | Confidence và bbox thay đổi | Khoảng cách ảnh hưởng nhận diện thế nào? |
| L7-S5 | Thử ánh sáng yếu hoặc ngược sáng | Detection sai/giảm confidence | Vì sao dữ liệu ảnh thực tế khó hơn ảnh demo? |
| L7-S6 | Đọc `detection_log.csv` và tìm `threshold_used` | Thấy threshold trong log | Vì sao cần truy vết cấu hình inference? |
| L7-S7 | Đọc `vision_event_log.csv` | Thấy event_type/severity | Vì sao model output cần chuyển thành event? |
| L7-S8 | Chạy `/threshold-experiment` | Có `threshold_experiment_log.csv` | Threshold tăng làm hệ thống thận trọng hơn hay nhạy hơn? |
