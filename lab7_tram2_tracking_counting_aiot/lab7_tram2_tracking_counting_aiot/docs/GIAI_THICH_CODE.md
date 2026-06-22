# Giải thích code trọng tâm

## `detect_objects()`

Nhận frame và trả danh sách detection gồm bbox, class, confidence. Hàm ưu tiên YOLO; nếu không tải được thì chuyển sang contour fallback.

## `update_tracks()`

Ghép detection mới với track cũ bằng khoảng cách giữa hai centroid. Nếu khoảng cách nhỏ hơn `max_distance`, detection được gắn lại ID cũ; nếu không, tạo ID mới.

## `count_crossings()`

Chia ảnh thành hai phía bởi một vạch. Khi `stable_side` của cùng một track đổi từ -1 sang +1 hoặc ngược lại, hệ thống tạo `LINE_CROSSING_DETECTED` và tăng bộ đếm.

## Hysteresis và cooldown

Centroid có thể rung gần vạch. Hysteresis tạo vùng chết quanh vạch; cooldown ngăn cùng một track bị đếm nhiều lần liên tiếp.

## `process_tracking_frame()`

Đây là hàm lõi:

```text
frame → detection → tracking → crossing → annotation → records/events/metadata
```

## `app.py`

- `/video_feed`: camera stream.
- `/stats`: số đếm và trạng thái hiện tại.
- `/reset`: xóa track và đưa bộ đếm về 0.
- `/model-info`: kiểm tra YOLO hay fallback.
- `/logs/{filename}`: đọc log cho dashboard.
