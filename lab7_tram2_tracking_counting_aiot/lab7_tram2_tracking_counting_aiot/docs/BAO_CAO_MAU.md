# Báo cáo ngắn — Trạm 2 Tracking & Counting

## 1. Mục tiêu

Nâng cấp Lab 7 Object Detection thành hệ thống theo dõi và đếm đối tượng đi qua vạch.

## 2. Kiến trúc

```text
Camera → YOLO detection → Centroid Tracker → Object ID/Trajectory
       → Line Crossing Rule → Count IN/OUT → Snapshot/CSV/Dashboard
```

## 3. Kết quả chạy

- Backend sử dụng: ...
- Class filter: ...
- Confidence: ...
- Count IN: ...
- Count OUT: ...
- Inference time trung bình: ... ms

## 4. Bảng thí nghiệm

| Thí nghiệm | Tham số | Kết quả quan sát | Giải thích |
|---|---|---|---|
| 1 | confidence=0.25 | | |
| 2 | confidence=0.50 | | |
| 3 | confidence=0.70 | | |
| 4 | max_distance=40 | | |
| 5 | max_distance=90 | | |
| 6 | detect_every=3 | | |

## 5. Ít nhất ba trường hợp lỗi

1. Người bị che khuất: ...
2. Hai người đi sát nhau: ...
3. Ánh sáng yếu hoặc ngược sáng: ...

## 6. Kết luận AIoT

Model output chưa phải quyết định điều khiển cuối cùng. Hệ thống phải chuyển detection thành event, áp dụng rule an toàn và lưu log để kiểm tra.
