# Model cho Lab 7

Lab 7 dùng YOLO nano pretrained để chạy object detection. Trong code, model mặc định là:

```text
yolov8n.pt
```

Lần chạy đầu có thể cần Internet để ultralytics tải weights. Có thể tải trước bằng:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

Nếu chưa tải được model, app vẫn chạy fallback contour detector để kiểm tra pipeline, nhưng trải nghiệm object detection thật cần YOLO.
