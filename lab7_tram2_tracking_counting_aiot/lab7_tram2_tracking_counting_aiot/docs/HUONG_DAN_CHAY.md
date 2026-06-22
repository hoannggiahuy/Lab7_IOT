# Hướng dẫn chạy Trạm 2

## 1. Mở đúng thư mục

Terminal phải nằm tại thư mục có `app.py`, `tracking_engine.py` và `index.html`.

```powershell
Get-ChildItem
Test-Path .\app.py
```

`Test-Path` phải trả về `True`.

## 2. Cài đặt

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 3. Chạy test

```powershell
python run_tracking_demo.py
```

## 4. Chạy web

```powershell
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## 5. Kiểm tra YOLO

Mở `http://127.0.0.1:8000/model-info`.

- `yolo_ultralytics`: model thật.
- `fallback_contour`: pipeline demo, chưa có class COCO thật.
