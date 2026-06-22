"""
Lab 7 - Object Detection / Image AI Integration

Student-ready compact backend:
- live detection stream from laptop camera or IP camera URL
- upload image and run object detection
- snapshot from camera and run object detection
- save annotated images
- write detection_log.csv and vision_event_log.csv
- serve index.html dashboard

Run:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
Open:
    http://127.0.0.1:8000/

Main learning path:
    camera frame -> YOLO/fallback detector -> class + confidence + bbox
    -> visual event -> log -> dashboard
"""

from __future__ import annotations

import csv
import json
import time
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample_images"
INPUT_DIR = DATA_DIR / "input_images"
ANNOTATED_DIR = DATA_DIR / "annotated_images"
OUTPUT_DIR = ROOT / "outputs"
DETECTION_CSV = OUTPUT_DIR / "detection_log.csv"
EVENT_CSV = OUTPUT_DIR / "vision_event_log.csv"
THRESHOLD_CSV = OUTPUT_DIR / "threshold_experiment_log.csv"
INDEX_HTML = ROOT / "index.html"

for folder in [SAMPLE_DIR, INPUT_DIR, ANNOTATED_DIR, OUTPUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

DETECTION_FIELDS = [
    "detection_id", "image_id", "timestamp", "source_type", "model_name", "model_version",
    "threshold_used", "class_name", "confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "inference_time_ms", "annotated_image_path",
]

EVENT_FIELDS = [
    "event_id", "image_id", "timestamp", "event_type", "severity", "class_name", "confidence",
    "rule_used", "explanation", "action_hint", "annotated_image_path",
]

THRESHOLD_FIELDS = [
    "experiment_id", "timestamp", "image_id", "threshold", "num_detections", "top_class",
    "top_confidence", "inference_time_ms", "note",
]

DEFAULT_MODEL_NAME = "yolov8n.pt"  # light pretrained detector; downloaded by ultralytics on first use if internet is available
MODEL_VERSION = "lab7_yolo_nano_v1"

_detector = None
_detector_status: Dict[str, Any] = {
    "backend": "not_loaded",
    "model_name": DEFAULT_MODEL_NAME,
    "message": "Model has not been loaded yet.",
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def append_csv(path: Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def relative_url(path: Optional[Path]) -> Optional[str]:
    if not path:
        return None
    try:
        rel = path.resolve().relative_to(ROOT.resolve())
        return f"/files/{rel.as_posix()}"
    except Exception:
        return None


def validate_image_bytes(data: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def frame_to_jpeg_bytes(frame_bgr: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise RuntimeError("Could not encode frame as JPEG")
    return buffer.tobytes()


def create_sample_images() -> None:
    """Create simple sample images so the pipeline can run without internet or camera."""
    if any(SAMPLE_DIR.glob("*.jpg")):
        return
    img = np.full((420, 640, 3), 245, dtype=np.uint8)
    cv2.rectangle(img, (70, 110), (210, 330), (60, 140, 240), -1)
    cv2.putText(img, "sample object", (62, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.circle(img, (410, 210), 70, (80, 200, 120), -1)
    cv2.putText(img, "Lab 7 demo image", (170, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.imwrite(str(SAMPLE_DIR / "sample_objects.jpg"), img)

    img2 = np.full((420, 640, 3), 25, dtype=np.uint8)
    cv2.rectangle(img2, (90, 150), (260, 350), (180, 180, 180), -1)
    cv2.circle(img2, (460, 230), 60, (140, 140, 140), -1)
    cv2.putText(img2, "low light sample", (160, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
    cv2.imwrite(str(SAMPLE_DIR / "sample_low_light.jpg"), img2)


create_sample_images()


def parse_camera_source(source: str) -> Any:
    source = str(source).strip()
    return int(source) if source.isdigit() else source


def open_capture(source: str) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(parse_camera_source(source))
    if not cap.isOpened():
        return None
    return cap


def simulated_frame(counter: int = 0, width: int = 640, height: int = 360) -> np.ndarray:
    frame = np.full((height, width, 3), 245, dtype=np.uint8)
    x = 30 + (counter * 11) % max(1, width - 180)
    y = 90 + (counter * 6) % max(1, height - 180)
    cv2.rectangle(frame, (x, 100), (x + 135, 245), (45, 130, 245), -1)
    cv2.circle(frame, (width - 130, y), 48, (70, 190, 115), -1)
    cv2.putText(frame, "SIMULATED CAMERA - LAB 7", (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(frame, "Use source=0 for laptop camera; put object in front of camera", (25, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return frame


def read_one_frame(source: str = "0") -> Tuple[np.ndarray, str]:
    cap = open_capture(source)
    if cap is None:
        return simulated_frame(0), "simulated"
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return simulated_frame(0), "simulated"
    return frame, "camera"


def load_detector() -> Tuple[Optional[Any], Dict[str, Any]]:
    """Try to load YOLO. If unavailable, use deterministic fallback detection.

    The fallback keeps the lab runnable for smoke testing, but students should install
    ultralytics and use YOLO for the real object-detection experience.
    """
    global _detector, _detector_status
    if _detector_status["backend"] in {"ultralytics", "fallback"}:
        return _detector, _detector_status

    try:
        from ultralytics import YOLO  # type: ignore
        try:
            _detector = YOLO(DEFAULT_MODEL_NAME)
            _detector_status = {
                "backend": "ultralytics",
                "model_name": DEFAULT_MODEL_NAME,
                "model_version": MODEL_VERSION,
                "message": "YOLO nano model loaded. First run may download weights if needed.",
            }
        except Exception:
            # Try another common lightweight YOLO name for local environments.
            _detector = YOLO("yolo11n.pt")
            _detector_status = {
                "backend": "ultralytics",
                "model_name": "yolo11n.pt",
                "model_version": MODEL_VERSION,
                "message": "YOLO11 nano model loaded. First run may download weights if needed.",
            }
    except Exception as exc:
        _detector = None
        _detector_status = {
            "backend": "fallback",
            "model_name": "fallback_contour_detector",
            "model_version": "fallback_v1",
            "message": f"Ultralytics YOLO is not available or weights cannot be loaded. Fallback contour detector is active. Detail: {exc}",
        }
    return _detector, _detector_status


def parse_class_filter(classes: str = "") -> List[str]:
    return [c.strip().lower() for c in classes.split(",") if c.strip()]


def fallback_detect(frame_bgr: np.ndarray, conf: float, class_filter: List[str]) -> List[Dict[str, Any]]:
    """Simple contour-based fallback to keep the lab observable without YOLO.

    This is NOT a replacement for object detection. It only marks visually distinct
    regions as generic objects so the rest of the AIoT pipeline can be tested.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 140)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    h, w = gray.shape[:2]
    min_area = max(1200, int(0.01 * h * w))
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        score = min(0.95, max(0.25, area / float(h * w) * 8.0))
        label = "visual_object"
        if class_filter and label not in class_filter:
            continue
        if score < conf:
            continue
        detections.append({
            "class_name": label,
            "confidence": round(float(score), 4),
            "bbox": {"x1": int(x), "y1": int(y), "x2": int(x + bw), "y2": int(y + bh)},
        })
    detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)[:8]
    return detections


def yolo_detect(model: Any, frame_bgr: np.ndarray, conf: float, class_filter: List[str]) -> List[Dict[str, Any]]:
    results = model(frame_bgr, conf=conf, verbose=False)
    result = results[0]
    names = result.names if hasattr(result, "names") else {}
    detections: List[Dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        cls_id = int(box.cls[0].cpu().numpy().item())
        confidence = float(box.conf[0].cpu().numpy().item())
        class_name = str(names.get(cls_id, cls_id)).lower()
        if class_filter and class_name not in class_filter:
            continue
        detections.append({
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "bbox": {"x1": int(xyxy[0]), "y1": int(xyxy[1]), "x2": int(xyxy[2]), "y2": int(xyxy[3])},
        })
    return detections


def run_detection(frame_bgr: np.ndarray, conf: float = 0.35, classes: str = "") -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
    model, status = load_detector()
    class_filter = parse_class_filter(classes)
    start = time.perf_counter()
    if status["backend"] == "ultralytics" and model is not None:
        detections = yolo_detect(model, frame_bgr, conf=conf, class_filter=class_filter)
    else:
        detections = fallback_detect(frame_bgr, conf=conf, class_filter=class_filter)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    return detections, status, elapsed_ms


def severity_from_detections(detections: List[Dict[str, Any]]) -> Tuple[str, str, str, str]:
    if not detections:
        return "NO_OBJECT_DETECTED", "NORMAL", "no_object_rule", "No object was detected above the selected confidence threshold."
    top = detections[0]
    cls = top["class_name"]
    conf = float(top["confidence"])
    if conf < 0.45:
        return "LOW_CONFIDENCE_REVIEW", "WARNING", "low_confidence_rule", "Detected object has low confidence; human review is recommended."
    if cls == "person":
        return "PERSON_DETECTED", "WARNING", "person_rule", "A person was detected. In AIoT, this can become a visual event for monitoring."
    if cls in {"car", "truck", "bus", "motorcycle", "bicycle"}:
        return "VEHICLE_DETECTED", "WARNING", "vehicle_rule", "A vehicle was detected. This is useful for parking, traffic or gate monitoring."
    return "OBJECT_DETECTED", "NORMAL", "generic_object_rule", "At least one object was detected above the confidence threshold."


def draw_detections(frame_bgr: np.ndarray, detections: List[Dict[str, Any]], status: Dict[str, Any], conf: float) -> np.ndarray:
    out = frame_bgr.copy()
    color = (42, 180, 75) if detections else (60, 60, 220)
    for det in detections:
        bbox = det["bbox"]
        cls = det["class_name"]
        score = det["confidence"]
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        det_color = (50, 180, 80)
        if cls == "person":
            det_color = (50, 140, 250)
        elif score < 0.45:
            det_color = (0, 190, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), det_color, 3)
        label = f"{cls} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - 12)), (x1 + tw + 8, y1), det_color, -1)
        cv2.putText(out, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    header = f"{status['backend']} | conf={conf:.2f} | detections={len(detections)}"
    cv2.rectangle(out, (0, 0), (out.shape[1], 32), (255, 255, 255), -1)
    cv2.putText(out, header, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
    return out


def detect_and_log(frame_bgr: np.ndarray, source_type: str, device_id: str, conf: float = 0.35, classes: str = "", note: str = "") -> Dict[str, Any]:
    image_id = f"img_{uuid.uuid4().hex[:10]}"
    timestamp = now_iso()
    input_path = INPUT_DIR / f"{image_id}.jpg"
    cv2.imwrite(str(input_path), frame_bgr)

    detections, status, inference_time_ms = run_detection(frame_bgr, conf=conf, classes=classes)
    annotated = draw_detections(frame_bgr, detections, status, conf=conf)
    annotated_path = ANNOTATED_DIR / f"{image_id}_detected.jpg"
    cv2.imwrite(str(annotated_path), annotated)

    if detections:
        for det in detections:
            bbox = det["bbox"]
            row = {
                "detection_id": f"det_{uuid.uuid4().hex[:10]}",
                "image_id": image_id,
                "timestamp": timestamp,
                "source_type": source_type,
                "model_name": status["model_name"],
                "model_version": status["model_version"],
                "threshold_used": conf,
                "class_name": det["class_name"],
                "confidence": det["confidence"],
                "bbox_x1": bbox["x1"], "bbox_y1": bbox["y1"], "bbox_x2": bbox["x2"], "bbox_y2": bbox["y2"],
                "inference_time_ms": inference_time_ms,
                "annotated_image_path": str(annotated_path.relative_to(ROOT)),
            }
            append_csv(DETECTION_CSV, DETECTION_FIELDS, row)

    top = detections[0] if detections else {"class_name": "", "confidence": 0}
    event_type, severity, rule_used, explanation = severity_from_detections(detections)
    event_row = {
        "event_id": f"evt_{uuid.uuid4().hex[:10]}",
        "image_id": image_id,
        "timestamp": timestamp,
        "event_type": event_type,
        "severity": severity,
        "class_name": top.get("class_name", ""),
        "confidence": top.get("confidence", 0),
        "rule_used": rule_used,
        "explanation": explanation,
        "action_hint": "Display annotated image on dashboard; do not trigger actuator without a safety rule.",
        "annotated_image_path": str(annotated_path.relative_to(ROOT)),
    }
    append_csv(EVENT_CSV, EVENT_FIELDS, event_row)

    return {
        "image_id": image_id,
        "source_type": source_type,
        "device_id": device_id,
        "model_status": status,
        "threshold_used": conf,
        "class_filter": parse_class_filter(classes),
        "num_detections": len(detections),
        "detections": detections,
        "event": event_row,
        "inference_time_ms": inference_time_ms,
        "input_image_url": relative_url(input_path),
        "annotated_image_url": relative_url(annotated_path),
        "note": note,
    }


def stream_detect_frames(source: str = "0", conf: float = 0.35, classes: str = "") -> Iterable[bytes]:
    cap = open_capture(source)
    counter = 0
    while True:
        if cap is None:
            frame = simulated_frame(counter)
        else:
            ok, frame = cap.read()
            if not ok or frame is None:
                frame = simulated_frame(counter)
        detections, status, _ = run_detection(frame, conf=conf, classes=classes)
        annotated = draw_detections(frame, detections, status, conf=conf)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_to_jpeg_bytes(annotated) + b"\r\n"
        counter += 1
        time.sleep(0.12)


app = FastAPI(title="Lab 7 - Object Detection / Image AI Integration", description="Live camera object detection, annotated image, detection log, visual event and dashboard.")
app.mount("/files", StaticFiles(directory=str(ROOT)), name="files")


@app.get("/")
def home() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "lab": "Lab 7 - Object Detection / Image AI Integration", "outputs": {"detection_log": str(DETECTION_CSV.relative_to(ROOT)), "vision_event_log": str(EVENT_CSV.relative_to(ROOT))}}


@app.get("/model-info")
def model_info() -> Dict[str, Any]:
    _, status = load_detector()
    return {"task": "object_detection", "status": status, "default_threshold": 0.35, "main_source": "laptop camera source=0", "note": "If backend=fallback, install ultralytics and allow the YOLO nano weights to download for real object detection."}


@app.get("/video_feed")
def video_feed(source: str = Query("0"), conf: float = Query(0.35, ge=0.01, le=0.99), classes: str = Query("")) -> StreamingResponse:
    return StreamingResponse(stream_detect_frames(source=source, conf=conf, classes=classes), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/snapshot-detect")
def snapshot_detect(source: str = Query("0"), conf: float = Query(0.35, ge=0.01, le=0.99), classes: str = Query("")) -> Dict[str, Any]:
    frame, source_type = read_one_frame(source)
    return detect_and_log(frame, source_type=source_type, device_id=f"camera:{source}", conf=conf, classes=classes, note="snapshot-detect")


@app.post("/upload-detect")
async def upload_detect(file: UploadFile = File(...), conf: float = Query(0.35, ge=0.01, le=0.99), classes: str = Query("")) -> Dict[str, Any]:
    data = await file.read()
    img = validate_image_bytes(data)
    return detect_and_log(pil_to_bgr(img), source_type="upload", device_id="upload_client", conf=conf, classes=classes, note=f"filename={file.filename}")


@app.get("/detect-sample")
def detect_sample(sample: str = Query("sample_objects.jpg"), conf: float = Query(0.25, ge=0.01, le=0.99), classes: str = Query("")) -> Dict[str, Any]:
    path = SAMPLE_DIR / sample
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Sample image not found: {sample}")
    frame = cv2.imread(str(path))
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not read sample image")
    return detect_and_log(frame, source_type="sample", device_id="sample_image", conf=conf, classes=classes, note=f"sample={sample}")


@app.get("/threshold-experiment")
def threshold_experiment(sample: str = Query("sample_objects.jpg"), classes: str = Query("")) -> Dict[str, Any]:
    path = SAMPLE_DIR / sample
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Sample image not found: {sample}")
    frame = cv2.imread(str(path))
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not read sample image")
    experiment_id = f"exp_{uuid.uuid4().hex[:10]}"
    rows = []
    for threshold in [0.25, 0.50, 0.70]:
        result = detect_and_log(frame, source_type="threshold_experiment", device_id="sample_image", conf=threshold, classes=classes, note=f"experiment_id={experiment_id}")
        top = result["detections"][0] if result["detections"] else {"class_name": "", "confidence": 0}
        row = {
            "experiment_id": experiment_id,
            "timestamp": now_iso(),
            "image_id": result["image_id"],
            "threshold": threshold,
            "num_detections": result["num_detections"],
            "top_class": top.get("class_name", ""),
            "top_confidence": top.get("confidence", 0),
            "inference_time_ms": result["inference_time_ms"],
            "note": "Compare how threshold changes number of detections.",
        }
        append_csv(THRESHOLD_CSV, THRESHOLD_FIELDS, row)
        rows.append(row)
    return {"experiment_id": experiment_id, "items": rows}


@app.get("/detections")
def detections(limit: int = 30) -> Dict[str, Any]:
    rows = read_csv(DETECTION_CSV)
    return {"count": len(rows), "items": rows[-limit:]}


@app.get("/vision-events")
def vision_events(limit: int = 30) -> Dict[str, Any]:
    rows = read_csv(EVENT_CSV)
    return {"count": len(rows), "items": rows[-limit:]}


@app.get("/latest")
def latest() -> Dict[str, Any]:
    events = read_csv(EVENT_CSV)
    detections = read_csv(DETECTION_CSV)
    latest_event = events[-1] if events else None
    annotated_url = None
    if latest_event and latest_event.get("annotated_image_path"):
        annotated_url = relative_url(ROOT / latest_event["annotated_image_path"])
    return {
        "latest_event": latest_event,
        "latest_detections": detections[-10:],
        "event_count": len(events),
        "detection_count": len(detections),
        "annotated_image_url": annotated_url,
    }


if __name__ == "__main__":
    create_sample_images()
    sample = cv2.imread(str(SAMPLE_DIR / "sample_objects.jpg"))
    result = detect_and_log(sample, source_type="script", device_id="local_smoke", conf=0.25, note="python app.py smoke test")
    print(json.dumps(result, indent=2, ensure_ascii=False))
