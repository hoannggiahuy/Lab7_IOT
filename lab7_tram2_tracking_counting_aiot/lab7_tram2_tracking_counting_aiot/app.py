from __future__ import annotations

import csv
import json
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from tracking_engine import TrackingState, process_tracking_frame, try_load_yolo

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
OUTPUT_DIR = BASE_DIR / "outputs"
for directory in (DATA_DIR, SNAPSHOT_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Lab 7 mở rộng - Trạm 2 Tracking & Counting")
app.mount("/snapshots", StaticFiles(directory=str(SNAPSHOT_DIR)), name="snapshots")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

STATES: Dict[str, TrackingState] = {}
STATE_LOCK = threading.Lock()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def append_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fieldnames = list(row.keys())
    if exists:
        try:
            with path.open("r", encoding="utf-8", newline="") as file:
                fieldnames = next(csv.reader(file))
        except Exception:
            fieldnames = list(row.keys())
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def open_capture(source: str):
    try:
        capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return capture
    except Exception:
        pass
    return None


def synthetic_frame(counter: int) -> np.ndarray:
    """Stream mô phỏng: một vật đi xuống rồi đi lên qua vạch."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (18, 24, 31)
    period = 120
    phase = counter % period
    y = 60 + int((phase if phase < 60 else 120 - phase) * 5.5)
    x = 260
    cv2.rectangle(frame, (x, y), (x + 110, y + 130), (245, 245, 245), -1)
    cv2.putText(frame, "SIMULATED OBJECT", (200, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (150, 210, 255), 2)
    return frame


def encode_jpeg(frame: np.ndarray, quality: int = 78) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Không thể mã hóa frame thành JPEG")
    return buffer.tobytes()


def log_tracking_records(
    session_id: str,
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
    source: str,
) -> None:
    for record in records:
        append_csv(
            OUTPUT_DIR / "tracking_log.csv",
            {
                "timestamp": now_iso(),
                "session_id": session_id,
                "source": source,
                "frame_index": metadata["frame_index"],
                "track_id": record["track_id"],
                "class_name": record["class_name"],
                "confidence": record["confidence"],
                "centroid_x": record["centroid"][0],
                "centroid_y": record["centroid"][1],
                "bbox_x1": record["bbox"][0],
                "bbox_y1": record["bbox"][1],
                "bbox_x2": record["bbox"][2],
                "bbox_y2": record["bbox"][3],
                "age_frames": record["age_frames"],
                "count_in": record["count_in"],
                "count_out": record["count_out"],
                "backend": metadata["backend"],
                "model_path": metadata["model_path"],
                "confidence_threshold": metadata["confidence_threshold"],
                "class_filter": metadata["class_filter"],
                "inference_time_ms": metadata["inference_time_ms"],
                "processing_time_ms": metadata["processing_time_ms"],
            },
        )


def log_crossing_events(
    session_id: str,
    events: list[dict[str, Any]],
    metadata: dict[str, Any],
    snapshot_url: str,
    source: str,
) -> None:
    for event in events:
        append_csv(
            OUTPUT_DIR / "counting_event_log.csv",
            {
                "event_id": str(uuid.uuid4())[:10],
                "timestamp": now_iso(),
                "session_id": session_id,
                "source": source,
                "event_type": event["event_type"],
                "severity": event["severity"],
                "track_id": event["track_id"],
                "class_name": event["class_name"],
                "confidence": event["confidence"],
                "direction": event["direction"],
                "counter_updated": event["counter_updated"],
                "count_in": event["count_in"],
                "count_out": event["count_out"],
                "line_orientation": event["line_orientation"],
                "line_position": event["line_position"],
                "frame_index": event["frame_index"],
                "backend": metadata["backend"],
                "model_path": metadata["model_path"],
                "confidence_threshold": metadata["confidence_threshold"],
                "snapshot_url": snapshot_url,
                "explanation": event["explanation"],
                "action_hint": event["action_hint"],
            },
        )
        append_csv(
            OUTPUT_DIR / "vision_event_log.csv",
            {
                "timestamp": now_iso(),
                "session_id": session_id,
                "event_type": event["event_type"],
                "severity": event["severity"],
                "object": f"{event['class_name']}#{event['track_id']}",
                "direction": event["direction"],
                "count_in": event["count_in"],
                "count_out": event["count_out"],
                "explanation": event["explanation"],
                "action_hint": event["action_hint"],
            },
        )


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return (BASE_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "service": "lab7_station2_tracking_counting", "time": now_iso()}


@app.get("/model-info")
def model_info(model_path: str = Query("yolov8n.pt")) -> Dict[str, Any]:
    model = try_load_yolo(model_path)
    if model is None:
        return {
            "backend": "fallback_contour",
            "model_path": model_path,
            "message": "Chưa có Ultralytics/model. Pipeline vẫn chạy fallback nhưng không có class COCO thật.",
        }
    names = getattr(model, "names", {})
    return {
        "backend": "yolo_ultralytics",
        "model_path": model_path,
        "num_classes": len(names),
        "class_names": names,
    }


@app.get("/video_feed")
def video_feed(
    source: str = Query("0"),
    confidence: float = Query(0.35, ge=0.05, le=0.95),
    class_filter: str = Query("person"),
    model_path: str = Query("yolov8n.pt"),
    line_orientation: str = Query("horizontal"),
    line_position: float = Query(0.55, ge=0.10, le=0.90),
    max_distance: float = Query(90.0, ge=20.0, le=300.0),
    max_missed: int = Query(12, ge=1, le=60),
    detect_every: int = Query(1, ge=1, le=10),
    session_id: str = Query("default"),
):
    with STATE_LOCK:
        state = STATES.setdefault(session_id, TrackingState())

    def generate():
        capture = open_capture(source)
        counter = 0
        last_annotated: np.ndarray | None = None
        last_records: list[dict[str, Any]] = []
        last_metadata: dict[str, Any] = {}
        try:
            while True:
                if capture is None:
                    frame = synthetic_frame(counter)
                    force_fallback = True
                else:
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        frame = synthetic_frame(counter)
                        force_fallback = True
                    else:
                        force_fallback = False

                should_process = counter % max(1, detect_every) == 0 or last_annotated is None
                if should_process:
                    annotated, records, crossing_events, metadata = process_tracking_frame(
                        frame=frame,
                        state=state,
                        confidence=confidence,
                        class_filter=class_filter,
                        model_path=model_path,
                        line_orientation=line_orientation,
                        line_position=line_position,
                        max_distance=max_distance,
                        max_missed=max_missed,
                        force_fallback=force_fallback,
                    )
                    last_annotated = annotated
                    last_records = records
                    last_metadata = metadata

                    if metadata["frame_index"] % 15 == 0 and records:
                        log_tracking_records(session_id, records, metadata, source)

                    if crossing_events:
                        snapshot_name = f"crossing_{session_id}_{int(time.time() * 1000)}.jpg"
                        snapshot_path = SNAPSHOT_DIR / snapshot_name
                        cv2.imwrite(str(snapshot_path), annotated)
                        snapshot_url = f"/snapshots/{snapshot_name}"
                        log_crossing_events(session_id, crossing_events, metadata, snapshot_url, source)
                else:
                    annotated = last_annotated
                    records = last_records
                    metadata = last_metadata

                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encode_jpeg(annotated) + b"\r\n"
                counter += 1
                time.sleep(0.035)
        finally:
            if capture is not None:
                capture.release()

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/stats")
def stats(session_id: str = Query("default")) -> Dict[str, Any]:
    with STATE_LOCK:
        state = STATES.setdefault(session_id, TrackingState())
        return {
            "session_id": session_id,
            "count_in": state.count_in,
            "count_out": state.count_out,
            "occupancy_estimate": state.count_in - state.count_out,
            "active_tracks": len([track for track in state.tracks.values() if track.missed == 0]),
            "total_track_ids_created": state.next_id - 1,
            "backend": state.last_backend,
            "inference_time_ms": round(state.last_inference_ms, 2),
            "processing_time_ms": round(state.last_processing_ms, 2),
            "frame_index": state.frame_index,
            "last_event": state.last_event,
        }


@app.post("/reset")
def reset(session_id: str = Query("default")) -> Dict[str, Any]:
    with STATE_LOCK:
        state = STATES.setdefault(session_id, TrackingState())
        state.reset()
    append_csv(
        OUTPUT_DIR / "vision_event_log.csv",
        {
            "timestamp": now_iso(),
            "session_id": session_id,
            "event_type": "COUNTER_RESET",
            "severity": "INFO",
            "object": "system",
            "direction": "",
            "count_in": 0,
            "count_out": 0,
            "explanation": "User reset tracker IDs and counters.",
            "action_hint": "Start a new counting experiment.",
        },
    )
    return {"status": "reset", "session_id": session_id, "count_in": 0, "count_out": 0}


@app.get("/logs/{filename}")
def get_log(filename: str) -> Dict[str, Any]:
    safe_name = "".join(char for char in filename if char.isalnum() or char in "._-")
    path = OUTPUT_DIR / safe_name
    if not path.exists():
        return {"name": safe_name, "content": "", "message": "Log chưa được tạo."}
    return {"name": safe_name, "content": path.read_text(encoding="utf-8")[-20000:]}
