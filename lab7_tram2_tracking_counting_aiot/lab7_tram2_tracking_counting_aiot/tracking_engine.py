"""Tracking & Counting engine for Lab 7 mở rộng - Trạm 2.

Luồng xử lý:
    Camera frame -> YOLO detection (hoặc fallback) -> centroid tracker
    -> kiểm tra cắt vạch -> count_in/count_out -> visual event.

Engine cố gắng dùng YOLOv8n khi đã cài ``ultralytics``. Nếu chưa có,
chương trình dùng bộ phát hiện contour để người học vẫn kiểm tra được
pipeline, dashboard, tracker, counting và log.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]
Point = Tuple[int, int]


@dataclass
class Detection:
    bbox: BBox
    class_id: int
    class_name: str
    confidence: float

    @property
    def centroid(self) -> Point:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass
class Track:
    track_id: int
    bbox: BBox
    centroid: Point
    class_name: str
    confidence: float
    age: int = 1
    missed: int = 0
    stable_side: int = 0
    last_count_frame: int = -10_000
    history: Deque[Point] = field(default_factory=lambda: deque(maxlen=30))

    def __post_init__(self) -> None:
        self.history.append(self.centroid)


@dataclass
class TrackingState:
    tracks: Dict[int, Track] = field(default_factory=dict)
    next_id: int = 1
    count_in: int = 0
    count_out: int = 0
    frame_index: int = 0
    last_backend: str = "not_started"
    last_inference_ms: float = 0.0
    last_processing_ms: float = 0.0
    last_event: Dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = 1
        self.count_in = 0
        self.count_out = 0
        self.frame_index = 0
        self.last_event = {}


_YOLO_CACHE: Dict[str, Any] = {}


def resize_keep_width(frame: np.ndarray, width: int = 640) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= width:
        return frame.copy()
    scale = width / float(w)
    return cv2.resize(frame, (width, max(1, int(h * scale))))


def try_load_yolo(model_path: str = "yolov8n.pt") -> Optional[Any]:
    """Tải YOLO và cache theo đường dẫn model; trả ``None`` nếu chưa cài."""
    if model_path in _YOLO_CACHE:
        return _YOLO_CACHE[model_path]
    try:
        from ultralytics import YOLO  # type: ignore

        model = YOLO(model_path)
        _YOLO_CACHE[model_path] = model
        return model
    except Exception:
        return None


def _normalise_names(names: Any) -> Dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def _parse_class_filter(class_filter: str) -> List[str]:
    return [part.strip().lower() for part in class_filter.split(",") if part.strip()]


def _yolo_detections(
    frame: np.ndarray,
    model_path: str,
    confidence: float,
    class_filter: str,
) -> Tuple[List[Detection], str, float]:
    model = try_load_yolo(model_path)
    if model is None:
        return [], "fallback_contour", 0.0

    names = _normalise_names(getattr(model, "names", {}))
    wanted_names = _parse_class_filter(class_filter)
    wanted_ids = [class_id for class_id, name in names.items() if name.lower() in wanted_names]
    # Khi người dùng nhập class không tồn tại, trả rỗng thay vì chạy tất cả class.
    if wanted_names and not wanted_ids:
        return [], "yolo_ultralytics", 0.0

    start = time.perf_counter()
    kwargs: Dict[str, Any] = {
        "source": frame,
        "conf": float(confidence),
        "imgsz": 640,
        "verbose": False,
    }
    if wanted_ids:
        kwargs["classes"] = wanted_ids
    results = model.predict(**kwargs)
    inference_ms = (time.perf_counter() - start) * 1000.0

    detections: List[Detection] = []
    if not results:
        return detections, "yolo_ultralytics", inference_ms

    boxes = getattr(results[0], "boxes", None)
    if boxes is None:
        return detections, "yolo_ultralytics", inference_ms

    for box in boxes:
        score = float(box.conf[0].item())
        class_id = int(box.cls[0].item())
        x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
        class_name = names.get(class_id, f"class_{class_id}")
        if wanted_names and class_name.lower() not in wanted_names:
            continue
        detections.append(
            Detection(
                bbox=(x1, y1, x2, y2),
                class_id=class_id,
                class_name=class_name,
                confidence=score,
            )
        )
    return detections, "yolo_ultralytics", inference_ms


def _fallback_detections(
    frame: np.ndarray,
    confidence: float,
    min_area: int = 1200,
) -> Tuple[List[Detection], str, float]:
    """Phát hiện contour để test pipeline khi chưa có YOLO.

    Fallback chỉ trả class ``moving_object``; không giả mạo class COCO.
    """
    start = time.perf_counter()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = float(frame.shape[0] * frame.shape[1])
    detections: List[Detection] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 20 or h < 20:
            continue
        score = min(0.90, 0.45 + area / max(image_area, 1.0))
        if score < confidence:
            continue
        detections.append(
            Detection(
                bbox=(x, y, x + w, y + h),
                class_id=-1,
                class_name="moving_object",
                confidence=score,
            )
        )
        if len(detections) >= 10:
            break
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return detections, "fallback_contour", elapsed_ms


def detect_objects(
    frame: np.ndarray,
    model_path: str = "yolov8n.pt",
    confidence: float = 0.35,
    class_filter: str = "person",
    force_fallback: bool = False,
) -> Tuple[List[Detection], str, float]:
    if not force_fallback:
        detections, backend, inference_ms = _yolo_detections(
            frame=frame,
            model_path=model_path,
            confidence=confidence,
            class_filter=class_filter,
        )
        if backend == "yolo_ultralytics":
            return detections, backend, inference_ms
    return _fallback_detections(frame, confidence=confidence)


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def update_tracks(
    state: TrackingState,
    detections: Sequence[Detection],
    max_distance: float = 90.0,
    max_missed: int = 12,
) -> List[Track]:
    """Ghép detection hiện tại với track cũ bằng khoảng cách centroid.

    Đây là centroid tracker đơn giản, dễ đọc cho sinh viên. YOLO chịu trách nhiệm
    nhận diện; tracker chịu trách nhiệm giữ ID qua nhiều frame.
    """
    for track in state.tracks.values():
        track.missed += 1

    candidates: List[Tuple[float, int, int]] = []
    track_items = list(state.tracks.items())
    for det_index, detection in enumerate(detections):
        for track_id, track in track_items:
            same_class = (
                detection.class_name == track.class_name
                or detection.class_name == "moving_object"
                or track.class_name == "moving_object"
            )
            if not same_class:
                continue
            dist = _distance(detection.centroid, track.centroid)
            if dist <= max_distance:
                candidates.append((dist, det_index, track_id))

    assigned_detections: set[int] = set()
    assigned_tracks: set[int] = set()
    for _, det_index, track_id in sorted(candidates, key=lambda item: item[0]):
        if det_index in assigned_detections or track_id in assigned_tracks:
            continue
        detection = detections[det_index]
        track = state.tracks[track_id]
        track.bbox = detection.bbox
        track.centroid = detection.centroid
        track.class_name = detection.class_name
        track.confidence = detection.confidence
        track.age += 1
        track.missed = 0
        track.history.append(track.centroid)
        assigned_detections.add(det_index)
        assigned_tracks.add(track_id)

    for det_index, detection in enumerate(detections):
        if det_index in assigned_detections:
            continue
        track_id = state.next_id
        state.next_id += 1
        state.tracks[track_id] = Track(
            track_id=track_id,
            bbox=detection.bbox,
            centroid=detection.centroid,
            class_name=detection.class_name,
            confidence=detection.confidence,
        )

    stale_ids = [
        track_id
        for track_id, track in state.tracks.items()
        if track.missed > max_missed
    ]
    for track_id in stale_ids:
        del state.tracks[track_id]

    return [track for track in state.tracks.values() if track.missed == 0]


def _stable_side(
    centroid: Point,
    orientation: str,
    line_coordinate: int,
    hysteresis: int,
) -> int:
    value = centroid[1] if orientation == "horizontal" else centroid[0]
    if value < line_coordinate - hysteresis:
        return -1
    if value > line_coordinate + hysteresis:
        return 1
    return 0


def count_crossings(
    state: TrackingState,
    active_tracks: Iterable[Track],
    frame_shape: Tuple[int, int, int],
    line_orientation: str = "horizontal",
    line_position: float = 0.55,
    hysteresis: int = 12,
    cooldown_frames: int = 12,
) -> Tuple[List[Dict[str, Any]], int]:
    h, w = frame_shape[:2]
    orientation = "vertical" if line_orientation.lower() == "vertical" else "horizontal"
    ratio = max(0.10, min(0.90, float(line_position)))
    line_coordinate = int((w if orientation == "vertical" else h) * ratio)
    events: List[Dict[str, Any]] = []

    for track in active_tracks:
        side = _stable_side(track.centroid, orientation, line_coordinate, hysteresis)
        if side == 0:
            continue
        previous_side = track.stable_side
        if previous_side == 0:
            track.stable_side = side
            continue
        if side == previous_side:
            continue
        if state.frame_index - track.last_count_frame < cooldown_frames:
            track.stable_side = side
            continue

        if previous_side == -1 and side == 1:
            direction = "down" if orientation == "horizontal" else "right"
            state.count_in += 1
            counter_name = "count_in"
        else:
            direction = "up" if orientation == "horizontal" else "left"
            state.count_out += 1
            counter_name = "count_out"

        track.stable_side = side
        track.last_count_frame = state.frame_index
        events.append(
            {
                "event_type": "LINE_CROSSING_DETECTED",
                "severity": "WARNING",
                "track_id": track.track_id,
                "class_name": track.class_name,
                "confidence": round(track.confidence, 4),
                "direction": direction,
                "counter_updated": counter_name,
                "centroid": list(track.centroid),
                "bbox": list(track.bbox),
                "count_in": state.count_in,
                "count_out": state.count_out,
                "line_orientation": orientation,
                "line_position": ratio,
                "frame_index": state.frame_index,
                "explanation": f"Track {track.track_id} ({track.class_name}) crossed the {orientation} line toward {direction}.",
                "action_hint": "Update occupancy/count dashboard; require rule validation before controlling an actuator.",
            }
        )
    return events, line_coordinate


def _track_color(track_id: int) -> Tuple[int, int, int]:
    # BGR, sinh màu ổn định theo ID mà không cần thư viện ngoài.
    return (
        int(80 + (track_id * 47) % 175),
        int(80 + (track_id * 83) % 175),
        int(80 + (track_id * 131) % 175),
    )


def annotate_frame(
    frame: np.ndarray,
    active_tracks: Sequence[Track],
    state: TrackingState,
    line_orientation: str,
    line_coordinate: int,
    backend: str,
    processing_ms: float,
) -> np.ndarray:
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    orientation = "vertical" if line_orientation.lower() == "vertical" else "horizontal"

    if orientation == "horizontal":
        cv2.line(annotated, (0, line_coordinate), (w, line_coordinate), (0, 200, 255), 3)
        cv2.putText(annotated, "COUNTING LINE", (10, max(25, line_coordinate - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
    else:
        cv2.line(annotated, (line_coordinate, 0), (line_coordinate, h), (0, 200, 255), 3)
        cv2.putText(annotated, "COUNTING LINE", (min(w - 170, line_coordinate + 8), 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

    for track in active_tracks:
        color = _track_color(track.track_id)
        x1, y1, x2, y2 = track.bbox
        cx, cy = track.centroid
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
        label = f"ID {track.track_id} | {track.class_name} {track.confidence:.2f}"
        label_y = max(24, y1)
        text_width = max(155, len(label) * 8)
        cv2.rectangle(annotated, (x1, label_y - 22), (min(w - 1, x1 + text_width), label_y + 3), color, -1)
        cv2.putText(annotated, label, (x1 + 4, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 0, 0), 1, cv2.LINE_AA)
        points = list(track.history)
        for p1, p2 in zip(points, points[1:]):
            cv2.line(annotated, p1, p2, color, 2)

    cv2.rectangle(annotated, (0, 0), (w, 42), (25, 30, 38), -1)
    cv2.putText(annotated, "TRAM 2 - TRACKING & COUNTING", (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    stats = f"IN: {state.count_in}  OUT: {state.count_out}  ACTIVE: {len(active_tracks)}"
    cv2.rectangle(annotated, (0, h - 40), (w, h), (25, 30, 38), -1)
    cv2.putText(annotated, stats, (10, h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (0, 220, 255), 2)
    backend_text = f"{backend} | {processing_ms:.1f} ms"
    cv2.putText(annotated, backend_text, (max(10, w - 300), 27), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (210, 220, 230), 1)
    return annotated


def process_tracking_frame(
    frame: np.ndarray,
    state: TrackingState,
    confidence: float = 0.35,
    class_filter: str = "person",
    model_path: str = "yolov8n.pt",
    line_orientation: str = "horizontal",
    line_position: float = 0.55,
    max_distance: float = 90.0,
    max_missed: int = 12,
    hysteresis: int = 12,
    force_fallback: bool = False,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    start = time.perf_counter()
    resized = resize_keep_width(frame, 640)
    detections, backend, inference_ms = detect_objects(
        frame=resized,
        model_path=model_path,
        confidence=confidence,
        class_filter=class_filter,
        force_fallback=force_fallback,
    )
    active_tracks = update_tracks(
        state=state,
        detections=detections,
        max_distance=max_distance,
        max_missed=max_missed,
    )
    crossing_events, line_coordinate = count_crossings(
        state=state,
        active_tracks=active_tracks,
        frame_shape=resized.shape,
        line_orientation=line_orientation,
        line_position=line_position,
        hysteresis=hysteresis,
    )
    processing_ms = (time.perf_counter() - start) * 1000.0
    state.last_backend = backend
    state.last_inference_ms = inference_ms
    state.last_processing_ms = processing_ms

    records: List[Dict[str, Any]] = []
    for track in active_tracks:
        records.append(
            {
                "track_id": track.track_id,
                "class_name": track.class_name,
                "confidence": round(track.confidence, 4),
                "bbox": list(track.bbox),
                "centroid": list(track.centroid),
                "age_frames": track.age,
                "trajectory": [list(point) for point in track.history],
                "count_in": state.count_in,
                "count_out": state.count_out,
                "backend": backend,
                "frame_index": state.frame_index,
            }
        )

    if crossing_events:
        state.last_event = crossing_events[-1]
    elif records:
        state.last_event = {
            "event_type": "TRACKING_ACTIVE",
            "severity": "NORMAL",
            "num_tracks": len(records),
            "count_in": state.count_in,
            "count_out": state.count_out,
            "explanation": "Objects are being tracked; no new line crossing in this frame.",
            "action_hint": "Continue monitoring.",
        }
    else:
        state.last_event = {
            "event_type": "NO_OBJECT_TRACKED",
            "severity": "NORMAL",
            "num_tracks": 0,
            "count_in": state.count_in,
            "count_out": state.count_out,
            "explanation": "No object passed the current confidence/class filters.",
            "action_hint": "Check lighting, distance, confidence threshold and class filter.",
        }

    annotated = annotate_frame(
        frame=resized,
        active_tracks=active_tracks,
        state=state,
        line_orientation=line_orientation,
        line_coordinate=line_coordinate,
        backend=backend,
        processing_ms=processing_ms,
    )
    metadata = {
        "backend": backend,
        "model_path": model_path,
        "confidence_threshold": float(confidence),
        "class_filter": class_filter,
        "line_orientation": "vertical" if line_orientation.lower() == "vertical" else "horizontal",
        "line_position": max(0.10, min(0.90, float(line_position))),
        "line_coordinate": line_coordinate,
        "active_tracks": len(records),
        "count_in": state.count_in,
        "count_out": state.count_out,
        "inference_time_ms": round(inference_ms, 2),
        "processing_time_ms": round(processing_ms, 2),
        "frame_index": state.frame_index,
    }
    state.frame_index += 1
    return annotated, records, crossing_events, metadata
