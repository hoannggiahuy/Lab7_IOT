"""Smoke test Trạm 2 không cần camera và không cần YOLO.

Tạo một vật thể trắng đi xuống rồi đi lên qua vạch ngang. Kết quả hợp lệ
khi tracker tạo ID, có ít nhất một LINE_CROSSING_DETECTED và sinh ảnh/log.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from tracking_engine import TrackingState, process_tracking_frame

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
SNAPSHOT_DIR = BASE_DIR / "data" / "snapshots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def make_frame(y: int) -> np.ndarray:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (10, 10, 10)
    cv2.rectangle(frame, (250, y), (370, y + 110), (255, 255, 255), -1)
    return frame


def main() -> None:
    state = TrackingState()
    all_events = []
    final_metadata = {}
    final_records = []
    paths = []

    # Đi xuống qua vạch y=240 rồi đi ngược lên.
    sequence = list(range(40, 330, 12)) + list(range(330, 40, -12))
    for index, y in enumerate(sequence):
        annotated, records, events, metadata = process_tracking_frame(
            frame=make_frame(y),
            state=state,
            confidence=0.30,
            class_filter="",
            line_orientation="horizontal",
            line_position=0.50,
            max_distance=80,
            max_missed=5,
            force_fallback=True,
        )
        final_metadata = metadata
        final_records = records
        all_events.extend(events)
        if events:
            path = SNAPSHOT_DIR / f"demo_crossing_{index:03d}.jpg"
            cv2.imwrite(str(path), annotated)
            paths.append(str(path.relative_to(BASE_DIR)))

    report = {
        "status": "PASS" if len(all_events) >= 2 and state.count_in >= 1 and state.count_out >= 1 else "FAIL",
        "backend": final_metadata.get("backend"),
        "frames_processed": len(sequence),
        "count_in": state.count_in,
        "count_out": state.count_out,
        "crossing_events": all_events,
        "last_records": final_records,
        "snapshots": paths,
    }
    (OUTPUT_DIR / "tracking_demo_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with (OUTPUT_DIR / "tracking_demo_events.csv").open("w", encoding="utf-8", newline="") as file:
        fieldnames = ["event_type", "track_id", "class_name", "direction", "count_in", "count_out", "frame_index"]
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_events)

    line = "LOCAL_TRACKING_PIPELINE_TEST_PASS" if report["status"] == "PASS" else "LOCAL_TRACKING_PIPELINE_TEST_FAIL"
    (BASE_DIR / "RUN_TEST_LOG.txt").write_text(line + "\n", encoding="utf-8")
    print(line)
    print(json.dumps({k: report[k] for k in ["backend", "frames_processed", "count_in", "count_out", "snapshots"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
