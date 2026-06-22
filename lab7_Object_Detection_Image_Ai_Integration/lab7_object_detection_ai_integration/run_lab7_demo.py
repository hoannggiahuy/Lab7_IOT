"""
Quick smoke test for Lab 7.

This script does not require a real camera. It creates sample images and runs the
same detection pipeline used by the API. If YOLO/ultralytics is not available, the
fallback contour detector is used so students can still observe the logging flow.
"""

from pathlib import Path
import json
import cv2

import app

ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "RUN_TEST_LOG.txt"


def main() -> None:
    app.create_sample_images()
    sample_path = app.SAMPLE_DIR / "sample_objects.jpg"
    frame = cv2.imread(str(sample_path))
    if frame is None:
        raise RuntimeError(f"Cannot read sample image: {sample_path}")
    result = app.detect_and_log(frame, source_type="demo_script", device_id="local_demo", conf=0.25, note="run_lab7_demo.py")
    checks = {
        "annotated_image_exists": bool(result.get("annotated_image_url")),
        "detection_log_exists": app.DETECTION_CSV.exists(),
        "vision_event_log_exists": app.EVENT_CSV.exists(),
        "backend": result["model_status"]["backend"],
        "num_detections": result["num_detections"],
    }
    status = "LOCAL_PIPELINE_TEST_PASS" if checks["annotated_image_exists"] and checks["vision_event_log_exists"] else "LOCAL_PIPELINE_TEST_FAIL"
    LOG_PATH.write_text(status + "\n" + json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(status)
    print(json.dumps(checks, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
