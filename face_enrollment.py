"""Employee face enrollment using OpenCV YuNet + SFace and SQLite.

Run from the project root with the project's venv activated:
    python face_enrollment.py

Controls in the camera window:
    SPACE  Capture one valid face sample
    U      Undo the latest sample
    S      Save enrollment (requires the configured minimum samples)
    Q/ESC  Cancel without saving

For security, set FACE_RTSP_URL in PowerShell rather than editing this file:
    $env:FACE_RTSP_URL = "rtsp://username:password@camera-ip:554/..."
If FACE_RTSP_URL is not set, the script uses the project's configured EZVIZ RTSP URL.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "face_recognition"
YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
DB_PATH = ROOT / "data" / "face_database.sqlite3"

# Same camera stream configured in detect_v4_fixed.py. The @ in the password
# must be URL-encoded as %40. Prefer FACE_RTSP_URL environment variable.
DEFAULT_RTSP_URL = "rtsp://admin:Longtran2005%40@192.168.2.79:554/h264/ch1/sub/av_stream"
SOURCE = os.environ.get("FACE_RTSP_URL", DEFAULT_RTSP_URL)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

MIN_SAMPLES = 5
TARGET_SAMPLES = 10
MIN_FACE_WIDTH = 90
MIN_FACE_HEIGHT = 90
MIN_DETECTION_CONFIDENCE = 0.85
MIN_BLUR_SCORE = 35.0
WINDOW_NAME = "Employee Face Enrollment - YuNet + SFace"


def init_database() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS employees (
                employee_id TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS face_embeddings (
                embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT NOT NULL,
                sample_index INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                dimension INTEGER NOT NULL DEFAULT 128,
                captured_at TEXT NOT NULL,
                FOREIGN KEY(employee_id) REFERENCES employees(employee_id)
                    ON DELETE CASCADE,
                UNIQUE(employee_id, sample_index)
            )"""
        )


def load_models():
    for path in (YUNET_PATH, SFACE_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
    detector = cv2.FaceDetectorYN.create(
        str(YUNET_PATH), "", (320, 320), 0.85, 0.3, 5000
    )
    recognizer = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")
    return detector, recognizer


def face_blur_score(frame: np.ndarray, face_row: np.ndarray) -> float:
    x, y, w, h = face_row[:4].astype(int)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main() -> int:
    print("\n=== EMPLOYEE FACE ENROLLMENT ===")
    print(f"OpenCV: {cv2.__version__}")
    print(f"Database: {DB_PATH}")
    print("Enter one employee at a time. Only the intended employee should be in frame.")

    employee_id = input("Employee ID (e.g. NV001): ").strip()
    full_name = input("Full name: ").strip()
    if not employee_id or not full_name:
        print("[ERROR] Employee ID and full name are required.")
        return 1
    if len(employee_id) > 64 or len(full_name) > 150:
        print("[ERROR] Employee ID or name is too long.")
        return 1

    init_database()
    with sqlite3.connect(DB_PATH) as conn:
        exists = conn.execute(
            "SELECT 1 FROM employees WHERE employee_id = ?", (employee_id,)
        ).fetchone()
    if exists:
        print(f"[ERROR] Employee ID {employee_id!r} already exists. No data was changed.")
        print("Use a new ID, or ask to add a deliberate re-enrollment/update workflow.")
        return 1

    try:
        detector, recognizer = load_models()
    except Exception as exc:
        print(f"[ERROR] Could not load YuNet/SFace: {exc}")
        return 1

    print("[OK] YuNet loaded")
    print("[OK] SFace loaded")
    print("[INFO] Opening configured RTSP camera...")
    cap = cv2.VideoCapture(SOURCE, cv2.CAP_FFMPEG if SOURCE.lower().startswith("rtsp://") else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check FACE_RTSP_URL, network and camera credentials.")
        return 1

    embeddings: list[np.ndarray] = []
    last_capture_time = 0.0
    print("\nControls: SPACE=capture | U=undo | S=save | Q/ESC=cancel")
    print(f"Capture {TARGET_SAMPLES} samples if possible; at least {MIN_SAMPLES} are required.")
    print("Ask the employee to look at the camera, then turn slightly left/right between captures.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] Camera frame read failed.")
                time.sleep(0.1)
                continue

            height, width = frame.shape[:2]
            detector.setInputSize((width, height))
            _status, faces = detector.detect(frame)
            preview = frame.copy()
            face_count = 0 if faces is None else len(faces)
            selected_face = None

            if faces is not None:
                # Draw every detected face, but only allow enrollment if exactly one is present.
                for idx, face in enumerate(faces, start=1):
                    x, y, w, h = face[:4].astype(int)
                    confidence = float(face[14])
                    color = (0, 220, 0) if face_count == 1 else (0, 165, 255)
                    cv2.rectangle(preview, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(preview, f"Face {idx}: {confidence:.2f}",
                                (max(5, x), max(22, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, color, 2, cv2.LINE_AA)
                if face_count == 1:
                    selected_face = faces[0]

            blur_score = face_blur_score(frame, selected_face) if selected_face is not None else 0.0
            if face_count == 0:
                status_text = "No face detected - move into view"
            elif face_count > 1:
                status_text = "More than one face - keep only the employee in frame"
            else:
                fw, fh = selected_face[2:4]
                if float(selected_face[14]) < MIN_DETECTION_CONFIDENCE:
                    status_text = "Detection confidence too low - improve lighting/position"
                elif fw < MIN_FACE_WIDTH or fh < MIN_FACE_HEIGHT:
                    status_text = "Move closer - face is too small"
                elif blur_score < MIN_BLUR_SCORE:
                    status_text = "Image is blurry - hold still"
                else:
                    status_text = "Ready - press SPACE to capture"

            cv2.rectangle(preview, (0, 0), (width, 112), (0, 0, 0), -1)
            cv2.putText(preview, f"Employee: {full_name} ({employee_id})", (12, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(preview, f"Faces: {face_count} | Samples: {len(embeddings)}/{TARGET_SAMPLES}",
                        (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(preview, status_text, (12, 82), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (0, 255, 0) if status_text.startswith("Ready") else (0, 165, 255),
                        2, cv2.LINE_AA)
            cv2.putText(preview, "SPACE capture | U undo | S save | Q cancel", (12, 104),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("[CANCELLED] Enrollment cancelled; nothing saved.")
                return 0
            if key == ord("u"):
                if embeddings:
                    embeddings.pop()
                    print(f"[INFO] Removed latest sample. Current: {len(embeddings)}")
                else:
                    print("[INFO] No sample to undo.")
            elif key == ord(" "):
                now = time.monotonic()
                if now - last_capture_time < 0.5:
                    continue
                last_capture_time = now
                if selected_face is None:
                    print("[SKIP] Exactly one face must be visible.")
                    continue
                confidence = float(selected_face[14])
                fw, fh = selected_face[2:4]
                if confidence < MIN_DETECTION_CONFIDENCE or fw < MIN_FACE_WIDTH or fh < MIN_FACE_HEIGHT:
                    print("[SKIP] Face confidence/size is below the quality threshold.")
                    continue
                if blur_score < MIN_BLUR_SCORE:
                    print(f"[SKIP] Image is too blurry (score {blur_score:.1f}). Hold still and retry.")
                    continue
                aligned = recognizer.alignCrop(frame, selected_face)
                feature = recognizer.feature(aligned).reshape(-1).astype(np.float32)
                norm = float(np.linalg.norm(feature))
                if feature.size != 128 or not np.isfinite(feature).all() or norm == 0:
                    print("[SKIP] SFace returned an invalid embedding.")
                    continue
                # Normalize for consistent cosine-similarity comparisons later.
                feature /= norm
                embeddings.append(feature.copy())
                print(f"[OK] Captured sample {len(embeddings)}/{TARGET_SAMPLES} | confidence={confidence:.2f} | blur={blur_score:.1f}")
            elif key == ord("s"):
                if len(embeddings) < MIN_SAMPLES:
                    print(f"[WARN] Need at least {MIN_SAMPLES} valid samples; currently have {len(embeddings)}.")
                    continue
                created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                try:
                    with sqlite3.connect(DB_PATH) as conn:
                        conn.execute("PRAGMA foreign_keys = ON")
                        conn.execute("BEGIN IMMEDIATE")
                        conn.execute(
                            "INSERT INTO employees(employee_id, full_name, created_at) VALUES (?, ?, ?)",
                            (employee_id, full_name, created_at),
                        )
                        for index, vector in enumerate(embeddings, start=1):
                            conn.execute(
                                """INSERT INTO face_embeddings
                                   (employee_id, sample_index, embedding, dimension, captured_at)
                                   VALUES (?, ?, ?, 128, ?)""",
                                (employee_id, index, sqlite3.Binary(vector.tobytes()), created_at),
                            )
                        conn.commit()
                except sqlite3.IntegrityError as exc:
                    print(f"[ERROR] Could not save enrollment: {exc}")
                    return 1
                print("\n[SUCCESS] Employee enrolled successfully.")
                print(f"Employee ID : {employee_id}")
                print(f"Full name   : {full_name}")
                print(f"Samples     : {len(embeddings)}")
                print(f"Database    : {DB_PATH}")
                return 0

    except KeyboardInterrupt:
        print("\n[CANCELLED] Stopped by user; samples were not saved.")
        return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
