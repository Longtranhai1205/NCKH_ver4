"""Live employee face recognition with OpenCV YuNet + SFace + the enrollment SQLite DB.

Run from the project root with the project's virtual environment activated:
    python face_recognition.py

Optional:
    python face_recognition.py --threshold 0.363 --min-margin 0.05
    python face_recognition.py --source 0
    python face_recognition.py --source "rtsp://..."

The RTSP source and model/database paths are imported from face_enrollment.py so
this module uses the same camera configuration and exact SQLite schema.

Controls: Q/ESC = quit.
"""
from __future__ import annotations

import argparse
import sys
import time
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import numpy as np

# Reuse the paths and camera source defined by the already-tested enrollment module.
# Importing this module does not start enrollment because it has a main guard.
try:
    from face_enrollment import DB_PATH, YUNET_PATH, SFACE_PATH, SOURCE as ENROLLMENT_SOURCE
except Exception as exc:  # pragma: no cover - clear runtime diagnostic
    raise SystemExit(
        "Cannot import configuration from face_enrollment.py. "
        "Keep face_recognition.py beside face_enrollment.py and check its dependencies.\n"
        f"Details: {exc}"
    )

WINDOW_NAME = "Employee Face Recognition - YuNet + SFace"
DEFAULT_THRESHOLD = 0.363  # OpenCV SFace reference threshold; must be validated locally.
DEFAULT_MIN_MARGIN = 0.05  # Separation required between best and second-best employee.
MIN_FACE_WIDTH = 70
MIN_FACE_HEIGHT = 70


@dataclass
class EmployeeTemplate:
    employee_id: str
    full_name: str
    vector: np.ndarray
    sample_count: int


def parse_source(value: str) -> Union[int, str]:
    """Convert numeric source values such as '0' to webcam indices."""
    return int(value) if value.isdigit() else value


def load_employee_templates(db_path: Path) -> list[EmployeeTemplate]:
    """Load enrolled vectors and build one normalized mean template per employee."""
    if not db_path.is_file():
        raise FileNotFoundError(
            f"Face database not found: {db_path}\n"
            "Run face_enrollment.py and save at least one employee first."
        )

    grouped: dict[str, dict[str, object]] = {}
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        required = {"employees", "face_embeddings"}
        if not required.issubset(tables):
            raise RuntimeError(
                "The SQLite database does not have the expected tables: "
                "employees and face_embeddings. Use the database created by face_enrollment.py."
            )

        rows = conn.execute(
            """SELECT e.employee_id, e.full_name,
                      fe.embedding, fe.dimension
               FROM employees AS e
               INNER JOIN face_embeddings AS fe
                   ON fe.employee_id = e.employee_id
               ORDER BY e.employee_id, fe.sample_index"""
        ).fetchall()

    if not rows:
        raise RuntimeError(
            "The database contains no saved face samples. Register an employee first."
        )

    for employee_id, full_name, blob, dimension in rows:
        if int(dimension) != 128:
            print(
                f"[WARN] Skipping {employee_id}: stored embedding dimension is {dimension}, not 128."
            )
            continue
        vector = np.frombuffer(blob, dtype=np.float32)
        if vector.size != 128 or not np.isfinite(vector).all():
            print(f"[WARN] Skipping an invalid embedding for employee {employee_id}.")
            continue
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            print(f"[WARN] Skipping a zero-length embedding for employee {employee_id}.")
            continue
        vector = (vector / norm).astype(np.float32)

        if employee_id not in grouped:
            grouped[employee_id] = {"full_name": full_name, "vectors": []}
        grouped[employee_id]["vectors"].append(vector)

    templates: list[EmployeeTemplate] = []
    for employee_id, item in grouped.items():
        samples = np.vstack(item["vectors"])
        # Enrollment samples are individually L2-normalized. Average them to create
        # a stable per-employee template, then normalize the average again.
        mean_vector = samples.mean(axis=0).astype(np.float32)
        mean_norm = float(np.linalg.norm(mean_vector))
        if mean_norm <= 1e-12:
            print(f"[WARN] Skipping employee {employee_id}: invalid mean template.")
            continue
        template = (mean_vector / mean_norm).astype(np.float32)
        templates.append(
            EmployeeTemplate(
                employee_id=employee_id,
                full_name=str(item["full_name"]),
                vector=template,
                sample_count=len(samples),
            )
        )

    if not templates:
        raise RuntimeError("No valid 128-dimensional employee templates were found in the database.")
    return templates


def load_models():
    missing = [str(path) for path in (YUNET_PATH, SFACE_PATH) if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing ONNX model file(s):\n  " + "\n  ".join(missing))
    detector = cv2.FaceDetectorYN.create(
        str(YUNET_PATH), "", (320, 320), 0.85, 0.3, 5000
    )
    recognizer = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")
    return detector, recognizer


def match_embedding(
    query_vector: np.ndarray,
    templates: list[EmployeeTemplate],
    threshold: float,
    min_margin: float,
):
    """Return identity decision, best cosine score, runner-up score and margin."""
    scores = [
        (float(np.dot(query_vector, employee.vector)), employee)
        for employee in templates
    ]
    scores.sort(key=lambda item: item[0], reverse=True)
    best_score, best_employee = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else -1.0
    margin = best_score - second_score if len(scores) > 1 else 1.0

    # Require both a minimum similarity and a clear lead over the next employee.
    if best_score >= threshold and margin >= min_margin:
        return best_employee, best_score, second_score, margin
    return None, best_score, second_score, margin


def draw_label(
    image: np.ndarray,
    face_row: np.ndarray,
    label: str,
    score: float,
    known: bool,
) -> None:
    x, y, w, h = face_row[:4].astype(int)
    height, width = image.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width - 1, x + w), min(height - 1, y + h)
    color = (0, 220, 0) if known else (0, 0, 255)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    text = f"{label} | cos={score:.3f}"
    text_y = y1 - 9 if y1 >= 28 else min(height - 5, y1 + 22)
    cv2.putText(
        image, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
        0.52, color, 2, cv2.LINE_AA
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recognize enrolled employees using YuNet, SFace and SQLite"
    )
    parser.add_argument(
        "--source",
        default=ENROLLMENT_SOURCE,
        help="Camera index, video/image path, or RTSP URL (defaults to enrollment camera)",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Minimum cosine similarity (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--min-margin", type=float, default=DEFAULT_MIN_MARGIN,
        help=f"Minimum lead over the second-best employee (default: {DEFAULT_MIN_MARGIN})",
    )
    args = parser.parse_args()

    if not -1.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between -1 and 1")
    if not 0.0 <= args.min_margin <= 2.0:
        parser.error("--min-margin must be between 0 and 2")

    print("\n=== EMPLOYEE FACE RECOGNITION ===")
    print(f"OpenCV: {cv2.__version__}")
    print(f"Database: {DB_PATH}")
    print(f"Cosine threshold: {args.threshold:.3f} | Minimum margin: {args.min_margin:.3f}")

    try:
        employees = load_employee_templates(Path(DB_PATH))
        detector, recognizer = load_models()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    total_samples = sum(employee.sample_count for employee in employees)
    print(f"[OK] Loaded {len(employees)} employee(s), {total_samples} face sample(s) from SQLite")
    for employee in employees:
        print(f"  - {employee.employee_id}: {employee.full_name} ({employee.sample_count} samples)")
    print("[OK] YuNet loaded")
    print("[OK] SFace loaded")
    print("[INFO] Opening camera... Press Q or ESC to quit.")

    source = parse_source(args.source)
    backend = (
        cv2.CAP_FFMPEG
        if isinstance(source, str) and source.lower().startswith("rtsp://")
        else cv2.CAP_ANY
    )
    cap = cv2.VideoCapture(source, backend)
    if not cap.isOpened():
        print("[ERROR] Could not open camera/source. Check RTSP URL, network and credentials.")
        return 1
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    previous_time = time.perf_counter()
    smoothed_fps = 0.0
    frame_number = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] No frame received from camera. Stopping recognition.")
                break

            frame_number += 1
            height, width = frame.shape[:2]
            detector.setInputSize((width, height))
            _status, faces = detector.detect(frame)
            preview = frame.copy()
            face_count = 0 if faces is None else len(faces)
            known_count = 0
            unknown_count = 0

            if faces is not None:
                for face in faces:
                    x, y, w, h = face[:4].astype(int)
                    confidence = float(face[14])
                    if w < MIN_FACE_WIDTH or h < MIN_FACE_HEIGHT:
                        draw_label(preview, face, "FACE TOO SMALL", 0.0, False)
                        unknown_count += 1
                        continue

                    try:
                        aligned = recognizer.alignCrop(frame, face)
                        raw_feature = recognizer.feature(aligned).reshape(-1).astype(np.float32)
                        norm = float(np.linalg.norm(raw_feature))
                        if raw_feature.size != 128 or not np.isfinite(raw_feature).all() or norm <= 1e-12:
                            draw_label(preview, face, "INVALID FEATURE", 0.0, False)
                            unknown_count += 1
                            continue
                        query_vector = raw_feature / norm
                        employee, score, second_score, margin = match_embedding(
                            query_vector, employees, args.threshold, args.min_margin
                        )
                    except cv2.error as exc:
                        print(f"[WARN] Could not process a detected face: {exc}")
                        draw_label(preview, face, "PROCESSING ERROR", 0.0, False)
                        unknown_count += 1
                        continue

                    if employee is not None:
                        label = f"{employee.full_name} ({employee.employee_id})"
                        known_count += 1
                        draw_label(preview, face, label, score, True)
                    else:
                        draw_label(preview, face, "UNKNOWN", score, False)
                        unknown_count += 1

            now = time.perf_counter()
            elapsed = max(now - previous_time, 1e-9)
            current_fps = 1.0 / elapsed
            smoothed_fps = current_fps if smoothed_fps == 0.0 else 0.9 * smoothed_fps + 0.1 * current_fps
            previous_time = now

            cv2.rectangle(preview, (0, 0), (width, 68), (0, 0, 0), -1)
            cv2.putText(
                preview,
                f"Faces: {face_count} | Known: {known_count} | Unknown: {unknown_count}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.putText(
                preview, f"FPS: {smoothed_fps:.1f} | Registered employees: {len(employees)}",
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            if frame_number % 300 == 0:
                print(
                    f"[INFO] Frame {frame_number}: faces={face_count}, known={known_count}, "
                    f"unknown={unknown_count}, FPS={smoothed_fps:.1f}"
                )
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("[INFO] Camera released. Face recognition test finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
