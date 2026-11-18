"""Integrated PPE detection, ByteTrack person tracking, face recognition and alerts.

Based on the user's working detect_v4_fixed.py, face_enrollment_v2.py,
and face_recognition_v2.py.

Run from the project root with the venv activated:
    python detect_ppe_face_integrated_v2.py

Optional settings:
    python detect_ppe_face_integrated_v2.py --threshold 0.45 --min-margin 0.05
    python detect_ppe_face_integrated_v2.py --source 0
    python detect_ppe_face_integrated_v2.py --face-every 5

Camera credentials are requested securely (password is not echoed) if
PPE_RTSP_URL is not set. You can also set PPE_RTSP_URL in the environment.

Controls:
    Q / ESC = quit
"""
from __future__ import annotations

import argparse
import csv
import getpass
import os
import sqlite3
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
from urllib.parse import quote

# Set FFmpeg RTSP transport before opening VideoCapture.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml


# ============================================================
# 1. PROJECT / MODEL / CAMERA CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
YOLO_MODEL_PATH = ROOT / "best.onnx"
YUNET_PATH = ROOT / "models" / "face_recognition" / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = ROOT / "models" / "face_recognition" / "face_recognition_sface_2021dec.onnx"
FACE_DB_PATH = ROOT / "data" / "face_database.sqlite3"

CAMERA_IP = "192.168.2.79"
CAMERA_USERNAME = "admin"
CAMERA_RTSP_SUFFIX = "/h264/ch1/sub/av_stream"

IMGSZ = 416
# Run inference at a lower global confidence, then apply separate thresholds:
# Person retains the proven 0.70 threshold; PPE gets a lower threshold to reduce misses.
YOLO_INFER_CONF = 0.30
PERSON_CONF = 0.70
PPE_CONF = 0.45
OTHER_CONF = 0.50
PERSON_CLASS_ID = 5
WARMUP_FRAMES = 5
REPORT_INTERVAL = 100
WINDOW_NAME = "PPE + Face Recognition | YOLO + ByteTrack"

# Face recognition settings. The threshold is an initial value, not a final
# deployment threshold; calibrate it using this exact matching method/database.
DEFAULT_FACE_THRESHOLD = 0.363
DEFAULT_FACE_MIN_MARGIN = 0.05
DEFAULT_FACE_TOP_K = 2
FACE_DETECT_EVERY_N_FRAMES = 4
FACE_MIN_CONFIDENCE = 0.75
# The first uploaded frame had a face box about 80x55 px; the old 70 px
# minimum height rejected it, leaving the label at FACE: PENDING.
FACE_MIN_WIDTH = 55
FACE_MIN_HEIGHT = 50
IDENTITY_SWITCH_CONFIRMATIONS = 2

# Existing PPE alert behavior preserved from detect_v4_fixed.py.
SMOOTH_FRAMES = 5                 # Number of valid PPE observations used for voting
PPE_UNKNOWN_HOLD_SECONDS = 0.75   # Hold last known PPE state briefly through missed detections
VIOLATION_SECONDS = 2.0
ALERT_COOLDOWN = 30.0
TRACK_TIMEOUT = 10.0
GROUPS = ("Mu", "Khau trang", "Ao bao ho")
ALERT_GROUPS = ("Mu", "Khau trang", "Ao bao ho")
ALERT_DIR = ROOT / "alerts"
ALERT_LOG_FILE = ALERT_DIR / "alert_log_integrated_v2.csv"

COLOR_PPE_OK = (0, 255, 0)
COLOR_PPE_VIOLATION = (0, 0, 255)
COLOR_PERSON = (255, 180, 0)
COLOR_PERSON_OK = (0, 200, 0)
COLOR_PERSON_ALERT = (0, 0, 255)
COLOR_OTHER = (0, 220, 255)
COLOR_FACE = (255, 255, 0)
COLOR_UNKNOWN = (0, 165, 255)

PERSON_COLOR_PALETTE = [
    (255, 0, 255), (255, 128, 0), (128, 0, 255), (0, 165, 255),
    (255, 255, 0), (180, 105, 255), (128, 128, 0), (255, 0, 128),
    (0, 200, 255), (200, 80, 160), (255, 180, 80), (80, 120, 255),
]


def get_camera_source(cli_source: Optional[str]) -> Union[int, str]:
    """Resolve CLI source, environment URL, or request a hidden camera password."""
    if cli_source is not None:
        return int(cli_source) if cli_source.isdigit() else cli_source

    env_url = os.environ.get("PPE_RTSP_URL", "").strip()
    if env_url:
        return env_url

    password = getpass.getpass("Nhap mat khau camera EZVIZ (ky tu se duoc an): ")
    if not password:
        raise ValueError("Mat khau camera dang de trong.")
    encoded_password = quote(password, safe="")
    return (
        f"rtsp://{CAMERA_USERNAME}:{encoded_password}@{CAMERA_IP}:554"
        f"{CAMERA_RTSP_SUFFIX}"
    )


# ============================================================
# 2. CAMERA READER (same threaded latest-frame approach)
# ============================================================

class RTSPReader:
    def __init__(self, source: Union[int, str]):
        backend = (
            cv2.CAP_FFMPEG
            if isinstance(source, str) and source.lower().startswith("rtsp://")
            else cv2.CAP_ANY
        )
        self.cap = cv2.VideoCapture(source, backend)
        if not self.cap.isOpened():
            raise RuntimeError(
                "Khong mo duoc camera/video. Kiem tra RTSP URL, IP, mang va mat khau."
            )
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame: Optional[np.ndarray] = None
        self.ok = False
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while self.running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            with self.lock:
                self.ok = True
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ok, self.frame.copy()

    def release(self) -> None:
        self.running = False
        self.thread.join(timeout=2.0)
        self.cap.release()


# ============================================================
# 3. FACE DATABASE + YUNET / SFACE
# ============================================================

@dataclass
class EmployeeTemplate:
    employee_id: str
    full_name: str
    vectors: np.ndarray  # N x 128; each stored sample remains separate.
    sample_count: int


def load_employee_templates(db_path: Path) -> list[EmployeeTemplate]:
    """Load embeddings from the one shared V2 database created by enrollment V2."""
    if not db_path.is_file():
        raise FileNotFoundError(
            f"Khong tim thay database khuon mat: {db_path}\n"
            "Hay dang ky it nhat mot nhan vien bang face_enrollment_v2.py truoc."
        )

    with sqlite3.connect(str(db_path)) as conn:
        tables = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if not {"employees", "face_embeddings"}.issubset(tables):
            raise RuntimeError(
                "Database khong co du hai bang employees va face_embeddings. "
                "Hay tao database bang face_enrollment_v2.py."
            )
        embedding_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(face_embeddings)").fetchall()
        }
        # V2 enrollment adds pose_label. Refuse an old V1-only DB rather than
        # silently mixing unclassified/legacy embeddings into V2 recognition.
        if "pose_label" not in embedding_columns:
            raise RuntimeError(
                "Database thieu cot pose_label cua Enrollment V2. "
                "Hay xoa database cu va dang ky lai bang face_enrollment_v2.py."
            )
        rows = conn.execute(
            """SELECT e.employee_id, e.full_name, fe.embedding, fe.dimension, fe.pose_label
               FROM employees e
               INNER JOIN face_embeddings fe ON fe.employee_id = e.employee_id
               WHERE LOWER(COALESCE(fe.pose_label, 'unknown')) != 'unknown'
               ORDER BY e.employee_id, fe.sample_index"""
        ).fetchall()

    grouped: dict[str, dict[str, object]] = {}
    for employee_id, full_name, blob, dimension, pose_label in rows:
        if int(dimension) != 128:
            print(f"[WARN] Bo qua {employee_id}: embedding dimension={dimension}.")
            continue
        vector = np.frombuffer(blob, dtype=np.float32)
        if vector.size != 128 or not np.isfinite(vector).all():
            print(f"[WARN] Bo qua embedding khong hop le cua {employee_id}.")
            continue
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            print(f"[WARN] Bo qua embedding co norm bang 0 cua {employee_id}.")
            continue
        vector = (vector / norm).astype(np.float32)
        if employee_id not in grouped:
            grouped[employee_id] = {"full_name": str(full_name), "vectors": []}
        grouped[employee_id]["vectors"].append(vector)

    employees: list[EmployeeTemplate] = []
    for employee_id, data in grouped.items():
        matrix = np.vstack(data["vectors"]).astype(np.float32)
        employees.append(
            EmployeeTemplate(
                employee_id=str(employee_id),
                full_name=str(data["full_name"]),
                vectors=matrix,
                sample_count=len(matrix),
            )
        )
    if not employees:
        raise RuntimeError("Database chua co embedding hop le. Hay dang ky nhan vien truoc.")
    return employees


def load_face_models():
    missing = [str(p) for p in (YUNET_PATH, SFACE_PATH) if not p.is_file()]
    if missing:
        raise FileNotFoundError("Thieu model ONNX:\n  " + "\n  ".join(missing))
    detector = cv2.FaceDetectorYN.create(
        str(YUNET_PATH), "", (320, 320), 0.80, 0.3, 5000
    )
    recognizer = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")
    return detector, recognizer


def match_embedding(
    query_vector: np.ndarray,
    employees: list[EmployeeTemplate],
    threshold: float,
    min_margin: float,
    top_k: int,
):
    """Per-employee top-K mean, consistent with face_recognition_v2.py."""
    scores = []
    for employee in employees:
        similarities = employee.vectors @ query_vector
        k = min(max(1, int(top_k)), len(similarities))
        score = float(np.mean(np.sort(similarities)[-k:]))
        scores.append((score, employee))
    scores.sort(key=lambda item: item[0], reverse=True)
    best_score, best_employee = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else -1.0
    margin = best_score - second_score if len(scores) > 1 else 1.0
    if best_score >= threshold and margin >= min_margin:
        return best_employee, best_score, second_score, margin
    return None, best_score, second_score, margin


@dataclass
class IdentityState:
    employee_id: Optional[str] = None
    full_name: Optional[str] = None
    score: float = 0.0
    status: str = "PENDING"  # PENDING / UNKNOWN / KNOWN
    last_track_seen: float = 0.0
    last_face_seen: float = 0.0
    pending_employee_id: Optional[str] = None
    pending_name: Optional[str] = None
    pending_hits: int = 0


class IdentityManager:
    """Cache a recognized identity on each ByteTrack ID.

    Once recognized, identity survives temporary profile/occlusion frames. A
    different identity must be seen repeatedly before replacing a cached one.
    """
    def __init__(self):
        self.states: dict[int, IdentityState] = {}

    def touch_track(self, track_id: int, now: float) -> None:
        state = self.states.setdefault(track_id, IdentityState())
        state.last_track_seen = now

    def observe_face(
        self,
        track_id: int,
        employee: Optional[EmployeeTemplate],
        score: float,
        now: float,
    ) -> None:
        state = self.states.setdefault(track_id, IdentityState())
        state.last_track_seen = now
        state.last_face_seen = now
        state.score = float(score)

        if employee is None:
            if state.employee_id is None:
                state.status = "UNKNOWN"
            # Do not erase an already confirmed identity just because the face
            # turns sideways or is temporarily difficult to recognize.
            state.pending_employee_id = None
            state.pending_name = None
            state.pending_hits = 0
            return

        if state.employee_id is None or state.employee_id == employee.employee_id:
            state.employee_id = employee.employee_id
            state.full_name = employee.full_name
            state.status = "KNOWN"
            state.pending_employee_id = None
            state.pending_name = None
            state.pending_hits = 0
            return

        # A track should normally belong to one person. Require repeated
        # evidence before allowing a cached identity to switch.
        if state.pending_employee_id == employee.employee_id:
            state.pending_hits += 1
        else:
            state.pending_employee_id = employee.employee_id
            state.pending_name = employee.full_name
            state.pending_hits = 1
        if state.pending_hits >= IDENTITY_SWITCH_CONFIRMATIONS:
            state.employee_id = employee.employee_id
            state.full_name = employee.full_name
            state.status = "KNOWN"
            state.pending_employee_id = None
            state.pending_name = None
            state.pending_hits = 0

    def display(self, track_id: int) -> str:
        state = self.states.get(track_id)
        if state is None or state.status == "PENDING":
            return "DANG NHAN DIEN..."
        if state.status == "UNKNOWN" and state.employee_id is None:
            return "CHUA XAC DINH DANH TINH"
        return f"{state.employee_id} | {state.full_name}"

    def details(self, track_id: int) -> tuple[str, str]:
        state = self.states.get(track_id)
        if state is None or state.employee_id is None:
            return "UNKNOWN", "UNKNOWN"
        return state.employee_id, state.full_name or "UNKNOWN"

    def cleanup(self, now: float) -> None:
        stale = [
            track_id for track_id, state in self.states.items()
            if now - state.last_track_seen > TRACK_TIMEOUT
        ]
        for track_id in stale:
            del self.states[track_id]


def associate_face_to_track(face_row: np.ndarray, tracked_persons) -> Optional[int]:
    """Associate face center with one tracked Person bbox.

    When boxes overlap, choose the candidate whose upper-body center is closest
    to the face center. Face center must be inside the Person bbox.
    """
    fx, fy, fw, fh = face_row[:4].astype(float)
    face_cx = fx + fw / 2.0
    face_cy = fy + fh / 2.0
    candidates = []
    for track in tracked_persons:
        x1, y1, x2, y2 = map(float, track[:4])
        track_id = int(track[4])
        if x2 <= x1 or y2 <= y1:
            continue
        if x1 <= face_cx <= x2 and y1 <= face_cy <= y2:
            center_x = (x1 + x2) / 2.0
            upper_face_y = y1 + 0.25 * (y2 - y1)
            norm_dx = (face_cx - center_x) / max((x2 - x1) / 2.0, 1.0)
            norm_dy = (face_cy - upper_face_y) / max((y2 - y1) / 2.0, 1.0)
            distance = norm_dx * norm_dx + norm_dy * norm_dy
            candidates.append((distance, track_id))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def recognize_faces(
    frame: np.ndarray,
    tracked_persons,
    detector,
    recognizer,
    employees: list[EmployeeTemplate],
    identity_manager: IdentityManager,
    threshold: float,
    min_margin: float,
    top_k: int,
):
    """Detect faces, compute SFace features, match, and attach to ByteTrack IDs."""
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))
    _status, faces = detector.detect(frame)
    face_count = 0 if faces is None else len(faces)
    recognized_now = 0
    unknown_now = 0
    face_time_start = time.perf_counter()

    if faces is not None:
        for face in faces:
            x, y, w, h = face[:4].astype(int)
            confidence = float(face[14])
            if confidence < FACE_MIN_CONFIDENCE or w < FACE_MIN_WIDTH or h < FACE_MIN_HEIGHT:
                continue
            track_id = associate_face_to_track(face, tracked_persons)
            if track_id is None:
                continue
            try:
                aligned = recognizer.alignCrop(frame, face)
                raw_feature = recognizer.feature(aligned).reshape(-1).astype(np.float32)
                norm = float(np.linalg.norm(raw_feature))
                if raw_feature.size != 128 or not np.isfinite(raw_feature).all() or norm <= 1e-12:
                    identity_manager.observe_face(track_id, None, 0.0, time.perf_counter())
                    unknown_now += 1
                    continue
                query_vector = raw_feature / norm
                employee, score, _second, _margin = match_embedding(
                    query_vector, employees, threshold, min_margin, top_k
                )
                identity_manager.observe_face(track_id, employee, score, time.perf_counter())
                if employee is not None:
                    recognized_now += 1
                else:
                    unknown_now += 1
            except cv2.error as exc:
                print(f"[WARN] Loi xu ly khuon mat: {exc}")
                identity_manager.observe_face(track_id, None, 0.0, time.perf_counter())
                unknown_now += 1

    elapsed = time.perf_counter() - face_time_start
    face_fps = 1.0 / max(elapsed, 1e-6) if face_count else 0.0
    return face_count, recognized_now, unknown_now, face_fps, faces


# ============================================================
# 4. PPE DETECTION / BYTE TRACK HELPERS (preserved from V4)
# ============================================================

def get_person_color(track_id: int):
    return PERSON_COLOR_PALETTE[(int(track_id) - 1) % len(PERSON_COLOR_PALETTE)]


def draw_detection(image, box, label, color):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    text_y = y1 - 8 if y1 >= 18 else y1 + 20
    cv2.putText(image, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, color, 1, cv2.LINE_AA)


def draw_person_identity(image, person_box, identity_text):
    """Draw the employee identity only; tracker IDs stay internal."""
    x1, y1, x2, y2 = map(int, person_box)
    if identity_text.startswith("DANG NHAN DIEN"):
        color = (0, 200, 255)       # Orange: face has not been identified yet
    elif identity_text.startswith("CHUA XAC DINH"):
        color = COLOR_UNKNOWN      # Orange: no matching employee
    else:
        color = (0, 220, 0)        # Green: recognized employee
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    label_y = y1 - 9 if y1 >= 22 else y1 + 20
    cv2.putText(image, identity_text, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, color, 2, cv2.LINE_AA)


def get_ppe_group(class_id: int):
    mapping = {
        0: ("Mu", True),
        2: ("Mu", False),
        1: ("Khau trang", True),
        3: ("Khau trang", False),
        7: ("Ao bao ho", True),
        4: ("Ao bao ho", False),
    }
    return mapping.get(class_id)


def associate_ppe_to_person(person_box, ppe_box) -> bool:
    px1, py1, px2, py2 = person_box
    x1, y1, x2, y2 = ppe_box
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    return px1 <= center_x <= px2 and py1 <= center_y <= py2


def evaluate_person_ppe(person_box, ppe_detections):
    candidates = {name: [] for name in GROUPS}
    for detection in ppe_detections:
        group_info = get_ppe_group(detection["class_id"])
        if group_info is None:
            continue
        group_name, is_equipped = group_info
        if associate_ppe_to_person(person_box, detection["box"]):
            candidates[group_name].append({
                "is_equipped": is_equipped,
                "confidence": detection["confidence"],
            })
    statuses = {}
    for group_name, group_candidates in candidates.items():
        if not group_candidates:
            statuses[group_name] = None
        else:
            best = max(group_candidates, key=lambda item: item["confidence"])
            statuses[group_name] = best["is_equipped"]
    return statuses


def prepare_person_detections(result, frame_shape, person_conf: float = PERSON_CONF):
    """Keep the NumPy conversion used to avoid the user's earlier ByteTrack index error."""
    if result.boxes is None:
        return Boxes(np.empty((0, 6), dtype=np.float32), frame_shape)
    boxes_numpy = result.boxes.cpu().numpy()
    data = np.asarray(boxes_numpy.data, dtype=np.float32)
    if data.size == 0:
        data = np.empty((0, 6), dtype=np.float32)
    elif data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or data.shape[1] < 6:
        raise ValueError(f"result.boxes.data khong dung Nx6; shape={data.shape}")
    class_ids = data[:, 5].astype(np.int32)
    person_mask = (class_ids == PERSON_CLASS_ID) & (data[:, 4] >= float(person_conf))
    person_data = np.ascontiguousarray(data[person_mask, :6], dtype=np.float32)
    return Boxes(person_data, frame_shape)


# ============================================================
# 5. PPE TEMPORAL SMOOTHING + ALERTS (preserved from V4)
# ============================================================

def majority(values, previous):
    observed = [v for v in values if v is not None]
    if not observed:
        return None
    counts = Counter(observed)
    if counts[True] == counts[False]:
        return previous
    return counts[True] > counts[False]


class PersonState:
    def __init__(self, now: float):
        # Keep only real model observations; a missing detection is not a vote.
        self.history = {g: deque(maxlen=SMOOTH_FRAMES) for g in GROUPS}
        self.last_observed_at = {g: None for g in GROUPS}
        self.smoothed = {g: None for g in GROUPS}
        self.violation_since = {g: None for g in GROUPS}
        self.last_seen = now
        self.last_alert = None
        self.last_printed = None
        self.alerting = False

    def update(self, statuses, now: float):
        self.last_seen = now
        for group in GROUPS:
            observation = statuses[group]
            if observation is not None:
                self.history[group].append(observation)
                self.last_observed_at[group] = now
                voted = majority(self.history[group], self.smoothed[group])
                if voted is not None:
                    self.smoothed[group] = voted
            else:
                last_observed = self.last_observed_at[group]
                if last_observed is None or now - last_observed > PPE_UNKNOWN_HOLD_SECONDS:
                    # No fresh evidence for too long: report unknown and stop
                    # counting a continuous violation without visual evidence.
                    self.smoothed[group] = None
                    self.violation_since[group] = None
                    self.history[group].clear()
                    self.last_observed_at[group] = None

            if self.smoothed[group] is False:
                if self.violation_since[group] is None:
                    self.violation_since[group] = now
            elif self.smoothed[group] is True:
                self.violation_since[group] = None

    def snapshot(self):
        return tuple(self.smoothed[g] for g in GROUPS)

    def confirmed_violations(self, now: float):
        return [
            g for g in ALERT_GROUPS
            if self.violation_since[g] is not None
            and now - self.violation_since[g] >= VIOLATION_SECONDS
        ]

    def any_violation_running(self):
        return any(self.violation_since[g] is not None for g in ALERT_GROUPS)

    def color(self):
        if self.alerting:
            return COLOR_PERSON_ALERT
        if all(self.smoothed[g] is True for g in GROUPS):
            return COLOR_PERSON_OK
        return COLOR_PERSON


class AlertManager:
    def __init__(self):
        self.states: dict[int, PersonState] = {}
        ALERT_DIR.mkdir(parents=True, exist_ok=True)
        if not ALERT_LOG_FILE.exists():
            with ALERT_LOG_FILE.open("w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow([
                    "thoi_gian", "employee_id", "full_name", "vi_pham", "anh",
                ])

    def update(self, track_id: int, statuses, now: float) -> PersonState:
        state = self.states.get(track_id)
        if state is None:
            state = PersonState(now)
            self.states[track_id] = state
        state.update(statuses, now)
        return state

    def check_alert(self, track_id: int, state: PersonState, now: float):
        violations = state.confirmed_violations(now)
        if violations:
            in_cooldown = (
                state.last_alert is not None
                and now - state.last_alert < ALERT_COOLDOWN
            )
            state.alerting = True
            if not in_cooldown:
                return violations
            return []
        if state.alerting and not state.any_violation_running():
            state.alerting = False
            state.last_alert = None
            print(f"\n[{datetime.now():%H:%M:%S}] Trang thai PPE da duoc khac phuc.", flush=True)
        return []

    def fire(self, image, track_id: int, violations, identity_manager: IdentityManager):
        state = self.states[track_id]
        state.last_alert = time.perf_counter()
        stamp = datetime.now()
        employee_id, full_name = identity_manager.details(track_id)
        safe_employee = employee_id.replace("/", "_").replace("\\", "_")
        filename = f"{stamp:%Y%m%d_%H%M%S}_{safe_employee}.jpg"
        image_path = ALERT_DIR / filename
        cv2.imwrite(str(image_path), image)
        violation_text = ", ".join(violations)
        with ALERT_LOG_FILE.open("a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow([
                f"{stamp:%Y-%m-%d %H:%M:%S}",
                employee_id, full_name, violation_text, str(image_path),
            ])
        print("\n" + "!" * 58)
        print(f"CANH BAO [{stamp:%H:%M:%S}]")
        print(f"Nhan vien: {full_name} ({employee_id})")
        print(f"Vi pham: {violation_text}")
        print(f"Anh bang chung: {image_path}")
        print("!" * 58, flush=True)

    def cleanup(self, now: float):
        stale = [
            track_id for track_id, state in self.states.items()
            if now - state.last_seen > TRACK_TIMEOUT
        ]
        for track_id in stale:
            del self.states[track_id]


def print_person_status(track_id: int, statuses, identity_text: str):
    status_map = {True: "DAT", False: "VI PHAM", None: "CHUA XAC DINH"}
    print("\n" + "=" * 54)
    print(f"NHAN VIEN: {identity_text}")
    print("-" * 54)
    for group_name in GROUPS:
        print(f"{group_name:<20}: {status_map[statuses[group_name]]}")
    if any(status is False for status in statuses.values()):
        overall = "KHONG DAT"
    elif any(status is None for status in statuses.values()):
        overall = "CHUA DAY DU DU LIEU"
    else:
        overall = "DAT DAY DU PPE"
    print(f"TRANG THAI CHUNG: {overall}")
    print("=" * 54, flush=True)


def draw_fps_panel(image, inference_fps: float, processing_fps: float,
                   face_count: int, registered_count: int):
    panel_width, panel_height = 275, 68
    height, width = image.shape[:2]
    x_left = max(5, width - panel_width - 5)
    y_top = 5
    overlay = image.copy()
    cv2.rectangle(overlay, (x_left, y_top),
                  (x_left + panel_width, y_top + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.50, image, 0.50, 0, image)
    cv2.putText(image, f"YOLO FPS: {inference_fps:.1f} | Total FPS: {processing_fps:.1f}",
                (x_left + 7, y_top + 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.43, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(image, f"Faces: {face_count} | Employees: {registered_count}",
                (x_left + 7, y_top + 42), cv2.FONT_HERSHEY_SIMPLEX,
                0.43, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "Face matching runs every N frames",
                (x_left + 7, y_top + 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.34, (0, 220, 255), 1, cv2.LINE_AA)


# ============================================================
# 6. MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Integrated PPE + Employee Face Recognition")
    parser.add_argument("--source", default=None,
                        help="Camera index or video/RTSP path. Default: EZVIZ RTSP (asks password securely).")
    parser.add_argument("--threshold", type=float, default=DEFAULT_FACE_THRESHOLD,
                        help=f"SFace cosine threshold (default {DEFAULT_FACE_THRESHOLD})")
    parser.add_argument("--min-margin", type=float, default=DEFAULT_FACE_MIN_MARGIN,
                        help=f"Required lead over second employee (default {DEFAULT_FACE_MIN_MARGIN})")
    parser.add_argument("--top-k", type=int, default=DEFAULT_FACE_TOP_K,
                        help=f"Average top K samples for each employee (default {DEFAULT_FACE_TOP_K})")
    parser.add_argument("--face-every", type=int, default=FACE_DETECT_EVERY_N_FRAMES,
                        help=f"Run face detect/recognition every N frames (default {FACE_DETECT_EVERY_N_FRAMES})")
    parser.add_argument("--person-conf", type=float, default=PERSON_CONF,
                        help=f"Minimum Person confidence for ByteTrack (default {PERSON_CONF})")
    parser.add_argument("--ppe-conf", type=float, default=PPE_CONF,
                        help=f"Minimum PPE confidence for display/association (default {PPE_CONF})")
    args = parser.parse_args()
    if not -1.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between -1 and 1")
    if not 0.0 <= args.min_margin <= 2.0:
        parser.error("--min-margin must be between 0 and 2")
    if not 1 <= args.top_k <= 10:
        parser.error("--top-k must be between 1 and 10")
    if args.face_every < 1:
        parser.error("--face-every must be >= 1")
    if not 0.0 <= args.person_conf <= 1.0:
        parser.error("--person-conf must be between 0 and 1")
    if not 0.0 <= args.ppe_conf <= 1.0:
        parser.error("--ppe-conf must be between 0 and 1")
    return args


def main() -> int:
    args = parse_args()
    print("\n=== PPE DETECTION + FACE RECOGNITION ===")
    print(f"YOLO model: {YOLO_MODEL_PATH}")
    print(f"Face database: {FACE_DB_PATH}")
    print(f"Face threshold={args.threshold:.3f} | margin={args.min_margin:.3f} | top-k={args.top_k}")
    print(f"Face recognition interval: every {args.face_every} frame(s)")
    print(f"YOLO inference conf={YOLO_INFER_CONF:.2f} | Person conf={args.person_conf:.2f} | PPE conf={args.ppe_conf:.2f}")
    print(f"PPE missing-detection hold: {PPE_UNKNOWN_HOLD_SECONDS:.2f}s")

    for required_path in (YOLO_MODEL_PATH, YUNET_PATH, SFACE_PATH):
        if not required_path.is_file():
            print(f"[ERROR] Khong tim thay file: {required_path}")
            return 1

    try:
        employees = load_employee_templates(FACE_DB_PATH)
        print(f"[OK] Da nap {len(employees)} nhan vien, "
              f"{sum(e.sample_count for e in employees)} mau khuon mat V2.")
        for employee in employees:
            print(f"  - {employee.employee_id}: {employee.full_name} ({employee.sample_count} mau)")
        face_detector, face_recognizer = load_face_models()
        print("[OK] YuNet loaded")
        print("[OK] SFace loaded")
    except Exception as exc:
        print(f"[ERROR] Loi khoi tao Face Recognition: {exc}")
        return 1

    try:
        source = get_camera_source(args.source)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    reader: Optional[RTSPReader] = None
    try:
        print("\n[INFO] Dang ket noi camera...")
        reader = RTSPReader(source)
        print("[OK] Camera connected")

        print("\n[INFO] Dang khoi tao YOLOv8n PPE...")
        model = YOLO(str(YOLO_MODEL_PATH), task="detect")
        print("[OK] YOLO loaded. Cac class:")
        for class_id, class_name in model.names.items():
            print(f"  {class_id}: {class_name}")

        print("\n[INFO] Dang khoi tao ByteTrack...")
        tracker_config = yaml.safe_load(
            Path(check_yaml("bytetrack.yaml")).read_text(encoding="utf-8")
        )
        tracker = BYTETracker(IterableSimpleNamespace(**tracker_config))
        print("[OK] ByteTrack loaded; only Person class ID=5 is tracked.")

        alerts = AlertManager()
        identity_manager = IdentityManager()

        ok, first_frame = False, None
        wait_start = time.time()
        while time.time() - wait_start < 10.0:
            ok, first_frame = reader.read()
            if ok and first_frame is not None:
                break
            time.sleep(0.02)
        if first_frame is None:
            raise RuntimeError("Camera connected but no video frame arrived within 10 seconds.")

        print(f"[INFO] Frame size: {first_frame.shape[1]} x {first_frame.shape[0]}")
        print("[INFO] YOLO warm-up...")
        for index in range(WARMUP_FRAMES):
            model.predict(first_frame, imgsz=IMGSZ, conf=YOLO_INFER_CONF, verbose=False)
            print(f"  Warm-up {index + 1}/{WARMUP_FRAMES}")
        print("[OK] Warm-up completed")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        frame_count = 0
        stats_count = 0
        total_yolo_time = 0.0
        total_process_time = 0.0
        min_yolo_time = float("inf")
        max_yolo_time = 0.0
        min_process_time = float("inf")
        max_process_time = 0.0
        last_face_count = 0
        last_face_known_count = 0
        last_face_unknown_count = 0
        last_face_fps = 0.0
        last_detected_faces = None

        print("\n[RUNNING] PPE + ByteTrack + Face Recognition + Alerts")
        print("Nhan Q/ESC de thoat. Bao cao thong ke moi 100 frame.\n")

        while True:
            loop_start = time.perf_counter()
            ok, frame = reader.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            # YOLO inference is kept at the same settings as detect_v4_fixed.py.
            yolo_start = time.perf_counter()
            results = model.predict(frame, imgsz=IMGSZ, conf=YOLO_INFER_CONF, verbose=False)
            yolo_time = time.perf_counter() - yolo_start
            result = results[0]

            # ByteTrack receives only the Person class, converted to NumPy first.
            person_detections = prepare_person_detections(result, frame.shape[:2], args.person_conf)
            tracked_persons = tracker.update(person_detections, frame)
            now = time.perf_counter()
            for track in tracked_persons:
                track_id = int(track[4])
                identity_manager.touch_track(track_id, now)

            # Run YuNet/SFace periodically to limit additional CPU load.
            if frame_count % args.face_every == 0:
                (last_face_count, last_face_known_count,
                 last_face_unknown_count, last_face_fps,
                 last_detected_faces) = recognize_faces(
                    frame, tracked_persons, face_detector, face_recognizer,
                    employees, identity_manager, args.threshold,
                    args.min_margin, args.top_k,
                )

            annotated = frame.copy()
            ppe_detections = []

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    if class_id == PERSON_CLASS_ID:
                        continue
                    box_coords = box.xyxy[0].tolist()
                    confidence = float(box.conf[0].item())
                    class_name = model.names[class_id]
                    is_ppe = get_ppe_group(class_id) is not None
                    min_class_conf = args.ppe_conf if is_ppe else OTHER_CONF
                    if confidence < min_class_conf:
                        continue
                    label = f"{class_name} {confidence:.2f}"
                    if class_id in {0, 1, 7}:
                        box_color = COLOR_PPE_OK
                    elif class_id in {2, 3, 4}:
                        box_color = COLOR_PPE_VIOLATION
                    else:
                        box_color = COLOR_OTHER
                    draw_detection(annotated, box_coords, label, box_color)
                    if is_ppe:
                        ppe_detections.append({
                            "class_id": class_id,
                            "class_name": class_name,
                            "box": box_coords,
                            "confidence": confidence,
                        })

            pending_alerts = []
            for track in tracked_persons:
                person_box = list(map(float, track[:4]))
                track_id = int(track[4])
                confidence = float(track[5])
                statuses = evaluate_person_ppe(person_box, ppe_detections)
                state = alerts.update(track_id, statuses, now)
                identity_text = identity_manager.display(track_id)

                if state.last_printed != (state.snapshot(), identity_text):
                    print_person_status(track_id, state.smoothed, identity_text)
                    state.last_printed = (state.snapshot(), identity_text)

                violations = alerts.check_alert(track_id, state, now)
                if violations:
                    pending_alerts.append((track_id, violations))

                draw_person_identity(annotated, person_box, identity_text)

            # Draw current face boxes only on frames where YuNet was executed.
            # The identity itself remains cached against the ByteTrack ID between runs.
            if frame_count % args.face_every == 0 and last_detected_faces is not None:
                for face in last_detected_faces:
                    x, y, w, h = face[:4].astype(int)
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), COLOR_FACE, 1)

            for track_id, violations in pending_alerts:
                alerts.fire(annotated, track_id, violations, identity_manager)

            alerts.cleanup(now)
            identity_manager.cleanup(now)

            process_time = time.perf_counter() - loop_start
            yolo_fps = 1.0 / max(yolo_time, 1e-6)
            total_fps = 1.0 / max(process_time, 1e-6)
            draw_fps_panel(
                annotated, yolo_fps, total_fps,
                last_face_count, len(employees),
            )
            cv2.imshow(WINDOW_NAME, annotated)

            stats_count += 1
            total_yolo_time += yolo_time
            total_process_time += process_time
            min_yolo_time = min(min_yolo_time, yolo_time)
            max_yolo_time = max(max_yolo_time, yolo_time)
            min_process_time = min(min_process_time, process_time)
            max_process_time = max(max_process_time, process_time)

            if stats_count >= REPORT_INTERVAL:
                avg_yolo = total_yolo_time / stats_count
                avg_process = total_process_time / stats_count
                print("\n" + "=" * 54)
                print(f"THONG KE {stats_count} FRAME")
                print(f"YOLO inference trung binh: {avg_yolo:.4f}s ({1.0 / avg_yolo:.2f} FPS)")
                print(f"Tong xu ly trung binh:    {avg_process:.4f}s ({1.0 / avg_process:.2f} FPS)")
                print(f"YOLO min/max: {min_yolo_time:.4f}s / {max_yolo_time:.4f}s")
                print(f"Total min/max: {min_process_time:.4f}s / {max_process_time:.4f}s")
                print(f"Face pass gan nhat: {last_face_fps:.2f} FPS, faces={last_face_count}, "
                      f"known={last_face_known_count}, unknown={last_face_unknown_count}")
                print("=" * 54 + "\n")
                stats_count = 0
                total_yolo_time = 0.0
                total_process_time = 0.0
                min_yolo_time = float("inf")
                max_yolo_time = 0.0
                min_process_time = float("inf")
                max_process_time = 0.0

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("\n[INFO] Nguoi dung dung chuong trinh.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Nguoi dung dung chuong trinh.")
    except Exception as exc:
        print(f"\n[ERROR] Loi chuong trinh: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if reader is not None:
            reader.release()
        cv2.destroyAllWindows()
        print("[INFO] Da dong camera va giai phong tai nguyen.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
