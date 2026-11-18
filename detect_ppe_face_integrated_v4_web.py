"""Integrated PPE detection, ByteTrack person tracking, face recognition and alerts.

Chạy từ thư mục project với venv đã kích hoạt:
    python detect_ppe_face_integrated_v4_web.py

Tuỳ chọn:
    python detect_ppe_face_integrated_v4_web.py --threshold 0.45 --min-margin 0.05
    python detect_ppe_face_integrated_v4_web.py --source 0
    python detect_ppe_face_integrated_v4_web.py --face-every 5
    python detect_ppe_face_integrated_v4_web.py --no-face

Controls:
    Q / ESC = quit
"""
from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

# WEB integration
from web_server import (
    start_server, update_frame, set_enroll_context, is_enrolling, update_live_tracks,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
YOLO_MODEL_PATH = ROOT / "best.onnx"
YUNET_PATH = ROOT / "models" / "face_recognition" / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = ROOT / "models" / "face_recognition" / "face_recognition_sface_2021dec.onnx"
FACE_DB_PATH = ROOT / "data" / "face_database.sqlite3"

CAMERA_RTSP_URL = (
    "rtsp://admin:Longtran2005%40@192.168.2.79:554"
    "/h264/ch1/sub/av_stream"
)

IMGSZ = 416
YOLO_INFER_CONF = 0.30
PERSON_CONF = 0.70
PPE_CONF = 0.45
OTHER_CONF = 0.50
PERSON_CLASS_ID = 5
WARMUP_FRAMES = 5
REPORT_INTERVAL = 100
WINDOW_NAME = "PPE + Face Recognition | YOLO + ByteTrack"

DEFAULT_FACE_THRESHOLD = 0.363
DEFAULT_FACE_MIN_MARGIN = 0.05
DEFAULT_FACE_TOP_K = 2
FACE_DETECT_EVERY_N_FRAMES = 4
FACE_MIN_CONFIDENCE = 0.75
FACE_MIN_WIDTH = 55
FACE_MIN_HEIGHT = 50
IDENTITY_SWITCH_CONFIRMATIONS = 2

SMOOTH_FRAMES = 5
PPE_UNKNOWN_HOLD_SECONDS = 0.75
FACE_BOX_TTL_SECONDS = 0.50
HEAD_ROI_TOP_RATIO = -0.05
HEAD_ROI_BOTTOM_RATIO = 0.32
TORSO_ROI_TOP_RATIO = 0.25
TORSO_ROI_BOTTOM_RATIO = 0.95
MASK_ROI_TOP_RATIO = 0.38
MASK_ROI_BOTTOM_RATIO = 1.08
VIOLATION_SECONDS = 2.0

# Cập nhật thời gian chờ và timeout
ALERT_COOLDOWN = 60.0   # 1 phút
TRACK_TIMEOUT = 30.0    # 30s

GROUPS = ("Mu", "Khau trang", "Ao bao ho")
ALERT_GROUPS = ("Mu", "Khau trang", "Ao bao ho")
ALERT_DIR = ROOT / "alerts"
ALERT_LOG_FILE = ALERT_DIR / "alert_log_integrated_v3.csv"

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
    if cli_source is not None:
        return int(cli_source) if cli_source.isdigit() else cli_source
    return CAMERA_RTSP_URL


# ============================================================
# 2. CAMERA READER
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
                "Không mở được camera/video. Kiểm tra RTSP URL, IP, mạng hoặc --source."
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
# 3. FACE DATABASE + MIGRATION + YUNET/SFACE
# ============================================================

@dataclass
class EmployeeTemplate:
    employee_id: str
    full_name: str
    vectors: np.ndarray
    sample_count: int


def _migrate_face_database(db_path: Path) -> None:
    """Tự vá schema cho DB cũ. An toàn: chỉ ALTER khi thiếu cột."""
    with sqlite3.connect(str(db_path)) as conn:
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        if "employees" not in tables or "face_embeddings" not in tables:
            return

        emp_cols = {r[1] for r in conn.execute("PRAGMA table_info(employees)").fetchall()}
        if "is_active" not in emp_cols:
            conn.execute(
                "ALTER TABLE employees ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1"
            )
            print("[MIGRATION] Đã thêm cột employees.is_active (mặc định = 1).")

        fe_cols = {r[1] for r in conn.execute("PRAGMA table_info(face_embeddings)").fetchall()}
        if "pose_label" not in fe_cols:
            conn.execute(
                "ALTER TABLE face_embeddings ADD COLUMN pose_label TEXT NOT NULL DEFAULT 'unknown'"
            )
            print("[MIGRATION] Đã thêm cột face_embeddings.pose_label (mặc định = 'unknown').")

        conn.commit()


def load_employee_templates(db_path: Path) -> list[EmployeeTemplate]:
    if not db_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy database khuôn mặt: {db_path}\n"
            "Hãy đăng ký ít nhất một nhân viên qua dashboard hoặc face_enrollment_v2.py."
        )

    _migrate_face_database(db_path)

    with sqlite3.connect(str(db_path)) as conn:
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        if not {"employees", "face_embeddings"}.issubset(tables):
            raise RuntimeError(
                "Database thiếu bảng employees/face_embeddings. "
                "Hãy đăng ký nhân viên qua face_enrollment_v2.py."
            )

        rows = conn.execute(
            """SELECT e.employee_id, e.full_name, fe.embedding, fe.dimension, fe.pose_label
               FROM employees e
               INNER JOIN face_embeddings fe ON fe.employee_id = e.employee_id
               WHERE COALESCE(e.is_active, 1) = 1
                 AND LOWER(COALESCE(fe.pose_label, 'unknown')) != 'unknown'
               ORDER BY e.employee_id, fe.sample_index"""
        ).fetchall()

    grouped: dict[str, dict[str, object]] = {}
    for employee_id, full_name, blob, dimension, pose_label in rows:
        if int(dimension) != 128:
            print(f"[WARN] Bỏ qua {employee_id}: embedding dimension={dimension}.")
            continue
        vector = np.frombuffer(blob, dtype=np.float32)
        if vector.size != 128 or not np.isfinite(vector).all():
            print(f"[WARN] Bỏ qua embedding không hợp lệ của {employee_id}.")
            continue
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            continue
        vector = (vector / norm).astype(np.float32)
        if employee_id not in grouped:
            grouped[employee_id] = {"full_name": str(full_name), "vectors": []}
        grouped[employee_id]["vectors"].append(vector)

    employees: list[EmployeeTemplate] = []
    for emp_id, data in grouped.items():
        matrix = np.vstack(data["vectors"]).astype(np.float32)
        employees.append(EmployeeTemplate(
            employee_id=str(emp_id),
            full_name=str(data["full_name"]),
            vectors=matrix,
            sample_count=len(matrix),
        ))

    if not employees:
        raise RuntimeError("Chưa có embedding hợp lệ nào trong database.")
    return employees


def load_face_models():
    missing = [str(p) for p in (YUNET_PATH, SFACE_PATH) if not p.is_file()]
    if missing:
        raise FileNotFoundError("Thiếu model ONNX:\n  " + "\n  ".join(missing))
    detector = cv2.FaceDetectorYN.create(
        str(YUNET_PATH), "", (320, 320), 0.80, 0.3, 5000
    )
    recognizer = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")
    return detector, recognizer


def match_embedding(query_vector, employees, threshold, min_margin, top_k):
    scores = []
    for employee in employees:
        sims = employee.vectors @ query_vector
        k = min(max(1, int(top_k)), len(sims))
        score = float(np.mean(np.sort(sims)[-k:]))
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
    status: str = "PENDING"
    last_track_seen: float = 0.0
    last_face_seen: float = 0.0
    pending_employee_id: Optional[str] = None
    pending_name: Optional[str] = None
    pending_hits: int = 0


class IdentityManager:
    def __init__(self):
        self.states: dict[int, IdentityState] = {}

    def touch_track(self, track_id: int, now: float) -> None:
        state = self.states.setdefault(track_id, IdentityState())
        state.last_track_seen = now

    def observe_face(self, track_id, employee, score, now):
        state = self.states.setdefault(track_id, IdentityState())
        state.last_track_seen = now
        state.last_face_seen = now
        state.score = float(score)

        if employee is None:
            if state.employee_id is None:
                state.status = "UNKNOWN"
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
            return "ĐANG NHẬN DIỆN..."
        if state.status == "UNKNOWN" and state.employee_id is None:
            return "CHƯA XÁC ĐỊNH DANH TÍNH"
        return f"{state.employee_id} | {state.full_name}"

    def details(self, track_id: int) -> tuple[str, str]:
        state = self.states.get(track_id)
        if state is None or state.employee_id is None:
            return "UNKNOWN", "UNKNOWN"
        return state.employee_id, state.full_name or "UNKNOWN"

    def cleanup(self, now: float) -> None:
        stale = [
            tid for tid, st in self.states.items()
            if now - st.last_track_seen > TRACK_TIMEOUT
        ]
        for tid in stale:
            del self.states[tid]


def associate_face_to_track(face_row, tracked_persons) -> Optional[int]:
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


def recognize_faces(frame, tracked_persons, detector, recognizer, employees,
                    identity_manager, threshold, min_margin, top_k):
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
                raw = recognizer.feature(aligned).reshape(-1).astype(np.float32)
                norm = float(np.linalg.norm(raw))
                if raw.size != 128 or not np.isfinite(raw).all() or norm <= 1e-12:
                    identity_manager.observe_face(track_id, None, 0.0, time.perf_counter())
                    unknown_now += 1
                    continue
                qv = raw / norm
                employee, score, _s, _m = match_embedding(
                    qv, employees, threshold, min_margin, top_k
                )
                identity_manager.observe_face(track_id, employee, score, time.perf_counter())
                if employee is not None:
                    recognized_now += 1
                else:
                    unknown_now += 1
            except cv2.error:
                identity_manager.observe_face(track_id, None, 0.0, time.perf_counter())
                unknown_now += 1

    elapsed = time.perf_counter() - face_time_start
    face_fps = 1.0 / max(elapsed, 1e-6) if face_count else 0.0
    return face_count, recognized_now, unknown_now, face_fps, faces


# ============================================================
# 4. PPE helpers
# ============================================================

def draw_detection(image, box, label, color):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    text_y = y1 - 8 if y1 >= 18 else y1 + 20
    cv2.putText(image, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, color, 1, cv2.LINE_AA)


def draw_person_identity(image, person_box, identity_text):
    x1, y1, x2, y2 = map(int, person_box)
    if identity_text.startswith("ĐANG NHẬN DIỆN"):
        color = (0, 200, 255)
    elif identity_text.startswith("CHƯA XÁC ĐỊNH"):
        color = COLOR_UNKNOWN
    else:
        color = (0, 220, 0)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    label_y = y1 - 9 if y1 >= 22 else y1 + 20
    cv2.putText(image, identity_text, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, color, 2, cv2.LINE_AA)


def get_ppe_group(class_id: int):
    mapping = {
        0: ("Mu", True), 2: ("Mu", False),
        1: ("Khau trang", True), 3: ("Khau trang", False),
        7: ("Ao bao ho", True), 4: ("Ao bao ho", False),
    }
    return mapping.get(class_id)


def point_inside_box(px: float, py: float, box) -> bool:
    x1, y1, x2, y2 = map(float, box)
    return x1 <= px <= x2 and y1 <= py <= y2


def get_face_box_for_track(track_id, face_rows, tracked_persons):
    if face_rows is None:
        return None
    best = None
    best_area = -1.0
    for face in face_rows:
        if associate_face_to_track(face, tracked_persons) != track_id:
            continue
        x, y, w, h = map(float, face[:4])
        if w <= 0 or h <= 0:
            continue
        area = w * h
        if area > best_area:
            best = (x, y, x + w, y + h)
            best_area = area
    return best


def ppe_detection_matches_region(detection, person_box, face_box=None) -> bool:
    px1, py1, px2, py2 = map(float, person_box)
    pw = max(1.0, px2 - px1)
    ph = max(1.0, py2 - py1)
    x1, y1, x2, y2 = map(float, detection["box"])
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    if not point_inside_box(cx, cy, (px1, py1, px2, py2)):
        return False

    cid = int(detection["class_id"])

    if cid in (1, 3):
        if face_box is None:
            return False
        fx1, fy1, fx2, fy2 = map(float, face_box)
        fw = max(1.0, fx2 - fx1)
        fh = max(1.0, fy2 - fy1)
        roi = (
            fx1 - 0.10 * fw, fy1 + MASK_ROI_TOP_RATIO * fh,
            fx2 + 0.10 * fw, fy1 + MASK_ROI_BOTTOM_RATIO * fh,
        )
        return point_inside_box(cx, cy, roi)

    if cid in (0, 2):
        roi = (
            px1 + 0.08 * pw, py1 + HEAD_ROI_TOP_RATIO * ph,
            px2 - 0.08 * pw, py1 + HEAD_ROI_BOTTOM_RATIO * ph,
        )
        return point_inside_box(cx, cy, roi)

    if cid in (4, 7):
        roi = (
            px1 + 0.05 * pw, py1 + TORSO_ROI_TOP_RATIO * ph,
            px2 - 0.05 * pw, py1 + TORSO_ROI_BOTTOM_RATIO * ph,
        )
        return point_inside_box(cx, cy, roi)

    return False


def evaluate_person_ppe(person_box, ppe_detections, face_box=None):
    candidates = {name: [] for name in GROUPS}
    valid_detections = []
    for detection in ppe_detections:
        info = get_ppe_group(detection["class_id"])
        if info is None:
            continue
        if not ppe_detection_matches_region(detection, person_box, face_box):
            continue
        gname, equipped = info
        candidates[gname].append({"is_equipped": equipped, "confidence": detection["confidence"]})
        valid_detections.append(detection)
    statuses = {}
    for gname, lst in candidates.items():
        if not lst:
            statuses[gname] = None
        else:
            best = max(lst, key=lambda item: item["confidence"])
            statuses[gname] = best["is_equipped"]
    return statuses, valid_detections


def prepare_person_detections(result, frame_shape, person_conf: float = PERSON_CONF):
    if result.boxes is None:
        return Boxes(np.empty((0, 6), dtype=np.float32), frame_shape)
    boxes_np = result.boxes.cpu().numpy()
    data = np.asarray(boxes_np.data, dtype=np.float32)
    if data.size == 0:
        data = np.empty((0, 6), dtype=np.float32)
    elif data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or data.shape[1] < 6:
        raise ValueError(f"boxes không đúng Nx6; shape={data.shape}")
    cids = data[:, 5].astype(np.int32)
    mask = (cids == PERSON_CLASS_ID) & (data[:, 4] >= float(person_conf))
    pdata = np.ascontiguousarray(data[mask, :6], dtype=np.float32)
    return Boxes(pdata, frame_shape)


# ============================================================
# 5. PPE temporal smoothing + alerts
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
            obs = statuses[group]
            if obs is not None:
                self.history[group].append(obs)
                self.last_observed_at[group] = now
                voted = majority(self.history[group], self.smoothed[group])
                if voted is not None:
                    self.smoothed[group] = voted
            else:
                last = self.last_observed_at[group]
                if last is None or now - last > PPE_UNKNOWN_HOLD_SECONDS:
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


class AlertManager:
    """Theo dõi trạng thái PPE theo NHÂN VIÊN (employee_id), không theo track_id.
    
    Khi ByteTrack cấp ID mới do mất dấu, nhưng khuôn mặt vẫn nhận ra cùng 1
    nhân viên -> state vẫn được kế thừa (cooldown chung, violation_since chung).
    Điều này tránh alert spam cho cùng 1 người và gộp đủ vi phạm vào 1 lần báo.
    Với UNKNOWN, fallback dùng track_id (không thể biết là cùng người).
    """

    def __init__(self):
        self.states: dict[str, PersonState] = {}   # key: "EMP:xxx" hoặc "TRK:yyy"
        ALERT_DIR.mkdir(parents=True, exist_ok=True)
        if not ALERT_LOG_FILE.exists():
            with ALERT_LOG_FILE.open("w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow(
                    ["thoi_gian", "employee_id", "full_name", "vi_pham", "anh"]
                )

    @staticmethod
    def _key(track_id: int, identity_manager) -> str:
        eid, _ = identity_manager.details(track_id)
        return f"EMP:{eid}" if eid != "UNKNOWN" else f"TRK:{track_id}"

    def update(self, track_id, statuses, now, identity_manager):
        key = self._key(track_id, identity_manager)
        state = self.states.get(key)
        if state is None:
            state = PersonState(now)
            self.states[key] = state
        state.update(statuses, now)
        return state

    def check_alert(self, key, state, now):
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
        return []

    def fire(self, image, track_id, violations, identity_manager):
        key = self._key(track_id, identity_manager)
        state = self.states.get(key)
        if state is None:
            return
        state.last_alert = time.perf_counter()
        stamp = datetime.now()
        employee_id, full_name = identity_manager.details(track_id)
        safe = employee_id.replace("/", "_").replace("\\", "_")
        filename = f"{stamp:%Y%m%d_%H%M%S}_{safe}.jpg"
        image_path = ALERT_DIR / filename
        cv2.imwrite(str(image_path), image)
        with ALERT_LOG_FILE.open("a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow([
                f"{stamp:%Y-%m-%d %H:%M:%S}",
                employee_id, full_name, ", ".join(violations), str(image_path),
            ])
        print(
            f"\n[CẢNH BÁO {stamp:%H:%M:%S}] {full_name} ({employee_id}) "
            f"- {', '.join(violations)} - {image_path.name}",
            flush=True,
        )

    def cleanup(self, now):
        stale = [
            k for k, st in self.states.items()
            if now - st.last_seen > TRACK_TIMEOUT
        ]
        for k in stale:
            del self.states[k]


def print_person_status(track_id, statuses, identity_text):
    status_map = {True: "ĐẠT", False: "VI PHẠM", None: "CHƯA XÁC ĐỊNH"}
    print("\n" + "=" * 54)
    print(f"NHÂN VIÊN: {identity_text}")
    print("-" * 54)
    for gname in GROUPS:
        print(f"{gname:<20}: {status_map[statuses[gname]]}")
    print("=" * 54, flush=True)


def draw_fps_panel(image, inference_fps, processing_fps, face_count, registered):
    pw, ph = 275, 68
    h, w = image.shape[:2]
    x = max(5, w - pw - 5)
    y = 5
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + pw, y + ph), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.50, image, 0.50, 0, image)
    cv2.putText(image, f"YOLO FPS: {inference_fps:.1f} | Total FPS: {processing_fps:.1f}",
                (x + 7, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(image, f"Faces: {face_count} | Employees: {registered}",
                (x + 7, y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "Face matching runs every N frames",
                (x + 7, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 220, 255), 1, cv2.LINE_AA)


# ============================================================
# 6. MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Integrated PPE + Employee Face Recognition")
    p.add_argument("--source", default=None)
    p.add_argument("--threshold", type=float, default=DEFAULT_FACE_THRESHOLD)
    p.add_argument("--min-margin", type=float, default=DEFAULT_FACE_MIN_MARGIN)
    p.add_argument("--top-k", type=int, default=DEFAULT_FACE_TOP_K)
    p.add_argument("--no-face", action="store_true")
    p.add_argument("--face-every", type=int, default=FACE_DETECT_EVERY_N_FRAMES)
    p.add_argument("--person-conf", type=float, default=PERSON_CONF)
    p.add_argument("--ppe-conf", type=float, default=PPE_CONF)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print("\n=== PPE + FACE RECOGNITION ===")
    print(f"YOLO: {YOLO_MODEL_PATH}")
    print(f"DB: {FACE_DB_PATH}")
    print(f"threshold={args.threshold:.3f} margin={args.min_margin:.3f} top-k={args.top_k}")
    print(f"face-every={args.face_every} | person-conf={args.person_conf:.2f} | ppe-conf={args.ppe_conf:.2f}")

    required = (YOLO_MODEL_PATH,) if args.no_face else (YOLO_MODEL_PATH, YUNET_PATH, SFACE_PATH)
    for p in required:
        if not p.is_file():
            print(f"[ERROR] Không tìm thấy: {p}")
            return 1

    employees, face_detector, face_recognizer = [], None, None
    if args.no_face:
        print("[INFO] --no-face: bỏ qua nhận diện khuôn mặt.")
    else:
        try:
            employees = load_employee_templates(FACE_DB_PATH)
            print(
                f"[OK] {len(employees)} nhân viên đang hoạt động, "
                f"{sum(e.sample_count for e in employees)} mẫu khuôn mặt."
            )
            for emp in employees:
                print(f"  - {emp.employee_id}: {emp.full_name} ({emp.sample_count} mẫu)")
            face_detector, face_recognizer = load_face_models()
            print("[OK] YuNet loaded")
            print("[OK] SFace loaded")
        except Exception as exc:
            print(f"[ERROR] Face init: {exc}")
            return 1

    try:
        source = get_camera_source(args.source)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    reader: Optional[RTSPReader] = None
    try:
        print("\n[INFO] Đang kết nối camera...")
        reader = RTSPReader(source)
        print("[OK] Camera kết nối.")

        print("\n[INFO] Khởi tạo YOLO...")
        model = YOLO(str(YOLO_MODEL_PATH), task="detect")
        print(f"[OK] YOLO loaded. Classes: {list(model.names.values())}")

        print("[INFO] Khởi tạo ByteTrack...")
        cfg = yaml.safe_load(Path(check_yaml("bytetrack.yaml")).read_text(encoding="utf-8"))
        # Tăng buffer để giữ ID ổn định hơn khi mất dấu ngắn (~3s)
        cfg["track_buffer"] = max(int(cfg.get("track_buffer", 30)), 90)
        # Nới lỏng ngưỡng match nhẹ để tái nhận track khi di chuyển nhanh
        cfg["match_thresh"] = min(float(cfg.get("match_thresh", 0.8)), 0.9)
        tracker = BYTETracker(IterableSimpleNamespace(**cfg))
        print(f"[OK] ByteTrack: track_buffer={cfg['track_buffer']}, match_thresh={cfg['match_thresh']}")

        start_server()
        set_enroll_context(
            reader.read, employees, lambda: load_employee_templates(FACE_DB_PATH),
        )
        alerts = AlertManager()
        identity_manager = IdentityManager()

        ok, first = False, None
        wait_start = time.time()
        while time.time() - wait_start < 10.0:
            ok, first = reader.read()
            if ok and first is not None:
                break
            time.sleep(0.02)
        if first is None:
            raise RuntimeError("Camera kết nối nhưng không có frame trong 10 giây.")

        print(f"[INFO] Frame size: {first.shape[1]} x {first.shape[0]}")
        print("[INFO] Warm-up YOLO...")
        for i in range(WARMUP_FRAMES):
            model.predict(first, imgsz=IMGSZ, conf=YOLO_INFER_CONF, verbose=False)
            print(f"  Warm-up {i + 1}/{WARMUP_FRAMES}")
        print("[OK] Warm-up xong.")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        frame_count = 0
        stats_count = 0
        total_yolo_time = 0.0
        total_process_time = 0.0
        last_detected_faces = None
        last_face_detection_at = 0.0
        last_face_count = 0

        print("\n[RUNNING] Nhấn Q/ESC để thoát.\n")

        while True:
            loop_start = time.perf_counter()
            ok, frame = reader.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            yolo_start = time.perf_counter()
            results = model.predict(frame, imgsz=IMGSZ, conf=YOLO_INFER_CONF, verbose=False)
            yolo_time = time.perf_counter() - yolo_start
            result = results[0]

            person_detections = prepare_person_detections(
                result, frame.shape[:2], args.person_conf
            )
            tracked_persons = tracker.update(person_detections, frame)
            now = time.perf_counter()
            for track in tracked_persons:
                identity_manager.touch_track(int(track[4]), now)

            if not args.no_face and frame_count % args.face_every == 0:
                (last_face_count, _fk, _fu, _ffps, last_detected_faces) = recognize_faces(
                    frame, tracked_persons, face_detector, face_recognizer,
                    employees, identity_manager, args.threshold,
                    args.min_margin, args.top_k,
                )
                last_face_detection_at = now

            face_boxes_by_track = {}
            if (last_detected_faces is not None
                    and now - last_face_detection_at <= FACE_BOX_TTL_SECONDS):
                for track in tracked_persons:
                    tid = int(track[4])
                    fb = get_face_box_for_track(tid, last_detected_faces, tracked_persons)
                    if fb is not None:
                        face_boxes_by_track[tid] = fb

            annotated = frame.copy()
            ppe_detections = []
            non_ppe = []
            det_idx = 0

            if result.boxes is not None:
                for box in result.boxes:
                    cid = int(box.cls[0].item())
                    if cid == PERSON_CLASS_ID:
                        continue
                    coords = box.xyxy[0].tolist()
                    conf = float(box.conf[0].item())
                    cname = model.names[cid]
                    is_ppe = get_ppe_group(cid) is not None
                    if conf < (args.ppe_conf if is_ppe else OTHER_CONF):
                        continue
                    label = f"{cname} {conf:.2f}"
                    color = (
                        COLOR_PPE_OK if cid in {0, 1, 7}
                        else COLOR_PPE_VIOLATION if cid in {2, 3, 4}
                        else COLOR_OTHER
                    )
                    if is_ppe:
                        ppe_detections.append({
                            "detection_index": det_idx,
                            "class_id": cid,
                            "class_name": cname,
                            "box": coords,
                            "confidence": conf,
                            "label": label,
                            "color": color,
                        })
                        det_idx += 1
                    else:
                        non_ppe.append((coords, label, color))

            for coords, label, color in non_ppe:
                draw_detection(annotated, coords, label, color)

            valid_indices = set()
            pending_alerts = []
            for track in tracked_persons:
                pbox = list(map(float, track[:4]))
                tid = int(track[4])
                fb = face_boxes_by_track.get(tid)
                statuses, valid = evaluate_person_ppe(pbox, ppe_detections, fb)
                valid_indices.update(d["detection_index"] for d in valid)
                state = alerts.update(tid, statuses, now, identity_manager)
                identity_text = identity_manager.display(tid)

                if state.last_printed != (state.snapshot(), identity_text):
                    print_person_status(tid, state.smoothed, identity_text)
                    state.last_printed = (state.snapshot(), identity_text)

                key = AlertManager._key(tid, identity_manager)
                viols = alerts.check_alert(key, state, now)
                if viols:
                    pending_alerts.append((tid, viols))
                draw_person_identity(annotated, pbox, identity_text)

            for det in ppe_detections:
                if det["detection_index"] in valid_indices:
                    draw_detection(annotated, det["box"], det["label"], det["color"])

            if frame_count % args.face_every == 0 and last_detected_faces is not None:
                for face in last_detected_faces:
                    x, y, w, h = face[:4].astype(int)
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), COLOR_FACE, 1)

            for tid, viols in ([] if is_enrolling() else pending_alerts):
                alerts.fire(annotated, tid, viols, identity_manager)

            _live = []
            for track in tracked_persons:
                _tid = int(track[4])
                _st = alerts.states.get(AlertManager._key(_tid, identity_manager))
                _eid, _name = identity_manager.details(_tid)
                _live.append({
                    "track_id": _tid,
                    "employee_id": _eid,
                    "full_name": _name,
                    "hat": _st.smoothed["Mu"] if _st else None,
                    "mask": _st.smoothed["Khau trang"] if _st else None,
                    "vest": _st.smoothed["Ao bao ho"] if _st else None,
                })
            update_live_tracks(_live)

            alerts.cleanup(now)
            identity_manager.cleanup(now)

            process_time = time.perf_counter() - loop_start
            yolo_fps = 1.0 / max(yolo_time, 1e-6)
            total_fps = 1.0 / max(process_time, 1e-6)
            draw_fps_panel(
                annotated, yolo_fps, total_fps,
                last_face_count if not args.no_face else 0,
                len(employees),
            )
            update_frame(annotated)
            cv2.imshow(WINDOW_NAME, annotated)

            stats_count += 1
            total_yolo_time += yolo_time
            total_process_time += process_time
            if stats_count >= REPORT_INTERVAL:
                avg_y = total_yolo_time / stats_count
                avg_p = total_process_time / stats_count
                print(
                    f"\n[STATS {stats_count}f] "
                    f"YOLO {avg_y:.4f}s ({1 / avg_y:.2f} fps) | "
                    f"Total {avg_p:.4f}s ({1 / avg_p:.2f} fps)\n"
                )
                stats_count = 0
                total_yolo_time = 0.0
                total_process_time = 0.0

            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                print("\n[INFO] Dừng bởi người dùng.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Dừng bởi người dùng.")
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if reader is not None:
            reader.release()
        cv2.destroyAllWindows()
        print("[INFO] Đã đóng camera và giải phóng tài nguyên.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())