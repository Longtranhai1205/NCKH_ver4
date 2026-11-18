from __future__ import annotations

import os
import sqlite3
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "face_recognition"
YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
DB_PATH = ROOT / "data" / "face_database.sqlite3"

DEFAULT_RTSP_URL = "rtsp://admin:Longtran2005%40@192.168.2.79:554/h264/ch1/sub/av_stream"
SOURCE = os.environ.get("FACE_RTSP_URL", DEFAULT_RTSP_URL)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

TARGET_SAMPLES = 10
CAPTURE_INTERVAL_SECONDS = 2.0
NEXT_EMPLOYEE_TIMEOUT_SECONDS = 10.0
MIN_FACE_WIDTH = 90
MIN_FACE_HEIGHT = 90
MIN_DETECTION_CONFIDENCE = 0.75
MIN_BLUR_SCORE = 35.0
WINDOW_NAME = "Face Enrollment - Auto Capture"


def pose_for_sample(sample_number: int) -> str:
    if sample_number <= 3:
        return "Nhìn thẳng, giữ đầu tự nhiên"
    if sample_number <= 5:
        return "Quay nhẹ sang TRÁI khoảng 15-25 độ"
    if sample_number <= 7:
        return "Quay nhẹ sang PHẢI khoảng 15-25 độ"
    if sample_number == 8:
        return "Ngẩng cằm nhẹ"
    if sample_number == 9:
        return "Cúi cằm nhẹ"
    return "Trở về nhìn thẳng"


def pose_label_for_sample(sample_number: int) -> str:
    if sample_number <= 3 or sample_number == 10:
        return "frontal"
    if sample_number <= 5:
        return "left_slight"
    if sample_number <= 7:
        return "right_slight"
    if sample_number == 8:
        return "up_slight"
    return "down_slight"


def init_database() -> None:
    """Tạo bảng và thêm các cột mới nếu chưa có. An toàn cho DB cũ."""
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
        # Cột pose_label (V2)
        fe_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(face_embeddings)").fetchall()
        }
        if "pose_label" not in fe_cols:
            conn.execute(
                "ALTER TABLE face_embeddings ADD COLUMN pose_label TEXT NOT NULL DEFAULT 'unknown'"
            )

        # === MỚI === Cột is_active: 1 = đang hoạt động, 0 = đã xóa mềm
        # Khi = 0: vẫn giữ lịch sử, nhưng detect_ppe không nạp vào bộ nhận diện
        emp_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(employees)").fetchall()
        }
        if "is_active" not in emp_cols:
            conn.execute(
                "ALTER TABLE employees ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1"
            )
        conn.commit()


def get_existing_employee_name(employee_id: str) -> Optional[str]:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT full_name FROM employees WHERE employee_id = ? AND COALESCE(is_active,1)=1",
            (employee_id,),
        ).fetchone()
    return str(row[0]) if row else None


def prompt_employee() -> Optional[tuple[str, str, bool]]:
    while True:
        print("\n=== THÔNG TIN NHÂN VIÊN ===")
        employee_id = input("Mã nhân viên (nhập Q để thoát): ").strip()
        if employee_id.lower() == "q":
            return None
        if not employee_id or len(employee_id) > 64:
            print("[WARN] Mã nhân viên không được để trống và tối đa 64 ký tự.")
            continue

        existing_name = get_existing_employee_name(employee_id)
        if existing_name is not None:
            choice = input(
                f"Mã {employee_id} đã tồn tại. Nhập R để chụp lại và thay mẫu cũ, "
                "N để nhập mã khác, Q để thoát: "
            ).strip().lower()
            if choice == "q":
                return None
            if choice != "r":
                continue
            full_name = input(f"Họ và tên [{existing_name}] (Enter để giữ nguyên): ").strip()
            full_name = full_name or existing_name
            if len(full_name) > 150:
                print("[WARN] Họ tên tối đa 150 ký tự.")
                continue
            return employee_id, full_name, True

        full_name = input("Họ và tên: ").strip()
        if not full_name:
            print("[WARN] Họ tên không được để trống.")
            continue
        if len(full_name) > 150:
            print("[WARN] Họ tên tối đa 150 ký tự.")
            continue
        return employee_id, full_name, False


def load_models():
    for path in (YUNET_PATH, SFACE_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy model: {path}")
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


def ascii_display(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    plain = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return plain.replace("Đ", "D").replace("đ", "d")


def draw_text(image: np.ndarray, text: str, xy: tuple[int, int],
              scale: float = 0.48, color=(245, 245, 245), thickness: int = 1) -> None:
    cv2.putText(image, ascii_display(text), xy, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def save_employee(employee_id: str, full_name: str,
                  embeddings: list[np.ndarray], poses: list[str],
                  captured_times: list[str], replace_existing: bool = False) -> None:
    """Lưu nhân viên mới. replace_existing: xóa bản ghi cũ trước khi chèn."""
    if len(embeddings) != TARGET_SAMPLES:
        raise ValueError(f"Cần đủ {TARGET_SAMPLES} mẫu; hiện có {len(embeddings)}.")
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN IMMEDIATE")
        if replace_existing:
            conn.execute("DELETE FROM employees WHERE employee_id = ?", (employee_id,))
        conn.execute(
            "INSERT INTO employees(employee_id, full_name, created_at, is_active) "
            "VALUES (?, ?, ?, 1)",
            (employee_id, full_name, created_at),
        )
        for index, (vector, pose, captured_at) in enumerate(
            zip(embeddings, poses, captured_times), start=1
        ):
            conn.execute(
                """INSERT INTO face_embeddings
                   (employee_id, sample_index, embedding, dimension, captured_at, pose_label)
                   VALUES (?, ?, ?, 128, ?, ?)""",
                (employee_id, index, sqlite3.Binary(vector.tobytes()), captured_at, pose),
            )
        conn.commit()


def main() -> int:
    print("\n=== ĐĂNG KÝ KHUÔN MẶT NHÂN VIÊN – TỰ ĐỘNG ===")
    print(f"OpenCV: {cv2.__version__}")
    print(f"Database: {DB_PATH}")
    print(f"Tự chụp: mỗi {CAPTURE_INTERVAL_SECONDS:.0f} giây | {TARGET_SAMPLES} mẫu/người")

    init_database()
    employee = prompt_employee()
    if employee is None:
        print("[INFO] Người dùng thoát trước khi mở camera.")
        return 0

    try:
        detector, recognizer = load_models()
    except Exception as exc:
        print(f"[ERROR] Không thể tải YuNet/SFace: {exc}")
        return 1

    print("[OK] Đã tải YuNet và SFace.")
    source = SOURCE
    backend = cv2.CAP_FFMPEG if isinstance(source, str) and source.lower().startswith("rtsp://") else cv2.CAP_ANY
    cap = cv2.VideoCapture(source, backend)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("[ERROR] Không mở được camera.")
        return 1

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    embeddings: list[np.ndarray] = []
    pose_labels: list[str] = []
    capture_times: list[str] = []
    next_capture_at = time.monotonic() + 1.0
    saved_at: Optional[float] = None
    waiting_for_next_employee = False
    status_message = "Đưa khuôn mặt vào giữa khung hình."

    print("[INFO] Camera đã mở. U=lùi mẫu | R=làm lại | S=lưu | Q=thoát.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            height, width = frame.shape[:2]
            detector.setInputSize((width, height))
            _status, faces = detector.detect(frame)
            face_count = 0 if faces is None else len(faces)
            selected_face = faces[0] if face_count == 1 else None
            blur_score = face_blur_score(frame, selected_face) if selected_face is not None else 0.0
            face_conf = float(selected_face[14]) if selected_face is not None else 0.0
            face_w = float(selected_face[2]) if selected_face is not None else 0.0
            face_h = float(selected_face[3]) if selected_face is not None else 0.0

            preview = frame.copy()
            if faces is not None:
                for idx, face in enumerate(faces, start=1):
                    x, y, w, h = face[:4].astype(int)
                    color = (0, 220, 0) if face_count == 1 else (0, 165, 255)
                    cv2.rectangle(preview, (x, y), (x + w, y + h), color, 2)
                    draw_text(preview, f"Face {idx} {float(face[14]):.2f}",
                              (max(5, x), max(20, y - 7)), 0.43, color, 1)

            now = time.monotonic()
            if not waiting_for_next_employee and len(embeddings) < TARGET_SAMPLES and now >= next_capture_at:
                next_capture_at = now + CAPTURE_INTERVAL_SECONDS
                invalid = None
                if face_count == 0: invalid = "Không thấy khuôn mặt"
                elif face_count > 1: invalid = "Có nhiều khuôn mặt"
                elif face_conf < MIN_DETECTION_CONFIDENCE: invalid = "Độ tin cậy thấp"
                elif face_w < MIN_FACE_WIDTH or face_h < MIN_FACE_HEIGHT: invalid = "Khuôn mặt quá nhỏ"
                elif blur_score < MIN_BLUR_SCORE: invalid = "Ảnh bị mờ"

                if invalid:
                    status_message = f"Chưa chụp: {invalid}"
                else:
                    try:
                        aligned = recognizer.alignCrop(frame, selected_face)
                        feature = recognizer.feature(aligned).reshape(-1).astype(np.float32)
                        norm = float(np.linalg.norm(feature))
                        if feature.size != 128 or not np.isfinite(feature).all() or norm <= 1e-12:
                            raise ValueError("Embedding không hợp lệ")
                        feature /= norm
                        sample_number = len(embeddings) + 1
                        embeddings.append(feature.copy())
                        pose_labels.append(pose_label_for_sample(sample_number))
                        capture_times.append(datetime.now(timezone.utc).isoformat(timespec="seconds"))
                        status_message = f"Đã chụp {sample_number}/{TARGET_SAMPLES}"
                        if len(embeddings) == TARGET_SAMPLES:
                            status_message = "ĐỦ 10 MẪU - Nhấn S để lưu"
                    except (cv2.error, ValueError) as exc:
                        status_message = f"Lỗi: {exc}"

            cv2.rectangle(preview, (0, 0), (width, 62), (0, 0, 0), -1)
            employee_id, full_name, _replace = employee
            draw_text(preview, f"{employee_id} | {full_name} | Mau: {len(embeddings)}/{TARGET_SAMPLES}",
                      (10, 22), 0.48, (255, 255, 255), 1)

            if waiting_for_next_employee and saved_at is not None:
                remaining = max(0.0, NEXT_EMPLOYEE_TIMEOUT_SECONDS - (now - saved_at))
                draw_text(preview, f"DA LUU | N: tiep theo | Q: thoat | Tu dong {remaining:.1f}s",
                          (10, 47), 0.43, (0, 255, 120), 1)
                if remaining <= 0:
                    break
            elif len(embeddings) >= TARGET_SAMPLES:
                draw_text(preview, "DU 10/10 - S: luu | U: lui | R: lai | Q: thoat",
                          (10, 47), 0.40, (0, 255, 255), 1)
            else:
                pose = pose_for_sample(len(embeddings) + 1)
                wait_left = max(0.0, next_capture_at - now)
                draw_text(preview, f"{status_message} | {wait_left:.1f}s: {pose}",
                          (10, 47), 0.39, (0, 255, 255), 1)

            cv2.rectangle(preview, (0, height - 25), (width, height), (0, 0, 0), -1)
            draw_text(preview, "U: lui mau | R: lam lai | S: luu khi du 10 mau | Q: thoat",
                      (8, height - 7), 0.38, (230, 230, 230), 1)
            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if waiting_for_next_employee:
                if key == ord("n"):
                    next_employee = prompt_employee()
                    if next_employee is None:
                        break
                    employee = next_employee
                    embeddings.clear(); pose_labels.clear(); capture_times.clear()
                    waiting_for_next_employee = False
                    saved_at = None
                    next_capture_at = time.monotonic() + 1.0
                    status_message = "Bắt đầu đăng ký mới."
                continue

            if key == ord("u") and embeddings:
                embeddings.pop(); pose_labels.pop(); capture_times.pop()
                next_capture_at = time.monotonic() + CAPTURE_INTERVAL_SECONDS
                status_message = "Đã lùi 1 mẫu."
            elif key == ord("r"):
                embeddings.clear(); pose_labels.clear(); capture_times.clear()
                next_capture_at = time.monotonic() + 1.0
                status_message = "Đã xóa toàn bộ."
            elif key == ord("s"):
                if len(embeddings) != TARGET_SAMPLES:
                    continue
                try:
                    save_employee(employee[0], employee[1], embeddings, pose_labels,
                                  capture_times, replace_existing=employee[2])
                except sqlite3.IntegrityError as exc:
                    print(f"[ERROR] {exc}")
                    continue
                print(f"\n[SUCCESS] Đã lưu {employee[0]} - {employee[1]} ({len(embeddings)} mẫu).")
                waiting_for_next_employee = True
                saved_at = time.monotonic()
                status_message = "Đã lưu."

    except KeyboardInterrupt:
        print("\n[INFO] Dừng.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())