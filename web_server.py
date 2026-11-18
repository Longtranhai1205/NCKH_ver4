"""Web server cho hệ thống PPE. Đọc trực tiếp dữ liệu của chương trình detect:
   - alerts/alert_log_integrated_v3.csv + ảnh trong alerts/
   - data/face_database.sqlite3 (employees, face_embeddings)
   - alerts/assignments.json (gán công nhân cho UNKNOWN)
Không sửa gì trong dữ liệu gốc.
"""
from __future__ import annotations

import csv
import json
import sqlite3
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
ALERT_DIR = ROOT / "alerts"
CSV_FILE = ALERT_DIR / "alert_log_integrated_v3.csv"
ASSIGN_FILE = ALERT_DIR / "assignments.json"
DB_FILE = ROOT / "data" / "face_database.sqlite3"
LABELS = {"Mu": "Không đội mũ", "Khau trang": "Không đeo khẩu trang", "Ao bao ho": "Không mặc áo bảo hộ"}
TYPES = ("Mu", "Khau trang", "Ao bao ho")

try:
    import face_enrollment_v2 as fe
except Exception as _e:
    fe = None
    print(f"[WEB] Không nạp được face_enrollment_v2.py: {_e}")

app = FastAPI()
_started_at = time.time()
_latest = {"jpg": None, "t": 0.0}
_enroll = {"active": False, "cancel": False, "state": "idle", "count": 0, "total": 10,
           "pose": "", "message": "", "name": ""}
_ctx = {"get_frame": None, "employees": None, "reload": None}
_live = {"tracks": [], "updated_at": 0.0}
_lock = threading.Lock()
_last_enc = 0.0


# ============================================================
# Frame & live tracks (được detect loop gọi)
# ============================================================

def update_frame(frame, every: float = 1.0) -> None:
    global _last_enc
    now = time.time()
    if _enroll["active"]:
        return
    if now - _last_enc < every:
        return
    _last_enc = now
    h, w = frame.shape[:2]
    if w > 960:
        frame = cv2.resize(frame, (960, int(h * 960 / w)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ok:
        with _lock:
            _latest["jpg"], _latest["t"] = buf.tobytes(), now


def update_live_tracks(tracks: list) -> None:
    with _lock:
        _live["tracks"] = tracks
        _live["updated_at"] = time.time()


def _put_jpg(frame) -> None:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ok:
        with _lock:
            _latest["jpg"], _latest["t"] = buf.tobytes(), time.time()


def set_enroll_context(get_frame, employees, reload_fn) -> None:
    _ctx.update(get_frame=get_frame, employees=employees, reload=reload_fn)


def is_enrolling() -> bool:
    return _enroll["active"]


# ============================================================
# Enrollment
# ============================================================

def _enroll_loop(emp_id: str, name: str, replace: bool) -> None:
    st = _enroll
    try:
        detector, recognizer = fe.load_models()
        embs, poses, times = [], [], []
        next_at = time.monotonic() + 1.0
        while not st["cancel"]:
            ok, frame = _ctx["get_frame"]()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            h, w = frame.shape[:2]
            detector.setInputSize((w, h))
            _s, faces = detector.detect(frame)
            n = 0 if faces is None else len(faces)
            sel = faces[0] if n == 1 else None
            prev = frame.copy()
            if faces is not None:
                for f in faces:
                    x, y, fw, fh = f[:4].astype(int)
                    cv2.rectangle(prev, (x, y), (x + fw, y + fh),
                                  (0, 220, 0) if n == 1 else (0, 165, 255), 2)
            _put_jpg(prev)
            now = time.monotonic()
            if len(embs) < fe.TARGET_SAMPLES and now >= next_at:
                next_at = now + fe.CAPTURE_INTERVAL_SECONDS
                reason = None
                if n == 0: reason = "Không thấy khuôn mặt"
                elif n > 1: reason = "Có nhiều khuôn mặt"
                elif float(sel[14]) < fe.MIN_DETECTION_CONFIDENCE: reason = "Độ tin cậy thấp"
                elif float(sel[2]) < fe.MIN_FACE_WIDTH or float(sel[3]) < fe.MIN_FACE_HEIGHT: reason = "Khuôn mặt nhỏ"
                elif fe.face_blur_score(frame, sel) < fe.MIN_BLUR_SCORE: reason = "Ảnh mờ"
                if reason:
                    st["message"] = "Chưa chụp: " + reason
                else:
                    feat = recognizer.feature(recognizer.alignCrop(frame, sel)).reshape(-1).astype(np.float32)
                    norm = float(np.linalg.norm(feat))
                    if feat.size != 128 or not np.isfinite(feat).all() or norm <= 1e-12:
                        st["message"] = "Chưa chụp: mẫu không hợp lệ"
                    else:
                        feat /= norm
                        embs.append(feat.copy())
                        poses.append(fe.pose_label_for_sample(len(embs)))
                        times.append(datetime.now(timezone.utc).isoformat(timespec="seconds"))
                        st["message"] = f"Đã chụp {len(embs)}/{fe.TARGET_SAMPLES}"
            st["count"] = len(embs)
            st["pose"] = fe.pose_for_sample(len(embs) + 1) if len(embs) < fe.TARGET_SAMPLES else ""
            if len(embs) == fe.TARGET_SAMPLES:
                fe.save_employee(emp_id, name, embs, poses, times, replace_existing=replace)
                if _ctx["employees"] is not None and _ctx["reload"] is not None:
                    _ctx["employees"][:] = _ctx["reload"]()
                st["state"] = "done"
                st["message"] = f"Đã đăng ký {name}. Hệ thống nhận diện ngay."
                return
            time.sleep(0.1)
        st["state"], st["message"] = "cancelled", "Đã hủy."
    except Exception as e:
        st["state"], st["message"] = "error", f"Lỗi: {e}"
    finally:
        st["active"] = False


class EnrollReq(BaseModel):
    id: str
    name: str
    replace: bool = False


@app.post("/api/enroll/start")
def enroll_start(body: EnrollReq):
    if fe is None or _ctx["get_frame"] is None:
        raise HTTPException(503, "Chưa sẵn sàng đăng ký")
    if _enroll["active"]:
        raise HTTPException(409, "Đang có một lượt đăng ký khác")
    emp_id, name = body.id.strip(), body.name.strip()
    if not emp_id or len(emp_id) > 64 or not name or len(name) > 150:
        raise HTTPException(400, "Mã (≤64) và họ tên (≤150) không được để trống")
    fe.init_database()
    old = fe.get_existing_employee_name(emp_id)
    if old is not None and not body.replace:
        return Response(
            json.dumps({"exists": True, "name": old}, ensure_ascii=False),
            status_code=409, media_type="application/json",
        )
    _enroll.update(active=True, cancel=False, state="running", count=0,
                   pose="", message="Đưa khuôn mặt vào giữa khung hình", name=name)
    threading.Thread(target=_enroll_loop, args=(emp_id, name, old is not None), daemon=True).start()
    return {"ok": True}


@app.get("/api/enroll/status")
def enroll_status():
    return _enroll


@app.post("/api/enroll/cancel")
def enroll_cancel():
    _enroll["cancel"] = True
    return {"ok": True}


# ============================================================
# Workers
# ============================================================

def _all_workers(include_inactive: bool = False) -> list[dict]:
    if not DB_FILE.is_file():
        return []
    with sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True) as c:
        where = "" if include_inactive else "WHERE COALESCE(e.is_active, 1) = 1"
        rows = c.execute(
            f"""SELECT e.employee_id, e.full_name, COUNT(fe.employee_id),
                       COALESCE(e.is_active, 1)
                FROM employees e
                LEFT JOIN face_embeddings fe ON fe.employee_id = e.employee_id
                {where}
                GROUP BY e.employee_id, e.full_name
                ORDER BY e.employee_id"""
        ).fetchall()
    return [{"id": r[0], "name": r[1], "n": r[2], "active": bool(r[3])} for r in rows]


@app.get("/api/workers")
def workers(include_inactive: bool = Query(False)):
    return _all_workers(include_inactive)


class DeactivateReq(BaseModel):
    reason: str = "inactive"


@app.post("/api/workers/{emp_id}/deactivate")
def deactivate_worker(emp_id: str, body: DeactivateReq = DeactivateReq()):
    if not DB_FILE.is_file():
        raise HTTPException(404, "Chưa có database")
    with sqlite3.connect(DB_FILE) as c:
        row = c.execute("SELECT is_active FROM employees WHERE employee_id=?", (emp_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Không có công nhân này")
        c.execute("UPDATE employees SET is_active=0 WHERE employee_id=?", (emp_id,))
        c.commit()
    return {"ok": True, "reason": body.reason}


# ============================================================
# Snapshot / status
# ============================================================

@app.get("/api/snapshot.jpg")
def snapshot():
    with _lock:
        jpg = _latest["jpg"]
    if jpg is None:
        return Response(status_code=204)
    return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/status")
def status():
    age = time.time() - _latest["t"] if _latest["t"] else None
    return {"online": age is not None and age < 5, "age": age}


@app.get("/api/live/tracks")
def live_tracks():
    with _lock:
        tracks = list(_live["tracks"])
        ts = _live["updated_at"]
    return {"tracks": tracks, "updated_at": ts, "age": time.time() - ts if ts else None}


# ============================================================
# Violations
# ============================================================

def _assignments() -> dict:
    try:
        return json.loads(ASSIGN_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_all_violations() -> list[dict]:
    if not CSV_FILE.is_file():
        return []
    assign = _assignments()
    name_map = {w["id"]: w["name"] for w in _all_workers(include_inactive=True)}
    out: list[dict] = []
    with CSV_FILE.open(encoding="utf-8-sig", newline="") as f:
        for i, r in enumerate(csv.DictReader(f), start=1):
            stamp = (r.get("thoi_gian") or "").strip()
            if len(stamp) < 16:
                continue
            raw_codes = [x.strip() for x in (r.get("vi_pham") or "").split(",") if x.strip()]
            type_vi = ", ".join(LABELS.get(c, c) for c in raw_codes)
            csv_eid = (r.get("employee_id") or "").strip()
            csv_name = (r.get("full_name") or "").strip()

            if str(i) in assign and assign[str(i)]:
                wid = assign[str(i)]
                w_name = name_map.get(wid, "UNKNOWN")
            elif csv_eid and csv_eid.upper() != "UNKNOWN":
                wid = csv_eid
                w_name = csv_name if csv_name and csv_name.upper() != "UNKNOWN" else name_map.get(wid, "UNKNOWN")
            else:
                wid = None
                w_name = None

            out.append({
                "id": i,
                "date": stamp[:10],
                "time": stamp[11:16],
                "type": type_vi,
                "type_raw": raw_codes,
                "cam": "Camera 1",
                "wid": wid,
                "w_name": w_name,
                "img": "/evidence/" + Path(r.get("anh", "")).name,
                "status": "assigned" if wid else "unassigned",
            })
    return out


def _group_violations(items: list[dict], minutes: int = 5) -> list[dict]:
    """Gộp các vi phạm của cùng 1 nhân viên trong khung `minutes` phút.
    - UNKNOWN: giữ riêng từng dòng.
    - KNOWN: gộp type_raw, lấy ảnh mới nhất.
    """
    if minutes <= 0:
        return items

    result: list[dict] = []
    known_buckets: dict[tuple[str, str], int] = {}

    for v in items:
        if v["wid"] is None:
            result.append(dict(v))
            continue
        stamp = f"{v['date']} {v['time']}"
        try:
            dt = datetime.strptime(stamp, "%Y-%m-%d %H:%M")
        except ValueError:
            result.append(dict(v))
            continue
        
        # Chia bucket theo block `minutes` phút
        bucket_min = (dt.minute // minutes) * minutes
        bucket = dt.replace(minute=bucket_min).strftime("%Y-%m-%d %H:%M")
        key = (v["wid"], bucket)

        if key in known_buckets:
            existing = result[known_buckets[key]]
            merged = sorted(set(existing["type_raw"]) | set(v["type_raw"]))
            existing["type_raw"] = merged
            existing["type"] = ", ".join(LABELS.get(c, c) for c in merged)
            if v["id"] > existing["id"]:
                existing["id"] = v["id"]
                existing["img"] = v["img"]
                existing["date"] = v["date"]
                existing["time"] = v["time"]
        else:
            known_buckets[key] = len(result)
            result.append(dict(v))

    return result


@app.get("/api/violations")
def violations(
    page: int = Query(0, ge=0),
    limit: int = Query(0, ge=0),
    date: str = Query(""),
    worker_id: str = Query(""),
    type: str = Query(""),
    status_: str = Query("", alias="status"),
    group_minutes: int = Query(0, ge=0, le=60),
):
    items = _read_all_violations()
    if date:      items = [v for v in items if v["date"] == date]
    if worker_id: items = [v for v in items if v["wid"] == worker_id]
    if type:      items = [v for v in items if type in v["type_raw"]]
    if status_:   items = [v for v in items if v["status"] == status_]
    items = list(reversed(items))  # mới nhất trước
    if group_minutes > 0:
        items = _group_violations(items, group_minutes)
    if limit > 0:
        total = len(items)
        pages = max(1, (total + limit - 1) // limit)
        start = (max(1, page) - 1) * limit
        return {"items": items[start:start + limit], "total": total,
                "page": max(1, page), "limit": limit, "total_pages": pages}
    return items


class Assign(BaseModel):
    worker_id: str


@app.post("/api/violations/{vid}/assign")
def assign(vid: int, body: Assign):
    if body.worker_id not in {w["id"] for w in _all_workers()}:
        raise HTTPException(404, "Không có công nhân này")
    data = _assignments()
    data[str(vid)] = body.worker_id
    ALERT_DIR.mkdir(exist_ok=True)
    ASSIGN_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return {"ok": True}


# ============================================================
# Stats (Áp dụng gộp 5 phút)
# ============================================================

@app.get("/api/stats/overview")
def stats_overview():
    items = _group_violations(_read_all_violations(), 5)
    today = datetime.now().strftime("%Y-%m-%d")
    return {
        "total_workers": len(_all_workers()),
        "total_violations": len(items),
        "today_violations": sum(1 for v in items if v["date"] == today),
        "unassigned_violations": sum(1 for v in items if v["status"] == "unassigned"),
    }


@app.get("/api/stats/by-day")
def stats_by_day(days: int = Query(30, ge=1, le=365)):
    items = _group_violations(_read_all_violations(), 5)
    counter = Counter(v["date"] for v in items)
    today = datetime.now().date()
    out = []
    for i in range(days - 1, -1, -1):
        d = today.fromordinal(today.toordinal() - i).isoformat()
        out.append({"date": d, "count": counter.get(d, 0)})
    return out


@app.get("/api/stats/by-type")
def stats_by_type():
    items = _group_violations(_read_all_violations(), 5)
    c = Counter()
    for v in items:
        for t in v["type_raw"]:
            c[t] += 1
    return [{"type": t, "count": c.get(t, 0)} for t in TYPES]


@app.get("/api/stats/top-workers")
def stats_top_workers(limit: int = Query(5, ge=1, le=50)):
    items = _group_violations(_read_all_violations(), 5)
    counter = Counter(v["wid"] for v in items if v["wid"])
    names = {w["id"]: w["name"] for w in _all_workers(include_inactive=True)}
    return [{"id": k, "name": names.get(k, k), "count": n} for k, n in counter.most_common(limit)]


# ============================================================
# System info
# ============================================================

@app.get("/api/system/info")
def system_info():
    db_size = DB_FILE.stat().st_size if DB_FILE.is_file() else 0
    age = time.time() - _latest["t"] if _latest["t"] else None
    return {
        "camera": {"online": age is not None and age < 5, "age": age},
        "backend": {"running": True, "uptime": round(time.time() - _started_at)},
        "yolo": {"loaded": True, "model": "best.onnx", "imgsz": 416},
        "face": {"loaded": fe is not None, "employees": len(_all_workers(include_inactive=True))},
        "database": {"path": str(DB_FILE), "size_kb": round(db_size / 1024, 1),
                     "workers": len(_all_workers())},
    }


# ============================================================
# Static
# ============================================================

ALERT_DIR.mkdir(exist_ok=True)
app.mount("/evidence", StaticFiles(directory=str(ALERT_DIR)), name="evidence")
app.mount("/", StaticFiles(directory=str(ROOT / "static"), html=True), name="static")


def start_server(port: int = 8000) -> None:
    threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning"),
        daemon=True,
    ).start()
    print(f"[WEB] Dashboard: http://localhost:{port}")