"""
Rehabilitation AI System - Optimized Backend
FastAPI + MediaPipe Pose Estimation (Performance Tuned)
+ MJPEG /video_feed pipeline (reliable browser-native streaming)
"""

import os
import base64
import random
import asyncio
import datetime
import warnings
import time
import threading
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import mediapipe as mp
warnings.filterwarnings("ignore")

from fastapi import (
    FastAPI, HTTPException, Request, UploadFile,
    File, Form, WebSocket, WebSocketDisconnect
)
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from database import get_db, init_db
from models import (
    Patient, SessionModel, JointAngle, ExerciseResult,
    Report, Setting, History,
)

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

# ─── Directories ──────────────────────────────────────────────────────────────
for d in ("data", "reports", "uploads", "static", "templates", "assets"):
    os.makedirs(d, exist_ok=True)


def compress_photo(raw_bytes: bytes, max_dim: int = 800, quality: int = 80) -> bytes:
    """
    Downscale + JPEG-compress an uploaded patient photo before storing it.
    Raw camera/canvas captures can be several MB (e.g. 3200x4800 PNG); this
    keeps DB rows small and avoids MySQL column-size issues. Falls back to
    the original bytes if decoding fails (e.g. unsupported format).
    """
    try:
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return raw_bytes
        h, w = img.shape[:2]
        scale = max_dim / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else raw_bytes
    except Exception as exc:
        print(f"⚠️  Photo compression failed, storing original bytes: {exc}")
        return raw_bytes


init_db()  # creates all tables (and the MySQL database itself, if missing)

# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="Rehabilitation AI System", version="2.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
app.mount("/static",  StaticFiles(directory="static"),  name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/assets",  StaticFiles(directory="assets"),  name="assets")

# ─── MediaPipe init (optimized) ───────────────────────────────────────────────
# model_complexity=0  → fastest, lightest model
# smooth_landmarks=False → skip smoothing for speed
try:
    _mp_pose = mp.solutions.pose
    _pose    = _mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,           # ← 0 = fastest
        smooth_landmarks=False,       # ← skip smoothing
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    _mp_drawing        = mp.solutions.drawing_utils
    _mp_drawing_styles = mp.solutions.drawing_styles
    _POSE_CONNECTIONS  = _mp_pose.POSE_CONNECTIONS
    print("✅ MediaPipe Pose initialized (optimized: complexity=0)")
except Exception as exc:
    print(f"⚠️  MediaPipe init failed: {exc}")
    _pose = _mp_drawing = _mp_drawing_styles = _POSE_CONNECTIONS = None

# ─── Key landmarks only (13 joints instead of 33) ────────────────────────────
# Indices: nose=0, shoulders=11/12, elbows=13/14, wrists=15/16,
#          hips=23/24, knees=25/26, ankles=27/28
_KEY_LANDMARKS = {
    "nose": 0, "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14, "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}

# ─── Angle helper ─────────────────────────────────────────────────────────────
def _get_angle(p1, p2, p3) -> float:
    a = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    b = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))

# ─── Helpers ──────────────────────────────────────────────────────────────────
def calculate_recovery_score(sessions: list) -> float:
    if not sessions:
        return 0.0
    n = len(sessions)
    avg = lambda attr: sum(getattr(s, attr) or 0 for s in sessions) / n
    score = (
        avg("accuracy_percentage") * 0.30
        + avg("average_rom")       * 0.20
        + avg("stability_score")   * 0.25
        + avg("balance_score")     * 0.25
    )
    if n >= 3:
        score += min(n * 2, 10)
    return min(score, 100.0)


def calculate_improvement(sessions: list) -> float:
    if len(sessions) < 2:
        return 0.0
    ss = sorted(sessions, key=lambda s: s.start_time)
    def _avg(s):
        return (s.accuracy_percentage + s.average_rom + s.stability_score + s.balance_score) / 4
    first, last = _avg(ss[0]), _avg(ss[-1])
    if first == 0:
        return 0.0
    return max(-100.0, min(100.0, ((last - first) / first) * 100))


def session_summary(s: SessionModel) -> dict:
    return {
        "session_id":          s.id,
        "exercise_type":       s.exercise_type,
        "duration":            s.duration_seconds,
        "total_reps":          s.total_reps,
        "completed_reps":      s.completed_reps,
        "accuracy":            s.accuracy_percentage,
        "rom":                 s.average_rom,
        "stability":           s.stability_score,
        "balance":             s.balance_score,
        "smoothness":          s.movement_smoothness,
        "fatigue":             s.fatigue_estimation,
        "recovery_score":      s.recovery_score,
        "incorrect_movements": s.incorrect_movements,
        "date":                s.start_time.strftime("%Y-%m-%d %H:%M"),
    }

# ─── Camera Manager (WebSocket pipeline — kept for future use) ───────────────
class CameraManager:
    # Target FPS for streaming — caps how often we send frames
    TARGET_FPS   = 10
    FRAME_BUDGET = 1.0 / TARGET_FPS   # seconds per frame

    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running  = False
        self._frame_count = 0           # for frame skipping
        self._last_sent   = 0.0         # for FPS throttling
        self._draw_landmarks = True     # can disable for more speed

    def start(self) -> bool:
        if self.cap is not None:
            return False
        self.cap = cv2.VideoCapture(0)
        # ↓ 320×240 — halves pixel count vs 640×480, ~4× faster inference
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.is_running   = True
        self._frame_count = 0
        self._last_sent   = 0.0
        return True

    def stop(self) -> bool:
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        return True

    def get_frame(self):
        if not (self.cap and self.is_running):
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def should_process(self) -> bool:
        """Skip every other frame — process only even frames."""
        self._frame_count += 1
        return self._frame_count % 2 == 0

    def fps_throttle(self) -> bool:
        """Return True if enough time has passed to send next frame."""
        now = time.monotonic()
        if now - self._last_sent >= self.FRAME_BUDGET:
            self._last_sent = now
            return True
        return False

    def process_frame(self, frame):
        """
        Run MediaPipe on frame, return (annotated_frame, pose_data).
        pose_data contains only 13 key landmarks + joint angles.
        Returns (frame, None) when no pose is detected.
        """
        if frame is None or _pose is None:
            return frame, None

        # Convert BGR → RGB (MediaPipe needs RGB)
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False          # avoid unnecessary copy inside MP
        results = _pose.process(rgb)
        rgb.flags.writeable = True

        annotated = frame.copy()

        if not results.pose_landmarks:
            return annotated, None

        # Optional landmark drawing — disable if still slow
        if self._draw_landmarks and _mp_drawing:
            _mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                _POSE_CONNECTIONS,
                landmark_drawing_spec=(
                    _mp_drawing_styles.get_default_pose_landmarks_style()
                    if _mp_drawing_styles else None
                ),
            )

        lm = results.pose_landmarks.landmark

        # ── Only send 13 key landmarks (not all 33) ──────────────────────────
        pose_data = {}
        for name, idx in _KEY_LANDMARKS.items():
            if idx < len(lm) and lm[idx].visibility > 0.5:   # skip low-confidence
                pose_data[name] = {
                    "x": round(lm[idx].x, 4),
                    "y": round(lm[idx].y, 4),
                    "z": round(lm[idx].z, 4),
                    "v": round(lm[idx].visibility, 2),
                }

        # ── Joint angles ──────────────────────────────────────────────────────
        angles = {}
        try:
            if len(lm) > 16:
                angles["l_shoulder"] = round(_get_angle(lm[23], lm[11], lm[13]), 1)
                angles["r_shoulder"] = round(_get_angle(lm[24], lm[12], lm[14]), 1)
                angles["l_elbow"]    = round(_get_angle(lm[11], lm[13], lm[15]), 1)
                angles["r_elbow"]    = round(_get_angle(lm[12], lm[14], lm[16]), 1)
            if len(lm) > 28:
                angles["l_knee"]     = round(_get_angle(lm[23], lm[25], lm[27]), 1)
                angles["r_knee"]     = round(_get_angle(lm[24], lm[26], lm[28]), 1)
            if len(lm) > 26:
                angles["l_hip"]      = round(_get_angle(lm[11], lm[23], lm[25]), 1)
                angles["r_hip"]      = round(_get_angle(lm[12], lm[24], lm[26]), 1)
        except Exception:
            pass

        pose_data["angles"] = angles
        return annotated, pose_data


# ─── WebSocket Manager ────────────────────────────────────────────────────────
class WSManager:
    def __init__(self):
        self.connections: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)

    async def send(self, ws: WebSocket, data: dict):
        try:
            await ws.send_json(data)
        except Exception:
            pass


camera = CameraManager()
ws_mgr = WSManager()

# ─── WebSocket: Pose stream (kept, not removed — available for future use) ───
@app.websocket("/ws/pose")
async def ws_pose(websocket: WebSocket):
    await ws_mgr.connect(websocket)
    try:
        if not camera.is_running:
            camera.start()

        while True:
            # ── 1. Grab frame ─────────────────────────────────────────────────
            frame = camera.get_frame()

            # ── 2. Safety check ───────────────────────────────────────────────
            if frame is None:
                await asyncio.sleep(0.1)
                continue

            # ── 3. Frame skip — process only every 2nd frame ──────────────────
            if not camera.should_process():
                await asyncio.sleep(0.033)   # ~30fps read loop, skip odd frames
                continue

            # ── 4. FPS throttle — don't send faster than TARGET_FPS ───────────
            if not camera.fps_throttle():
                await asyncio.sleep(0.01)
                continue

            # ── 5. Pose processing ────────────────────────────────────────────
            annotated, pose_data = camera.process_frame(frame)

            if annotated is None:
                await asyncio.sleep(0.1)
                continue

            # ── 6. JPEG encode at lower quality (60) → smaller payload ────────
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 60]
            success, buf  = cv2.imencode(".jpg", annotated, encode_params)
            if not success:
                await asyncio.sleep(0.1)
                continue

            # ── 7. Send base64 frame + slim pose_data ─────────────────────────
            await ws_mgr.send(websocket, {
                "type":      "pose_data",
                "frame":     base64.b64encode(buf).decode(),
                "pose_data": pose_data,
                "ts":        datetime.datetime.now().isoformat(),
            })

            # ── 8. Async sleep — gives event loop room to breathe ─────────────
            await asyncio.sleep(0.1)   # ~10fps effective send rate

    except WebSocketDisconnect:
        ws_mgr.disconnect(websocket)
        if not ws_mgr.connections:
            camera.stop()
    except Exception as e:
        print(f"WebSocket error: {e}")
        ws_mgr.disconnect(websocket)


# ═══════════════════════════════════════════════════════════════════════════
# ─── MJPEG video feed pipeline (PRIMARY camera transport for session page) ───
# ═══════════════════════════════════════════════════════════════════════════
#
# Simpler + more reliable than WebSocket for this use case: browser <img> tag
# decodes the multipart stream natively, no JS frame-decoding / canvas loop
# needed. Runs as a sync generator (FastAPI runs sync generators in a thread
# pool), so it does not block the asyncio event loop used by /ws/pose above.

_mjpeg_cap: Optional[cv2.VideoCapture] = None
_mjpeg_active = False
_mjpeg_lock = threading.Lock()

# ── Background frame reader (fixes growing camera delay) ──────────────────────
# cv2.VideoCapture buffers frames internally. If the consumer (MediaPipe
# processing in gen_frames(), below) is slower than the camera's own frame
# rate, cap.read() starts returning older and older buffered frames — a
# "delay that keeps growing the longer the session runs" bug. Fix: a
# dedicated thread reads the camera as fast as it can and always overwrites
# a single shared "latest frame" slot. gen_frames() just grabs whatever's
# freshest each time and skips whatever it didn't get to in time, so it
# never falls behind — display always shows the current moment, not a
# backlog.
_mjpeg_latest_frame = None
_mjpeg_frame_id = 0
_mjpeg_frame_lock = threading.Lock()
_mjpeg_reader_thread: Optional[threading.Thread] = None
_mjpeg_reader_running = False

_latest_pose_data: Dict[str, Any] = {
    "detected": False, "angles": {}, "ts": None,
    "reps": 0, "stability": 100.0, "primary_angle": None,
    "smoothness": 100.0, "balance": 100.0, "fatigue": 0.0,
}

# ─── ✅ Real rep-counting + stability tracking (no randomness) ────────────────
import collections
import statistics

_rep_lock = threading.Lock()
_rep_count = 0
_rep_stage: Optional[str] = None      # "up" / "down"
_current_target_rom = 90.0

_stability_lock = threading.Lock()
_landmark_jitter_buffer: "collections.deque" = collections.deque(maxlen=20)
_current_stability_score = 100.0

# ─── ✅ Real smoothness tracking — frame-to-frame angular jerk variance ───────
# Jerky/shaky movement produces large, erratic frame-to-frame angle deltas;
# a controlled rep produces small, consistent deltas. Low delta-variance =
# high smoothness.
_smoothness_lock = threading.Lock()
_angle_velocity_buffer: "collections.deque" = collections.deque(maxlen=15)
_last_primary_angle: Optional[float] = None
_current_smoothness_score = 100.0

# ─── ✅ Real balance tracking — shoulder-midpoint lateral sway ────────────────
# Distinct signal from hip-based stability above: tracks horizontal drift of
# the shoulder midpoint, which picks up upper-body swaying/compensation that
# hip jitter alone wouldn't catch.
_balance_lock = threading.Lock()
_shoulder_sway_buffer: "collections.deque" = collections.deque(maxlen=20)
_current_balance_score = 100.0

# ─── ✅ Real fatigue tracking — rep-quality decline over the session ──────────
# Each completed rep's peak angle (as % of target ROM) is logged. Fatigue is
# derived from how much rep quality has dropped in the second half of the
# session vs. the first half — a real physiological proxy (form degrades as
# the patient tires), not a random number or a copy of another metric.
_fatigue_lock = threading.Lock()
_rep_quality_buffer: "collections.deque" = collections.deque(maxlen=30)
_current_fatigue_score = 0.0
_last_fatigue_rep_count = 0


def _compute_primary_angle(angles: Dict[str, float], exercise_type: str) -> Optional[float]:
    """Pick the joint-angle relevant to the active exercise (mirrors frontend logic)."""
    ex = (exercise_type or "").lower()

    def avg(*vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    if "shoulder" in ex or "arm" in ex:
        return avg(angles.get("l_shoulder"), angles.get("r_shoulder"))
    if "elbow" in ex:
        return avg(angles.get("l_elbow"), angles.get("r_elbow"))
    if "knee" in ex or "squat" in ex or "leg" in ex:
        return avg(angles.get("l_knee"), angles.get("r_knee"))
    if "hip" in ex:
        return avg(angles.get("l_hip"), angles.get("r_hip"))
    return avg(angles.get("l_elbow"), angles.get("r_elbow"),
               angles.get("l_knee"), angles.get("r_knee"))


def _update_rep_count(primary_angle: Optional[float], target_rom: float) -> int:
    """Hysteresis-based rep counter: counts one rep per full down→up cycle."""
    global _rep_count, _rep_stage
    if primary_angle is None or target_rom <= 0:
        with _rep_lock:
            return _rep_count

    high_thresh = target_rom * 0.80
    low_thresh  = target_rom * 0.35

    with _rep_lock:
        if primary_angle >= high_thresh:
            if _rep_stage == "down":
                _rep_count += 1
            _rep_stage = "up"
        elif primary_angle <= low_thresh:
            _rep_stage = "down"
        return _rep_count


def _update_stability(landmarks) -> float:
    """Real stability score from hip-center landmark jitter over a short
    rolling window — steadier patient = less x/y variance = higher score."""
    global _current_stability_score
    try:
        hip_x = (landmarks[23].x + landmarks[24].x) / 2
        hip_y = (landmarks[23].y + landmarks[24].y) / 2
    except Exception:
        return _current_stability_score

    with _stability_lock:
        _landmark_jitter_buffer.append((hip_x, hip_y))
        if len(_landmark_jitter_buffer) >= 5:
            xs = [p[0] for p in _landmark_jitter_buffer]
            ys = [p[1] for p in _landmark_jitter_buffer]
            jitter = statistics.pstdev(xs) + statistics.pstdev(ys)
            # jitter is in normalized [0,1] frame coords; scale empirically to 0-100
            score = max(0.0, min(100.0, 100.0 - jitter * 4000))
            _current_stability_score = round(score, 1)
        return _current_stability_score


def _update_smoothness(primary_angle: Optional[float]) -> float:
    """Real smoothness score from frame-to-frame angular velocity variance.
    A shaky/jerky movement swings the primary angle around erratically
    frame-to-frame (high variance in the deltas); a controlled, smooth rep
    changes angle steadily (low variance). No randomness, no copy of
    accuracy — this is its own independent signal."""
    global _current_smoothness_score, _last_primary_angle
    if primary_angle is None:
        return _current_smoothness_score

    with _smoothness_lock:
        if _last_primary_angle is not None:
            delta = abs(primary_angle - _last_primary_angle)
            _angle_velocity_buffer.append(delta)
        _last_primary_angle = primary_angle

        if len(_angle_velocity_buffer) >= 5:
            jerk_variance = statistics.pstdev(_angle_velocity_buffer)
            # empirically scaled: ~0-2°/frame deltas (smooth) -> near 100,
            # large erratic swings -> drops toward 0
            score = max(0.0, min(100.0, 100.0 - jerk_variance * 8))
            _current_smoothness_score = round(score, 1)
        return _current_smoothness_score


def _update_balance(landmarks) -> float:
    """Real balance score from shoulder-midpoint horizontal sway over a
    rolling window — separate signal from the hip-based stability score
    above, since upper-body swaying/compensation shows up at the shoulders
    before it shows up at the hips."""
    global _current_balance_score
    try:
        sh_x = (landmarks[11].x + landmarks[12].x) / 2
    except Exception:
        return _current_balance_score

    with _balance_lock:
        _shoulder_sway_buffer.append(sh_x)
        if len(_shoulder_sway_buffer) >= 5:
            sway = statistics.pstdev(_shoulder_sway_buffer)
            # normalized [0,1] frame coords; scale empirically to 0-100
            score = max(0.0, min(100.0, 100.0 - sway * 5000))
            _current_balance_score = round(score, 1)
        return _current_balance_score


def _record_rep_quality(peak_angle: Optional[float], target_rom: float):
    """Called once per completed rep. Logs how close that rep's peak angle
    got to the target ROM, then derives fatigue from the drop-off between
    the first half and second half of the session's rep quality — real
    physiological signal (form degrades as the patient tires), not a
    random number or an inverted copy of the stability score."""
    global _current_fatigue_score
    if peak_angle is None or target_rom <= 0:
        return

    quality = max(0.0, min(100.0, (peak_angle / target_rom) * 100))
    with _fatigue_lock:
        _rep_quality_buffer.append(quality)
        n = len(_rep_quality_buffer)
        if n >= 4:
            half   = n // 2
            buf    = list(_rep_quality_buffer)
            early  = buf[:half]
            recent = buf[half:]
            early_avg  = sum(early)  / len(early)
            recent_avg = sum(recent) / len(recent)
            decline = max(0.0, early_avg - recent_avg)  # >0 if quality dropped
            # scale a 0-40pt quality drop across the session to 0-100 fatigue
            _current_fatigue_score = round(min(100.0, decline * 2.5), 1)


def _reset_rep_and_stability_state():
    global _rep_count, _rep_stage, _current_stability_score
    global _current_smoothness_score, _current_balance_score
    global _current_fatigue_score, _last_primary_angle, _last_fatigue_rep_count
    with _rep_lock:
        _rep_count = 0
        _rep_stage = None
    with _stability_lock:
        _landmark_jitter_buffer.clear()
        _current_stability_score = 100.0
    with _smoothness_lock:
        _angle_velocity_buffer.clear()
        _last_primary_angle = None
        _current_smoothness_score = 100.0
    with _balance_lock:
        _shoulder_sway_buffer.clear()
        _current_balance_score = 100.0
    with _fatigue_lock:
        _rep_quality_buffer.clear()
        _current_fatigue_score = 0.0
        _last_fatigue_rep_count = 0

# Bright, high-visibility drawing specs (default MediaPipe style is dim on
# a 320x240 frame) — used only as a fallback; primary drawing is now done
# manually per-exercise-type below (see _get_active_connections).
_MJPEG_LANDMARK_SPEC = (
    _mp_drawing.DrawingSpec(color=(0, 230, 120), thickness=1, circle_radius=3)
    if _mp_drawing else None
)
_MJPEG_CONNECTION_SPEC = (
    _mp_drawing.DrawingSpec(color=(255, 160, 0), thickness=1)
    if _mp_drawing else None
)

# ─── Exercise-type → active joints/lines filter ──────────────────────────────
# Instead of drawing the full 33-point skeleton, only draw the connections
# relevant to whichever exercise is currently selected on the session page.
_current_exercise_type = "Shoulder Rehab"
_exercise_lock = threading.Lock()

_LS, _RS = "left_shoulder", "right_shoulder"
_LE, _RE = "left_elbow", "right_elbow"
_LW, _RW = "left_wrist", "right_wrist"
_LH, _RH = "left_hip", "right_hip"
_LK, _RK = "left_knee", "right_knee"
_LA, _RA = "left_ankle", "right_ankle"


def _get_active_connections(exercise_type: str):
    """Return the small set of (point_a, point_b) name-pairs to draw for the
    currently selected exercise, instead of the full body skeleton."""
    ex = (exercise_type or "").lower()

    if "elbow" in ex:
        return [(_LS, _LE), (_LE, _LW), (_RS, _RE), (_RE, _RW)]
    if "hand" in ex:
        return [(_LE, _LW), (_RE, _RW)]
    if "ankle" in ex:
        return [(_LK, _LA), (_RK, _RA)]
    if "knee" in ex or "squat" in ex or "leg" in ex:
        return [(_LH, _LK), (_LK, _LA), (_RH, _RK), (_RK, _RA)]
    if "hip" in ex:
        return [(_LS, _LH), (_RS, _RH), (_LH, _LK), (_RH, _RK)]
    if "shoulder" in ex or "arm" in ex:
        return [(_LS, _LE), (_RS, _RE), (_LS, _LH), (_RS, _RH)]
    if "balance" in ex:
        return [
            (_LS, _RS), (_LS, _LH), (_RS, _RH), (_LH, _RH),
            (_LS, _LE), (_LE, _LW), (_RS, _RE), (_RE, _RW),
            (_LH, _LK), (_LK, _LA), (_RH, _RK), (_RK, _RA),
        ]
    # Fallback: arms + legs, no torso clutter
    return [(_LS, _LE), (_LE, _LW), (_RS, _RE), (_RE, _RW),
            (_LH, _LK), (_LK, _LA), (_RH, _RK), (_RK, _RA)]


def _draw_filtered_skeleton(frame, landmarks, exercise_type: str):
    """Draw only the joints/lines relevant to the active exercise, at a
    smaller, cleaner size than the MediaPipe default style."""
    h, w = frame.shape[:2]

    def _pt(name):
        idx = _KEY_LANDMARKS.get(name)
        if idx is None or idx >= len(landmarks):
            return None
        lmk = landmarks[idx]
        if lmk.visibility < 0.5:
            return None
        return (int(lmk.x * w), int(lmk.y * h))

    active_conns = _get_active_connections(exercise_type)
    active_points = set()
    for a, b in active_conns:
        active_points.add(a)
        active_points.add(b)

    # Lines first (so dots sit on top), thin + clean
    for a, b in active_conns:
        pa, pb = _pt(a), _pt(b)
        if pa and pb:
            cv2.line(frame, pa, pb, (255, 160, 0), 1, cv2.LINE_AA)

    # Small joint dots
    for name in active_points:
        p = _pt(name)
        if p:
            cv2.circle(frame, p, 3, (0, 230, 120), -1, cv2.LINE_AA)
            cv2.circle(frame, p, 3, (255, 255, 255), 1, cv2.LINE_AA)


import sys as _sys

def _mjpeg_open_camera() -> bool:
    """Open the MJPEG-pipeline camera device. Returns False (with logging) on failure."""
    global _mjpeg_cap
    with _mjpeg_lock:
        if _mjpeg_cap is not None and _mjpeg_cap.isOpened():
            return True

        # Windows: CAP_DSHOW is far more reliable than the default (MSMF) backend
        # for OpenCV running inside a FastAPI/uvicorn worker thread — MSMF can
        # silently hang or fail to open in this context even though it works
        # fine in a plain standalone script.
        if _sys.platform == "win32":
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            backend_name = "CAP_DSHOW"
        else:
            cap = cv2.VideoCapture(0)
            backend_name = "default"

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_FPS, 30)
        try:
            # Ask the driver to keep only 1 frame buffered (not all backends
            # honor this, but it helps on the ones that do — combined with
            # the reader-thread pattern below, it's belt-and-suspenders
            # against the buffered-delay problem).
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if not cap.isOpened():
            print(f"❌ MJPEG: could not open camera with backend={backend_name} "
                  f"(cv2.VideoCapture(0).isOpened() == False — no webcam device found, "
                  f"already in use by another process, or blocked by OS permissions)")
            cap.release()
            _mjpeg_cap = None
            return False

        # Confirm we can actually read a frame, not just that the handle opened
        ok, _test_frame = cap.read()
        if not ok:
            print(f"❌ MJPEG: camera opened (backend={backend_name}) but first read() failed")
            cap.release()
            _mjpeg_cap = None
            return False

        _mjpeg_cap = cap
        print(f"✅ MJPEG: camera opened (backend={backend_name}, 320x240 @30fps requested)")

        # Start the background reader thread that keeps _mjpeg_latest_frame
        # fresh — see the comment on the globals above for why this exists.
        global _mjpeg_reader_thread, _mjpeg_reader_running, _mjpeg_latest_frame, _mjpeg_frame_id
        _mjpeg_latest_frame = None
        _mjpeg_frame_id = 0
        _mjpeg_reader_running = True
        _mjpeg_reader_thread = threading.Thread(target=_mjpeg_reader_loop, daemon=True)
        _mjpeg_reader_thread.start()

        return True


def _mjpeg_reader_loop():
    """
    Runs in its own daemon thread for as long as the camera is open.
    Continuously reads frames as fast as the camera/driver produces them and
    always keeps only the LATEST one in the shared slot (_mjpeg_latest_frame),
    overwriting whatever was there before. This is what decouples capture
    speed from MediaPipe processing speed in gen_frames() and stops the
    display from falling further and further behind over a session.
    """
    global _mjpeg_latest_frame, _mjpeg_frame_id
    consecutive_failures = 0
    while _mjpeg_reader_running:
        with _mjpeg_lock:
            cap = _mjpeg_cap
        if cap is None:
            break
        ok, frame = cap.read()
        if not ok or frame is None:
            consecutive_failures += 1
            if consecutive_failures >= 10:
                print("⚠️  MJPEG reader: camera stopped returning frames, stopping reader thread")
                break
            time.sleep(0.01)
            continue
        consecutive_failures = 0
        with _mjpeg_frame_lock:
            _mjpeg_latest_frame = frame
            _mjpeg_frame_id += 1


def _mjpeg_release_camera():
    """Release the MJPEG-pipeline camera device and reset state."""
    global _mjpeg_cap, _mjpeg_active, _mjpeg_reader_running, _mjpeg_latest_frame
    _mjpeg_reader_running = False   # signal reader thread to stop
    with _mjpeg_lock:
        _mjpeg_active = False
        if _mjpeg_cap is not None:
            _mjpeg_cap.release()
            print("🛑 MJPEG: camera released")
        _mjpeg_cap = None
    _mjpeg_latest_frame = None
    _latest_pose_data["detected"] = False
    _latest_pose_data["angles"] = {}

def gen_frames():
    """
    Sync generator (pattern borrowed from the reference Flask app):
    open camera -> loop: read -> flip -> MediaPipe pose -> draw -> JPEG encode
    -> yield multipart chunk. Also updates the module-level _latest_pose_data
    dict each frame so /api/pose_data can serve it independently.
    """
    global _mjpeg_active

    if not _mjpeg_open_camera():
        # Yield one explanatory frame instead of just hanging/breaking silently
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(blank, "Camera unavailable", (15, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(blank, "Check device / permissions", (15, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", blank)
        if ok:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        return

    _mjpeg_active = True
    print("▶️  MJPEG: stream starting")

    global _last_fatigue_rep_count

    try:
        last_frame_id = -1
        while _mjpeg_active:
            with _mjpeg_lock:
                cap = _mjpeg_cap
            if cap is None:
                print("ℹ️  MJPEG: camera handle gone, stopping generator")
                break

            # Pull whatever's freshest from the reader thread instead of
            # calling cap.read() here directly — this is what prevents the
            # processing loop (MediaPipe is the slow part) from falling
            # behind and showing an increasingly stale/delayed frame.
            with _mjpeg_frame_lock:
                frame = _mjpeg_latest_frame
                frame_id = _mjpeg_frame_id

            if frame is None:
                time.sleep(0.01)   # camera just opened, reader hasn't produced a frame yet
                continue
            if frame_id == last_frame_id:
                time.sleep(0.005)  # no new frame since last loop — avoid reprocessing/duplicating it
                continue
            last_frame_id = frame_id
            frame = frame.copy()  # reader thread may overwrite the shared slot while we work on this one

            frame = cv2.flip(frame, 1)  # mirror, like the reference app

            results = None
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                if _pose is not None:
                    results = _pose.process(rgb)
                rgb.flags.writeable = True
            except Exception as e:
                print(f"⚠️  MJPEG: MediaPipe processing failed: {e}")
                results = None

            detected = bool(results and results.pose_landmarks)
            angles: Dict[str, float] = {}
            primary_angle: Optional[float] = None
            reps = 0
            with _rep_lock:
                stability = _current_stability_score
            with _smoothness_lock:
                smoothness = _current_smoothness_score
            with _balance_lock:
                balance = _current_balance_score
            with _fatigue_lock:
                fatigue = _current_fatigue_score

            if detected and _mp_drawing is not None:
                with _exercise_lock:
                    active_exercise = _current_exercise_type
                _draw_filtered_skeleton(frame, results.pose_landmarks.landmark, active_exercise)

                lm = results.pose_landmarks.landmark
                try:
                    if len(lm) > 16:
                        angles["l_elbow"] = round(_get_angle(lm[11], lm[13], lm[15]), 1)
                        angles["r_elbow"] = round(_get_angle(lm[12], lm[14], lm[16]), 1)
                    if len(lm) > 28:
                        angles["l_knee"] = round(_get_angle(lm[23], lm[25], lm[27]), 1)
                        angles["r_knee"] = round(_get_angle(lm[24], lm[26], lm[28]), 1)
                    if len(lm) > 26:
                        angles["l_hip"] = round(_get_angle(lm[11], lm[23], lm[25]), 1)
                        angles["r_hip"] = round(_get_angle(lm[12], lm[24], lm[26]), 1)
                    if len(lm) > 14:
                        # Shoulder flexion/abduction: hip → shoulder → elbow,
                        # measures how far the arm is raised relative to the torso.
                        angles["l_shoulder"] = round(_get_angle(lm[23], lm[11], lm[13]), 1)
                        angles["r_shoulder"] = round(_get_angle(lm[24], lm[12], lm[14]), 1)
                except Exception as e:
                    print(f"⚠️  MJPEG: angle calculation failed: {e}")

                # ✅ Real rep counting + stability + smoothness + balance
                # (no randomness, no duplicated/copied metrics)
                primary_angle = _compute_primary_angle(angles, active_exercise)
                reps       = _update_rep_count(primary_angle, _current_target_rom)
                stability  = _update_stability(lm)
                smoothness = _update_smoothness(primary_angle)
                balance    = _update_balance(lm)

                # ✅ Real fatigue — record rep quality the instant a new rep
                # is detected, then fatigue score reflects the actual
                # early-vs-recent quality trend for this session
                if reps > _last_fatigue_rep_count:
                    _record_rep_quality(primary_angle, _current_target_rom)
                    _last_fatigue_rep_count = reps
                with _fatigue_lock:
                    fatigue = _current_fatigue_score

            _latest_pose_data["detected"]      = detected
            _latest_pose_data["angles"]        = angles
            _latest_pose_data["ts"]            = datetime.datetime.now().isoformat()
            _latest_pose_data["reps"]          = reps
            _latest_pose_data["stability"]     = stability
            _latest_pose_data["smoothness"]    = smoothness
            _latest_pose_data["balance"]       = balance
            _latest_pose_data["fatigue"]       = fatigue
            _latest_pose_data["primary_angle"] = primary_angle

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                print("⚠️  MJPEG: JPEG encode failed, skipping this frame")
                continue

            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

    except GeneratorExit:
        # Client closed the <img> / navigated away
        print("ℹ️  MJPEG: client disconnected, closing generator")
    except Exception as e:
        print(f"❌ MJPEG: generator crashed unexpectedly: {e}")
    finally:
        _mjpeg_release_camera()
        print("🏁 MJPEG: stream ended, camera released")


@app.get("/video_feed")
async def video_feed():
    """Primary live camera feed with pose skeleton drawn server-side. <img src='/video_feed'>"""
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/pose_data")
async def api_pose_data():
    """Latest joint angles + detection flag, updated every frame by gen_frames()."""
    return JSONResponse(_latest_pose_data)


@app.post("/api/camera/stop")
async def api_camera_stop():
    """Explicitly release the MJPEG camera device (called on Stop / page unload)."""
    _mjpeg_release_camera()
    return JSONResponse({"success": True, "message": "Camera stopped"})


@app.post("/api/exercise_type")
async def set_exercise_type(payload: Dict[str, Any]):
    """Frontend calls this whenever the exercise dropdown (or target ROM)
    changes, so the MJPEG stream draws the right joints and rep-counts
    against the right threshold."""
    global _current_exercise_type, _current_target_rom
    ex = payload.get("exercise_type")
    if ex:
        with _exercise_lock:
            _current_exercise_type = ex
    rom = payload.get("target_rom")
    if rom is not None:
        try:
            _current_target_rom = float(rom)
        except (TypeError, ValueError):
            pass
    return JSONResponse({
        "success": True,
        "exercise_type": _current_exercise_type,
        "target_rom": _current_target_rom,
    })


@app.post("/api/session/reset")
async def api_session_reset():
    """Call this right before a session starts so rep count + stability
    buffer don't carry over stale data from a previous session/patient."""
    _reset_rep_and_stability_state()
    return JSONResponse({"success": True})


@app.get("/api/camera/status")
async def api_camera_status():
    """Quick status check — useful for frontend polling / debugging."""
    with _mjpeg_lock:
        running = _mjpeg_active and _mjpeg_cap is not None and _mjpeg_cap.isOpened()
    return JSONResponse({"active": running})


# ─── Startup seed ─────────────────────────────────────────────────────────────
# Off by default — set SEED_DEMO_DATA=true in the environment if you ever want
# the 3 sample patients + random demo sessions back (e.g. for a fresh demo).
@app.on_event("startup")
async def seed():
    if os.getenv("SEED_DEMO_DATA", "false").lower() != "true":
        return
    with get_db() as db:
        if db.query(Patient).count() > 0:
            return

        sample_patients = [
            Patient(name="John Smith",      age=45, gender="Male",   weight=82.5, height=178.0,
                    diagnosis="Rotator Cuff Tear",       affected_body_part="Right Shoulder",
                    doctor_name="Dr. Sarah Johnson",     therapist_name="Michael Brown",
                    phone="+1 (555) 123-4567",           email="john.smith@email.com"),
            Patient(name="Maria Garcia",    age=62, gender="Female", weight=68.0, height=165.0,
                    diagnosis="Knee Osteoarthritis",     affected_body_part="Left Knee",
                    doctor_name="Dr. Robert Chen",       therapist_name="Lisa Wong",
                    phone="+1 (555) 234-5678",           email="maria.garcia@email.com"),
            Patient(name="Robert Williams", age=38, gender="Male",   weight=90.0, height=183.0,
                    diagnosis="Lumbar Disc Herniation",  affected_body_part="Lower Back",
                    doctor_name="Dr. Emily Davis",       therapist_name="James Wilson",
                    phone="+1 (555) 345-6789",           email="robert.williams@email.com"),
        ]

        for p in sample_patients:
            db.add(p)
        db.flush()

        exercises = ["Shoulder Rehab", "Knee Flexion", "Arm Raise", "Balance Exercise"]
        joints    = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_knee", "right_knee"]

        for p in sample_patients:
            for _ in range(5):
                sess = SessionModel(
                    patient_id          = p.id,
                    exercise_type       = random.choice(exercises),
                    start_time          = datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 30)),
                    duration_seconds    = random.randint(180, 600),
                    total_reps          = random.randint(10, 20),
                    completed_reps      = random.randint(5, 18),
                    accuracy_percentage = random.uniform(60, 95),
                    average_rom         = random.uniform(40, 85),
                    incorrect_movements = random.randint(0, 5),
                    stability_score     = random.uniform(50, 90),
                    balance_score       = random.uniform(45, 85),
                    movement_smoothness = random.uniform(50, 90),
                    fatigue_estimation  = random.uniform(10, 40),
                    recovery_score      = random.uniform(40, 80),
                )
                db.add(sess)
                db.flush()

                for joint in joints:
                    db.add(JointAngle(
                        session_id   = sess.id,
                        joint_name   = joint,
                        angle_value  = random.uniform(30, 120),
                        target_angle = random.uniform(40, 110),
                        deviation    = random.uniform(-15, 15),
                        is_correct   = random.choice([True, True, True, False]),
                    ))

        for key, val, desc in [
            ("fps_target",           "10",      "Target FPS for pose streaming"),
            ("camera_resolution",    "320x240", "Camera resolution"),
            ("confidence_threshold", "0.5",     "Min confidence threshold"),
            ("rom_warning",          "30",      "Min ROM warning threshold"),
        ]:
            db.add(Setting(key=key, value=val, description=desc))

        db.commit()
        print("✅ Seed data inserted.")

# ─── API: Patients ────────────────────────────────────────────────────────────
@app.get("/api/patients")
async def list_patients(request: Request):
    search = request.query_params.get("search", "").strip()
    include_inactive = request.query_params.get("include_inactive", "false").lower() == "true"
    with get_db() as db:
        q = db.query(Patient)
        if not include_inactive:
            q = q.filter(Patient.is_active == True)
        if search:
            q = q.filter(
                Patient.name.contains(search)
                | Patient.id.contains(search)
                | Patient.phone.contains(search)
            )
        patients = q.order_by(Patient.date_created.desc()).all()
        out = []
        for p in patients:
            n = len(p.sessions)
            avg_acc = (sum(s.accuracy_percentage for s in p.sessions) / n) if n else 0.0
            out.append({
                "id":                p.id,
                "name":              p.name,
                "age":               p.age,
                "gender":            p.gender,
                "diagnosis":         p.diagnosis,
                "affected_body_part": p.affected_body_part,
                "therapist_name":    p.therapist_name,
                "date_created":      p.date_created.isoformat(),
                "is_active":         p.is_active,
                "sessions_count":    n,
                "avg_accuracy":      round(avg_acc, 1),
                "photo":             base64.b64encode(p.photo).decode() if p.photo else None,
                "email":             p.email,
                "phone":             p.phone,
            })
        return JSONResponse(out)


@app.post("/api/patients")
async def create_patient(
    name: str = Form(...), age: int = Form(...), gender: str = Form(...),
    weight: float = Form(None), height: float = Form(None),
    diagnosis: str = Form(None), affected_body_part: str = Form(None),
    doctor_name: str = Form(None), therapist_name: str = Form(None),
    phone: str = Form(None), email: str = Form(None),
    medical_history: str = Form(None), previous_injury: str = Form(None),
    current_treatment: str = Form(None), exercise_plan: str = Form(None),
    photo: UploadFile = File(None),
):
    with get_db() as db:
        if email and db.query(Patient).filter(Patient.email == email).first():
            raise HTTPException(400, "Email already registered")
        if phone and db.query(Patient).filter(Patient.phone == phone).first():
            raise HTTPException(400, "Phone number already registered")
        photo_data = compress_photo(await photo.read()) if photo and photo.filename else None
        p = Patient(
            name=name, age=age, gender=gender, weight=weight, height=height,
            diagnosis=diagnosis, affected_body_part=affected_body_part,
            doctor_name=doctor_name, therapist_name=therapist_name,
            phone=phone, email=email, medical_history=medical_history,
            previous_injury=previous_injury, current_treatment=current_treatment,
            exercise_plan=exercise_plan, photo=photo_data,
        )
        db.add(p)
        db.flush()
        db.add(History(patient_id=p.id, action="Patient Created", details=f"{p.name} registered"))
        db.commit()
        return JSONResponse({"success": True, "message": "Patient created", "patient_id": p.id})


@app.get("/api/patients/{patient_id}")
async def get_patient(patient_id: str):
    with get_db() as db:
        p = db.query(Patient).filter(Patient.id == patient_id).first()
        if not p:
            raise HTTPException(404, "Patient not found")
        return JSONResponse({
            "id": p.id, "name": p.name, "age": p.age, "gender": p.gender,
            "weight": p.weight, "height": p.height, "diagnosis": p.diagnosis,
            "affected_body_part": p.affected_body_part, "doctor_name": p.doctor_name,
            "therapist_name": p.therapist_name, "phone": p.phone, "email": p.email,
            "medical_history": p.medical_history, "previous_injury": p.previous_injury,
            "current_treatment": p.current_treatment, "exercise_plan": p.exercise_plan,
            "photo": base64.b64encode(p.photo).decode() if p.photo else None,
            "date_created": p.date_created.isoformat(), "is_active": p.is_active,
            "sessions": [session_summary(s) for s in p.sessions],
        })


@app.put("/api/patients/{patient_id}")
async def update_patient(
    patient_id: str,
    name: str = Form(...), age: int = Form(...), gender: str = Form(...),
    weight: float = Form(None), height: float = Form(None),
    diagnosis: str = Form(None), affected_body_part: str = Form(None),
    doctor_name: str = Form(None), therapist_name: str = Form(None),
    phone: str = Form(None), email: str = Form(None),
    medical_history: str = Form(None), previous_injury: str = Form(None),
    current_treatment: str = Form(None), exercise_plan: str = Form(None),
    is_active: bool = Form(True), photo: UploadFile = File(None),
):
    with get_db() as db:
        p = db.query(Patient).filter(Patient.id == patient_id).first()
        if not p:
            raise HTTPException(404, "Patient not found")
        for field, val in [
            ("name", name), ("age", age), ("gender", gender), ("weight", weight),
            ("height", height), ("diagnosis", diagnosis), ("affected_body_part", affected_body_part),
            ("doctor_name", doctor_name), ("therapist_name", therapist_name),
            ("phone", phone), ("email", email), ("medical_history", medical_history),
            ("previous_injury", previous_injury), ("current_treatment", current_treatment),
            ("exercise_plan", exercise_plan), ("is_active", is_active),
        ]:
            setattr(p, field, val)
        if photo and photo.filename:
            p.photo = compress_photo(await photo.read())
        db.add(History(patient_id=patient_id, action="Patient Updated", details=f"{p.name} updated"))
        db.commit()
        return JSONResponse({"success": True, "message": "Patient updated"})


@app.delete("/api/patients/{patient_id}")
async def delete_patient(patient_id: str):
    with get_db() as db:
        p = db.query(Patient).filter(Patient.id == patient_id).first()
        if not p:
            raise HTTPException(404, "Patient not found")
        p.is_active = False
        db.add(History(patient_id=patient_id, action="Patient Deactivated", details=f"{p.name} deactivated"))
        db.commit()
        return JSONResponse({"success": True, "message": "Patient deactivated"})

# ─── API: Sessions ────────────────────────────────────────────────────────────
@app.get("/api/sessions/{patient_id}")
async def get_sessions(patient_id: str):
    with get_db() as db:
        if not db.query(Patient).filter(Patient.id == patient_id).first():
            raise HTTPException(404, "Patient not found")
        sessions = (
            db.query(SessionModel)
            .filter(SessionModel.patient_id == patient_id)
            .order_by(SessionModel.start_time.desc())
            .all()
        )
        out = []
        for s in sessions:
            out.append({
                "session_id":          s.id,
                "patient_id":          s.patient_id,
                "exercise_type":       s.exercise_type,
                "start_time":          s.start_time.isoformat(),
                "end_time":            s.end_time.isoformat() if s.end_time else None,
                "duration_seconds":    s.duration_seconds,
                "total_reps":          s.total_reps,
                "completed_reps":      s.completed_reps,
                "accuracy_percentage": s.accuracy_percentage,
                "average_rom":         s.average_rom,
                "incorrect_movements": s.incorrect_movements,
                "stability_score":     s.stability_score,
                "balance_score":       s.balance_score,
                "movement_smoothness": s.movement_smoothness,
                "fatigue_estimation":  s.fatigue_estimation,
                "recovery_score":      s.recovery_score,
                "joint_angles": [
                    {"joint_name": ja.joint_name, "angle": ja.angle_value,
                     "target": ja.target_angle, "is_correct": ja.is_correct}
                    for ja in s.joint_angles[:20]
                ],
            })
        return JSONResponse(out)


@app.post("/api/sessions")
async def save_session(payload: Dict[str, Any]):
    with get_db() as db:
        pid = payload.get("patient_id")
        if not db.query(Patient).filter(Patient.id == pid).first():
            raise HTTPException(404, "Patient not found")

        def _dt(key):
            v = payload.get(key)
            if not v:
                return None
            # JS toISOString() emits a trailing 'Z', which Python 3.10's
            # fromisoformat() can't parse directly (only 3.11+ supports it).
            if v.endswith("Z"):
                v = v[:-1] + "+00:00"
            return datetime.datetime.fromisoformat(v)

        sess = SessionModel(
            patient_id          = pid,
            exercise_type       = payload.get("exercise_type", "General Exercise"),
            start_time          = _dt("start_time") or datetime.datetime.now(),
            end_time            = _dt("end_time"),
            duration_seconds    = payload.get("duration_seconds", 0),
            total_reps          = payload.get("total_reps", 0),
            completed_reps      = payload.get("completed_reps", 0),
            accuracy_percentage = payload.get("accuracy_percentage", 0.0),
            average_rom         = payload.get("average_rom", 0.0),
            incorrect_movements = payload.get("incorrect_movements", 0),
            stability_score     = payload.get("stability_score", 0.0),
            balance_score       = payload.get("balance_score", 0.0),
            movement_smoothness = payload.get("movement_smoothness", 0.0),
            fatigue_estimation  = payload.get("fatigue_estimation", 0.0),
            recovery_score      = payload.get("recovery_score", 0.0),
            session_data        = payload.get("session_data", {}),
        )
        db.add(sess)
        db.flush()

        for ja in payload.get("joint_angles", []):
            db.add(JointAngle(
                session_id   = sess.id,
                joint_name   = ja.get("joint_name", "Unknown"),
                angle_value  = ja.get("angle_value", 0.0),
                target_angle = ja.get("target_angle"),
                deviation    = ja.get("deviation"),
                is_correct   = ja.get("is_correct", True),
            ))

        for er in payload.get("exercise_results", []):
            db.add(ExerciseResult(
                session_id         = sess.id,
                exercise_name      = er.get("exercise_name", "Unknown"),
                repetition_number  = er.get("repetition_number", 0),
                accuracy           = er.get("accuracy", 0.0),
                rom_achieved       = er.get("rom_achieved", 0.0),
                speed              = er.get("speed", 0.0),
                hold_duration      = er.get("hold_duration", 0.0),
                compensation_score = er.get("compensation_score", 0.0),
                is_completed       = er.get("is_completed", False),
                feedback           = er.get("feedback", ""),
            ))

        db.add(History(
            patient_id = pid,
            action     = "Session Saved",
            details    = f"Session {sess.id} — {sess.completed_reps} reps",
        ))
        db.commit()
        return JSONResponse({"success": True, "message": "Session saved", "session_id": sess.id})

# ─── API: Dashboard ───────────────────────────────────────────────────────────
@app.get("/api/dashboard")
async def dashboard():
    with get_db() as db:
        total_patients = db.query(Patient).filter(Patient.is_active == True).count()
        total_sessions = db.query(SessionModel).count()

        today     = datetime.date.today()
        today_min = datetime.datetime.combine(today, datetime.time.min)
        today_max = datetime.datetime.combine(today, datetime.time.max)
        today_sessions = (
            db.query(SessionModel)
            .filter(SessionModel.start_time.between(today_min, today_max))
            .count()
        )

        all_sessions = db.query(SessionModel).all()
        n = len(all_sessions)

        def _avg(attr):
            return sum(getattr(s, attr) or 0 for s in all_sessions) / n if n else 0

        patients       = db.query(Patient).filter(Patient.is_active == True).all()
        patient_scores = [calculate_recovery_score(p.sessions) for p in patients]
        recovery_score = sum(patient_scores) / len(patient_scores) if patient_scores else 0

        recent = db.query(SessionModel).order_by(SessionModel.start_time.desc()).limit(10).all()
        progress = [
            {
                "date":      s.start_time.strftime("%Y-%m-%d"),
                "accuracy":  s.accuracy_percentage,
                "rom":       s.average_rom,
                "stability": s.stability_score or 0,
                "balance":   s.balance_score or 0,
                "exercise":  s.exercise_type,
            }
            for s in recent
        ]

        def _period(days):
            cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
            ss = [s for s in all_sessions if s.start_time >= cutoff]
            m  = len(ss)
            return {
                "sessions": m,
                "accuracy": round(sum(s.accuracy_percentage for s in ss) / m if m else 0, 1),
                "rom":      round(sum(s.average_rom for s in ss) / m if m else 0, 1),
            }

        return JSONResponse({
            "statistics": {
                "total_patients":      total_patients,
                "total_sessions":      total_sessions,
                "today_sessions":      today_sessions,
                "completed_exercises": sum(s.completed_reps for s in all_sessions),
                "avg_accuracy":        round(_avg("accuracy_percentage"), 1),
                "avg_rom":             round(_avg("average_rom"), 1),
                "incorrect_movements": sum(s.incorrect_movements for s in all_sessions),
                "recovery_score":      round(recovery_score, 1),
                "avg_stability":       round(_avg("stability_score"), 1),
                "avg_balance":         round(_avg("balance_score"), 1),
                "avg_smoothness":      round(_avg("movement_smoothness"), 1),
            },
            "progress": progress,
            "weekly":   _period(7),
            "monthly":  _period(30),
        })

# ─── API: Analytics (today vs previous day comparison) ────────────────────────
@app.get("/api/analytics")
async def analytics(patient_id: Optional[str] = None):
    with get_db() as db:
        q = db.query(SessionModel)
        if patient_id:
            if not db.query(Patient.id).filter(Patient.id == patient_id).first():
                raise HTTPException(404, "Patient not found")
            q = q.filter(SessionModel.patient_id == patient_id)
        sessions = q.order_by(SessionModel.start_time).all()

        today      = datetime.date.today()
        yesterday  = today - datetime.timedelta(days=1)

        def _day_range(d):
            return (
                datetime.datetime.combine(d, datetime.time.min),
                datetime.datetime.combine(d, datetime.time.max),
            )

        def _bucket(d):
            lo, hi = _day_range(d)
            return [s for s in sessions if lo <= s.start_time <= hi]

        def _summary(bucket):
            n = len(bucket)
            avg = lambda attr: round(sum(getattr(s, attr) or 0 for s in bucket) / n, 1) if n else 0.0
            return {
                "sessions":       n,
                "total_reps":     sum(s.completed_reps or 0 for s in bucket),
                "avg_accuracy":   avg("accuracy_percentage"),
                "avg_rom":        avg("average_rom"),
                "avg_stability":  avg("stability_score"),
                "avg_balance":    avg("balance_score"),
                "avg_smoothness": avg("movement_smoothness"),
                "recovery_score": round(calculate_recovery_score(bucket), 1),
                "incorrect_movements": sum(s.incorrect_movements or 0 for s in bucket),
            }

        today_bucket     = _bucket(today)
        yesterday_bucket = _bucket(yesterday)
        today_sum        = _summary(today_bucket)
        yesterday_sum    = _summary(yesterday_bucket)

        def _delta(key):
            t, y = today_sum[key], yesterday_sum[key]
            if y == 0:
                return None if t == 0 else 100.0
            return round(((t - y) / y) * 100, 1)

        deltas = {
            key: _delta(key)
            for key in ("sessions", "total_reps", "avg_accuracy", "avg_rom",
                        "avg_stability", "avg_balance", "recovery_score")
        }

        # Last 7 days trend, for context below the today-vs-yesterday cards
        trend = []
        for i in range(6, -1, -1):
            d = today - datetime.timedelta(days=i)
            b = _bucket(d)
            s = _summary(b)
            trend.append({"date": d.isoformat(), **s})

        return JSONResponse({
            "today":     {"date": today.isoformat(),     **today_sum},
            "yesterday": {"date": yesterday.isoformat(), **yesterday_sum},
            "deltas":    deltas,
            "trend":     trend,
            "today_sessions":     [session_summary(s) for s in today_bucket],
            "yesterday_sessions": [session_summary(s) for s in yesterday_bucket],
        })

# ─── API: Reports (isolated from WS — runs in thread pool via asyncio) ────────
@app.get("/api/reports/{patient_id}")
async def generate_report(patient_id: str, report_type: str = "weekly"):
    """
    PDF generation runs synchronously in a thread pool executor so it
    does NOT block the event loop while WebSocket streaming is active.
    """
    import asyncio

    # Pre-check so the user gets a clean JSON error instead of a raw 404
    # blank tab when the patient doesn't exist / has no sessions yet.
    with get_db() as db:
        p = db.query(Patient).filter(Patient.id == patient_id).first()
        if not p:
            raise HTTPException(404, "Patient not found")
        has_sessions = (
            db.query(SessionModel.id)
            .filter(SessionModel.patient_id == patient_id)
            .first()
            is not None
        )
        if not has_sessions:
            raise HTTPException(404, "No sessions found for this patient yet — complete a session first")

    loop = asyncio.get_event_loop()
    filepath = await loop.run_in_executor(
        None, _build_report_sync, patient_id, report_type
    )
    return FileResponse(
        filepath,
        media_type="application/pdf",
        filename=f"report_{patient_id}_{report_type}.pdf",
    )


@app.get("/api/reports/history/{patient_id}")
async def report_history(patient_id: str):
    """List previously generated reports for a patient, newest first."""
    with get_db() as db:
        if not db.query(Patient.id).filter(Patient.id == patient_id).first():
            raise HTTPException(404, "Patient not found")
        rows = (
            db.query(Report)
            .filter(Report.patient_id == patient_id)
            .order_by(Report.generated_date.desc())
            .all()
        )
        out = []
        for r in rows:
            out.append({
                "id":             r.id,
                "report_type":    r.report_type,
                "generated_date": r.generated_date.isoformat(),
                "available":      bool(r.file_path and os.path.exists(r.file_path)),
            })
        return JSONResponse(out)


@app.get("/api/reports/file/{report_id}")
async def download_report_file(report_id: int):
    """Download a specific, already-generated report by its Report row id."""
    with get_db() as db:
        r = db.query(Report).filter(Report.id == report_id).first()
        if not r:
            raise HTTPException(404, "Report record not found")
        if not r.file_path or not os.path.exists(r.file_path):
            raise HTTPException(410, "This report file is no longer available — please regenerate it")
        return FileResponse(
            r.file_path,
            media_type="application/pdf",
            filename=f"report_{r.patient_id}_{r.report_type}.pdf",
        )


def _build_report_sync(patient_id: str, report_type: str) -> str:
    """Pure sync function — safe to run in executor alongside async WS loop."""
    with get_db() as db:
        p = db.query(Patient).filter(Patient.id == patient_id).first()
        if not p:
            raise HTTPException(404, "Patient not found")

        sessions = (
            db.query(SessionModel)
            .filter(SessionModel.patient_id == patient_id)
            .order_by(SessionModel.start_time)
            .all()
        )
        if not sessions:
            raise HTTPException(404, "No sessions found for this patient")

        report_dir = f"reports/{patient_id}"
        os.makedirs(report_dir, exist_ok=True)
        ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(report_dir, f"{patient_id}_{report_type}_{ts}.pdf")

        doc    = SimpleDocTemplate(filepath, pagesize=A4,
                                   rightMargin=72, leftMargin=72,
                                   topMargin=72,  bottomMargin=18)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("T", parent=styles["Heading1"],
                                     fontSize=22, spaceAfter=20, alignment=TA_CENTER)
        h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14, spaceAfter=10)

        n           = len(sessions)
        avg_acc     = sum(s.accuracy_percentage for s in sessions) / n
        avg_rom     = sum(s.average_rom for s in sessions) / n
        total_reps  = sum(s.completed_reps for s in sessions)
        rec_score   = calculate_recovery_score(sessions)
        improvement = calculate_improvement(sessions)

        story = [
            Paragraph("Rehabilitation AI System — Medical Report", title_style),
            Spacer(1, 0.2 * inch),
            Paragraph(f"<b>Patient:</b> {p.name} &nbsp; <b>ID:</b> {p.id}", styles["Normal"]),
            Paragraph(f"<b>Age:</b> {p.age} | <b>Gender:</b> {p.gender}", styles["Normal"]),
            Paragraph(f"<b>Diagnosis:</b> {p.diagnosis or '—'}", styles["Normal"]),
            Paragraph(f"<b>Affected area:</b> {p.affected_body_part or '—'}", styles["Normal"]),
            Paragraph(f"<b>Therapist:</b> {p.therapist_name or '—'}", styles["Normal"]),
            Spacer(1, 0.2 * inch),
            Paragraph("Session Summary (last 10)", h2),
        ]

        tbl_data = [["Session", "Date", "Exercise", "Accuracy", "ROM", "Reps", "Stability"]]
        for s in sessions[-10:]:
            tbl_data.append([
                s.id[:8],
                s.start_time.strftime("%Y-%m-%d"),
                s.exercise_type[:20],
                f"{s.accuracy_percentage:.1f}%",
                f"{s.average_rom:.1f}°",
                str(s.completed_reps),
                f"{s.stability_score:.1f}" if s.stability_score else "N/A",
            ])

        col_w = [1.1*inch, 1.2*inch, 1.5*inch, 0.9*inch, 0.8*inch, 0.7*inch, 0.9*inch]
        tbl = Table(tbl_data, colWidths=col_w)
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F3F4")]),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story += [tbl, Spacer(1, 0.2 * inch), Paragraph("Statistical Analysis", h2)]

        stats_tbl = Table(
            [
                ["Metric",         "Value"],
                ["Total Sessions", str(n)],
                ["Avg Accuracy",   f"{avg_acc:.1f}%"],
                ["Avg ROM",        f"{avg_rom:.1f}°"],
                ["Total Reps",     str(total_reps)],
                ["Recovery Score", f"{rec_score:.1f}%"],
                ["Improvement",    f"{improvement:+.1f}%"],
            ],
            colWidths=[2.5*inch, 2*inch],
        )
        stats_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F3F4")]),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story += [stats_tbl, Spacer(1, 0.2 * inch), Paragraph("AI Recommendations", h2)]

        recs = []
        if avg_acc  < 70:  recs.append("• Focus on improving movement accuracy — consider slower, controlled repetitions.")
        if avg_rom  < 60:  recs.append("• Work on increasing range of motion with gentle stretching before sessions.")
        if rec_score < 50: recs.append("• Continue therapy with increased frequency (3–4 sessions/week recommended).")
        if n < 5:          recs.append("• Consistent practice is key — aim for at least 10 sessions before re-evaluation.")
        if avg_acc >= 85 and avg_rom >= 80:
            recs.append("• Excellent progress! Consider introducing advanced functional exercises.")
        elif avg_acc >= 70 and avg_rom >= 70:
            recs.append("• Good progress — maintain current routine and gradually increase intensity.")
        if improvement > 15:
            recs.append(f"• Strong improvement trend ({improvement:+.1f}%) — keep up the momentum.")
        if not recs:
            recs = ["• Continue current therapy plan.", "• Regular monitoring is recommended."]

        for r in recs:
            story.append(Paragraph(r, styles["Normal"]))

        story += [
            Spacer(1, 0.3 * inch),
            Paragraph(f"<b>Generated:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]),
            Spacer(1, 0.3 * inch),
            Paragraph("<b>Therapist Signature:</b> _________________________", styles["Normal"]),
            Paragraph("<b>Date:</b> _________________________", styles["Normal"]),
        ]

        doc.build(story)

        db.add(Report(patient_id=patient_id, report_type=report_type, file_path=filepath))
        db.commit()

    return filepath

# ─── HTML routes ──────────────────────────────────────────────────────────────
@app.get("/",          response_class=HTMLResponse)
async def page_index(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.get("/dashboard",  response_class=HTMLResponse)
async def page_dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html")

@app.get("/patients",   response_class=HTMLResponse)
async def page_patients(request: Request):
    return templates.TemplateResponse(request, "patients.html")

@app.get("/session",    response_class=HTMLResponse)
async def page_session(request: Request):
    return templates.TemplateResponse(request, "session.html")

@app.get("/reports",    response_class=HTMLResponse)
async def page_reports(request: Request):
    return templates.TemplateResponse(request, "reports.html")

@app.get("/analytics",  response_class=HTMLResponse)
async def page_analytics(request: Request):
    return templates.TemplateResponse(request, "analytics.html")

# ─── Health check ─────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    with _mjpeg_lock:
        mjpeg_running = _mjpeg_active and _mjpeg_cap is not None and _mjpeg_cap.isOpened()
    return {
        "status":          "healthy",
        "timestamp":       datetime.datetime.now().isoformat(),
        "camera_running":  camera.is_running,
        "mjpeg_running":   mjpeg_running,
        "ws_connections":  len(ws_mgr.connections),
        "mediapipe_ready": _pose is not None,
        "target_fps":      CameraManager.TARGET_FPS,
        "resolution":      "320x240",
    }

# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # reload=False here on purpose: running `python app.py` loads this file
    # as module "__main__". If reload=True, uvicorn re-imports the file a
    # SECOND time under the name "app" to power its file-watcher, which
    # redefines every SQLAlchemy model on the same Base.metadata and crashes
    # with "Table 'patients' is already defined for this MetaData instance."
    #
    # For live-reload during development, run this from the terminal instead:
    #   uvicorn app:app --reload --host 0.0.0.0 --port 8000
    # (that command imports the file once, under the name "app", so there's
    # no double-definition.)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)