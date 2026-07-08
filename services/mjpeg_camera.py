"""
Primary live camera pipeline for the session page (<img src="/video_feed">).
Simpler + more reliable than the WebSocket pipeline in camera_ws.py: the
browser <img> tag decodes the multipart MJPEG stream natively, no JS
frame-decoding / canvas loop needed. Runs as a sync generator (FastAPI runs
sync generators in a thread pool), so it does not block the asyncio event
loop used by the legacy /ws/pose route.
"""
import sys
import time
import threading
import datetime
from typing import Dict, Any, Optional

import cv2
import numpy as np

from config import pose, mp_drawing, KEY_LANDMARKS, get_angle
from services import metrics

# Bright, high-visibility drawing specs (default MediaPipe style is dim on a
# 320x240 frame) — used only as a fallback; primary drawing is done manually
# per-exercise-type via _draw_filtered_skeleton() below.
_MJPEG_LANDMARK_SPEC = (
    mp_drawing.DrawingSpec(color=(0, 230, 120), thickness=1, circle_radius=3)
    if mp_drawing else None
)
_MJPEG_CONNECTION_SPEC = (
    mp_drawing.DrawingSpec(color=(255, 160, 0), thickness=1)
    if mp_drawing else None
)

# Exercise-type → active joints/lines filter
# Instead of drawing the full 33-point skeleton, only draw the connections
# relevant to whichever exercise is currently selected on the session page.
_LS, _RS = "left_shoulder", "right_shoulder"
_LE, _RE = "left_elbow", "right_elbow"
_LW, _RW = "left_wrist", "right_wrist"
_LH, _RH = "left_hip", "right_hip"
_LK, _RK = "left_knee", "right_knee"
_LA, _RA = "left_ankle", "right_ankle"

# Camera device state 
_mjpeg_cap: Optional[cv2.VideoCapture] = None
_mjpeg_active = False
_mjpeg_lock = threading.Lock()

# Background frame reader (fixes growing camera delay)
_mjpeg_latest_frame = None
_mjpeg_frame_id = 0
_mjpeg_frame_lock = threading.Lock()
_mjpeg_reader_thread: Optional[threading.Thread] = None
_mjpeg_reader_running = False

latest_pose_data: Dict[str, Any] = {
    "detected": False, "angles": {}, "ts": None,
    "reps": 0, "stability": 100.0, "primary_angle": None,
    "smoothness": 100.0, "balance": 100.0, "fatigue": 0.0,
}


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


def _relevant_landmarks_visible(landmarks, exercise_type: str) -> bool:
    active_conns = _get_active_connections(exercise_type)
    active_points = set()
    for a, b in active_conns:
        active_points.add(a)
        active_points.add(b)
    if not active_points:
        return True

    visible_count = 0
    for name in active_points:
        idx = KEY_LANDMARKS.get(name)
        if idx is not None and idx < len(landmarks) and landmarks[idx].visibility > 0.5:
            visible_count += 1

    required = max(1, (len(active_points) + 1) // 2)  # majority, rounded up
    return visible_count >= required


def _draw_filtered_skeleton(frame, landmarks, exercise_type: str):
    """Draw only the joints/lines relevant to the active exercise, at a
    smaller, cleaner size than the MediaPipe default style."""
    h, w = frame.shape[:2]

    def _pt(name):
        idx = KEY_LANDMARKS.get(name)
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
        if sys.platform == "win32":
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
            print(f"MJPEG: could not open camera with backend={backend_name} "
                  f"(cv2.VideoCapture(0).isOpened() == False — no webcam device found, "
                  f"already in use by another process, or blocked by OS permissions)")
            cap.release()
            _mjpeg_cap = None
            return False

        # Confirm we can actually read a frame, not just that the handle opened
        ok, _test_frame = cap.read()
        if not ok:
            print(f"MJPEG: camera opened (backend={backend_name}) but first read() failed")
            cap.release()
            _mjpeg_cap = None
            return False

        _mjpeg_cap = cap
        print(f"MJPEG: camera opened (backend={backend_name}, 320x240 @30fps requested)")

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
                print("MJPEG reader: camera stopped returning frames, stopping reader thread")
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
            print("MJPEG: camera released")
        _mjpeg_cap = None
    _mjpeg_latest_frame = None
    latest_pose_data["detected"] = False
    latest_pose_data["angles"] = {}


def gen_frames():
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
    print("MJPEG: stream starting")

    try:
        last_frame_id = -1
        while _mjpeg_active:
            with _mjpeg_lock:
                cap = _mjpeg_cap
            if cap is None:
                print("MJPEG: camera handle gone, stopping generator")
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
                if pose is not None:
                    results = pose.process(rgb)
                rgb.flags.writeable = True
            except Exception as e:
                print(f"MJPEG: MediaPipe processing failed: {e}")
                results = None

            person_found = bool(results and results.pose_landmarks)
            active_exercise, target_rom = metrics.get_exercise_state()

            # "detected" now means "the joints THIS exercise needs are
            # visible" rather than "MediaPipe found some person somewhere
            # in frame" — so an Elbow exercise no longer needs the
            # patient's legs in shot to register as connected.
            detected = person_found and _relevant_landmarks_visible(
                results.pose_landmarks.landmark, active_exercise
            )
            angles: Dict[str, float] = {}
            primary_angle: Optional[float] = None
            reps = 0
            stability  = metrics.get_stability()
            smoothness = metrics.get_smoothness()
            balance    = metrics.get_balance()
            fatigue    = metrics.get_current_fatigue()

            if person_found and mp_drawing is not None:
                # Skeleton is still drawn whenever MediaPipe found a person
                # at all, even if the exercise-relevant joints aren't all
                # visible yet — the doctor can see the patient adjusting
                # into frame instead of a blank video.
                _draw_filtered_skeleton(frame, results.pose_landmarks.landmark, active_exercise)

                lm = results.pose_landmarks.landmark
                try:
                    if len(lm) > 16:
                        angles["l_elbow"] = round(get_angle(lm[11], lm[13], lm[15]), 1)
                        angles["r_elbow"] = round(get_angle(lm[12], lm[14], lm[16]), 1)
                    if len(lm) > 28:
                        angles["l_knee"] = round(get_angle(lm[23], lm[25], lm[27]), 1)
                        angles["r_knee"] = round(get_angle(lm[24], lm[26], lm[28]), 1)
                    if len(lm) > 26:
                        angles["l_hip"] = round(get_angle(lm[11], lm[23], lm[25]), 1)
                        angles["r_hip"] = round(get_angle(lm[12], lm[24], lm[26]), 1)
                    if len(lm) > 14:
                        # Shoulder flexion/abduction: hip → shoulder → elbow,
                        # measures how far the arm is raised relative to the torso.
                        angles["l_shoulder"] = round(get_angle(lm[23], lm[11], lm[13]), 1)
                        angles["r_shoulder"] = round(get_angle(lm[24], lm[12], lm[14]), 1)
                except Exception as e:
                    print(f"MJPEG: angle calculation failed: {e}")

                # Real rep counting + stability + smoothness + balance
                # (no randomness, no duplicated/copied metrics)
                primary_angle = metrics.compute_primary_angle(angles, active_exercise)
                reps       = metrics.update_rep_count(primary_angle, target_rom)
                stability  = metrics.update_stability(lm)
                smoothness = metrics.update_smoothness(primary_angle)
                balance    = metrics.update_balance(lm)

                # Real fatigue — record rep quality the instant a new rep
                # is detected, then fatigue score reflects the actual
                # early-vs-recent quality trend for this session
                fatigue = metrics.maybe_record_rep_quality(reps, primary_angle, target_rom)

            latest_pose_data["detected"]      = detected
            latest_pose_data["angles"]        = angles
            latest_pose_data["ts"]            = datetime.datetime.now().isoformat()
            latest_pose_data["reps"]          = reps
            latest_pose_data["stability"]     = stability
            latest_pose_data["smoothness"]    = smoothness
            latest_pose_data["balance"]       = balance
            latest_pose_data["fatigue"]       = fatigue
            latest_pose_data["primary_angle"] = primary_angle

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                print("MJPEG: JPEG encode failed, skipping this frame")
                continue

            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

    except GeneratorExit:
        # Client closed the <img> / navigated away
        print("MJPEG: client disconnected, closing generator")
    except Exception as e:
        print(f"MJPEG: generator crashed unexpectedly: {e}")
    finally:
        _mjpeg_release_camera()
        print("MJPEG: stream ended, camera released")


def is_active() -> bool:
    """Quick status check — used by /api/camera/status and /api/health."""
    with _mjpeg_lock:
        return _mjpeg_active and _mjpeg_cap is not None and _mjpeg_cap.isOpened()


def stop_camera():
    """Explicitly release the MJPEG camera device (called on Stop / page unload)."""
    _mjpeg_release_camera()
