import cv2
import numpy as np

from models import SessionModel


def compress_photo(raw_bytes: bytes, max_dim: int = 800, quality: int = 80) -> bytes:
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
        print(f"Photo compression failed, storing original bytes: {exc}")
        return raw_bytes


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
