import time
from typing import List, Optional

import cv2

from fastapi import WebSocket

from config import pose as _pose, mp_drawing as _mp_drawing
from config import mp_drawing_styles as _mp_drawing_styles
from config import POSE_CONNECTIONS as _POSE_CONNECTIONS
from config import KEY_LANDMARKS as _KEY_LANDMARKS
from config import get_angle as _get_angle


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

        # Only send 13 key landmarks (not all 33)
        pose_data = {}
        for name, idx in _KEY_LANDMARKS.items():
            if idx < len(lm) and lm[idx].visibility > 0.5:   # skip low-confidence
                pose_data[name] = {
                    "x": round(lm[idx].x, 4),
                    "y": round(lm[idx].y, 4),
                    "z": round(lm[idx].z, 4),
                    "v": round(lm[idx].visibility, 2),
                }

        # Joint angles
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
