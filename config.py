"""
Shared configuration / singletons used across the app:
- PDF report color tokens
- required directories
- MediaPipe Pose model (initialized once)
- key landmark index map + joint-angle helper
- shared Jinja2Templates instance (with the `tojson` filter registered)
"""
import os
import json as _json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import mediapipe as mp
from reportlab.lib import colors
from fastapi.templating import Jinja2Templates

# PDF report design tokens (2-color system: navy + grey, white bg) 
PDF_NAVY        = colors.HexColor("#1B2A4A")   # headings / emphasis
PDF_GREY_BORDER = colors.HexColor("#B7BEC9")   # borders / rules
PDF_GREY_BG     = colors.HexColor("#EEF1F5")   # section header band
PDF_GREY_TEXT   = colors.HexColor("#5A6472")   # secondary/meta text
PDF_ROW_ALT     = colors.HexColor("#F7F8FA")   # alternating table row
PDF_BODY_TEXT   = colors.HexColor("#2B2B2B")   # body copy

# Directories 
for d in ("data", "reports", "uploads", "static", "templates", "assets"):
    os.makedirs(d, exist_ok=True)

# Shared Jinja2 templates instance (import this everywhere instead of
# creating a new Jinja2Templates(...) so the `tojson` filter is available
# in every router that renders HTML) 
templates = Jinja2Templates(directory="templates")
templates.env.filters["tojson"] = lambda obj: _json.dumps(obj)

# MediaPipe init (optimized) 
try:
    _mp_pose = mp.solutions.pose
    pose = _mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,           # ← 0 = fastest
        smooth_landmarks=False,       # ← skip smoothing
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_drawing        = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    POSE_CONNECTIONS  = _mp_pose.POSE_CONNECTIONS
    print("MediaPipe Pose initialized (optimized: complexity=0)")
except Exception as exc:
    print(f"MediaPipe init failed: {exc}")
    pose = mp_drawing = mp_drawing_styles = POSE_CONNECTIONS = None

# Key landmarks only (13 joints instead of 33) 
# Indices: nose=0, shoulders=11/12, elbows=13/14, wrists=15/16,
#          hips=23/24, knees=25/26, ankles=27/28
KEY_LANDMARKS = {
    "nose": 0, "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14, "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}


def get_angle(p1, p2, p3) -> float:
    a = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    b = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))
