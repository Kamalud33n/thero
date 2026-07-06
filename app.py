import os
import random
import datetime

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from database import get_db, init_db
from models import Patient, SessionModel, JointAngle, Setting
from telehealth import router as telehealth_router

from config import pose as _pose  # for /api/health mediapipe_ready flag
from services.camera_ws import camera as camera_manager, ws_mgr
from services import mjpeg_camera

from routers import pages, patients, sessions, dashboard, analytics, reports, camera, ws

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

app.mount("/static",  StaticFiles(directory="static"),  name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/assets",  StaticFiles(directory="assets"),  name="assets")

app.include_router(telehealth_router)
app.include_router(pages.router)
app.include_router(patients.router)
app.include_router(sessions.router)
app.include_router(dashboard.router)
app.include_router(analytics.router)
app.include_router(reports.router)
app.include_router(camera.router)
app.include_router(ws.router)


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
        print("Seed data inserted.")


# ─── Health check ─────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {
        "status":          "healthy",
        "timestamp":       datetime.datetime.now().isoformat(),
        "camera_running":  camera_manager.is_running,
        "mjpeg_running":   mjpeg_camera.is_active(),
        "ws_connections":  len(ws_mgr.connections),
        "mediapipe_ready": _pose is not None,
        "target_fps":      camera_manager.TARGET_FPS,
        "resolution":      "320x240",
    }


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
