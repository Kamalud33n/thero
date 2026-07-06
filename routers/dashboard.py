import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from database import get_db
from models import Patient, SessionModel
from services.helpers import calculate_recovery_score

router = APIRouter()


@router.get("/api/dashboard")
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
