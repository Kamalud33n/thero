import os
import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from database import get_db
from models import Patient, SessionModel, Report
from services.report_builder import build_report_sync

router = APIRouter()


@router.get("/api/reports/{patient_id}")
async def generate_report(patient_id: str, report_type: str = "weekly"):
    """
    PDF generation runs synchronously in a thread pool executor so it
    does NOT block the event loop while WebSocket streaming is active.
    """
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
        None, build_report_sync, patient_id, report_type
    )
    return FileResponse(
        filepath,
        media_type="application/pdf",
        filename=f"report_{patient_id}_{report_type}.pdf",
    )


@router.get("/api/reports/history/{patient_id}")
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


@router.get("/api/reports/file/{report_id}")
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
