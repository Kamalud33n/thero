from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from config import templates

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def page_index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@router.get("/dashboard", response_class=HTMLResponse)
async def page_dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html")


@router.get("/patients", response_class=HTMLResponse)
async def page_patients(request: Request):
    return templates.TemplateResponse(request, "patients.html")


@router.get("/session", response_class=HTMLResponse)
async def page_session(request: Request):
    return templates.TemplateResponse(request, "session.html")


@router.get("/join/{room_id}", response_class=HTMLResponse)
async def page_join(request: Request, room_id: str):
    return templates.TemplateResponse(request, "patient.html", {"room_id": room_id})


@router.get("/reports", response_class=HTMLResponse)
async def page_reports(request: Request):
    return templates.TemplateResponse(request, "reports.html")


@router.get("/analytics", response_class=HTMLResponse)
async def page_analytics(request: Request):
    return templates.TemplateResponse(request, "analytics.html")
