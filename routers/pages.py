from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from config import templates

router = APIRouter()

NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


# NOTE: index.html (welcome page) commented out for now — using /dashboard
# as the main landing page instead. Uncomment if index page needed later.
#
# @router.get("/", response_class=HTMLResponse)
# async def page_index(request: Request):
#     return templates.TemplateResponse(request, "index.html", headers=NO_CACHE_HEADERS)

@router.get("/", response_class=HTMLResponse)
async def page_index(request: Request):
    return RedirectResponse(url="/dashboard")


@router.get("/dashboard", response_class=HTMLResponse)
async def page_dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html", headers=NO_CACHE_HEADERS)


@router.get("/patients", response_class=HTMLResponse)
async def page_patients(request: Request):
    return templates.TemplateResponse(request, "patients.html", headers=NO_CACHE_HEADERS)


@router.get("/session", response_class=HTMLResponse)
async def page_session(request: Request):
    return templates.TemplateResponse(request, "session.html", headers=NO_CACHE_HEADERS)


@router.get("/join/{room_id}", response_class=HTMLResponse)
async def page_join(request: Request, room_id: str):
    return templates.TemplateResponse(request, "patient.html", {"room_id": room_id}, headers=NO_CACHE_HEADERS)


@router.get("/reports", response_class=HTMLResponse)
async def page_reports(request: Request):
    return templates.TemplateResponse(request, "reports.html", headers=NO_CACHE_HEADERS)


@router.get("/analytics", response_class=HTMLResponse)
async def page_analytics(request: Request):
    return templates.TemplateResponse(request, "analytics.html", headers=NO_CACHE_HEADERS)