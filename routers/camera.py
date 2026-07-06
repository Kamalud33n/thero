from typing import Dict, Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse

from services import mjpeg_camera, metrics

router = APIRouter()


@router.get("/video_feed")
async def video_feed():
    """Primary live camera feed with pose skeleton drawn server-side. <img src='/video_feed'>"""
    return StreamingResponse(
        mjpeg_camera.gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/api/pose_data")
async def api_pose_data():
    """Latest joint angles + detection flag, updated every frame by gen_frames()."""
    return JSONResponse(mjpeg_camera.latest_pose_data)


@router.post("/api/camera/stop")
async def api_camera_stop():
    """Explicitly release the MJPEG camera device (called on Stop / page unload)."""
    mjpeg_camera.stop_camera()
    return JSONResponse({"success": True, "message": "Camera stopped"})


@router.post("/api/exercise_type")
async def set_exercise_type(payload: Dict[str, Any]):
    """Frontend calls this whenever the exercise dropdown (or target ROM)
    changes, so the MJPEG stream draws the right joints and rep-counts
    against the right threshold."""
    ex  = payload.get("exercise_type")
    rom = payload.get("target_rom")
    metrics.set_exercise_state(exercise_type=ex, target_rom=rom)
    current_ex, current_rom = metrics.get_exercise_state()
    return JSONResponse({
        "success": True,
        "exercise_type": current_ex,
        "target_rom": current_rom,
    })


@router.post("/api/session/reset")
async def api_session_reset():
    """Call this right before a session starts so rep count + stability
    buffer don't carry over stale data from a previous session/patient."""
    metrics.reset_state()
    return JSONResponse({"success": True})


@router.get("/api/camera/status")
async def api_camera_status():
    """Quick status check — useful for frontend polling / debugging."""
    return JSONResponse({"active": mjpeg_camera.is_active()})
