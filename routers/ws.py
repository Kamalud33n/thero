import base64
import asyncio
import datetime

import cv2
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from services.camera_ws import camera, ws_mgr

router = APIRouter()


@router.websocket("/ws/pose")
async def ws_pose(websocket: WebSocket):
    await ws_mgr.connect(websocket)
    try:
        if not camera.is_running:
            camera.start()

        while True:
            # 1. Grab frame
            frame = camera.get_frame()

            # 2. Safety check
            if frame is None:
                await asyncio.sleep(0.1)
                continue

            # 3. Frame skip — process only every 2nd frame
            if not camera.should_process():
                await asyncio.sleep(0.033)   # ~30fps read loop, skip odd frames
                continue

            # 4. FPS throttle — don't send faster than TARGET_FPS
            if not camera.fps_throttle():
                await asyncio.sleep(0.01)
                continue

            # 5. Pose processing
            annotated, pose_data = camera.process_frame(frame)

            if annotated is None:
                await asyncio.sleep(0.1)
                continue

            # 6. JPEG encode at lower quality (60) → smaller payload
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 60]
            success, buf  = cv2.imencode(".jpg", annotated, encode_params)
            if not success:
                await asyncio.sleep(0.1)
                continue

            # 7. Send base64 frame + slim pose_data
            await ws_mgr.send(websocket, {
                "type":      "pose_data",
                "frame":     base64.b64encode(buf).decode(),
                "pose_data": pose_data,
                "ts":        datetime.datetime.now().isoformat(),
            })

            # 8. Async sleep — gives event loop room to breathe
            await asyncio.sleep(0.1)   # ~10fps effective send rate

    except WebSocketDisconnect:
        ws_mgr.disconnect(websocket)
        if not ws_mgr.connections:
            camera.stop()
    except Exception as e:
        print(f"WebSocket error: {e}")
        ws_mgr.disconnect(websocket)
