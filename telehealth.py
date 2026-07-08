import os
import secrets
import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from database import get_db
from models import Patient, TelehealthRoom

router = APIRouter()

#In-memory room registry for WebRTC signaling relay
class RoomManager:

    def __init__(self):
        self.rooms: Dict[str, Dict[str, Optional[WebSocket]]] = {}

    def register(self, room_id: str, role: str, ws: WebSocket):
        self.rooms.setdefault(room_id, {"doctor": None, "patient": None})
        self.rooms[room_id][role] = ws

    def unregister(self, room_id: str, role: str, ws: WebSocket):
        room = self.rooms.get(room_id)
        if room and room.get(role) is ws:
            room[role] = None
        if room and not room["doctor"] and not room["patient"]:
            self.rooms.pop(room_id, None)

    async def relay(self, room_id: str, from_role: str, data: dict):
        room = self.rooms.get(room_id)
        if not room:
            return
        target_ws = room.get("patient" if from_role == "doctor" else "doctor")
        if target_ws is None:
            return
        try:
            await target_ws.send_json(data)
        except Exception:
            pass

    async def broadcast(self, room_id: str, data: dict):
        room = self.rooms.get(room_id)
        if not room:
            return
        for ws in (room.get("doctor"), room.get("patient")):
            if ws is not None:
                try:
                    await ws.send_json(data)
                except Exception:
                    pass


room_mgr = RoomManager()


#REST: room lifecycle
@router.post("/api/telehealth/create-room")
async def create_room(payload: Dict[str, Any]):
    patient_id    = payload.get("patient_id")
    exercise_type = payload.get("exercise_type")
    doctor_name   = payload.get("doctor_name")

    with get_db() as db:
        if not db.query(Patient).filter(Patient.id == patient_id).first():
            raise HTTPException(404, "Patient not found")

        room = TelehealthRoom(
            token         = secrets.token_urlsafe(24),
            patient_id    = patient_id,
            doctor_name   = doctor_name,
            exercise_type = exercise_type,
            status        = "pending",
        )
        db.add(room)
        db.commit()
        db.refresh(room)

        return JSONResponse({
            "room_id":  room.id,
            "token":    room.token,
            "join_url": f"/join/{room.id}?token={room.token}",
            "status":   room.status,
        })


@router.get("/api/telehealth/room/{room_id}")
async def get_room(room_id: str, token: str):
    with get_db() as db:
        room = db.query(TelehealthRoom).filter(TelehealthRoom.id == room_id).first()
        if not room or room.token != token:
            raise HTTPException(404, "Room not found or invalid link")
        if room.status == "closed":
            raise HTTPException(410, "This session has ended")

        patient = db.query(Patient).filter(Patient.id == room.patient_id).first()
        return JSONResponse({
            "room_id":       room.id,
            "status":        room.status,
            "patient_name":  patient.name if patient else "Patient",
            "exercise_type": room.exercise_type,
            "doctor_name":   room.doctor_name,
        })


@router.post("/api/telehealth/close-room/{room_id}")
async def close_room(room_id: str):
    """
    Doctor's "End Session" button. Marks the room closed (link stops
    working from here on) and notifies any connected socket so the
    patient's page can show a "session ended" state immediately.
    """
    with get_db() as db:
        room = db.query(TelehealthRoom).filter(TelehealthRoom.id == room_id).first()
        if not room:
            raise HTTPException(404, "Room not found")
        room.status    = "closed"
        room.closed_at = datetime.datetime.now()
        db.commit()

    await room_mgr.broadcast(room_id, {"type": "session_closed"})
    return JSONResponse({"success": True, "message": "Room closed"})


@router.get("/api/telehealth/turn-credentials")
async def turn_credentials():
    ice_servers = [{"urls": "stun:stun.l.google.com:19302"}]

    turn_url  = os.getenv("METERED_TURN_URL")
    turn_user = os.getenv("METERED_TURN_USERNAME")
    turn_cred = os.getenv("METERED_TURN_CREDENTIAL")

    if turn_url and turn_user and turn_cred:
        ice_servers.append({
            "urls":       turn_url,
            "username":   turn_user,
            "credential": turn_cred,
        })

    return JSONResponse({"iceServers": ice_servers})


#WebSocket: signaling + live pose-metric relay 
@router.websocket("/ws/signal/{room_id}")
async def ws_signal(websocket: WebSocket, room_id: str, role: str, token: str):

    if role not in ("doctor", "patient"):
        await websocket.close(code=4000)
        return

    with get_db() as db:
        room = db.query(TelehealthRoom).filter(TelehealthRoom.id == room_id).first()
        if not room or room.token != token:
            await websocket.close(code=4001)
            return
        if room.status == "closed":
            await websocket.close(code=4002)
            return

        await websocket.accept()
        room_mgr.register(room_id, role, websocket)

        # First patient connection flips the room live and stamps started_at
        if role == "patient" and room.status == "pending":
            room.status = "live"
            room.started_at = datetime.datetime.now()
            db.commit()

    # If the other side is already in the room, tell the socket that just
    # joined right away — this is what triggers the patient side to create
    # the WebRTC offer without waiting for a fresh "peer_joined" event.
    other_role = "patient" if role == "doctor" else "doctor"
    other_ws = room_mgr.rooms.get(room_id, {}).get(other_role)
    if other_ws is not None:
        try:
            await websocket.send_json({"type": "peer_already_present", "role": other_role})
        except Exception:
            pass

    await room_mgr.broadcast(room_id, {"type": "peer_joined", "role": role})

    try:
        while True:
            data = await websocket.receive_json()
            await room_mgr.relay(room_id, role, data)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"ws_signal error ({role}, room={room_id}): {e}")
    finally:
        room_mgr.unregister(room_id, role, websocket)

        # If the PATIENT's socket drops (network loss, tab closed, phone
        # locked, etc.) — not just when the doctor explicitly clicks "End
        # Remote Session" — mark the room closed in the DB too. Without
        # this, TelehealthRoom.status stays "live" forever in the database
        # even though nobody is actually connected anymore, and the old
        # link would still validate as active if reopened.
        if role == "patient":
            with get_db() as db:
                r = db.query(TelehealthRoom).filter(TelehealthRoom.id == room_id).first()
                if r and r.status != "closed":
                    r.status = "closed"
                    r.closed_at = datetime.datetime.now()
                    db.commit()

        await room_mgr.broadcast(room_id, {"type": "peer_left", "role": role})