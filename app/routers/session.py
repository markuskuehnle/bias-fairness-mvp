from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import uuid, datetime

router = APIRouter()
sessions = {}

class SessionEndRequest(BaseModel):
    session_id: str

@router.post("/session/start", tags=["Session"])
def start_session(user_id: str = None):
    session_id = str(uuid.uuid4())
    start_time = datetime.datetime.utcnow()
    sessions[session_id] = {"start": start_time, "end": None}
    return {"session_id": session_id, "start": start_time.isoformat()}

@router.post("/session/end", tags=["Session"])
def end_session(payload: SessionEndRequest):
    session_id = payload.session_id
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    end_time = datetime.datetime.utcnow()
    sessions[session_id]["end"] = end_time
    elapsed = (end_time - sessions[session_id]["start"]).total_seconds()
    return {
        "session_id": session_id,
        "end": end_time.isoformat(),
        "elapsed_seconds": elapsed
    }
