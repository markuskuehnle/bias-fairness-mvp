from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import uuid
import datetime
from supabase import create_client
import os
from dotenv import load_dotenv
from postgrest.exceptions import APIError

router = APIRouter()

# Load env
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase URL or Key missing. Check .env file.")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# In-memory session store
sessions = {}
MAX_ROUNDS = 6
SESSION_STORE = {}


# ----------- MODELS -----------

class SessionEndRequest(BaseModel):
    session_id: str
    user_group: str
    rounds: list
    feedback_time: float
    feedback_answers: dict

class SessionIdRequest(BaseModel):
    session_id: str


# ----------- ROUTES -----------

@router.post("/session/start", tags=["Session"])
def start_session(user_id: str = None):
    session_id = str(uuid.uuid4())
    start_time = datetime.datetime.utcnow()
    user_groups = ["no-xai", "badge", "predictions", "interactive"]

    # Defensive: prevent duplicate session_id use
    while session_id in sessions:
        session_id = str(uuid.uuid4())

    assigned_group = user_groups[uuid.uuid4().int % len(user_groups)]

    sessions[session_id] = {
        "start": start_time,
        "end": None,
        "user_id": user_id,
        "rounds_played": 0,
        "user_group": assigned_group
    }

    return {
        "session_id": session_id,
        "start": start_time.isoformat(),
        "user_group": assigned_group
    }



@router.post("/session/reinit", tags=["Session"])
def reinit_session(payload: SessionIdRequest):
    session_id = payload.session_id

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found in memory")

    sessions[session_id]["rounds_played"] = 0
    return {"success": True, "message": "Session state reset."}


@router.post("/session/end", tags=["Session"])
def end_session(payload: SessionEndRequest):
    session_id = payload.session_id
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    end_time = datetime.datetime.utcnow()
    elapsed = (end_time - sessions[session_id]["start"]).total_seconds()

    session_data = {
        "session_id": session_id,
        "user_group": payload.user_group,
        "session_time": elapsed,
        "rounds": payload.rounds,
        "feedback_time": payload.feedback_time,
        "feedback_answers": payload.feedback_answers,
        "created_at": end_time.isoformat()
    }

    try:
        print("Attempting to insert:", session_data)
        response = supabase.table("session_results").insert(session_data).execute()
        return {"success": True, "data": response.data}
    except APIError as e:
        print("Supabase error:", e)
        raise HTTPException(status_code=500, detail=f"Supabase insert error: {e}")


@router.post("/round/start", tags=["Round"])
def start_round(payload: SessionIdRequest):
    session_id = payload.session_id

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found in memory")

    if sessions[session_id]["rounds_played"] >= MAX_ROUNDS:
        raise HTTPException(status_code=400, detail="Maximum number of rounds reached")

    sessions[session_id]["rounds_played"] += 1
    return {"success": True, "round_number": sessions[session_id]["rounds_played"]}


@router.post("/candidates/reset", tags=["Round"])
def reset_candidates(payload: SessionIdRequest):
    session_id = payload.session_id

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found in memory")

    sessions[session_id]["rounds_played"] = 0
    print(f"[RESET] rounds_played reset for session {session_id}")
    return {"success": True, "message": f"Session {session_id} reset."}


@router.get("/session/group", tags=["Session"])
def get_user_group(session_id: str = Query(...)):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    return {"user_group": sessions[session_id]["user_group"]}
