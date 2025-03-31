# session.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import uuid
import datetime
from supabase import create_client
import os
from dotenv import load_dotenv
from postgrest.exceptions import APIError

router = APIRouter()

load_dotenv()

# Supabase Setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase URL or Key missing. Check .env file.")

# Temporary in-memory storage
sessions = {}

# Pydantic Model with aliases for frontend camelCase fields
class SessionEndRequest(BaseModel):
    session_id: str
    user_group: str
    rounds: list
    feedback_time: float   # snake_case
    feedback_answers: dict # snake_case

    class Config:
        allow_population_by_field_name = True


@router.post("/session/start", tags=["Session"])
def start_session(user_id: str = None):
    session_id = str(uuid.uuid4())
    start_time = datetime.datetime.utcnow()

    sessions[session_id] = {
        "start": start_time,
        "end": None,
        "user_id": user_id
    }

    return {
        "session_id": session_id,
        "start": start_time.isoformat()
    }


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
        print("Attempting to insert:", session_data)  # debug
        response = supabase.table("session_results").insert(session_data).execute()
        return {"success": True, "data": response.data}
    except APIError as e:
        print("Supabase error:", e)
        raise HTTPException(status_code=500, detail=f"Supabase insert error: {e}")
    