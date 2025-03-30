# session.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from uuid import UUID  # Important: match Supabase uuid type exactly
import datetime
from supabase import create_client
import os

router = APIRouter()

# Supabase Setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Temporary in-memory storage
sessions = {}

# Pydantic Model with aliases for frontend camelCase fields
class SessionEndRequest(BaseModel):
    session_id: UUID  # Correct type is UUID, not str
    user_group: str
    rounds: list
    feedback_time: float = Field(alias="feedbackTime")  # Fix camelCase
    feedback_answers: dict = Field(alias="feedbackAnswers")  # Fix camelCase

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
    session_id_str = str(payload.session_id)  # Convert UUID -> str if needed for Supabase

    if session_id_str not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    end_time = datetime.datetime.utcnow()
    sessions[session_id_str]["end"] = end_time
    elapsed = (end_time - sessions[session_id_str]["start"]).total_seconds()

    session_data = {
        "session_id": session_id_str,  # pass as string
        "user_group": payload.user_group,
        "session_time": elapsed,
        "rounds": payload.rounds,
        "feedback_time": payload.feedback_time,
        "feedback_answers": payload.feedback_answers,
        "created_at": end_time.isoformat()
    }

    response = supabase.table("results").insert(session_data).execute()

    if response.data is None:
        raise HTTPException(status_code=500, detail="Failed to write to Supabase.")

    return {
        "session_id": session_id_str,
        "end": end_time.isoformat(),
        "elapsed_seconds": elapsed,
        "db_response": response.data
    }
