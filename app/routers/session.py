# session.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import uuid
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
    session_id: uuid.UUID
    user_group: str
    rounds: list
    feedback_time: float = Field(alias="feedbackTime")
    feedback_answers: dict = Field(alias="feedbackAnswers")

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
    end_time = datetime.datetime.utcnow()

    session_data = {
        "session_id": str(payload.session_id),
        "user_group": payload.user_group,
        "session_time": sum(round["round_duration"] for round in payload.rounds),
        "rounds": payload.rounds,
        "feedback_time": payload.feedback_time,
        "feedback_answers": payload.feedback_answers,
        "created_at": end_time.isoformat()
    }

    response = supabase.table("session_results").insert(session_data).execute()

    if response.data is None:
        raise HTTPException(status_code=500, detail="Failed to write to Supabase.")

    return {
        "session_id": payload.session_id,
        "db_response": response.data
    }
