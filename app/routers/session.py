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

# Correctly mapped Pydantic model with aliases for frontend camelCase
class SessionEndRequest(BaseModel):
    session_id: str
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

    # Store temporarily in-memory
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

    # Prepare session data to save in Supabase
    session_data = {
        "session_id": session_id,
        "user_group": payload.user_group,
        "session_time": elapsed,
        "rounds": payload.rounds,
        "feedback_time": payload.feedback_time,
        "feedback_answers": payload.feedback_answers,
        "created_at": end_time.isoformat()
    }

    # Attempt to save to Supabase and handle response
    response = supabase.table("session_results").insert(session_data).execute()

    if hasattr(response, 'data') and response.data is None:
        print("❌ Supabase error details:", response)
        raise HTTPException(status_code=500, detail="Failed to write session results to Supabase.")
    elif hasattr(response, 'error') and response.error is not None:
        print("❌ Supabase returned an error:", response.error)
        raise HTTPException(status_code=500, detail=f"Supabase Error: {response.error}")

    return {
        "session_id": session_id,
        "end": end_time.isoformat(),
        "elapsed_seconds": elapsed,
        "db_response": response.data
    }
