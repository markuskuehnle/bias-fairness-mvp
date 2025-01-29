from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import logging

from app.routers import candidates, prediction

app = FastAPI()

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)

@app.middleware("http")
async def log_requests(request, call_next):
    logging.info(f"📥 Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logging.info(f"📤 Response status: {response.status_code}")
    return response

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(candidates.router)
app.include_router(prediction.router)

# Serve the frontend folder
app.mount("/frontend", StaticFiles(directory="app/frontend"), name="frontend")
app.mount("/data", StaticFiles(directory="app/data"), name="data")

@app.get("/candidates", response_class=HTMLResponse)
def show_candidates_frontend():
    with open("app/frontend/index.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Bias Fairness MVP API!"}
