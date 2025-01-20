from fastapi import FastAPI
from app.routers import candidates, prediction

app = FastAPI()

# Include routers
app.include_router(candidates.router)
app.include_router(prediction.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Bias Fairness MVP API!"}
