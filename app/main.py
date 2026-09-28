import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import assistant
from .features import SPECS
from .forecast import demo_history, forecast
from .model import load_metrics, predict, validate

try:  # optional: load .env for local development
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(title="MediMind API", version="1.0.0")


class PredictIn(BaseModel):
    disease: str
    values: dict[str, float]


class ForecastIn(BaseModel):
    history: list[float] = Field(min_length=14, max_length=730)
    days: int = Field(7, ge=1, le=30)
    capacity: float = Field(200, gt=0)


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/schema")
def schema():
    return SPECS


@app.post("/api/predict")
def predict_risk(body: PredictIn):
    if body.disease not in SPECS:
        raise HTTPException(404, f"Unknown disease. Choose one of: {', '.join(SPECS)}")
    clean, errors = validate(body.disease, body.values)
    if errors:
        raise HTTPException(422, "; ".join(errors))
    return predict(body.disease, clean)


@app.get("/api/metrics")
def metrics():
    return load_metrics()


@app.get("/api/forecast/beds")
def beds_demo(days: int = 7, capacity: float = 200):
    return forecast(demo_history(), max(1, min(days, 30)), max(capacity, 1))


@app.post("/api/forecast/beds")
def beds_custom(body: ForecastIn):
    return forecast(body.history, body.days, body.capacity)


@app.post("/api/assistant")
def ask(body: AskIn):
    return assistant.ask(body.question)


# Local dev only: on Vercel, files in public/ are served by the CDN.
if not os.getenv("VERCEL"):
    app.mount("/", StaticFiles(directory=Path(__file__).resolve().parent.parent / "public", html=True), name="site")
