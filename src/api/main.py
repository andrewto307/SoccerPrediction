"""FastAPI service exposing live La Liga match predictions.

Run from the src/ directory:
    ../myenv/bin/uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /health              liveness + readiness (model/scaler/api-key loaded)
    GET  /fixtures/upcoming   upcoming fixtures (names mapped to model vocabulary)
    POST /predict             predict one fixture (by fixture_id, or home/away/date)
    GET  /predict/upcoming    predict all upcoming fixtures in one call
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from live import config
from live.predictor import Predictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

state: dict = {"predictor": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model + scaler once at startup. Provider is built lazily on first use,
    # so the app still starts (and /health responds) without an API key.
    try:
        state["predictor"] = Predictor()
        logger.info("Predictor ready (provider=%s)", config.PROVIDER)
    except Exception as exc:  # missing artifact, etc.
        state["error"] = str(exc)
        logger.error("Predictor failed to initialize: %s", exc)
    yield


app = FastAPI(title="Soccer Prediction — Live La Liga", version="1.0", lifespan=lifespan)


# --- schemas ---------------------------------------------------------------
class Probabilities(BaseModel):
    home: float
    draw: float
    away: float


class PredictionResponse(BaseModel):
    fixture_id: int | None = None
    date: str | None = None
    home_team: str
    away_team: str
    prediction: int = Field(description="0=Away, 1=Draw, 2=Home")
    label: str
    probabilities: Probabilities
    missing_bookmakers: list[str] = []
    unmapped_teams: list[str] = []
    home_matches_used: int = 0
    away_matches_used: int = 0


class OddsInput(BaseModel):
    """Decimal (European) odds for the match, supplied by the user."""
    home: float = Field(gt=1.0, description="Decimal odds for a home win, e.g. 2.10")
    draw: float = Field(gt=1.0, description="Decimal odds for a draw, e.g. 3.40")
    away: float = Field(gt=1.0, description="Decimal odds for an away win, e.g. 3.50")


class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    date: datetime
    odds: OddsInput  # required: the free data plan does not provide odds


def _predictor() -> Predictor:
    if state["predictor"] is None:
        raise HTTPException(503, detail=f"Predictor not ready: {state['error'] or 'unknown error'}")
    return state["predictor"]


# --- routes ----------------------------------------------------------------
@app.get("/")
def index():
    """Small landing index so hitting the root isn't a bare 404."""
    return {
        "service": "Soccer Prediction — Live La Liga",
        "docs": "/docs",
        "endpoints": {
            "GET /health": "readiness (model / scaler / api key)",
            "GET /fixtures/upcoming?days=7": "upcoming fixtures (needs accessible season / paid plan)",
            "POST /predict": "predict one match from body {home_team, away_team, date, odds:{home,draw,away}}",
            "GET /predict/upcoming?days=7": "predict upcoming fixtures using provider odds (paid plans)",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if state["predictor"] else "degraded",
        "model_loaded": state["predictor"] is not None,
        "model": config.MODEL_PATH.name,  # fixed model served for live prediction
        "provider": config.PROVIDER,
        "api_key_present": bool(config.API_FOOTBALL_KEY),
        "league_id": config.LEAGUE_ID,
        "error": state["error"],
    }


@app.get("/fixtures/upcoming")
def upcoming(days: int = Query(7, ge=1, le=30)):
    try:
        return {"fixtures": _predictor().list_upcoming(days)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, detail=f"Provider error: {exc}")


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictRequest):
    """Predict a single match from user-supplied odds + best-effort form from the API."""
    p = _predictor()
    try:
        return p.predict_manual(
            req.home_team, req.away_team, req.date,
            req.odds.home, req.odds.draw, req.odds.away,
        )
    except Exception as exc:
        raise HTTPException(502, detail=f"Prediction failed: {exc}")


@app.get("/predict/upcoming")
def predict_upcoming(days: int = Query(7, ge=1, le=30)):
    try:
        return {"predictions": _predictor().predict_upcoming(days)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, detail=f"Provider error: {exc}")
