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
import secrets
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
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
    if not config.APP_API_KEY:
        logger.warning(
            "APP_API_KEY is not set — the prediction endpoints are UNAUTHENTICATED. "
            "Fine for local/demo; set APP_API_KEY before exposing this service."
        )
    yield


app = FastAPI(title="Soccer Prediction — Live La Liga", version="1.0", lifespan=lifespan)

# --- CORS (only if origins are configured) ---------------------------------
if config.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


# --- API-key authentication -------------------------------------------------
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(provided: str | None = Security(_api_key_header)) -> None:
    """Require the X-API-Key header when APP_API_KEY is configured.

    If APP_API_KEY is unset the check is skipped (keyless local/demo mode).
    Comparison is constant-time to avoid leaking the key via timing.
    """
    expected = config.APP_API_KEY
    if not expected:
        return
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# --- Simple in-memory rate limiter -----------------------------------------
# Single-instance / fixed-window. For multiple replicas move this to Redis.
_rate_hits: dict[str, list[float]] = {}
_rate_lock = threading.Lock()
_RATE_EXEMPT = {"/health", "/", "/docs", "/openapi.json", "/redoc"}


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if config.RATE_LIMIT_MAX <= 0 or request.url.path in _RATE_EXEMPT:
        return await call_next(request)
    # Key by API key when present, else client IP (accurate behind a proxy that
    # sets X-Forwarded-For, since uvicorn runs with proxy headers enabled).
    key = request.headers.get("X-API-Key") or (request.client.host if request.client else "anon")
    now = time.monotonic()
    cutoff = now - config.RATE_LIMIT_WINDOW
    with _rate_lock:
        hits = [t for t in _rate_hits.get(key, ()) if t > cutoff]
        if len(hits) >= config.RATE_LIMIT_MAX:
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Please slow down."})
        hits.append(now)
        _rate_hits[key] = hits
    return await call_next(request)


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    """Never leak internals: log the detail server-side, return a generic 500."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


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
        "api_key_present": bool(config.API_FOOTBALL_KEY),  # upstream (API-Football) key
        "auth_enabled": bool(config.APP_API_KEY),          # this service's own auth
        "league_id": config.LEAGUE_ID,
        "error": state["error"],
    }


@app.get("/fixtures/upcoming", dependencies=[Depends(require_api_key)])
def upcoming(days: int = Query(7, ge=1, le=30)):
    try:
        return {"fixtures": _predictor().list_upcoming(days)}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Provider error on /fixtures/upcoming")
        raise HTTPException(502, detail="Upstream data provider error. Please try again later.")


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(require_api_key)])
def predict(req: PredictRequest):
    """Predict a single match from user-supplied odds + best-effort form from the API."""
    p = _predictor()
    try:
        return p.predict_manual(
            req.home_team, req.away_team, req.date,
            req.odds.home, req.odds.draw, req.odds.away,
        )
    except Exception:
        logger.exception("Prediction failed on /predict")
        raise HTTPException(502, detail="Prediction failed. Please try again later.")


@app.get("/predict/upcoming", dependencies=[Depends(require_api_key)])
def predict_upcoming(days: int = Query(7, ge=1, le=30)):
    try:
        return {"predictions": _predictor().predict_upcoming(days)}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Provider error on /predict/upcoming")
        raise HTTPException(502, detail="Upstream data provider error. Please try again later.")
