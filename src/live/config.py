"""Central configuration for the live prediction layer.

All values can be overridden via environment variables (loaded from the project
`.env` file). Defaults target La Liga via API-Football, but LEAGUE_ID / SEASON are
intentionally externalized so other competitions can be added without code changes.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = .../SoccerPrediction (config.py lives at src/live/config.py)
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

# Load .env from the project root if present (no-op if missing).
load_dotenv(ROOT / ".env")


def _get(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name, default)
    return val.strip() if isinstance(val, str) else val


# --- Model / artifact paths -------------------------------------------------
# cb_final.pkl is a raw CatBoostClassifier trained on the `odds_form_teams`
# feature set (21 features: HomeTeam, AwayTeam, 4 form, 15 odds).
MODEL_PATH = Path(_get("MODEL_PATH", str(MODELS_DIR / "cb_final.pkl")))
# Reconstructed MinMaxScaler for the form features (see live/build_scaler.py).
SCALER_PATH = Path(_get("SCALER_PATH", str(MODELS_DIR / "form_scaler.pkl")))

# --- Competition ------------------------------------------------------------
# API-Football league id 140 = La Liga (Primera Division).
LEAGUE_ID = int(_get("LEAGUE_ID", "140"))
# API-Football encodes a season by its starting year (2025 => 2025/26).
# Empty -> the provider auto-selects the most relevant season.
_season = _get("SEASON", "")
SEASON: int | None = int(_season) if _season else None

# --- Provider ---------------------------------------------------------------
# "api_football" (live) or "mock" (local JSON fixtures, no network/quota).
PROVIDER = _get("LIVE_PROVIDER", "api_football")

# API-Football direct host (header auth: x-apisports-key).
API_FOOTBALL_BASE_URL = _get("API_FOOTBALL_BASE_URL", "https://v3.football.api-sports.io")
API_FOOTBALL_KEY = _get("API_FOOTBALL_KEY")

# Local sample payloads used by the mock provider / tests.
MOCK_DATA_DIR = Path(_get("MOCK_DATA_DIR", str(ROOT / "data" / "live_samples")))

# --- Behaviour --------------------------------------------------------------
# Rolling window for form features (matches training: last 5 matches).
FORM_WINDOW = int(_get("FORM_WINDOW", "5"))
# Cache TTL (seconds) for fixtures/odds to protect the free-tier request quota.
CACHE_TTL_SECONDS = int(_get("CACHE_TTL_SECONDS", "900"))

# --- Security ---------------------------------------------------------------
# When APP_API_KEY is set, the prediction endpoints require it in the
# `X-API-Key` request header. Left empty => auth is DISABLED (keyless), which
# keeps local/demo runs and the offline tests frictionless.
APP_API_KEY = _get("APP_API_KEY")

# Allowed browser origins for CORS (comma-separated). Empty => the CORS
# middleware is not installed at all (server-to-server callers don't need it).
_cors = _get("CORS_ORIGINS", "") or ""
CORS_ORIGINS = [o.strip() for o in _cors.split(",") if o.strip()]

# Lightweight per-client rate limit on the prediction endpoints: at most
# RATE_LIMIT_MAX requests per RATE_LIMIT_WINDOW seconds, keyed by API key
# (or client IP if unauthenticated). Set RATE_LIMIT_MAX=0 to disable.
RATE_LIMIT_MAX = int(_get("RATE_LIMIT_MAX", "120"))
RATE_LIMIT_WINDOW = int(_get("RATE_LIMIT_WINDOW", "60"))
