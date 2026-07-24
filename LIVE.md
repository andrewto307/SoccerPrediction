# Live Prediction Service

Predict **La Liga matches** by combining recent results from an external provider
(API-Football) with **odds the user supplies** (the free data plan has no odds),
rebuilding the exact features the trained CatBoost model expects, and serving
predictions over a REST API — **no retraining**.

## How it works

```
provider (API-Football)                 src/live/feature_builder.py
  ├─ upcoming_fixtures() ──┐            reuses the TRAINING preprocessing:
  ├─ recent_results()  ────┼──► build ► • team_last_matches_performance()  (form)
  └─ match_odds(id)    ────┘            • normalize_betting_odds()          (odds)
                                        • models/form_scaler.pkl            (scaling)
                                              │
                                              ▼  21-feature odds_form_teams row
                                    models/cb_final.pkl  (CatBoost)
                                              │
                                              ▼
                                    {label, probabilities{home,draw,away}}
```

The live feature builder **reuses the training functions verbatim**, so a rebuilt
feature row reproduces the committed `X_test.csv` byte-for-byte. The test suite
proves this: rebuilding all 180 test fixtures through the live path yields the
model's exact **62.2%** accuracy (`tests/test_parity.py::test_full_pipeline_accuracy`).

## One-time setup

```bash
# 1. Install deps (into the project venv)
pip install -r requirements.txt

# 2. Reconstruct the form scaler the model was trained with
#    (not persisted with the model; regenerates models/form_scaler.pkl)
cd src && python -m live.build_scaler

# 3. Configure the data provider
cp .env.example .env        # then set API_FOOTBALL_KEY
```

## Run

Easiest — from the project root (`SoccerPrediction/`), using the project venv:

```bash
../myenv/bin/python src/serve.py          # http://127.0.0.1:8000  (set PORT=… to change)
```

Or invoke uvicorn directly — but you **must** run it from `src/` so the `api`
package resolves (otherwise: `ModuleNotFoundError: No module named 'api'`):

```bash
cd src && ../../myenv/bin/uvicorn api.main:app --reload --port 8000
```

| Endpoint | Description |
|---|---|
| `GET /health` | liveness + readiness (model / scaler / api-key) |
| `GET /fixtures/upcoming?days=7` | upcoming fixtures (needs a paid plan / accessible season) |
| `POST /predict` | predict one match from **user-supplied odds** + best-effort form from the API |
| `GET /predict/upcoming?days=7` | predict upcoming fixtures using provider odds (paid plans) |

`POST /predict` — odds are a required input (the free data plan does not provide them):

```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
  "home_team": "Atletico Madrid",
  "away_team": "Real Madrid",
  "date": "2025-02-08T20:00:00Z",
  "odds": { "home": 2.50, "draw": 3.30, "away": 2.80 }
}'
```

The one odds triplet is normalized and broadcast across the model's 5 bookmaker
slots. Form is pulled from the provider when available; if the configured season
is locked (free plan) or it's the off-season, the prediction proceeds without form
(`home_matches_used`/`away_matches_used` will be 0).

## Streamlit app (calls the API)

The Streamlit app (`src/app.py`) is a thin client: pick the teams + date, type the
pre-match odds, and it POSTs to the API's `/predict`. Start **both**, in two
terminals, from the project root (`SoccerPrediction/`):

```bash
# terminal 1 — the API backend
../myenv/bin/python src/serve.py                 # http://127.0.0.1:8000

# terminal 2 — the Streamlit UI
../myenv/bin/streamlit run src/app.py            # http://localhost:8501
```

The app surfaces the API connection status and reads `API_URL` (default
`http://127.0.0.1:8000`) if the backend runs elsewhere.

## Offline / testing

Set `LIVE_PROVIDER=mock` to serve from `data/live_samples/*.json` — no network or
API key required (used by the test suite). Run tests with:

```bash
python -m pytest tests/ -v
```

## Notes & approximations

- **Pre-match only.** The model has no in-play features.
- **API-Football free tier.** The free plan exposes only seasons **2022–2024** and
  returns **no odds**. So by default odds are supplied manually via `POST /predict`,
  and form is best-effort. To compute form from the API on the free plan, set
  `SEASON` to an accessible season (e.g. `2024`). A paid plan unlocks the current
  season + live odds, after which `/fixtures/upcoming` and `/predict/upcoming` work
  with provider odds — no code change needed.
- **Bookmaker mapping.** The model uses 5 specific bookmakers (Bwin, Interwetten,
  William Hill, BetVictor/VC, Pinnacle-closing) — all present in API-Football's
  catalog. A single user-supplied odds triplet is broadcast across all 5 slots; with
  provider odds, any slot not supplied is filled with the market consensus (never 0)
  and listed in `missing_bookmakers`.
- **Team names** are mapped to the football-data.co.uk training vocabulary
  (`src/live/team_mapping.py`); unmapped teams are reported in `unmapped_teams`.
- **Expanding leagues.** Set `LEAGUE_ID` / `SEASON` in `.env`; add team-name
  entries for the new league. The provider/feature/model layers are league-agnostic.
```
