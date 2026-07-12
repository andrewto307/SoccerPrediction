# 🐳 Docker Setup

The system runs as **three containers** from one multi-target image:

| Service | What it is | URL |
|---|---|---|
| `api` | FastAPI backend — loads the CatBoost model, serves `/predict` | http://localhost:8000 (`/docs`) |
| `app` | Streamlit UI — a thin client that calls the API | http://localhost:8501 |
| `caddy` | TLS reverse proxy — public HTTPS entrypoint for the UI | https://localhost (or your `DOMAIN`) |

The UI reaches the API over the compose network (`API_URL=http://api:8000`). In
this setup `api` and `app` are bound to `127.0.0.1` — only **`caddy`** is public.

> **Security & HTTPS** (API-key auth, rate limiting, Caddy/Let's Encrypt certs)
> are off by default — enable them via `.env` (`APP_API_KEY`, `DOMAIN`).

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (Desktop, with Compose v2)
- An [API-Football](https://www.api-football.com) key

## 1. Configure the key

The API reads `API_FOOTBALL_KEY` from a `.env` file (passed at runtime — **not** baked
into the image):

```bash
cp .env.example .env      # then edit .env and set API_FOOTBALL_KEY=...
```

## 2. Run

```bash
./docker-run.sh           # convenience wrapper, or:
docker compose up --build
```

Then open **http://localhost:8501** for the UI, or **http://localhost:8000/docs** for the API.

## Common commands

```bash
docker compose up --build -d     # run in the background
docker compose logs -f api       # follow API logs
docker compose logs -f app       # follow UI logs
docker compose down              # stop and remove the containers
```

## Notes

- **Images are split by role.** The `api` image installs `requirements-api.txt`
  (no Streamlit); the `app` image installs `requirements-app.txt` (no ML libraries).
  Neither installs `torch` — that's training-only and lives in `proof_of_concept/`.
- **The form scaler is generated at build time** (`python -m live.build_scaler`), so the
  image is self-contained even though `models/form_scaler.pkl` isn't committed.
- **Free-tier data:** API-Football's free plan has no current-season odds. The UI takes
  odds as manual input; recent form is fetched when the configured `SEASON` is accessible
  (set `SEASON` in `.env`, e.g. `2024`). See [LIVE.md](LIVE.md).
- **No key / offline:** set `LIVE_PROVIDER=mock` in `.env` to run against the bundled
  sample data without any external API calls.
- **Secrets & size:** `.env` and `proof_of_concept/` are excluded via `.dockerignore`.

## Troubleshooting

- **Port already in use** — change the host port mappings in `docker-compose.yml`
  (e.g. `"8600:8501"`).
- **UI shows "Cannot reach the prediction API"** — the `api` service isn't healthy yet;
  check `docker compose logs api` (often a missing/invalid `API_FOOTBALL_KEY`).
- **Rebuild after code changes** — `docker compose up --build`.
