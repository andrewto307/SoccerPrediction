# Multi-target build: one shared base, two runtime images (api, app).
# Build a specific target with `--target api` / `--target app`
# (docker-compose selects the target per service).

# ---------- base ----------
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
# curl is used by the container healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# ---------- api: FastAPI prediction service ----------
FROM base AS api
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
# Code + artifacts the API needs at runtime.
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/
# form_scaler.pkl is generated (not committed) — build it so the model loads at startup.
RUN cd src && python -m live.build_scaler
ENV LIVE_PROVIDER=api_football \
    HOST=0.0.0.0 \
    PORT=8000
EXPOSE 8000
WORKDIR /app/src
CMD ["python", "serve.py"]

# ---------- app: Streamlit UI (thin client, calls the API) ----------
FROM base AS app
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt
COPY src/ ./src/
ENV API_URL=http://api:8000
EXPOSE 8501
WORKDIR /app/src
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
