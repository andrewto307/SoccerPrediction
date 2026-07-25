"""API-key auth on the prediction endpoints (opt-in via APP_API_KEY).

Uses the FastAPI TestClient against the mock provider (conftest forces
LIVE_PROVIDER=mock), so no network or real key is needed.
"""

from fastapi.testclient import TestClient

from api.main import app
from live import config

_BODY = {
    "home_team": "Barcelona",
    "away_team": "Real Madrid",
    "date": "2026-08-22T19:00:00+00:00",
    "odds": {"home": 2.10, "draw": 3.50, "away": 3.40},
}


def test_health_open_even_with_auth_on(monkeypatch):
    monkeypatch.setattr(config, "APP_API_KEY", "secret")
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["auth_enabled"] is True


def test_predict_rejected_without_key(monkeypatch):
    monkeypatch.setattr(config, "APP_API_KEY", "secret")
    with TestClient(app) as client:
        assert client.post("/predict", json=_BODY).status_code == 401


def test_predict_rejected_with_wrong_key(monkeypatch):
    monkeypatch.setattr(config, "APP_API_KEY", "secret")
    with TestClient(app) as client:
        r = client.post("/predict", json=_BODY, headers={"X-API-Key": "nope"})
        assert r.status_code == 401


def test_predict_accepted_with_key(monkeypatch):
    monkeypatch.setattr(config, "APP_API_KEY", "secret")
    with TestClient(app) as client:
        r = client.post("/predict", json=_BODY, headers={"X-API-Key": "secret"})
        assert r.status_code == 200
        assert r.json()["prediction"] in (0, 1, 2)


def test_predict_open_when_key_unset(monkeypatch):
    monkeypatch.setattr(config, "APP_API_KEY", None)
    with TestClient(app) as client:
        assert client.post("/predict", json=_BODY).status_code == 200
