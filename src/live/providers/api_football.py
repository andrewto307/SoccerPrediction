"""API-Football (api-sports.io) implementation of FixtureProvider.

Free tier is ~100 requests/day, so responses are cached in-memory with a TTL
(config.CACHE_TTL_SECONDS). Auth uses the direct host header `x-apisports-key`.

Docs: https://www.api-football.com/documentation-v3
"""

import logging
import time
from datetime import datetime, timedelta, timezone

import httpx

from live import config
from live.providers.base import (
    Fixture,
    FinishedMatch,
    FixtureProvider,
    OddsByBookmaker,
)

logger = logging.getLogger(__name__)

# API-Football "Match Winner" bet id.
_MATCH_WINNER_BET_ID = 1


def default_season(today: datetime | None = None) -> int:
    """La Liga season start-year for a date (season runs Aug->May)."""
    today = today or datetime.now(timezone.utc)
    return today.year if today.month >= 7 else today.year - 1


class _TTLCache:
    """Minimal time-bounded cache to conserve the request quota."""

    def __init__(self, ttl: int):
        self.ttl = ttl
        self._store: dict[str, tuple[float, object]] = {}

    def get(self, key: str):
        hit = self._store.get(key)
        if hit and (time.monotonic() - hit[0]) < self.ttl:
            return hit[1]
        return None

    def set(self, key: str, value) -> None:
        self._store[key] = (time.monotonic(), value)


class ApiFootballProvider(FixtureProvider):
    def __init__(
        self,
        api_key: str | None = None,
        league_id: int | None = None,
        season: int | None = None,
        base_url: str | None = None,
    ):
        self.api_key = api_key or config.API_FOOTBALL_KEY
        if not self.api_key:
            raise RuntimeError(
                "API_FOOTBALL_KEY is not set. Add it to the project .env or pass api_key."
            )
        self.league_id = league_id or config.LEAGUE_ID
        self.season = season or config.SEASON or default_season()
        self.base_url = (base_url or config.API_FOOTBALL_BASE_URL).rstrip("/")
        self._cache = _TTLCache(config.CACHE_TTL_SECONDS)

    # -- HTTP -----------------------------------------------------------------
    def _get(self, path: str, params: dict) -> list[dict]:
        """GET a paginated API-Football endpoint, returning the merged response[]."""
        cache_key = f"{path}?{sorted(params.items())}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        headers = {"x-apisports-key": self.api_key}
        results: list[dict] = []
        page = 1
        with httpx.Client(base_url=self.base_url, headers=headers, timeout=20.0) as client:
            while True:
                # API-Football rejects an explicit page=1; only send it when paging.
                req_params = {**params, "page": page} if page > 1 else dict(params)
                resp = client.get(path, params=req_params)
                resp.raise_for_status()
                payload = resp.json()
                errors = payload.get("errors")
                if errors:
                    raise RuntimeError(f"API-Football error on {path}: {errors}")
                results.extend(payload.get("response", []))
                paging = payload.get("paging", {}) or {}
                if page >= int(paging.get("total", 1) or 1):
                    break
                page += 1

        self._cache.set(cache_key, results)
        return results

    # -- FixtureProvider ------------------------------------------------------
    def upcoming_fixtures(self, days: int = 7) -> list[Fixture]:
        now = datetime.now(timezone.utc)
        params = {
            "league": self.league_id,
            "season": self.season,
            "from": now.date().isoformat(),
            "to": (now + timedelta(days=days)).date().isoformat(),
            "timezone": "UTC",
        }
        fixtures: list[Fixture] = []
        for item in self._get("/fixtures", params):
            fx = item["fixture"]
            status = (fx.get("status") or {}).get("short", "NS")
            if status not in ("NS", "TBD"):  # only not-yet-started
                continue
            fixtures.append(
                Fixture(
                    fixture_id=fx["id"],
                    date=datetime.fromisoformat(fx["date"].replace("Z", "+00:00")),
                    home_team=item["teams"]["home"]["name"],
                    away_team=item["teams"]["away"]["name"],
                    status=status,
                )
            )
        fixtures.sort(key=lambda f: f.date)
        return fixtures

    def recent_results(self) -> list[FinishedMatch]:
        matches: list[FinishedMatch] = []
        # Current season plus the previous one, so every team has >=5 prior games
        # even early in the campaign.
        for season in (self.season, self.season - 1):
            params = {
                "league": self.league_id,
                "season": season,
                "status": "FT",
                "timezone": "UTC",
            }
            for item in self._get("/fixtures", params):
                goals = item["goals"]
                if goals["home"] is None or goals["away"] is None:
                    continue
                matches.append(
                    FinishedMatch(
                        date=datetime.fromisoformat(
                            item["fixture"]["date"].replace("Z", "+00:00")
                        ),
                        home_team=item["teams"]["home"]["name"],
                        away_team=item["teams"]["away"]["name"],
                        home_goals=int(goals["home"]),
                        away_goals=int(goals["away"]),
                    )
                )
        matches.sort(key=lambda m: m.date)
        return matches

    def match_odds(self, fixture_id: int) -> OddsByBookmaker:
        params = {
            "fixture": fixture_id,
            "bet": _MATCH_WINNER_BET_ID,
            "league": self.league_id,
            "season": self.season,
        }
        out: OddsByBookmaker = {}
        for item in self._get("/odds", params):
            for bm in item.get("bookmakers", []):
                name = bm.get("name", "")
                triplet = {}
                for bet in bm.get("bets", []):
                    if int(bet.get("id", -1)) != _MATCH_WINNER_BET_ID:
                        continue
                    for v in bet.get("values", []):
                        triplet[v["value"].lower()] = float(v["odd"])
                if {"home", "draw", "away"} <= set(triplet):
                    out[name] = (triplet["home"], triplet["draw"], triplet["away"])
        return out
