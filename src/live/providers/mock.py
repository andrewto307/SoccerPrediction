"""Offline FixtureProvider backed by local JSON files.

Lets the API and feature pipeline run without network access or burning the
API-Football quota — used by tests/CI and local development. Reads three files
from `config.MOCK_DATA_DIR` (all optional; missing -> empty):

  upcoming.json : [{fixture_id, date, home_team, away_team, status?}, ...]
  results.json  : [{date, home_team, away_team, home_goals, away_goals}, ...]
  odds.json     : {"<fixture_id>": {"<bookmaker>": [home, draw, away], ...}, ...}
"""

import json
from datetime import datetime
from pathlib import Path

from live import config
from live.providers.base import (
    Fixture,
    FinishedMatch,
    FixtureProvider,
    OddsByBookmaker,
)


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class MockProvider(FixtureProvider):
    def __init__(self, data_dir: Path | None = None):
        self.data_dir = Path(data_dir or config.MOCK_DATA_DIR)

    def _load(self, name: str, default):
        path = self.data_dir / name
        if not path.exists():
            return default
        with open(path) as f:
            return json.load(f)

    def upcoming_fixtures(self, days: int = 7) -> list[Fixture]:
        rows = self._load("upcoming.json", [])
        return sorted(
            (
                Fixture(
                    fixture_id=int(r["fixture_id"]),
                    date=_parse_dt(r["date"]),
                    home_team=r["home_team"],
                    away_team=r["away_team"],
                    status=r.get("status", "NS"),
                )
                for r in rows
            ),
            key=lambda f: f.date,
        )

    def recent_results(self) -> list[FinishedMatch]:
        rows = self._load("results.json", [])
        return sorted(
            (
                FinishedMatch(
                    date=_parse_dt(r["date"]),
                    home_team=r["home_team"],
                    away_team=r["away_team"],
                    home_goals=int(r["home_goals"]),
                    away_goals=int(r["away_goals"]),
                )
                for r in rows
            ),
            key=lambda m: m.date,
        )

    def match_odds(self, fixture_id: int) -> OddsByBookmaker:
        odds = self._load("odds.json", {})
        entry = odds.get(str(fixture_id), {})
        return {bm: (float(v[0]), float(v[1]), float(v[2])) for bm, v in entry.items()}
