"""Offline tests for the live layer: mappings, mock provider, end-to-end predict."""

import pytest

from model_configs import OUTCOME_MAP
from live.bookmaker_mapping import bookmaker_to_slot
from live.predictor import Predictor
from live.providers.mock import MockProvider
from live.team_mapping import map_team


@pytest.mark.parametrize(
    "external, expected",
    [
        ("Atletico Madrid", "Ath Madrid"),
        ("Athletic Club", "Ath Bilbao"),
        ("Real Sociedad", "Sociedad"),
        ("Espanyol", "Espanol"),
        ("Rayo Vallecano", "Vallecano"),
        ("Celta Vigo", "Celta"),
        ("Barcelona", "Barcelona"),  # already canonical
    ],
)
def test_team_mapping(external, expected):
    mapped, matched = map_team(external)
    assert mapped == expected
    assert matched is True


def test_team_mapping_unknown():
    mapped, matched = map_team("Some Fake FC")
    assert mapped == "Some Fake FC"
    assert matched is False


@pytest.mark.parametrize(
    "name, slot",
    [
        ("Bwin", "BW"),
        ("Interwetten", "IW"),
        ("William Hill", "WH"),
        ("BetVictor", "VC"),
        ("Pinnacle", "PSC"),
        ("Bet365", None),  # not one of the model slots
    ],
)
def test_bookmaker_mapping(name, slot):
    assert bookmaker_to_slot(name) == slot


def test_mock_provider_contract():
    prov = MockProvider()
    fixtures = prov.upcoming_fixtures()
    assert len(fixtures) == 2
    assert {f.home_team for f in fixtures} == {"Barcelona", "Sevilla"}
    assert len(prov.recent_results()) == 6
    odds = prov.match_odds(900001)
    assert "Pinnacle" in odds and len(odds["Pinnacle"]) == 3


def test_predict_upcoming_offline():
    predictor = Predictor(provider=MockProvider())
    results = predictor.predict_upcoming(days=30)
    assert len(results) == 2

    for r in results:
        assert r["prediction"] in (0, 1, 2)
        assert r["label"] == OUTCOME_MAP[r["prediction"]]
        p = r["probabilities"]
        assert sum(p.values()) == pytest.approx(1.0, abs=1e-6)
        assert r["unmapped_teams"] == []

    # Fixture 900002 omits Interwetten and BetVictor -> those slots must be flagged
    # missing (consensus-filled), but the prediction is still produced.
    second = next(r for r in results if r["fixture_id"] == 900002)
    assert set(second["missing_bookmakers"]) == {"IW", "VC"}


def test_predict_manual_offline():
    """User-supplied odds path: all 5 slots filled from the one triplet."""
    predictor = Predictor(provider=MockProvider())
    r = predictor.predict_manual("Barcelona", "Real Madrid", "2026-08-22T19:00:00+00:00", 2.10, 3.50, 3.40)
    assert r["prediction"] in (0, 1, 2)
    assert r["label"] == OUTCOME_MAP[r["prediction"]]
    assert sum(r["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
    assert r["missing_bookmakers"] == []
    assert r["unmapped_teams"] == []


class _NoResultsProvider(MockProvider):
    def recent_results(self):
        raise RuntimeError("season locked (free plan)")


def test_predict_manual_resilient_without_history():
    """Free-tier reality: prediction still works when the API can't supply form."""
    predictor = Predictor(provider=_NoResultsProvider())
    r = predictor.predict_manual("Barcelona", "Real Madrid", "2026-08-22T19:00:00+00:00", 2.10, 3.50, 3.40)
    assert r["prediction"] in (0, 1, 2)
    assert r["home_matches_used"] == 0 and r["away_matches_used"] == 0
    assert sum(r["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
