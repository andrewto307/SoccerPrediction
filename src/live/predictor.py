"""Load the trained model + scaler once and predict upcoming fixtures.

Orchestrates: provider -> feature_builder -> CatBoost -> labeled result. The
CatBoost artifact (cb_final.pkl) is a raw CatBoostClassifier; we call it directly
(it stores feature names and categorical columns internally, and consumes team
names as strings), so we don't go through SoccerPredictionModel at all.
"""

import logging
import pickle
from datetime import datetime

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import MinMaxScaler

from model_configs import OUTCOME_MAP
from live import config
from live.feature_builder import build_features, build_history_frame, FeatureResult
from live.providers.api_football import ApiFootballProvider, season_for_date
from live.providers.base import Fixture, FixtureProvider, OddsByBookmaker
from live.providers.mock import MockProvider

logger = logging.getLogger(__name__)

# CatBoost predict_proba column order is classes_ == [0, 1, 2] == [Away, Draw, Home].
_AWAY_IDX, _DRAW_IDX, _HOME_IDX = 0, 1, 2

# The provider/bookmaker names that map onto the model's 5 odds slots. Used to
# broadcast a single user-supplied H/D/A odds triplet onto every slot.
_MODEL_BOOKMAKER_NAMES = ["Bwin", "Interwetten", "William Hill", "BetVictor", "Pinnacle"]


def odds_from_triplet(home: float, draw: float, away: float) -> dict[str, tuple[float, float, float]]:
    """Broadcast one decimal H/D/A odds triplet across the model's 5 bookmaker slots."""
    return {name: (home, draw, away) for name in _MODEL_BOOKMAKER_NAMES}


def make_provider() -> FixtureProvider:
    """Instantiate the provider selected by config.PROVIDER."""
    if config.PROVIDER == "mock":
        return MockProvider()
    return ApiFootballProvider()


class Predictor:
    def __init__(
        self,
        provider: FixtureProvider | None = None,
        model_path=None,
        scaler_path=None,
        window: int | None = None,
    ):
        self.model: CatBoostClassifier = self._load_model(model_path or config.MODEL_PATH)
        self.scaler: MinMaxScaler = self._load_pickle(scaler_path or config.SCALER_PATH)
        self.window = window or config.FORM_WINDOW
        self._provider = provider  # lazy: only built when actually needed

    # -- loading --------------------------------------------------------------
    @staticmethod
    def _load_pickle(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_model(self, path) -> CatBoostClassifier:
        model = self._load_pickle(path)
        if not isinstance(model, CatBoostClassifier):
            raise TypeError(
                f"Expected a CatBoostClassifier at {path}, got {type(model).__name__}. "
                "Set MODEL_PATH to the odds_form_teams CatBoost artifact (models/cb_final.pkl)."
            )
        logger.info("Loaded model %s (%d features)", path, len(model.feature_names_))
        return model

    @property
    def provider(self) -> FixtureProvider:
        if self._provider is None:
            self._provider = make_provider()
        return self._provider

    # -- core prediction ------------------------------------------------------
    def _predict_from_features(self, fr: FeatureResult, fixture: Fixture | None) -> dict:
        proba = self.model.predict_proba(fr.features)[0]
        predicted = int(proba.argmax())
        return {
            "fixture_id": fixture.fixture_id if fixture else None,
            "date": fixture.date.isoformat() if fixture else None,
            "home_team": fr.home_team,
            "away_team": fr.away_team,
            "prediction": predicted,
            "label": OUTCOME_MAP[predicted],
            "probabilities": {
                "home": float(proba[_HOME_IDX]),
                "draw": float(proba[_DRAW_IDX]),
                "away": float(proba[_AWAY_IDX]),
            },
            "missing_bookmakers": fr.missing_bookmakers,
            "unmapped_teams": fr.unmapped_teams,
            "home_matches_used": fr.home_matches_used,
            "away_matches_used": fr.away_matches_used,
            "home_form_dates": fr.home_form_dates,
            "away_form_dates": fr.away_form_dates,
            "form_stale": fr.form_stale,
        }

    def predict_fixture(self, fixture: Fixture, history: pd.DataFrame) -> dict:
        odds = self.provider.match_odds(fixture.fixture_id)
        fr = build_features(
            fixture.home_team, fixture.away_team, fixture.date, odds, history, self.scaler, self.window
        )
        return self._predict_from_features(fr, fixture)

    def _safe_history(self, season: int | None = None) -> pd.DataFrame:
        """Build match history from the provider, best-effort.

        `season` selects which season's results to pull (default: the provider's
        own season). Form is a 'take what the API can give' feature: if the
        provider can't return recent results, we proceed with empty history and
        the model falls back to no-recent-form defaults rather than failing.
        """
        try:
            return build_history_frame(self.provider.recent_results(season))
        except Exception as exc:
            logger.warning("Recent results unavailable; predicting without form history: %s", exc)
            return build_history_frame([])

    def predict_upcoming(self, days: int = 7) -> list[dict]:
        fixtures = self.provider.upcoming_fixtures(days)
        history = self._safe_history()
        results = []
        for fx in fixtures:
            try:
                results.append(self.predict_fixture(fx, history))
            except Exception as exc:  # one bad fixture shouldn't sink the batch
                logger.warning("Skipping fixture %s (%s vs %s): %s",
                               fx.fixture_id, fx.home_team, fx.away_team, exc)
        return results

    def predict_fixture_id(self, fixture_id: int, days: int = 30) -> dict | None:
        """Look up a fixture by id in the upcoming window and predict it."""
        fixture = next(
            (f for f in self.provider.upcoming_fixtures(days) if f.fixture_id == fixture_id), None
        )
        if fixture is None:
            return None
        return self.predict_fixture(fixture, self._safe_history())

    def find_fixture(self, home_team: str, away_team: str, date) -> Fixture | None:
        """Return the real fixture matching (home, away) on `date` in that exact
        orientation, or None. Names are compared in the model's vocabulary, so a
        home/away swap does not match (reported as 'not accurate' upstream).
        """
        from live.team_mapping import map_team

        ts = pd.Timestamp(date)
        season = season_for_date(ts)
        want_home, _ = map_team(home_team)
        want_away, _ = map_team(away_team)
        for fx in self.provider.fixtures_on_date(ts.to_pydatetime(), season):
            if map_team(fx.home_team)[0] == want_home and map_team(fx.away_team)[0] == want_away:
                return fx
        return None

    def _provider_odds(self, fixture_id: int) -> OddsByBookmaker:
        """A fixture's odds from the provider, best-effort ({} if unavailable).

        Live odds only exist for imminent fixtures; historical and far-future
        fixtures return nothing, which the caller handles as a manual fallback.
        """
        if not fixture_id or fixture_id < 0:
            return {}
        try:
            return self.provider.match_odds(fixture_id)
        except Exception as exc:
            logger.warning("Odds unavailable for fixture %s: %s", fixture_id, exc)
            return {}

    def resolve_odds(
        self, fixture_id: int, manual: tuple[float, float, float] | None = None
    ) -> tuple[OddsByBookmaker | None, str]:
        """Prefer real provider odds; fall back to a manual (home, draw, away) triplet.

        Returns (odds_by_bookmaker | None, source) with source one of 'provider'
        (live odds), 'manual' (user-supplied), or 'none' (neither available).
        """
        provider_odds = self._provider_odds(fixture_id)
        if provider_odds:
            return provider_odds, "provider"
        if manual is not None:
            return odds_from_triplet(*manual), "manual"
        return None, "none"

    def predict_manual(
        self,
        home_team: str,
        away_team: str,
        date: datetime,
        home_odds: float,
        draw_odds: float,
        away_odds: float,
    ) -> dict:
        """Predict from user-supplied decimal odds + best-effort form from the API.

        This is the primary path when the data plan doesn't include odds: the
        caller provides the H/D/A decimal odds; form/team data is pulled from the
        provider when available (else defaulted).
        """
        season = season_for_date(pd.Timestamp(date))
        history = self._safe_history(season)
        odds = odds_from_triplet(home_odds, draw_odds, away_odds)
        fr = build_features(home_team, away_team, date, odds, history, self.scaler, self.window)
        fixture = Fixture(-1, pd.Timestamp(date).to_pydatetime(), home_team, away_team)
        return self._predict_from_features(fr, fixture)

    def list_upcoming(self, days: int = 7) -> list[dict]:
        """Upcoming fixtures with names mapped to the model vocabulary (no odds calls)."""
        from live.team_mapping import map_team

        out = []
        for fx in self.provider.upcoming_fixtures(days):
            home, _ = map_team(fx.home_team)
            away, _ = map_team(fx.away_team)
            out.append(
                {
                    "fixture_id": fx.fixture_id,
                    "date": fx.date.isoformat(),
                    "home_team": home,
                    "away_team": away,
                }
            )
        return out
