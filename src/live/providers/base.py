"""Provider abstraction: fixtures, recent results and odds for a competition.

Implementations translate an external data source into these neutral dataclasses
so the rest of the pipeline (feature builder, predictor, API) is provider-agnostic.
Team names here are the provider's *raw* names; translation to the model's
vocabulary happens later in the feature builder via team_mapping.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Fixture:
    """An upcoming (not-yet-played) match."""
    fixture_id: int
    date: datetime          # kickoff, timezone-aware (UTC)
    home_team: str          # raw provider name
    away_team: str          # raw provider name
    status: str = "NS"      # provider status code (NS = not started)


@dataclass(frozen=True)
class FinishedMatch:
    """A completed match, used to compute rolling form."""
    date: datetime
    home_team: str          # raw provider name
    away_team: str          # raw provider name
    home_goals: int
    away_goals: int


# bookmaker name -> (home, draw, away) decimal odds for the Match-Winner market
OddsByBookmaker = dict[str, tuple[float, float, float]]


class FixtureProvider(ABC):
    """Source of upcoming fixtures, recent results and pre-match odds."""

    @abstractmethod
    def upcoming_fixtures(self, days: int = 7) -> list[Fixture]:
        """Fixtures kicking off within the next `days` days."""

    @abstractmethod
    def recent_results(self) -> list[FinishedMatch]:
        """Finished matches (current season, plus prior season for early-season form)."""

    @abstractmethod
    def match_odds(self, fixture_id: int) -> OddsByBookmaker:
        """Pre-match 1X2 odds for a fixture, keyed by bookmaker name."""
