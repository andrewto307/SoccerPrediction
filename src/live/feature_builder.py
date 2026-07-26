"""Turn live match data into the exact 21-column `odds_form_teams` feature row.

This is the parity-critical component. It reuses the *training* preprocessing
functions (DataPreprocessing.team_last_matches_performance, .normalize_betting_odds)
and the reconstructed MinMaxScaler so that, given identical raw inputs, it
reproduces a training/X_test row byte-for-byte. Any divergence here makes the
model silently wrong, so the logic intentionally delegates rather than reimplements.

Feature order (== cb_final.feature_names_ == FEATURE_GROUPS['odds_form_teams']):
    HomeTeam, AwayTeam,
    HomeTeam_points, AwayTeam_points, HomeTeam_avg_goal_diff, AwayTeam_avg_goal_diff,
    BWH,BWD,BWA, IWH,IWD,IWA, WHH,WHD,WHA, VCH,VCD,VCA, PSCH,PSCD,PSCA
"""

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data_preprocessing import DataPreprocessing
from model_configs import get_feature_list
from live.bookmaker_mapping import MODEL_BOOKMAKERS, bookmaker_to_slot
from live.providers.base import FinishedMatch, OddsByBookmaker
from live.team_mapping import map_team

# Authoritative ordered column list for the model.
ODDS_FORM_TEAMS_COLUMNS = get_feature_list("odds_form_teams")
_FORM_USED = [
    "HomeTeam_points",
    "AwayTeam_points",
    "HomeTeam_avg_goal_diff",
    "AwayTeam_avg_goal_diff",
]
# FTR_encoded convention (LabelEncoder on sorted ['A','D','H']): A=0, D=1, H=2.
_AWAY, _DRAW, _HOME = 0, 1, 2

# Stateless reuse of the training preprocessing methods.
_PRE = DataPreprocessing(pd.DataFrame(), pd.DataFrame())


@dataclass
class FeatureResult:
    features: pd.DataFrame                       # 1 row, 21 columns, model order
    home_team: str                               # mapped (training vocabulary)
    away_team: str
    missing_bookmakers: list[str] = field(default_factory=list)
    unmapped_teams: list[str] = field(default_factory=list)
    home_matches_used: int = 0
    away_matches_used: int = 0
    home_form_dates: list[str] = field(default_factory=list)  # dates of the matches used (recent first)
    away_form_dates: list[str] = field(default_factory=list)
    form_stale: bool = False                     # most recent form match is long before the fixture


def _to_naive(dt: datetime) -> pd.Timestamp:
    """Drop timezone so all date comparisons are consistent (UTC, tz-naive)."""
    ts = pd.Timestamp(dt)
    return ts.tz_convert("UTC").tz_localize(None) if ts.tz is not None else ts


def build_history_frame(results: list[FinishedMatch]) -> pd.DataFrame:
    """Build the match-history frame team_last_matches_performance expects.

    Team names are mapped to the training vocabulary so the form filter matches
    the (also-mapped) fixture teams. HST/AST are filled with 0 — they only feed
    the ShotOnTarget output, which odds_form_teams does not use.
    """
    rows = []
    for m in results:
        home, _ = map_team(m.home_team)
        away, _ = map_team(m.away_team)
        if m.home_goals > m.away_goals:
            ftr = _HOME
        elif m.home_goals < m.away_goals:
            ftr = _AWAY
        else:
            ftr = _DRAW
        rows.append(
            {
                "HomeTeam": home,
                "AwayTeam": away,
                "Date": _to_naive(m.date),
                "FTHG": m.home_goals,
                "FTAG": m.away_goals,
                "FTR_encoded": ftr,
                "HST": 0,
                "AST": 0,
            }
        )
    df = pd.DataFrame(rows, columns=["HomeTeam", "AwayTeam", "Date", "FTHG", "FTAG", "FTR_encoded", "HST", "AST"])
    if not df.empty:
        df = df.sort_values("Date", kind="mergesort").reset_index(drop=True)
    return df


def _scaled_form(
    home: str, away: str, date: pd.Timestamp, history: pd.DataFrame, scaler: MinMaxScaler, window: int
) -> tuple[dict[str, float], int, int]:
    """Compute and scale the 4 form features for one fixture."""
    h_gd, h_pts, h_sot = _PRE.team_last_matches_performance(history, home, date, window)
    a_gd, a_pts, a_sot = _PRE.team_last_matches_performance(history, away, date, window)

    # Assemble the 6-column block in the scaler's fitted order, scale, then keep
    # only the 4 columns the model uses (MinMax is per-column independent).
    raw = {
        "HomeTeam_avg_goal_diff": h_gd,
        "HomeTeam_points": h_pts,
        "HomeTeam_ShotOnTarget": h_sot,
        "AwayTeam_avg_goal_diff": a_gd,
        "AwayTeam_points": a_pts,
        "AwayTeam_ShotOnTarget": a_sot,
    }
    block = pd.DataFrame([[raw[c] for c in scaler.feature_names_in_]], columns=list(scaler.feature_names_in_))
    scaled = pd.DataFrame(scaler.transform(block), columns=list(scaler.feature_names_in_))
    form = {c: float(scaled.iloc[0][c]) for c in _FORM_USED}

    if history.empty:
        return form, 0, 0
    h_used = int((((history["HomeTeam"] == home) | (history["AwayTeam"] == home)) & (history["Date"] < date)).sum())
    a_used = int((((history["HomeTeam"] == away) | (history["AwayTeam"] == away)) & (history["Date"] < date)).sum())
    return form, min(h_used, window), min(a_used, window)


def _implied_probs(h: float, d: float, a: float) -> tuple[float, float, float]:
    """Decimal odds -> normalized implied probabilities (sum to 1)."""
    inv = [1.0 / h, 1.0 / d, 1.0 / a]
    s = sum(inv)
    return inv[0] / s, inv[1] / s, inv[2] / s


def _odds_columns(odds: OddsByBookmaker) -> tuple[dict[str, float], list[str]]:
    """Map provider odds onto the 15 normalized bookmaker columns.

    Slots backed by a recognized bookmaker are normalized via the *training*
    function (parity-critical). Missing slots are filled with the market
    consensus across all provided bookmakers — never 0.0.
    """
    # First bookmaker wins per slot.
    slot_decimals: dict[str, tuple[float, float, float]] = {}
    all_norm: list[tuple[float, float, float]] = []
    for name, (h, d, a) in odds.items():
        if h > 0 and d > 0 and a > 0:
            all_norm.append(_implied_probs(h, d, a))
        slot = bookmaker_to_slot(name)
        if slot and slot not in slot_decimals:
            slot_decimals[slot] = (h, d, a)

    # Normalize the recognized slots through the exact training routine.
    df = pd.DataFrame([{}])
    for slot, (h, d, a) in slot_decimals.items():
        df.loc[0, f"{slot}H"], df.loc[0, f"{slot}D"], df.loc[0, f"{slot}A"] = h, d, a
        _PRE.normalize_betting_odds(df, [f"{slot}H", f"{slot}D", f"{slot}A"], prefix=slot)

    if all_norm:
        consensus = tuple(sum(x) / len(all_norm) for x in zip(*all_norm))
    else:
        consensus = (1 / 3, 1 / 3, 1 / 3)

    cols: dict[str, float] = {}
    missing: list[str] = []
    for slot in MODEL_BOOKMAKERS:
        if slot in slot_decimals:
            cols[f"{slot}H"] = float(df.loc[0, f"{slot}H"])
            cols[f"{slot}D"] = float(df.loc[0, f"{slot}D"])
            cols[f"{slot}A"] = float(df.loc[0, f"{slot}A"])
        else:
            cols[f"{slot}H"], cols[f"{slot}D"], cols[f"{slot}A"] = consensus
            missing.append(slot)
    return cols, missing


_STALE_FORM_DAYS = 45  # a fixture this far after the last form match -> form is "stale"


def _recent_match_dates(history: pd.DataFrame, team: str, date: pd.Timestamp, window: int) -> list[str]:
    """Dates (most-recent first, up to `window`) of a team's matches before `date`."""
    if history.empty:
        return []
    mask = ((history["HomeTeam"] == team) | (history["AwayTeam"] == team)) & (history["Date"] < date)
    dts = history.loc[mask, "Date"].sort_values(ascending=False).head(window)
    return [d.date().isoformat() for d in dts]


def _form_is_stale(matchday: pd.Timestamp, dates: list[str]) -> bool:
    """True if the most recent form match is > _STALE_FORM_DAYS before the fixture.

    Empty `dates` means form is *absent* (reported via matches_used), not stale.
    """
    if not dates:
        return False
    most_recent = max(pd.Timestamp(d) for d in dates)
    return (matchday - most_recent).days > _STALE_FORM_DAYS


def build_features(
    home_team: str,
    away_team: str,
    date: datetime,
    odds: OddsByBookmaker,
    history: pd.DataFrame,
    scaler: MinMaxScaler,
    window: int = 5,
) -> FeatureResult:
    """Assemble the single-row 21-feature DataFrame for one upcoming fixture.

    `history` is a frame from build_history_frame(). `home_team`/`away_team` are
    raw provider names (mapped here).
    """
    mapped_home, home_ok = map_team(home_team)
    mapped_away, away_ok = map_team(away_team)
    unmapped = [t for t, ok in ((home_team, home_ok), (away_team, away_ok)) if not ok]

    ts = _to_naive(date)
    form, h_used, a_used = _scaled_form(mapped_home, mapped_away, ts, history, scaler, window)
    home_dates = _recent_match_dates(history, mapped_home, ts, window)
    away_dates = _recent_match_dates(history, mapped_away, ts, window)
    stale = _form_is_stale(ts, home_dates + away_dates)
    odds_cols, missing = _odds_columns(odds)

    record = {"HomeTeam": mapped_home, "AwayTeam": mapped_away, **form, **odds_cols}
    features = pd.DataFrame([record])[ODDS_FORM_TEAMS_COLUMNS]
    # Categoricals as strings (CatBoost), numerics as float.
    features["HomeTeam"] = features["HomeTeam"].astype(str)
    features["AwayTeam"] = features["AwayTeam"].astype(str)
    for c in ODDS_FORM_TEAMS_COLUMNS[2:]:
        features[c] = features[c].astype(float)

    return FeatureResult(
        features=features,
        home_team=mapped_home,
        away_team=mapped_away,
        missing_bookmakers=missing,
        unmapped_teams=unmapped,
        home_matches_used=h_used,
        away_matches_used=a_used,
        home_form_dates=home_dates,
        away_form_dates=away_dates,
        form_stale=stale,
    )
