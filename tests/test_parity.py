"""Parity tests: the live feature builder must reproduce training preprocessing.

These are the single most important tests in the system. If the live pipeline
diverges from how X_test.csv was produced, the CatBoost model returns
silently-wrong answers. We therefore reconstruct the raw inputs for a real
2019/20 match and assert the rebuilt 21-feature vector — and the resulting
prediction — match the committed X_test.csv row.
"""

import pickle
from pathlib import Path

import pandas as pd
import pytest

import data_cleaning
import data_collection
from live.build_scaler import AWAY_COLS, DRAW_COLS, HOME_COLS  # type: ignore
from live.feature_builder import ODDS_FORM_TEAMS_COLUMNS, build_features, build_history_frame
from live.providers.base import FinishedMatch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"

# Bookmaker name (recognized by bookmaker_mapping) -> the column prefix in the data.
SLOT_TO_NAME = {"BW": "Bwin", "IW": "Interwetten", "WH": "William Hill", "VC": "BetVictor", "PSC": "Pinnacle"}
SLOTS = ["BW", "IW", "WH", "VC", "PSC"]


@pytest.fixture(scope="module")
def cleaned_history() -> pd.DataFrame:
    """The exact cleaned train+test history the training pipeline operated on."""
    collector = data_collection.DataCollection(str(DATA / "training.yaml"), str(DATA / "testing.yaml"))
    train, test = collector.data_collection(
        collector.get_training_file_path(), collector.get_testing_file_path()
    )
    cleaning = data_cleaning.DataCleaning(train, test)
    cleaning.clean(HOME_COLS, DRAW_COLS, AWAY_COLS)
    full = pd.concat([cleaning.get_training_ds(), cleaning.get_testing_df()])
    return full


@pytest.fixture(scope="module")
def scaler():
    with open(MODELS / "form_scaler.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def model():
    with open(MODELS / "cb_final.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def xtest() -> pd.DataFrame:
    df = pd.read_csv(DATA / "X_test.csv", index_col=0)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def _history_records(full: pd.DataFrame) -> list[FinishedMatch]:
    return [
        FinishedMatch(
            date=row.Date,
            home_team=row.HomeTeam,
            away_team=row.AwayTeam,
            home_goals=int(row.FTHG),
            away_goals=int(row.FTAG),
        )
        for row in full.itertuples()
    ]


def _pick_target(full: pd.DataFrame, xtest: pd.DataFrame):
    """Find a mid/late-season test match with all 5 bookmaker odds present."""
    test_rows = full[full["Date"] >= xtest["Date"].min()].sort_values("Date").reset_index(drop=True)
    for i in range(len(test_rows) - 1, 30, -1):  # search from late season backwards
        row = test_rows.iloc[i]
        cols = [f"{s}{o}" for s in SLOTS for o in ("H", "D", "A")]
        if row[cols].notna().all():
            xrow = xtest[
                (xtest["HomeTeam"] == row["HomeTeam"])
                & (xtest["AwayTeam"] == row["AwayTeam"])
                & (xtest["Date"] == row["Date"])
            ]
            if len(xrow) == 1:
                return row, xrow.iloc[0]
    pytest.skip("No suitable target match found")


def test_feature_parity(cleaned_history, scaler, xtest):
    raw_row, xrow = _pick_target(cleaned_history, xtest)

    # Raw decimal odds for this match -> provider-style dict.
    odds = {
        SLOT_TO_NAME[s]: (float(raw_row[f"{s}H"]), float(raw_row[f"{s}D"]), float(raw_row[f"{s}A"]))
        for s in SLOTS
    }
    history = build_history_frame(_history_records(cleaned_history))

    fr = build_features(
        home_team=raw_row["HomeTeam"],
        away_team=raw_row["AwayTeam"],
        date=raw_row["Date"],
        odds=odds,
        history=history,
        scaler=scaler,
    )

    built = fr.features.iloc[0]
    expected = xrow[ODDS_FORM_TEAMS_COLUMNS]

    assert fr.missing_bookmakers == [], "all 5 bookmakers supplied; none should be missing"
    assert built["HomeTeam"] == expected["HomeTeam"]
    assert built["AwayTeam"] == expected["AwayTeam"]
    for col in ODDS_FORM_TEAMS_COLUMNS[2:]:  # numeric features
        assert built[col] == pytest.approx(float(expected[col]), abs=1e-6), f"mismatch in {col}"


def test_full_pipeline_accuracy(cleaned_history, scaler, model, xtest):
    """End-to-end: rebuild every test fixture via the live path and reproduce the
    model's committed 62.2% accuracy. This proves the whole live pipeline (scaler
    + odds normalization + form + assembly) matches what cb_final was trained on.
    """
    from sklearn.metrics import accuracy_score

    history = build_history_frame(_history_records(cleaned_history))
    y_test = pd.read_csv(DATA / "y_test.csv", index_col=0).squeeze()
    xt = xtest.copy()
    xt["y"] = y_test.values  # y_test aligns positionally with X_test.csv

    test_rows = cleaned_history[cleaned_history["Date"] >= xtest["Date"].min()]
    y_true, y_pred = [], []
    feature_mismatches = 0
    rows_used = 0
    for raw_row in test_rows.itertuples():
        cols = [f"{s}{o}" for s in SLOTS for o in ("H", "D", "A")]
        if any(pd.isna(getattr(raw_row, c)) for c in cols):
            continue  # skip incomplete-odds rows (would trigger consensus fill)
        match = xt[(xt["HomeTeam"] == raw_row.HomeTeam) & (xt["AwayTeam"] == raw_row.AwayTeam) & (xt["Date"] == raw_row.Date)]
        if len(match) != 1:
            continue
        xrow = match.iloc[0]
        odds = {SLOT_TO_NAME[s]: (getattr(raw_row, f"{s}H"), getattr(raw_row, f"{s}D"), getattr(raw_row, f"{s}A")) for s in SLOTS}
        fr = build_features(raw_row.HomeTeam, raw_row.AwayTeam, raw_row.Date, odds, history, scaler)

        # feature parity at scale
        for col in ODDS_FORM_TEAMS_COLUMNS[2:]:
            if abs(float(fr.features.iloc[0][col]) - float(xrow[col])) > 1e-6:
                feature_mismatches += 1
                break

        y_pred.append(int(model.predict_proba(fr.features)[0].argmax()))
        y_true.append(int(xrow["y"]))
        rows_used += 1

    assert rows_used >= 150, f"expected most of 180 test rows, got {rows_used}"
    assert feature_mismatches == 0, f"{feature_mismatches} rows diverged from committed features"
    acc = accuracy_score(y_true, y_pred)
    assert acc == pytest.approx(0.6222, abs=0.02), f"live-path accuracy {acc:.4f} != committed 0.6222"


def test_prediction_parity(cleaned_history, scaler, model, xtest):
    """Prediction from rebuilt features must equal prediction from the X_test row."""
    raw_row, xrow = _pick_target(cleaned_history, xtest)
    odds = {
        SLOT_TO_NAME[s]: (float(raw_row[f"{s}H"]), float(raw_row[f"{s}D"]), float(raw_row[f"{s}A"]))
        for s in SLOTS
    }
    history = build_history_frame(_history_records(cleaned_history))
    fr = build_features(raw_row["HomeTeam"], raw_row["AwayTeam"], raw_row["Date"], odds, history, scaler)

    proba_built = model.predict_proba(fr.features)[0]
    proba_xtest = model.predict_proba(pd.DataFrame([xrow[ODDS_FORM_TEAMS_COLUMNS]]))[0]

    assert proba_built == pytest.approx(proba_xtest, abs=1e-6)
    assert sum(proba_built) == pytest.approx(1.0, abs=1e-6)
