"""Reconstruct and persist the MinMaxScaler the model was trained with.

The training pipeline fits a MinMaxScaler on the form columns but `save_model()`
never persists it. CatBoost's tree thresholds were learned in scaled space, so
live form features MUST be scaled with the *identical* min/max or predictions are
silently wrong (we measured an ~8pt accuracy swing from the wrong scaler).

Subtlety: re-running the current pipeline does NOT reproduce the committed
X_train.csv / X_test.csv — the committed files were generated from a slightly
different training-season set, so the form min/max differ (e.g.
HomeTeam_avg_goal_diff range [-3, 5] committed vs [-4, 5] today). cb_final.pkl
scores 62.2% on the committed scaling but only 54.4% on a fresh fit, so the
committed scaling is the ground truth.

We therefore RECOVER the committed scaler exactly. The committed *training* set
used a different season range (its early-season form windows differ), so we
recover from the TEST split only: there the raw form computation is provably
unchanged (affine-fit residual ~1e-16), so each committed scaled form value is an
exact affine function of the current raw value. We fit that affine map
(raw -> committed_scaled) per column and invert it to get the committed min/max.

Run once (from src/):  ../myenv/bin/python -m live.build_scaler
"""

import logging
import pickle
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import data_cleaning
import data_collection
import data_preprocessing
from live import config

logger = logging.getLogger(__name__)

# Bookmaker odds columns whose NaNs the cleaning step backfills (identical to main.py).
HOME_COLS = ["GBH", "IWH", "LBH", "SBH", "PSH", "SJH", "VCH", "BSH", "PSCH"]
DRAW_COLS = ["GBD", "IWD", "LBD", "SBD", "PSD", "SJD", "VCD", "BSD", "PSCD"]
AWAY_COLS = ["GBA", "IWA", "LBA", "SBA", "PSA", "SJA", "VCA", "BSA", "PSCA"]

# The 6 form columns the scaler was fit on, in the pipeline's fitted order.
FORM_COLUMNS = [
    "HomeTeam_avg_goal_diff",
    "HomeTeam_points",
    "HomeTeam_ShotOnTarget",
    "AwayTeam_avg_goal_diff",
    "AwayTeam_points",
    "AwayTeam_ShotOnTarget",
]
_KEYS = ["HomeTeam", "AwayTeam", "Date"]


def _raw_form() -> pd.DataFrame:
    """Compute the unscaled form features for every match (current pipeline)."""
    collector = data_collection.DataCollection(
        str(config.DATA_DIR / "training.yaml"), str(config.DATA_DIR / "testing.yaml")
    )
    train, test = collector.data_collection(
        collector.get_training_file_path(), collector.get_testing_file_path()
    )
    cleaning = data_cleaning.DataCleaning(train, test)
    cleaning.clean(HOME_COLS, DRAW_COLS, AWAY_COLS)
    pre = data_preprocessing.DataPreprocessing(cleaning.get_training_ds(), cleaning.get_testing_df())

    ds = pd.concat([cleaning.get_training_ds(), cleaning.get_testing_df()])
    ds = pre.match_encode(ds)
    ds = pre.add_team_performance_features(ds)
    out = ds[_KEYS + FORM_COLUMNS].copy()
    out["Date"] = pd.to_datetime(out["Date"])
    return out


def _committed_scaled() -> pd.DataFrame:
    """Committed scaled form values from the TEST split (raw provably unchanged there).

    X_train is intentionally excluded: the committed training set used a different
    season range, so current raw form for early-season train rows differs and the
    affine recovery would be invalid. The recovered min/max are a property of the
    scaler, independent of which (raw-stable) rows we fit on.
    """
    df = pd.read_csv(config.DATA_DIR / "X_test.csv", index_col=0)
    df["Date"] = pd.to_datetime(df["Date"])
    return df[_KEYS + FORM_COLUMNS]


def build_scaler(save: bool = True) -> MinMaxScaler:
    """Recover the committed scaler and (optionally) persist it."""
    raw = _raw_form()
    committed = _committed_scaled()

    merged = raw.merge(committed, on=_KEYS, suffixes=("_raw", "_scaled"))
    if len(merged) < 0.9 * len(committed):
        logger.warning("Only %d/%d committed rows aligned for scaler recovery", len(merged), len(committed))

    mins, maxs = [], []
    for col in FORM_COLUMNS:
        x = merged[f"{col}_raw"].to_numpy(dtype=float)
        y = merged[f"{col}_scaled"].to_numpy(dtype=float)
        # committed_scaled = a*raw + b  =>  range = 1/a, min = -b/a
        a, b = np.polyfit(x, y, 1)
        resid = float(np.abs(y - (a * x + b)).max())
        if resid > 1e-6:
            raise RuntimeError(
                f"Form column {col} is not an affine rescale of raw (resid={resid:.2e}); "
                "the raw form computation changed and the scaler cannot be recovered this way."
            )
        rng = 1.0 / a
        cmin = -b / a
        mins.append(cmin)
        maxs.append(cmin + rng)
        logger.info("Recovered %-26s min=%.4f max=%.4f (resid=%.1e)", col, cmin, cmin + rng, resid)

    # Build a scaler with exactly these min/max by fitting on the two extreme rows.
    bounds = pd.DataFrame([mins, maxs], columns=FORM_COLUMNS)
    scaler = MinMaxScaler().fit(bounds)

    _validate(scaler, raw, committed)

    if save:
        config.SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(config.SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
        logger.info("Saved recovered form scaler -> %s", config.SCALER_PATH)
    return scaler


def _validate(scaler: MinMaxScaler, raw: pd.DataFrame, committed: pd.DataFrame) -> None:
    """Assert the recovered scaler reproduces the committed scaled form values."""
    merged = raw.merge(committed, on=_KEYS, suffixes=("_raw", "_scaled"))
    raw_block = merged[[f"{c}_raw" for c in FORM_COLUMNS]].astype(float)
    raw_block.columns = FORM_COLUMNS  # keep feature names so MinMaxScaler stays quiet
    got = scaler.transform(raw_block)
    want = merged[[f"{c}_scaled" for c in FORM_COLUMNS]].to_numpy(dtype=float)
    max_diff = float(np.abs(got - want).max())
    if max_diff > 1e-6:
        raise RuntimeError(f"Recovered scaler does not reproduce committed data (max_diff={max_diff:.2e})")
    logger.info("Scaler validation OK (max reproduction diff=%.1e over %d rows)", max_diff, len(merged))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    s = build_scaler(save=True)
    print("feature_names_in_:", list(s.feature_names_in_))
    print("data_min_:", s.data_min_)
    print("data_max_:", s.data_max_)
