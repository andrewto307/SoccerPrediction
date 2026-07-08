# Proof of Concept — model training & selection

This folder preserves the experimentation behind the live app: how the data was
explored, which models were compared, and how the "best" model was chosen. **None of
this is part of the user-facing app** — the app ([../src/app.py](../src/app.py)) only does
live prediction. This is kept for anyone who wants to understand or reproduce the modelling.

## The story

- **Goal:** predict La Liga match outcomes (Home / Draw / Away).
- **Data:** 12 seasons (2008–2020) from football-data.co.uk — results, bookmaker odds,
  and engineered features (recent form, Elo, market consensus).
- **Models compared:** CatBoost, Random Forest, Gradient Boosting, XGBoost, SGD,
  Stacking, and an MLP.
- **Winner:** **CatBoost ≈ 62% accuracy** on the held-out 2019/20 season (with SMOTE for
  class balance), using the `odds_form_teams` feature set (teams + 4 form features + 15
  bookmaker-odds columns). That model — `models/cb_final.pkl` — is what the live app serves.

## Layout

```
proof_of_concept/
├── notebooks/     # exploration: data cleaning, preprocessing, model & MLP experiments
├── training/      # the training pipeline + trainer classes (moved out of src/)
│   ├── main.py             # data pipeline + optional --train-model
│   ├── model.py            # SoccerPredictionModel: train / evaluate / save / load
│   ├── base_trainer.py     # shared SMOTE + categorical encoding
│   ├── model_trainer.py    # sklearn/XGBoost trainer
│   └── catboost_trainer.py # CatBoost trainer
└── models/        # every trained comparison model + metrics.json
                   #   rf/gb/xgb/sgd _final.pkl, mlp_model.ts, soccer_prediction_model*.pkl
```

## Shared with the live pipeline (kept in `src/`)

The feature-engineering modules stay in `src/` because the **live prediction reuses them
verbatim** — that is exactly what guarantees live features match what the model was
trained on (see the parity tests in [../tests/test_parity.py](../tests/test_parity.py)):

- `src/data_preprocessing.py` — form, Elo, odds normalization, scaling
- `src/data_collection.py`, `src/data_cleaning.py` — load & clean season CSVs
- `src/model_configs.py` — model hyperparameters, feature groups, `OUTCOME_MAP`

`models/cb_final.pkl` (the chosen model) and `models/form_scaler.pkl` (its reconstructed
scaler) also stay in `models/` because the live service loads them.

## Reproducing the training

Run from the repo root, using the project venv (a path shim in `main.py` exposes the
shared `src/` modules automatically):

```bash
# data pipeline only
../myenv/bin/python proof_of_concept/training/main.py

# pipeline + train + save a model
../myenv/bin/python proof_of_concept/training/main.py \
    --train-model --model-type catboost --save-model models/cb_final.pkl
```

The interactive experiments and model comparison are in `notebooks/`.

For the production system built on top of the winning model, see [../LIVE.md](../LIVE.md).
