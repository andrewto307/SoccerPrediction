"""Live data + inference layer for the Soccer Prediction System.

Pulls upcoming La Liga fixtures and pre-match odds from an external provider,
rebuilds the exact `odds_form_teams` features the trained CatBoost model expects,
and produces predictions without retraining.
"""
