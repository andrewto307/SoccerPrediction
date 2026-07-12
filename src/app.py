"""Streamlit UI — a single live match-prediction page.

Thin client: it holds no model and does no training. It collects a matchup + odds
and calls the prediction API (`POST /predict`), which serves the fixed CatBoost model.

Model-training and model-selection experiments live in `proof_of_concept/` and are
intentionally not part of this UI.

Run (from the project root, with the API already running):
    ../myenv/bin/streamlit run src/app.py
"""

import logging
import os

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

from live.team_mapping import TRAINING_TEAMS

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Soccer Match Prediction", page_icon="⚽", layout="wide")

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")

# Sent on every API call. Must match the server's APP_API_KEY when auth is on;
# harmless (ignored) when the server runs keyless.
API_KEY = os.environ.get("APP_API_KEY", "").strip()
_HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}


def main():
    st.title("⚽ Soccer Match Prediction")
    st.caption(
        "Predict a La Liga match outcome from the pre-match odds. Served by a trained "
        "CatBoost model via the prediction API; recent team form is pulled live when available."
    )

    # --- API status: only surface a problem; stay quiet when healthy --------
    try:
        health = httpx.get(f"{API_URL}/health", headers=_HEADERS, timeout=5).json()
        if not health.get("model_loaded"):
            st.warning(f"The prediction service isn't ready yet: {health.get('error')}")
    except Exception:
        st.error(
            f"Cannot reach the prediction service at {API_URL}. "
            "Make sure it's running, then reload."
        )

    # --- Match inputs -------------------------------------------------------
    st.subheader("Match")
    teams = sorted(TRAINING_TEAMS)
    c1, c2, c3 = st.columns(3)
    with c1:
        home = st.selectbox("Home team", teams, index=teams.index("Barcelona"))
    with c2:
        away = st.selectbox("Away team", teams, index=teams.index("Real Madrid"))
    with c3:
        match_date = st.date_input("Match date")

    st.subheader("Pre-match odds (decimal)")
    o1, o2, o3 = st.columns(3)
    with o1:
        home_odds = st.number_input("Home win", min_value=1.01, value=2.10, step=0.05)
    with o2:
        draw_odds = st.number_input("Draw", min_value=1.01, value=3.40, step=0.05)
    with o3:
        away_odds = st.number_input("Away win", min_value=1.01, value=3.50, step=0.05)

    # --- Predict ------------------------------------------------------------
    if st.button("🔮 Predict", type="primary"):
        if home == away:
            st.warning("Home and away teams must be different.")
        else:
            payload = {
                "home_team": home,
                "away_team": away,
                "date": pd.Timestamp(match_date).isoformat(),
                "odds": {"home": float(home_odds), "draw": float(draw_odds), "away": float(away_odds)},
            }
            resp = None
            with st.spinner("Calling the prediction API…"):
                try:
                    resp = httpx.post(f"{API_URL}/predict", json=payload, headers=_HEADERS, timeout=30)
                except Exception:
                    st.error(f"Could not reach the prediction service at {API_URL}. Is it running?")

            if resp is not None and resp.status_code != 200:
                # Surface a friendly message; never dump raw server text to the UI.
                if resp.status_code in (401, 403):
                    st.error("Authentication failed — check that APP_API_KEY matches the API service.")
                elif resp.status_code == 429:
                    st.warning("Too many requests — please wait a moment and try again.")
                else:
                    st.error(f"The prediction service returned an error (HTTP {resp.status_code}). Please try again.")
            elif resp is not None:
                res = resp.json()
                st.subheader("Prediction")
                st.metric("Predicted Outcome", res["label"])

                p = res["probabilities"]
                pc1, pc2, pc3 = st.columns(3)
                pc1.metric("Home Win", f"{p['home']:.1%}")
                pc2.metric("Draw", f"{p['draw']:.1%}")
                pc3.metric("Away Win", f"{p['away']:.1%}")

                prob_data = pd.DataFrame({
                    'Outcome': ['Home Win', 'Draw', 'Away Win'],
                    'Probability': [p['home'], p['draw'], p['away']],
                })
                fig = px.bar(prob_data, x='Outcome', y='Probability', title="Prediction Probabilities",
                             color='Probability', color_continuous_scale='RdYlGn')
                fig.update_layout(yaxis_tickformat='.1%')
                st.plotly_chart(fig, width='content')

                st.caption(
                    f"Recent form matches used — home: {res['home_matches_used']}, "
                    f"away: {res['away_matches_used']}."
                )
                if res["home_matches_used"] == 0 and res["away_matches_used"] == 0:
                    st.info(
                        "No recent form available from the provider (free-tier season limit or "
                        "off-season) — this prediction is driven by the odds you entered."
                    )


if __name__ == "__main__":
    main()
