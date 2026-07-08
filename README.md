# Soccer Match Prediction System

A machine learning system that predicts soccer match outcomes using historical data and betting market intelligence.

## Overview

This system provides accurate predictions for soccer match results using multiple machine learning algorithms. It processes historical match data, betting odds, and team performance metrics to generate reliable predictions with up to 62% accuracy.

## Key Capabilities

- **Match Outcome Prediction**: Predict home wins, draws, and away wins
- **Multiple Algorithms**: Random Forest, XGBoost, Gradient Boosting, SGD Classifier, Stacking Classifier, MLP Classifier
- **Interactive Dashboard**: Web-based interface for model training and predictions
- **Historical Analysis**: Analyze past matches and model performance
- **Production Deployment**: Docker containerization for easy deployment

## Performance

CatBoost achieves the best performance with 62% accuracy using SMOTE for class balancing. The system supports multiple algorithms including Random Forest, XGBoost, Gradient Boosting, SGD Classifier, Stacking Classifier and MLP Classifier for comparison and experimentation.

## Quick Start

Get up and running in minutes:

```bash
# Clone and run with Docker (recommended)
git clone <repository-url>
cd SoccerPrediction
./docker-run.sh
```

The application will be available at `http://localhost:8000`.

### Alternative: Local Installation

```bash
# Clone the repository
git clone <repository-url>
cd SoccerPrediction

# Install dependencies
pip install -r requirements.txt

# 1) start the prediction API (loads the trained model)
python src/serve.py

# 2) in another terminal, start the app (it calls the API)
streamlit run src/app.py
```

See [LIVE.md](LIVE.md) for the full live-prediction service + API documentation.

## Usage

The app is a single **live prediction** page:

1. Pick the **home** and **away** teams and the **match date**
2. Enter the **pre-match odds** (decimal)
3. Click **Predict** — the app calls the API, which runs the CatBoost model and returns the outcome + probabilities

> The model-training and model-selection experiments (how CatBoost was chosen as the
> best model) are preserved under [`proof_of_concept/`](proof_of_concept/) and are
> intentionally not part of the app.

## Data Sources
<https://sports-statistics.com/sports-data/soccer-datasets/>

The system uses comprehensive match data of 12 La Liga seasons from 2008-2020, including:

- **Match Results**: Historical outcomes from multiple seasons
- **Betting Odds**: Multiple bookmaker odds for market intelligence
- **Team Performance**: Historical performance metrics and statistics
- **Elo Ratings**: Dynamic team strength calculations
- **Market Consensus**: Aggregated betting market data

## Project Structure

```
SoccerPrediction/
├── src/                          # live prediction service (what the app uses)
│   ├── app.py                    # Streamlit UI — single live-prediction page (calls the API)
│   ├── serve.py                  # launcher for the API
│   ├── api/                      # FastAPI service (/predict, /health, ...)
│   ├── live/                     # providers, feature builder, predictor, team/bookmaker maps
│   ├── data_collection.py        # load season CSVs        ┐
│   ├── data_cleaning.py          # clean / standardize     │ shared feature engine
│   ├── data_preprocessing.py     # form / Elo / odds       │ (reused by live prediction)
│   └── model_configs.py          # feature groups, OUTCOME_MAP  ┘
├── models/                       # cb_final.pkl (served model) + form_scaler.pkl
├── data/                         # season CSVs, train/test splits, live samples
├── proof_of_concept/             # model training & selection (NOT part of the app)
│   ├── notebooks/                # exploration notebooks
│   ├── training/                 # main.py, model.py, trainers
│   └── models/                   # all comparison models + metrics.json
├── LIVE.md                       # live prediction service documentation
├── requirements.txt
└── Dockerfile / docker-compose.yml / docker-run.sh / DOCKER.md   # (predate the API split)
```

## Docker Deployment

The system is fully containerized for production deployment:

```bash
# Quick deployment
./docker-run.sh

# Using Docker Compose
docker-compose up

# Build custom image
docker build -t soccer-prediction .
```

For detailed Docker configuration, see [DOCKER.md](DOCKER.md).

## Requirements

- Python 3.11+
- Docker (for containerized deployment)

## License

MIT
