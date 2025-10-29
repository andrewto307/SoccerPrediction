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

# Run the application
streamlit run src/app.py
```

## Usage

1. **Select Model**: Choose from available algorithms in the sidebar
2. **Train Model**: Click "Train Model" to train the selected algorithm
3. **Make Predictions**: Navigate to "Match Prediction" tab to predict outcomes for historical matches
4. **View Results**: See prediction probabilities and model performance metrics

## Data Sources
<https://sports-statistics.com/sports-data/soccer-datasets/>

The system uses comprehensive match data of 12 La Liga seasons from 2008-202, including:

- **Match Results**: Historical outcomes from multiple seasons
- **Betting Odds**: Multiple bookmaker odds for market intelligence
- **Team Performance**: Historical performance metrics and statistics
- **Elo Ratings**: Dynamic team strength calculations
- **Market Consensus**: Aggregated betting market data

## Project Structure

```
SoccerPrediction/
├── src/
│   ├── app.py                 # Streamlit web application
│   ├── main.py               # Command-line interface
│   ├── model.py              # Core ML model implementation
│   ├── data_collection.py    # Data collection module
│   ├── data_cleaning.py      # Data cleaning pipeline
│   ├── data_preprocessing.py # Data preprocessing pipeline
│   ├── model_configs.py      # Model configurations
│   ├── base_trainer.py       # Base trainer class
│   ├── model_trainer.py      # General model trainer
│   └── catboost_trainer.py   # CatBoost-specific trainer
├── data/                     # Training datasets and configurations
├── notebook/                 # Jupyter notebooks for development
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose setup
├── docker-run.sh           # Quick Docker deployment script
└── DOCKER.md               # Docker documentation
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
