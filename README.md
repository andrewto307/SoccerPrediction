# Soccer Match Prediction System

A comprehensive machine learning system for predicting soccer match outcomes using multiple algorithms and advanced feature engineering.

## Features

- **6 ML Algorithms**: CatBoost, Random Forest, XGBoost, Gradient Boosting, Naive Bayes, and Stacking Classifier
- **63% Accuracy**: Best performance achieved with CatBoost on recent data
- **Interactive Web Interface**: Streamlit-based dashboard for model selection and predictions
- **Real-time Predictions**: Select from historical matches and get instant predictions
- **Model Comparison**: Compare performance across different algorithms
- **Feature Engineering**: Advanced features including Elo ratings, market consensus, and team performance metrics
- **Production Ready**: Comprehensive error handling, logging, and model persistence
- **Docker Support**: Fully containerized for easy deployment and reproducibility

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| CatBoost | 63.0% | 62.5% | 61.8% | 62.1% |
| XGBoost | 61.0% | 60.2% | 59.8% | 60.0% |
| Random Forest | 58.0% | 57.1% | 56.9% | 57.0% |
| Gradient Boosting | 59.5% | 58.7% | 58.3% | 58.5% |
| Naive Bayes | 55.2% | 54.8% | 54.5% | 54.6% |
| Stacking | 60.5% | 59.8% | 59.2% | 59.5% |

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd SoccerPrediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**

**Option A: Using Docker (Recommended)**
```bash
# Quick start with Docker
./docker-run.sh

# Or using docker-compose
docker-compose up
```

**Option B: Local Python environment**
```bash
streamlit run src/app.py
```

## Project Structure

```
SoccerPrediction/
├── src/
│   ├── app.py                 # Streamlit web application
│   ├── model.py              # ML model implementation
│   ├── data_preprocessing.py # Data preprocessing pipeline
│   ├── data_cleaning.py      # Data cleaning utilities
│   └── data_collection.py    # Data collection module
├── data/
│   ├── X_train.csv           # Training features
│   ├── X_test.csv            # Test features
│   ├── y_train.csv           # Training labels
│   └── y_test.csv            # Test labels
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Usage

### Web Application
1. Launch the Streamlit app: `streamlit run src/app.py`
2. Select a model type from the sidebar
3. Click "Train Model" to train the selected algorithm
4. Go to "Match Prediction" tab to make predictions on historical matches
5. View "Analytics" tab for model performance and feature importance

### Programmatic Usage
```python
from src.model import SoccerPredictionModel

# Initialize model
model = SoccerPredictionModel(model_type='catboost')

# Load data
X_train, X_test, y_train, y_test = model.load_data()

# Train model
model.train(X_train, y_train, X_test, y_test, feature_set="odds_form_teams")

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

## Docker Deployment

The application is fully containerized for easy deployment and reproducibility.

### Quick Start with Docker
```bash
# Build and run the application
./docker-run.sh

# Or using docker-compose
docker-compose up
```

### Docker Features
- **One-command setup**: No need to install Python or dependencies
- **Data persistence**: Training data and models are preserved
- **Health monitoring**: Built-in health checks
- **Development mode**: Live code reloading for development
- **Production ready**: Optimized for deployment

For detailed Docker usage, see [DOCKER.md](DOCKER.md).

# Evaluate model
metrics = model.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.2%}")
```

## Technical Details

### Data Pipeline
1. **Data Collection**: Raw match data from multiple seasons
2. **Data Cleaning**: Handle missing values, standardize formats
3. **Feature Engineering**: Create advanced features (Elo ratings, market consensus, team performance)
4. **Preprocessing**: Encode categorical variables, scale features
5. **Model Training**: Train multiple algorithms with cross-validation

### Key Features
- **Betting Odds**: Multiple bookmaker odds with normalization
- **Team Performance**: Historical performance metrics
- **Elo Ratings**: Dynamic team strength ratings
- **Market Consensus**: Aggregated betting market intelligence
- **Home Advantage**: Statistical home field advantage
- **Draw Tightness**: Market expectation for draw outcomes

### Model Architecture
- **Feature Selection**: Automated feature selection based on importance
- **Class Balancing**: SMOTE for CatBoost, class weights for other models
- **Hyperparameter Tuning**: Optimized parameters for each algorithm
- **Cross-Validation**: 10-fold cross-validation for robust evaluation

## Performance Analysis

The system achieves 63% accuracy on recent data (2018-2020) using CatBoost with the following key insights:

- **Feature Importance**: Betting odds and Elo ratings are most predictive
- **Temporal Robustness**: Performance varies across different time periods
- **League Characteristics**: Different leagues show varying predictability
- **Market Efficiency**: More recent data shows better prediction accuracy

## License

This project is licensed under the MIT License - see the LICENSE file for details.
