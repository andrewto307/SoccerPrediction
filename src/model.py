"""
Model class for soccer prediction. Main interface for training and predicting using all the specialized trainers.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB

from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# Import our custom trainers and configurations
from catboost_trainer import CatBoostTrainer
from model_trainer import ModelTrainer
from model_configs import MODEL_CONFIGS, get_feature_list, get_categorical_features


class SoccerPredictionModel:
    """
    Soccer match prediction model with support for multiple ML algorithms.
    
    This model predicts match outcomes (Home Win, Draw, Away Win) using:
    - Team performance features
    - Bookmaker odds and market consensus
    
    Supported models:
    - catboost: CatBoost Classifier (default, best performance)
    - random_forest: Random Forest Classifier
    - gradient_boosting: Gradient Boosting Classifier
    - naive_bayes: Gaussian Naive Bayes
    - xgboost: XGBoost Classifier
    - stacking: Stacking Classifier (combines RF, GB, and NB)
    """
    
    def __init__(self, model_type: str = 'catboost'):
        """
        Initialize the soccer prediction model.
        
        Args:
            model_type: Type of model to use. Options: 'catboost', 'random_forest', 
                       'gradient_boosting', 'naive_bayes', 'xgboost', 'stacking'
        """
        if model_type not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model type: {model_type}. Available options: {list(MODEL_CONFIGS.keys())}")
        
        self.model_type = model_type
        self.model = None
        self.feature_columns = None
        self.label_encoders = {}
        self.smote = None
        self.categorical_features = ["HomeTeam", "AwayTeam"]
    
    def load_data(self, data_dir: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load training and testing data from CSV files.
        
        Args:
            data_dir: Directory containing the data files. If None, auto-detect.
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if data_dir is None:
            # Auto-detect data directory
            current_dir = Path(__file__).parent
            possible_paths = [
                current_dir / "data",  # When running from src/
                current_dir.parent / "data",  # When running from SoccerPrediction/
                current_dir.parent.parent / "SoccerPrediction" / "data"  # When running from root
            ]
            
            for path in possible_paths:
                if path.exists() and (path / "X_train.csv").exists():
                    data_path = path
                    break
            else:
                raise FileNotFoundError("Could not find data directory with X_train.csv")
        else:
            data_path = Path(data_dir)
        
        # Load full datasets
        X_train_full = pd.read_csv(data_path / "X_train.csv", index_col=0)
        X_test_full = pd.read_csv(data_path / "X_test.csv", index_col=0)
        y_train = pd.read_csv(data_path / "y_train.csv", index_col=0).squeeze()
        y_test = pd.read_csv(data_path / "y_test.csv", index_col=0).squeeze()
        
        # Remove non-feature columns
        non_features = ["Div", "Date", "HomeTeam_ShotOnTarget", "AwayTeam_ShotOnTarget"]
        
        X_train_features = X_train_full.drop(columns=non_features, errors='ignore')
        X_test_features = X_test_full.drop(columns=non_features, errors='ignore')

        # Load feature order from CSV
        try:
            feat_order = pd.read_csv(data_path / "feature_columns.csv")["feature"].tolist()
        except FileNotFoundError:
            feat_order = X_train_features.columns.tolist()
        
        # Align features to match training order 
        X_train = self.align_features(X_train_features, feat_order)
        X_test = self.align_features(X_test_features, feat_order)
        
        return X_train, X_test, y_train, y_test
    
    def align_features(self, df: pd.DataFrame, feat_order: List[str]) -> pd.DataFrame:
        """
        Align DataFrame columns to match training feature order.
        
        Args:
            df: DataFrame to align
            feat_order: List of feature names in training order
            
        Returns:
            Aligned DataFrame
        """
        
        # Add any missing training columns as zeros
        missing = [c for c in feat_order if c not in df.columns]
        if missing:
            for c in missing:
                df[c] = 0.0
        
        # Drop any extras not used in training
        extra = [c for c in df.columns if c not in feat_order]
        if extra:
            df = df.drop(columns=extra, errors="ignore")
        
        # Put in the exact training order with fill_value=0.0
        df = df.reindex(columns=feat_order, fill_value=0.0)
        
        return df
    
    def prepare_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                        feature_set: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare features based on the selected feature set.
        
        Args:
            X_train: Training features
            X_test: Testing features
            feature_set: Feature set to use ("odds_form_teams", "odds_form_teams_elo", etc.)
            
        Returns:
            Tuple of (X_train_processed, X_test_processed)
        """
        # Get feature list from centralized configuration
        feat_all = get_feature_list(feature_set)
        feat_cats = get_categorical_features(feature_set)

        # Select only the features for this feature set
        X_train = X_train[feat_all]
        X_test = X_test[feat_all]

        # Convert categorical features to strings
        for c in feat_cats:
            if c in X_train.columns:
                X_train.loc[:, c] = X_train.loc[:, c].astype("string")
                X_test.loc[:, c] = X_test.loc[:, c].astype("string")
        
        return X_train, X_test
    
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_test: pd.DataFrame, y_test: pd.Series,
              feature_set: str,
              apply_smote: bool = True,
              hyperparameters: Optional[Dict] = None) -> None:
        """
        Train the selected model type using specialized trainers.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Testing features
            y_test: Testing labels
            feature_set: Feature set to use (required - no default)
            apply_smote: Whether to apply SMOTE balancing
            hyperparameters: Custom hyperparameters (optional)
        """
        # Always prepare features based on the specified feature set
        X_train, X_test = self.prepare_features(X_train, X_test, feature_set)
        
        # Store feature columns for later use
        self.feature_columns = X_train.columns.tolist()
        
        # Train model based on type using specialized trainers
        if self.model_type == 'catboost':
            # Use CatBoost trainer
            trainer = CatBoostTrainer(self.categorical_features, random_state=42)
            self.model = trainer.train(X_train, y_train, X_test, y_test, hyperparameters)
            
            # Store trainer for predictions
            self.catboost_trainer = trainer
            
        else:
            # Use general model trainer for sklearn models
            model_config = MODEL_CONFIGS[self.model_type]
            use_smote = model_config.get('use_smote', apply_smote)
            use_class_weights = model_config.get('use_class_weights', True)
            
            trainer = ModelTrainer(self.categorical_features, random_state=42)
            Xtr_bal, ytr_bal, X_test, y_test = trainer.prepare_data(
                X_train, y_train, X_test, y_test, use_smote, use_class_weights
            )
            
            # Create and train model
            model_class = model_config['class']
            default_params = model_config['params'].copy()
            
            # Handle class imbalance
            sample_weight = None
            if use_class_weights:
                config_class_weight = model_config.get('class_weight')
                if self.model_type == 'xgboost':
                    # XGBoost: convert configured class weights to per-sample weights
                    if config_class_weight is not None:
                        from sklearn.utils.class_weight import compute_sample_weight
                        sample_weight = compute_sample_weight(class_weight=config_class_weight, y=ytr_bal)
                else:
                    # sklearn-compatible models: pass class_weight directly
                    if config_class_weight is not None:
                        default_params['class_weight'] = config_class_weight
                    elif 'class_weight' not in default_params:
                        class_weights = trainer.get_class_weights(y_train)
                        default_params['class_weight'] = class_weights
            
            # Use custom hyperparameters if provided
            if hyperparameters:
                default_params.update(hyperparameters)
            
            self.model = model_class(**default_params)
            if sample_weight is not None:
                self.model.fit(Xtr_bal, ytr_bal, sample_weight=sample_weight)
            else:
                self.model.fit(Xtr_bal, ytr_bal)
            
            # Store trainer for predictions
            self.model_trainer = trainer
        
        print("Model training completed!")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data using specialized trainers.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Align features to training order
        X_aligned = self.align_features(X, self.feature_columns)
        
        # Use appropriate trainer for predictions
        if self.model_type == 'catboost':
            return self.catboost_trainer.predict(self.model, X_aligned)
        else:
            return self.model_trainer.predict(self.model, X_aligned)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities using specialized trainers.
        
        Args:
            X: Features to predict on
            
        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Align features to training order
        X_aligned = self.align_features(X, self.feature_columns)
        
        # Use appropriate trainer for predictions
        if self.model_type == 'catboost':
            return self.catboost_trainer.predict_proba(self.model, X_aligned)
        else:
            return self.model_trainer.predict_proba(self.model, X_aligned)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        logloss = log_loss(y_test, y_proba)
        
        print(f"Test Set Accuracy: {accuracy:.2f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # Get best score if available (only for CatBoost and XGBoost)
        best_score = None
        if hasattr(self.model, 'get_best_score'):
            best_score = self.model.get_best_score()
            print(f"Best Score: {best_score}")
        
        # Calculate additional metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "log_loss": logloss,
            "best_score": best_score
        }
    
    
    def save_model(self, filepath: str) -> None:
        """Save complete model with metadata"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        from datetime import datetime
        
        model_data = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "label_encoders": self.label_encoders,
            "smote": self.smote,
            "categorical_features": self.categorical_features,
            "model_type": self.model_type,
            "accuracy": getattr(self, 'accuracy', None),
            "training_date": datetime.now().isoformat(),
            "data_version": "v1.0",
            "model_version": "1.0"
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model and metadata.
        
        Args:
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data["model"]
        self.feature_columns = model_data["feature_columns"]
        self.label_encoders = model_data["label_encoders"]
        self.smote = model_data["smote"]
        self.categorical_features = model_data["categorical_features"]
        
        print(f"Model loaded from {filepath}")


