"""
CatBoost-specific training logic for soccer prediction.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from typing import Tuple, Dict, Any
from base_trainer import BaseTrainer


class CatBoostTrainer(BaseTrainer):
    """Handles CatBoost-specific training logic."""
        
    def prepare_data(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepare data for CatBoost training (same as notebook Cell 26).
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features  
            y_test: Test labels
            
        Returns:
            Tuple of (Xtr_bal, ytr_bal, Xte_eval, y_test)
        """
        # Convert categorical features to strings (same as notebook)
        Xtr, Xte = self.convert_categorical_to_strings(X_train, X_test)
        
        # Encode categorical features for SMOTE (fit only on training data for CatBoost)
        Xtr_enc, Xte_enc = self.encode_categorical_features(Xtr, Xte, fit_on_combined=False)
        
        # Apply SMOTENC
        Xtr_bal, ytr_bal = self.apply_smote(Xtr_enc, y_train)
        
        # For eval_set, use original test data with strings (same as notebook)
        Xte_eval = Xte.copy()
        
        return Xtr_bal, ytr_bal, Xte_eval, y_test
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_test: pd.DataFrame, y_test: pd.Series,
              hyperparameters: Dict[str, Any] = None) -> CatBoostClassifier:
        """
        Train CatBoost model (same as notebook Cell 26).
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            hyperparameters: Optional hyperparameters to override defaults
            
        Returns:
            Trained CatBoost model
        """
        # Import default parameters from centralized config
        from model_configs import MODEL_CONFIGS
        default_params = MODEL_CONFIGS['catboost']['params'].copy()
        default_params['random_state'] = self.random_state
        
        # Override with custom hyperparameters if provided
        if hyperparameters:
            default_params.update(hyperparameters)
        
        # Prepare data
        Xtr_bal, ytr_bal, Xte_eval, y_test = self.prepare_data(X_train, y_train, X_test, y_test)
        
        # Create and train model
        model = CatBoostClassifier(**default_params)
        
        # Train with original test data for eval_set (same as notebook)
        model.fit(
            Xtr_bal,
            ytr_bal,
            eval_set=(Xte_eval, y_test),
            cat_features=self.categorical_features,  # Use column names, not indices
            verbose=False
        )
        
        print("CatBoost model training completed!")
        return model
    
    def predict(self, model: CatBoostClassifier, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using CatBoost model (same as notebook).
        
        Args:
            model: Trained CatBoost model
            X: Features to predict on
            
        Returns:
            Predicted labels
        """
        # Ensure categorical features are strings (same as notebook)
        X_original = self.prepare_categorical_for_prediction(X)
        return model.predict(X_original)
    
    def predict_proba(self, model: CatBoostClassifier, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities using CatBoost model (same as notebook).
        
        Args:
            model: Trained CatBoost model
            X: Features to predict on
            
        Returns:
            Prediction probabilities
        """
        # Ensure categorical features are strings (same as notebook)
        X_original = self.prepare_categorical_for_prediction(X)
        return model.predict_proba(X_original)
