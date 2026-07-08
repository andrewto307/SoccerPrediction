"""
General model training logic for non-CatBoost models.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from base_trainer import BaseTrainer


class ModelTrainer(BaseTrainer):
    """Handles training logic for sklearn-compatible models."""
        
    def prepare_data(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_test: pd.DataFrame, y_test: pd.Series,
                    use_smote: bool = True, use_class_weights: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepare data for sklearn model training.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            use_smote: Whether to apply SMOTE balancing
            use_class_weights: Whether to use class weights (for models that support it)
            
        Returns:
            Tuple of (Xtr_bal, ytr_bal, Xte_encoded, y_test)
        """
        # Convert categorical features to strings
        Xtr, Xte = self.convert_categorical_to_strings(X_train, X_test)
        
        # Encode categorical features (fit on combined data for sklearn models)
        Xtr_encoded, Xte_encoded = self.encode_categorical_features(Xtr, Xte, fit_on_combined=True)
        
        # Apply SMOTE if requested
        if use_smote:
            Xtr_bal, ytr_bal = self.apply_smote(Xtr_encoded, y_train)
        else:
            Xtr_bal = Xtr_encoded
            ytr_bal = y_train
        
        return Xtr_bal, ytr_bal, Xte_encoded, y_test
    
    def get_class_weights(self, y_train: pd.Series) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced datasets.
        
        Args:
            y_train: Training labels
            
        Returns:
            Dictionary of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        return dict(zip(classes, class_weights))
    
    def predict(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using sklearn model.
        
        Args:
            model: Trained sklearn model
            X: Features to predict on
            
        Returns:
            Predicted labels
        """
        X_encoded = self.encode_categorical_for_prediction(X)
        return model.predict(X_encoded)
    
    def predict_proba(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities using sklearn model.
        
        Args:
            model: Trained sklearn model
            X: Features to predict on
            
        Returns:
            Prediction probabilities
        """
        X_encoded = self.encode_categorical_for_prediction(X)
        return model.predict_proba(X_encoded)
