#!/usr/bin/env python3
"""
General model training logic for non-CatBoost models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC
from collections import Counter
from sklearn.utils import shuffle
from typing import Tuple, Dict, Any, List


class ModelTrainer:
    """Handles training logic for sklearn-compatible models."""
    
    def __init__(self, categorical_features: list, random_state: int = 42):
        self.categorical_features = categorical_features
        self.random_state = random_state
        self.label_encoders = {}
        self.smote = None
        
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
        Xtr = X_train.copy()
        Xte = X_test.copy()
        
        for c in self.categorical_features:
            if c in Xtr.columns:
                Xtr[c] = Xtr[c].astype(str)
            if c in Xte.columns:
                Xte[c] = Xte[c].astype(str)
        
        # Encode categorical features
        Xtr_encoded = Xtr.copy()
        Xte_encoded = Xte.copy()
        
        for c in self.categorical_features:
            if c in Xtr_encoded.columns:
                le = LabelEncoder()
                # Fit on combined data to handle unseen categories
                combined_cats = pd.concat([Xtr_encoded[c], Xte_encoded[c]]).astype(str)
                le.fit(combined_cats)
                Xtr_encoded[c] = le.transform(Xtr_encoded[c].astype(str))
                Xte_encoded[c] = le.transform(Xte_encoded[c].astype(str))
                self.label_encoders[c] = le
        
        # Apply SMOTE if requested
        if use_smote:
            cat_idx = [Xtr_encoded.columns.get_loc(c) for c in self.categorical_features 
                      if c in Xtr_encoded.columns]
            
            self.smote = SMOTENC(
                categorical_features=cat_idx,
                sampling_strategy="not majority",
                random_state=self.random_state
            )
            
            Xtr_bal, ytr_bal = self.smote.fit_resample(Xtr_encoded, y_train)
            
            print("Before SMOTE:", Counter(y_train))
            print("After SMOTE:", Counter(ytr_bal))
            
            # Convert back to DataFrame and shuffle
            Xtr_bal = pd.DataFrame(Xtr_bal, columns=Xtr_encoded.columns)
            Xtr_bal, ytr_bal = shuffle(Xtr_bal, ytr_bal, random_state=self.random_state)
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
        # Encode categorical features
        X_encoded = X.copy()
        for c in self.categorical_features:
            if c in X_encoded.columns and X_encoded[c].dtype == 'object':
                if c in self.label_encoders:
                    X_encoded[c] = self.label_encoders[c].transform(X_encoded[c])
        
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
        # Encode categorical features
        X_encoded = X.copy()
        for c in self.categorical_features:
            if c in X_encoded.columns and X_encoded[c].dtype == 'object':
                if c in self.label_encoders:
                    X_encoded[c] = self.label_encoders[c].transform(X_encoded[c])
        
        return model.predict_proba(X_encoded)
