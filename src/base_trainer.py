"""
Base trainer class with common functionality for all model trainers.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC
from collections import Counter
from sklearn.utils import shuffle
from typing import Tuple, Dict, Any, List


class BaseTrainer:
    """Base class with common functionality for all model trainers."""
    
    def __init__(self, categorical_features: list, random_state: int = 42):
        self.categorical_features = categorical_features
        self.random_state = random_state
        self.label_encoders = {}
        self.smote = None
    
    def convert_categorical_to_strings(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert categorical features to strings.
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            Tuple of (X_train_converted, X_test_converted)
        """
        Xtr = X_train.copy()
        Xte = X_test.copy()
        
        for c in self.categorical_features:
            if c in Xtr.columns:
                Xtr[c] = Xtr[c].astype(str)
            if c in Xte.columns:
                Xte[c] = Xte[c].astype(str)
        
        return Xtr, Xte
    
    def encode_categorical_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                  fit_on_combined: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Encode categorical features using LabelEncoder.
        
        Args:
            X_train: Training features
            X_test: Test features
            fit_on_combined: Whether to fit encoder on combined train+test data
            
        Returns:
            Tuple of (X_train_encoded, X_test_encoded)
        """
        Xtr_encoded = X_train.copy()
        Xte_encoded = X_test.copy()
        
        for c in self.categorical_features:
            if c in Xtr_encoded.columns:
                le = LabelEncoder()
                
                if fit_on_combined:
                    # Fit on combined data to handle unseen categories
                    combined_cats = pd.concat([Xtr_encoded[c], Xte_encoded[c]]).astype(str)
                    le.fit(combined_cats)
                else:
                    # Fit only on training data
                    le.fit(Xtr_encoded[c].astype(str))
                
                Xtr_encoded[c] = le.transform(Xtr_encoded[c].astype(str))
                Xte_encoded[c] = le.transform(Xte_encoded[c].astype(str))
                self.label_encoders[c] = le
        
        return Xtr_encoded, Xte_encoded
    
    def apply_smote(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTENC balancing.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Tuple of (balanced_X_train, balanced_y_train)
        """
        cat_idx = [X_train.columns.get_loc(c) for c in self.categorical_features 
                  if c in X_train.columns]
        
        self.smote = SMOTENC(
            categorical_features=cat_idx,
            sampling_strategy="not majority",
            random_state=self.random_state
        )
        
        Xtr_bal, ytr_bal = self.smote.fit_resample(X_train, y_train)
        
        print("Before SMOTE:", Counter(y_train))
        print("After SMOTE:", Counter(ytr_bal))
        
        # Convert back to DataFrame and shuffle
        Xtr_bal = pd.DataFrame(Xtr_bal, columns=X_train.columns)
        Xtr_bal, ytr_bal = shuffle(Xtr_bal, ytr_bal, random_state=self.random_state)
        
        return Xtr_bal, ytr_bal
    
    def prepare_categorical_for_prediction(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare categorical features for prediction (convert to strings).
        
        Args:
            X: Features to prepare
            
        Returns:
            Prepared features
        """
        X_prepared = X.copy()
        for c in self.categorical_features:
            if c in X_prepared.columns:
                X_prepared[c] = X_prepared[c].astype(str)
        return X_prepared
    
    def encode_categorical_for_prediction(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features for prediction using stored encoders.
        
        Args:
            X: Features to encode
            
        Returns:
            Encoded features
        """
        X_encoded = X.copy()
        for c in self.categorical_features:
            if c in X_encoded.columns and X_encoded[c].dtype == 'object':
                if c in self.label_encoders:
                    X_encoded[c] = self.label_encoders[c].transform(X_encoded[c])
        return X_encoded
