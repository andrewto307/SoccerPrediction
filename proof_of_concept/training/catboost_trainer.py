"""
CatBoost-specific training logic for soccer prediction.
"""

import logging
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from typing import Tuple, Dict, Any
from base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class CatBoostTrainer(BaseTrainer):
    """Handles CatBoost-specific training logic."""

    def prepare_data(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels

        Returns:
            Tuple of (Xtr_bal, ytr_bal, Xte_eval, y_test)
        """
        Xtr, Xte = self.convert_categorical_to_strings(X_train, X_test)

        # Encode categoricals to integers so SMOTENC can operate on numeric data
        Xtr_enc, _ = self.encode_categorical_features(Xtr, Xte, fit_on_combined=False)

        # Apply SMOTENC on encoded data
        Xtr_bal, ytr_bal = self.apply_smote(Xtr_enc, y_train)

        # Decode categoricals back to strings so CatBoost sees the same vocabulary
        # in training as in eval (CatBoost handles categoricals natively from strings)
        for c in self.categorical_features:
            if c in Xtr_bal.columns and c in self.label_encoders:
                le = self.label_encoders[c]
                Xtr_bal[c] = le.inverse_transform(Xtr_bal[c].astype(int))

        Xte_eval = Xte.copy()

        return Xtr_bal, ytr_bal, Xte_eval, y_test

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_test: pd.DataFrame, y_test: pd.Series,
              hyperparameters: Dict[str, Any] = None) -> CatBoostClassifier:
        """

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

        model.fit(
            Xtr_bal,
            ytr_bal,
            eval_set=(Xte_eval, y_test),
            cat_features=self.categorical_features,  # Use column names, not indices
            verbose=False
        )

        logger.info("CatBoost model training completed!")
        return model

    def predict(self, model: CatBoostClassifier, X: pd.DataFrame) -> np.ndarray:
        """

        Args:
            model: Trained CatBoost model
            X: Features to predict on

        Returns:
            Predicted labels
        """
        X_original = self.prepare_categorical_for_prediction(X)
        return model.predict(X_original)

    def predict_proba(self, model: CatBoostClassifier, X: pd.DataFrame) -> np.ndarray:
        """

        Args:
            model: Trained CatBoost model
            X: Features to predict on

        Returns:
            Prediction probabilities
        """
        X_original = self.prepare_categorical_for_prediction(X)
        return model.predict_proba(X_original)
