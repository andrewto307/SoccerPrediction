import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import BaseEstimator

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class Model:
    def __init__(self, classifier: BaseEstimator, X_train: pd.DataFrame, 
                 y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.classifier = classifier

    def get_classifier(self) -> BaseEstimator:
        return self.classifier
    
    def run_model(self, params: dict) -> BaseEstimator:
        model = self.get_classifier(**params)
        model.fit(self.X_train, self.y_train)

        return model

    def predict(self, model: BaseEstimator) -> np.ndarray:
        y_pred = model.predict(self.X_test)
        return y_pred

    def eval(y_test: pd.Series, y_pred: pd.Series, model) -> None:
        
        print(f"Test Set Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        print(model.get_best_score())

    def grid_search_cv(self, classifier: BaseEstimator, param_grid: dict) -> dict:
        grid_search = grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, scoring='balanced_accuracy',
                               cv=5, verbose=1)
        
        grid_search.fit(self.X_train, self.y_train)
        model = grid_search.best_estimator_
        y_pred = model.predict(self.X_test)
        eval(self.y_test, y_pred, model)

        return grid_search.best_params_

class Catboost(Model):
    def run_model(self, params: dict) -> BaseEstimator:
        model = CatBoostClassifier(**params)
        model.fit(self.X_train,
          self.y_train,
          eval_set=(self.X_test, self.y_test),
          verbose=False)
        
        return model
    
class StackingClassifier(Model):
    def __init__(self, classifier: BaseEstimator, X_train: pd.DataFrame, y_train: pd.Series, 
                 X_test: pd.DataFrame, y_test: pd.Series, classifier_list: list[BaseEstimator]) -> None:
        super().__init__(classifier, X_train, y_train, X_test, y_test)
        self.classifier_list = classifier_list

    def run_model(self, params, final_estimator: BaseEstimator) -> BaseEstimator:
        model = StackingClassifier(final_estimator=final_estimator, **params)
        model.fit(self.X_train, self.y_train)
        
        return model
        




