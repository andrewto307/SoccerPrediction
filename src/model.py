import pandas as pd

from sklearn.metrics import accuracy_score, classification_report

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class Model:
    def __init__(self, classifier, X_train, y_train, X_test, y_test) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.classifier = classifier

    def get_classifier(self):
        return self.classifier
    
    def run_model(self, params):
        model = self.get_classifier(**params)
        model.fit(self.X_train, self.y_train)

        return model

    def predict(self, model):
        y_pred = model.predict(self.X_test)
        return y_pred

    def eval(y_test, y_pred, model):
        print(f"Test Set Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        print(model.get_best_score())

    def grid_search_cv(self, classifier, param_grid):
        grid_search = grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, scoring='balanced_accuracy',
                               cv=5, verbose=1)
        
        grid_search.fit(self.X_train, self.y_train)
        model = grid_search.best_estimator_
        y_pred = model.predict(self.X_test)
        eval(self.y_test, y_pred, model)

        return grid_search.best_params_

class Catboost(Model):
    def run_model(self, params):
        model = CatBoostClassifier(**params)
        model.fit(self.X_train,
          self.y_train,
          eval_set=(self.X_test, self.y_test),
          verbose=False)
        
        return model
    
class StackingClassifier(Model):
    def __init__(self, classifier, X_train, y_train, X_test, y_test, classifier_list):
        super().__init__(classifier, X_train, y_train, X_test, y_test)
        self.classifier_list = classifier_list

    def run_model(self, params, final_estimator):
        model = StackingClassifier(**params, final_estimator)
        model.fit(self.X_train, self.y_train)
        
        return model
        




