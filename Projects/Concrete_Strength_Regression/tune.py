import pandas as pd
from constants import parameters
from utils import timeit
import joblib
import os.path

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from bartpy.sklearnmodel import SklearnModel

N_JOBS = -1

class TuneTrain(object):
    def __init__(self, model, cv, scoring):
        self.model = model
        self.param_grid = parameters[self.model.__class__.__name__]
        self.cv = cv
        self.scoring = scoring

    @timeit
    def optimizeHyperParams(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Perform grid search to find the best hyperparameters for the model. Updates the model with the best hyperparameters.

        Args:
            X_train (pd.DataFrame): training x values
            y_train (pd.Series): training y values

        Returns:
            best_params_: optimized hyperparameters
            best_score_: optimized CV score
        """
        grid_search = GridSearchCV(
            self.model, self.param_grid, cv=self.cv, scoring=self.scoring, n_jobs=N_JOBS)
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        print(f'CV Score ({self.scoring}): ', grid_search.best_score_)
    
    def trainModel(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model

        Args:
            X (pd.DataFrame): training x values
            y (pd.Series): training y values
        """
        self.model.fit(X, y)
    
    def evaluateModelMSE(self, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        """Evaluate the model on the test set

        Args:
            X_test (pd.DataFrame): test x values
            y_test (pd.Series): test y values

        Returns:
            float: score of the model
        """
        mse = mean_squared_error(y_test, self.model.predict(X_test))
        print(f'Model Test MSE: {mse}')
        return mse
    
    def saveModel(self, version: str) -> None:
        """Save the model to a file

        Args:
            version (str): version of the model to save
        """
        complete_path = f'../models/{self.model.__class__.__name__}_{version}.joblib'
        
        if os.path.isfile(complete_path):
            print(f'File already exists at {complete_path}')
            self.saveModel(input('Enter a new version number: '))
        else:
            joblib.dump(self.model, complete_path)
            print(f'Model saved to {complete_path}')