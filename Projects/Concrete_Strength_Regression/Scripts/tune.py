import pandas as pd
import numpy as np
from constants import parameters
from utils import timeit
import argparse

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from bartpy.sklearnmodel import SklearnModel

N_JOBS = -1

class Tune(object):
    def __init__(self, model, cv, scoring):
        self.model = model
        self.param_grid = parameters[self.model.__class__.__name__]
        self.cv = cv
        self.scoring = scoring

    @timeit
    def grid_search(self, X_train: pd.DataFrame, y_train: pd.Series) -> list:
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
        return [grid_search.best_params_, grid_search.best_score_]

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--cv', type=int, required=True)
parser.add_argument('--scoring', type=str, required=True)
args = parser.parse_args()

if args.model in parameters.keys():
    model = eval(args.model + '()')
else:
    print('Model not found')
    raise SystemExit()

tuner = Tune(model, args.cv, args.scoring)