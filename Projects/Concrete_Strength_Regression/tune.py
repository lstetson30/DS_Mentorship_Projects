import pandas as pd
from constants import PARAMETERS, MODELSPATH
from utils import timeit
import joblib
import os.path
from csv import DictWriter

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from bartpy.sklearnmodel import SklearnModel

N_JOBS = -1


class TuneTrain(object):
    """Class to tune and train a model. Also evaluates the model on the training and test sets and saves the model to a file.

    Args:
        model (sklearn model): model to tune and train
        cv (int): number of cross validation folds
        cv_scoring (str): scoring metric to use for cross validation
        scaled_data (bool): whether the data has been scaled or not

    Attributes:
        model (sklearn model): model to tune and train
        param_grid (dict): hyperparameter grid to search over
        cv (int): number of cross validation folds
        cv_scoring (str): scoring metric to use for cross validation
        scoring_metric (str): scoring metric used to evaluate the model
        train_score (float): score of the model on the training set
        test_score (float): score of the model on the test set
        scaled_data (bool): whether the data has been scaled or not

    Methods:
        optimizeHyperParams: perform grid search to find the best hyperparameters for the model
        trainModel: train the model
        evaluateModel: evaluate the model on the training or test set
        saveModel: save the model to a joblib file and saves the model scores to a csv file
    """

    def __init__(self, model, cv, cv_scoring, scaled_data):
        self.model = model
        self.param_grid = PARAMETERS[self.model.__class__.__name__]
        self.cv = cv
        self.cv_scoring = cv_scoring
        self.scoring_metric = None
        self.train_score = None
        self.test_score = None
        self.scaled_data = scaled_data

    @timeit
    def optimizeHyperParams(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Perform grid search to find the best hyperparameters for the model. Updates the model with the best hyperparameters.

        Args:
            X_train (pd.DataFrame): training x values
            y_train (pd.Series): training y values

        Returns:
            best_params_: optimized hyperparameters
            best_score_: optimized CV score
        """
        grid_search = GridSearchCV(
            self.model,
            self.param_grid,
            cv=self.cv,
            scoring=self.cv_scoring,
            n_jobs=N_JOBS,
        )
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        print(f"CV Score ({self.cv_scoring}): ", grid_search.best_score_)

    def trainModel(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model

        Args:
            X (pd.DataFrame): training x values
            y (pd.Series): training y values
        """
        self.model.fit(X, y)

    def evaluateModel(
        self, X: pd.DataFrame, y: pd.Series, test_func: callable, test_set=True
    ) -> None:
        """Evaluate the model on the training or test set

        Args:
            X (pd.DataFrame): x values
            y (pd.Series): y values
            test_func (function): scoring function to use
            test_set (bool): whether X and y are from a test set (True) or training set (False)
        """
        self.scoring_metric = test_func.__name__
        if test_set:
            self.test_score = test_func(y, self.model.predict(X))
            print(f"Model Test Score ({self.scoring_metric}): {self.test_score}")
        else:
            self.train_score = test_func(y, self.model.predict(X))
            print(f"Model Train Score ({self.scoring_metric}): {self.train_score}")

    def saveModel(self, version: str) -> None:
        """Save the model to a file. Saves the model's test score as well.

        Args:
            version (str): version of the model to save
        """
        complete_path = f"{MODELSPATH}{self.model.__class__.__name__}_{version}.joblib"

        if os.path.isfile(complete_path):
            print(f"File already exists at {complete_path}")
            self.saveModel(input("Enter a new version number: "))
        else:
            joblib.dump(self.model, complete_path)
            print(f"Model saved to {complete_path}")

            score_dict = {
                "model": self.model.__class__.__name__,
                "version": version,
                "scaled_data": self.scaled_data,
                "scoring_method": self.scoring_metric,
                "train_score": self.train_score,
                "test_score": self.test_score,
            }

            headers = score_dict.keys()

            test_score_file_exists = os.path.isfile(f"{MODELSPATH}test_score.csv")

            with open(f"{MODELSPATH}test_score.csv", "a") as file:
                writer = DictWriter(file, fieldnames=headers)

                if not test_score_file_exists:
                    writer.writeheader()

                writer.writerow(score_dict)
