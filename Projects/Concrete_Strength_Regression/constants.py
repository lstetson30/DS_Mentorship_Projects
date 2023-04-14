import numpy as np
import argparse

DATAPATH = 'Projects/Concrete_Strength_Regression/data/'
RAWDATAPATH = 'Projects/Concrete_Strength_Regression/data/raw_data/'
MODELSPATH = 'Projects/Concrete_Strength_Regression/models/'

PARAMETERS = {
    'LinearRegression': {'fit_intercept': [True, False]},
    'Ridge': {'alpha': list(10**np.linspace(10, -2, 10)*0.5), 'fit_intercept': [True, False]},
    'Lasso': {'alpha': list(10**np.linspace(10, -2, 10)*0.5), 'fit_intercept': [True, False]},
    'ElasticNet': {'alpha': list(10**np.linspace(10, -2, 10)*0.5),
                   'l1_ratio': np.linspace(0, 1, 10), 'fit_intercept': [True, False]},
    'DecisionTreeRegressor': {'max_depth': range(2, 21, 2), 'min_samples_split': [2, 5, 10]},
    'RandomForestRegressor': {'n_estimators': [5, 10, 25, 50, 100, 150, 200, 300, 400, 500],
                              'max_depth': range(6, 21, 2),
                              'min_samples_split': [2, 5, 10],
                              'max_features': [1, 'sqrt', 0.33],
                              'criterion': ['squared_error', 'absolute_error']
                              },
    'GradientBoostingRegressor': {'n_estimators': [25, 50, 75, 100, 200, 250, 500, 1000],
                                  'loss': ['squared_error', 'absolute_error'],
                                  'learning_rate': [0.01, 0.05, 0.1],
                                  'subsample': [0.8, 0.9, 1.0],
                                  'min_samples_split': [2, 5, 10],
                                  'max_depth': [2, 4, 8],
                                  'max_features': [1, 'auto', 'sqrt']
                                  },
    'LGBMRegressor': {'boosting_type': ['gbdt'],
                      'n_estimators': [25, 50, 75, 100, 200, 250, 500, 1000],
                      'learning_rate': [0.01, 0.05, 0.1],
                      'subsample': [0.8, 0.9, 1.0],
                      'min_child_samples': [2, 4, 16],
                      'max_depth': [2, 4, 8],
                      'colsample_bytree': [0.25, 0.5, 1.0]
                      },
    #This is the BART model from the bartpy package
    'SklearnModel': {'n_samples': range(50, 1001, 50),
                  'n_burn': range(50, 201, 50),
                  'n_trees': range(50, 201, 50),
                  'n_chains': [1]
                  }
}

parser = argparse.ArgumentParser(description='Run entire modeling pipeline from reading and\
                                 transforming data to training and evaluating model')
parser.add_argument('-r', '--reload', type=str, help='Reload model from joblib file')
parser.add_argument('-m', '--model', type=str, help='Model to use from the following list:\
                    LinearRegression, Ridge, Lasso, ElasticNet, DecisionTreeRegressor,\
                    RandomForestRegressor, GradientBoostingRegressor, LGBMRegressor, SklearnModel')
parser.add_argument('--printsummary', help='Print summary statistics of the data',\
                    action='store_true')
parser.add_argument('--testsize', type=float, help='Test size for train test split', default=0.2)
parser.add_argument('-s', '--scale', help='Scale the data', action='store_true')
parser.add_argument('--cv', type=int, help='Number of folds for cross validation', default=5)
parser.add_argument('--scoring', type=str, help='Scoring metric for cross validation',\
                    default='neg_mean_squared_error')
parser.add_argument('--modelversion', type=str, help='Version of the model to save', required=True)