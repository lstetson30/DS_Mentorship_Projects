import read, transform
from tune import TuneTrain
import argparse
from constants import parameters
import joblib
import os

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from bartpy.sklearnmodel import SklearnModel

parser = argparse.ArgumentParser(description='Run entire modeling pipeline from reading and\
                                 transforming data to training and evaluating model')
parser.add_argument('-r', '--reload', help='Reload model from joblib file')
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

args = parser.parse_args()

os.chdir(os.path.dirname(os.path.abspath(__file__)))

if args.reload:
    try:
        model = joblib.load(f'../models/{args.reload}')
    except FileNotFoundError:
        print('Model file not found')
        raise SystemExit()
else:
    if args.model in parameters.keys():
       model = eval(args.model + '()')
    else:
        print(f'No model called {args.model}')
        raise SystemExit()

# Read the data
# file_in = input("Enter the name of the file (in data folder)to be read: ").strip()
file_in = 'Concrete_Data_Yeh.csv'
df = read.readData(f"../data/{file_in}")

if args.printsummary:
    read.printSummaryStats(df)

read.saveToRawData(df)

#Transform the data
X_train, X_test, y_train, y_test = transform.splitData(df, 'csMPa', args.testsize)

#Scale the data with standard scaler
if args.scale:
    X_train_unscaled, X_test_unscaled, y_train_unscaled, y_test_unscaled=\
        X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()
    X_train, X_test = transform.standardScaleDataframe(X_train, X_test)
    y_train, y_test = transform.standardScaleSeries(y_train, y_test)

#Tune the model to find the best Hyperparameters
tuner = TuneTrain(model, args.cv, args.scoring)
tuner.optimizeHyperParams(X_train, y_train)

#Train the model on the training set
tuner.trainModel(X_train, y_train)

#Evaluate the model on the test set
test_mse = tuner.evaluateModelMSE(X_test, y_test)
if args.scale:
    eval_model = tuner.model
    y_predicted = eval_model.predict(X_test)
    y_predicted = transform.inverseStandardScale(y_predicted, y_train_unscaled)
    test_mse = mean_squared_error(y_test_unscaled, y_predicted)
    print(f'Model Test MSE Unscaled: {test_mse}')
    
#Train the model on the entire dataset and save it
X, y = transform.separateFeaturesTarget(df, 'csMPa')
tuner.trainModel(X, y)
tuner.saveModel(args.modelversion)