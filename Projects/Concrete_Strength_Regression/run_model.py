import read, utils
from transform import splitData, standardScaleDataframe, separateFeaturesTarget
from tune import TuneTrain
import constants
import joblib
import os

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from bartpy.sklearnmodel import SklearnModel


def run_model(model, file_name):
    # Read the data
    df = read.readData(file_name)

    if args.printsummary:
        utils.printSummaryStats(df)

    # Transform the data
    X_train, X_test, y_train, y_test = splitData(df, "csMPa", args.testsize)

    # Scale the data with standard scaler
    if args.scale:
        X_train, X_test = standardScaleDataframe(X_train, X_test)

    # Tune the model to find the best Hyperparameters
    tuner = TuneTrain(model, args.cv, args.cvscoring, args.scale)
    tuner.optimizeHyperParams(X_train, y_train)

    # Train the model on the training set
    tuner.trainModel(X_train, y_train)

    # Evaluate the model on the training and test sets
    tuner.evaluateModel(X_train, y_train, args.testscoring, test_set=False)
    tuner.evaluateModel(X_test, y_test, args.testscoring, test_set=True)

    # Train the model on the entire dataset and save it
    X, y = separateFeaturesTarget(df, "csMPa")
    tuner.trainModel(X, y)
    tuner.saveModel(args.modelversion)


if __name__ == "__main__":
    args = constants.model_parser.parse_args()

    if args.reload:
        model = utils.loadJoblibModel(args.reload)
    else:
        if args.model in constants.PARAMETERS.keys():
            model = eval(args.model + "()")
        else:
            print(f"No model called {args.model}")
            raise SystemExit()

    run_model(model, constants.DATAFILE)
