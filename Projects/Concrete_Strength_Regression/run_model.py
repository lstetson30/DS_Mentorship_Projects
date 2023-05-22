import read, transform
from tune import TuneTrain
from constants import PARAMETERS, MODELSPATH, parser
import joblib
import os

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from bartpy.sklearnmodel import SklearnModel


def run_model(model, file_name):

    # Read the data
    # file_in = input("Enter the name of the file (in data folder)to be read: ").strip()
    df = read.readData(file_name)

    if args.printsummary:
        read.printSummaryStats(df)

    read.saveToRawData(df)

    #Transform the data
    X_train, X_test, y_train, y_test = transform.splitData(df, 'csMPa', args.testsize)

    #Scale the data with standard scaler
    if args.scale:
        X_train_unscaled, X_test_unscaled = X_train.copy(), X_test.copy()
        X_train, X_test = transform.standardScaleDataframe(X_train, X_test)

    #Tune the model to find the best Hyperparameters
    tuner = TuneTrain(model, args.cv, args.scoring)
    tuner.optimizeHyperParams(X_train, y_train)

    #Train the model on the training set
    tuner.trainModel(X_train, y_train)

    #Evaluate the model on the test set
    test_mse = tuner.evaluateModelMSE(X_test, y_test)
    # if args.scale:
    #     eval_model = tuner.model
    #     y_predicted = eval_model.predict(X_test)
    #     y_predicted = transform.inverseStandardScale(y_predicted, y_train_unscaled)
    #     test_mse = mean_squared_error(y_test_unscaled, y_predicted)
    #     print(f'Model Test MSE Unscaled: {test_mse}')

    #Train the model on the entire dataset and save it
    X, y = transform.separateFeaturesTarget(df, 'csMPa')
    tuner.trainModel(X, y)
    tuner.saveModel(args.modelversion)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.reload:
        try:
            model = joblib.load(MODELSPATH + args.reload)
        except FileNotFoundError:
            print('Model file not found')
            raise SystemExit()
    else:
        if args.model in PARAMETERS.keys():
            model = eval(args.model + '()')
        else:
            print(f'No model called {args.model}')
            raise SystemExit()

    run_model(model, args.DATA_FILE)
