import read
import utils
from transform import standardScaleDataframe
from tune import TuneTrain
import constants

from sklearn.model_selection import train_test_split


def run_model(model, file_name):
    # Read the data
    df = read.readData(file_name)

    if args.printsummary:
        utils.printSummaryStats(df)

    # Transform the data
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("csMPa", axis=1), df["csMPa"], test_size=args.testsize
    )

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
    X = df.drop("csMPa", axis=1)
    y = df["csMPa"]

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
