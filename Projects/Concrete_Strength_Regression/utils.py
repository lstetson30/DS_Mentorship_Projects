from time import time
import pandas as pd
import joblib
from constants import MODELSPATH


def timeit(func):
    """Decorator to time the execution of a function"""

    def wrapper(*args, **kwargs):
        start = time()
        func(*args, **kwargs)
        end = time()
        print(f"Time taken to run {func.__name__}: {end-start} seconds")

    return wrapper


def printShape(func):
    """Decorator to print the shape of the output of a function"""

    def wrapper(*args, **kwargs):
        df = func(*args, **kwargs)
        print(f"Shape of {func.__name__} output: ", end=" ")
        if type(df) is not pd.DataFrame:
            for output in df:
                print(output.shape, end=" ")
            print()
        else:
            print(df.shape)
        return df

    return wrapper


def printSummaryStats(df: pd.DataFrame, columns: list = None):
    """Prints the summary statistics of the columns of the dataframe

    Args:
        df (pd.DataFrame): Dataframe containing the data
        columns (list, optional): List of columns for which the summary statistics are to be printed. Defaults to None.
    """
    print("Summary Statistics:")
    if columns is None:
        columns = df.columns
    print(df[columns].describe())


def loadJoblibModel(filename: str) -> object:
    """Loads the model from the specified path

    Args:
        filename (str): Name of the file to load excluding the extension

    Returns:
        model: The loaded model
    """

    try:
        model = joblib.load(MODELSPATH + filename + ".joblib")
    except FileNotFoundError:
        print("Model file not found")
        raise SystemExit()

    return model
