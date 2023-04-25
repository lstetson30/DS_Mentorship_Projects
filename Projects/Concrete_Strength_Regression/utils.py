from time import time
import pandas as pd


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time()
        func(*args, **kwargs)
        end = time()
        print(f'Time taken to run {func.__name__}: {end-start} seconds')
    return wrapper


def printShape(func):
    def wrapper(*args, **kwargs):
        df = func(*args, **kwargs)
        print(f'Shape of {func.__name__} output: ', end=' ')
        if type(df) is not pd.DataFrame:
            for output in df:
                print(output.shape, end=' ')
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