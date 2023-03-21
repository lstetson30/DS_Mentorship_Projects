from datetime import datetime
import pandas as pd
from utils import printShape

@printShape
def readData(filename: str):
    """Reads the data from the csv file and returns a pandas dataframe

    Args:
        filename (str): Name of the csv file

    Returns:
        pandas.DataFrame: Dataframe containing the data
    """
    frame = pd.read_csv(filename)
    print("Raw Data:")
    print(frame.head())
    return frame

def printSummaryStats(df: pd.DataFrame, columns: list):
    """Prints the summary statistics of the columns of the dataframe

    Args:
        df (pd.DataFrame): Dataframe containing the data
        columns (list): List of columns for which the summary statistics are to be printed
    """
    print("Summary Statistics:")
    print(df[columns].describe())

def saveToRawData(df: pd.DataFrame, filename: str = None):
    """Saves the dataframe to the raw_data folder with a timestamp in the filename

    Args:
        df (pd.DataFrame): Dataframe to be saved
        filename (str, optional): Name of the file. Defaults to None.
    """
    name = filename if filename is not None else 'raw_data'
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    df.to_csv(f"./Projects/Concrete_Strength_Regression/data/raw_data/{name}_{ts}.csv", index=False)
    print(f"Data saved to ./Projects/Concrete_Strength_Regression/data/raw_data/{name}_{ts}.csv")