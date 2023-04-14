from datetime import datetime
import pandas as pd
from utils import printShape
from constants import DATAPATH, RAWDATAPATH

@printShape
def readData(filename: str) -> pd.DataFrame:
    """Reads the data from the csv file and returns a pandas dataframe

    Args:
        filename (str): Name of the csv file

    Returns:
        pandas.DataFrame: Dataframe containing the data
    """
    frame = pd.read_csv(DATAPATH + filename)
    print("Raw Data:")
    print(frame.head())
    return frame

def printSummaryStats(df: pd.DataFrame, columns: list = None) -> None:
    """Prints the summary statistics of the columns of the dataframe

    Args:
        df (pd.DataFrame): Dataframe containing the data
        columns (list, optional): List of columns for which the summary statistics are to be printed. Defaults to None.
    """
    print("Summary Statistics:")
    if columns is None:
        columns = df.columns
    print(df[columns].describe())

def saveToRawData(df: pd.DataFrame, filename: str = None) -> None:
    """Saves the dataframe to the raw_data folder with a timestamp in the filename

    Args:
        df (pd.DataFrame): Dataframe to be saved
        filename (str, optional): Name of the file. Defaults to None.
    """
    name = filename if filename is not None else 'raw_data'
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f"{RAWDATAPATH}{name}_{ts}.csv"
    df.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}")