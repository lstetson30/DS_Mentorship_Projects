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


def saveToRawData(df: pd.DataFrame, filename: str = None):
    """Saves the dataframe to the raw_data folder with a timestamp in the filename

    Args:
        df (pd.DataFrame): Dataframe to be saved
        filename (str, optional): Name of the file. Defaults to None.
    """
    name = filename if filename is not None else 'raw_data'
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f"../data/raw_data/{name}_{ts}.csv"
    df.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}")