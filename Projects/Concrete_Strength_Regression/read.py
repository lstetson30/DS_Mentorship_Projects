from datetime import datetime
import pandas as pd
from utils import printShape
import constants


@printShape
def readData(filename: str) -> pd.DataFrame:
    """Reads the data from the csv file and returns a pandas dataframe

    Args:
        filename (str): Name of the csv file

    Returns:
        pandas.DataFrame: Dataframe containing the data
    """
    frame = pd.read_csv(constants.DATAPATH + filename)
    print("Raw Data:")
    print(frame.head())
    return frame


def saveToRawData(df: pd.DataFrame, filename: str = None) -> None:
    """Saves the dataframe to the raw_data folder with a timestamp in the filename

    Args:
        df (pd.DataFrame): Dataframe to be saved
        filename (str, optional): Name of the file. Defaults to None.
    """
    name = filename if filename else "raw_data"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{constants.RAWDATAPATH}{name}_{ts}.csv"
    df.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}")
