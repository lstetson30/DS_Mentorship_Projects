import pandas as pd
from sklearn.model_selection import train_test_split
from utils import printShape, timeit
from sklearn.preprocessing import StandardScaler

def separateFeaturesTarget(df: pd.DataFrame, target: str) -> tuple:
    """Separates the features and target columns

    Args:
        df (pd.DataFrame): Dataframe containing the data
        target (str): Name of the target column

    Returns:
        tuple: Features and target columns
    """
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y

@printShape
def splitData(df: pd.DataFrame, target: str, test_size: float, random_state=0):
    """Splits the data into train and test sets

    Args:
        df (pd.DataFrame): Dataframe containing the data
        target (str): Name of the target column
        test_size (float): Test size
        random_state (int): random state

    Returns:
        pd.DataFrame: Train set
        pd.DataFrame: Test set
    """
    X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1), df[target],
                                                        test_size=test_size, random_state=random_state)
    # print(f'Size of Train: X-{X_train.shape}, Y-{y_train.shape}')
    # print(f'Size of Test: X-{X_test.shape}, Y-{y_test.shape}')
    
    return X_train, X_test, y_train, y_test


def printMissingValues(df: pd.DataFrame, columns: list = None):
    """Prints the columns with missing values and how many missing values they have

    Args:
        df (pd.DataFrame): Dataframe containing the data
        columns (list, optional): List of columns to be checked. Defaults to None.
    """
    print("Missing Values:")
    if columns is None:
        columns = df.columns
    for column in columns:
        if df[column].isnull().sum() > 0:
            print(f'{column} has {df[column].isnull().sum()} missing values')
    print('Done')

def printDataTypes(df: pd.DataFrame, columns: list = None):
    """Prints the data types of the columns

    Args:
        df (pd.DataFrame): Dataframe containing the data
        columns (list, optional): List of columns to be checked. Defaults to None.
    """
    print("Data types:")
    if columns is None:
        columns = df.columns
    for column in columns:
        print(f'{column}: {df[column].dtype}')
    print('Done')

def imputeMissing(*dfs, columns: list = None, strategy: str):
    """Imputes missing values in the columns of the dataframes

    Args:
        *dfs (pd.DataFrame): Dataframes containing the data
        columns (list, optional): List of columns to be imputed. Defaults to None.
        strategy (str): Strategy to be used for imputation. Typically 'mean' or 'median'

    Returns:
        pd.DataFrame: Dataframe with imputed values
    """
    for df in dfs:
        if columns is None:
            columns = df.columns
        for column in columns:
            if df[column].isnull().sum() > 0:
                df[column].fillna(df[column].agg(strategy), inplace=True)
    return dfs

def standardScaleDataframe(*dfs):
    """Fits and transforms all columns of multiple dataframes using StandardScaler

    Args:
        *dfs (pd.DataFrame): Dataframes containing the data

    Returns:
        pd.DataFrame: Dataframe with scaled values
    """
    dfs_scaled = []
    for df in dfs:
        scaler = StandardScaler()
        dfs_scaled.append(pd.DataFrame(scaler.fit_transform(df), columns=df.columns))
    return dfs_scaled

def standardScaleSeries(*series):
    """Fits and transforms multiple pandas series using StandardScaler

    Args:
        *series (pd.Series): Series containing the data

    Returns:
        pd.Series: Series with scaled values
    """
    series_scaled = []
    for s in series:
        scaler = StandardScaler()
        series_scaled.append(pd.Series(scaler.fit_transform(s.values.reshape(-1, 1)).flatten(), name=s.name))
    return series_scaled

def inverseStandardScale(scaled_results, unscaled_series_ref: pd.Series):
    """Inverse standard scale on pandas series using StandardScaler

    Args:
        series (pd.Series): Series containing the data

    Returns:
        pd.Series: Series with unscaled values
    """
    scaler = StandardScaler()
    scaler.fit(unscaled_series_ref.values.reshape(-1, 1))
    result = scaler.inverse_transform(scaled_results.reshape(-1, 1))
    return result