import pandas as pd
from sklearn.model_selection import train_test_split
from utils import printShape
from sklearn.preprocessing import StandardScaler


def separateFeaturesTarget(df: pd.DataFrame, target: str) -> tuple:
    """Separates the features and target columns

    Args:
        df (pd.DataFrame): Dataframe containing the data
        target (str): Name of the target column

    Returns:
        tuple: Features and target columns (X, y)
    """
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y


@printShape
def splitData(df: pd.DataFrame, target: str, test_size: float, random_state=0) -> tuple:
    """Splits the data into train and test sets

    Args:
        df (pd.DataFrame): Dataframe containing the data
        target (str): Name of the target column
        test_size (float): Test size
        random_state (int): random state

    Returns:
        tuple: Train and test sets (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(target, axis=1),
        df[target],
        test_size=test_size,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test


def printMissingValues(df: pd.DataFrame, columns: list = None) -> None:
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
            print(f"{column} has {df[column].isnull().sum()} missing values")
    print("Done")


def printDataTypes(df: pd.DataFrame, columns: list = None) -> None:
    """Prints the data types of the columns

    Args:
        df (pd.DataFrame): Dataframe containing the data
        columns (list, optional): List of columns to be checked. Defaults to None.
    """
    print("Data types:")
    if columns is None:
        columns = df.columns
    for column in columns:
        print(f"{column}: {df[column].dtype}")
    print("Done")


def imputeMissing(*dfs: pd.DataFrame, columns: list = None, strategy: str) -> tuple:
    """Imputes missing values in the columns of the dataframes

    Args:
        *dfs (pd.DataFrame): Dataframes containing the data
        columns (list, optional): List of columns to be imputed. Defaults to None.
        strategy (str): Strategy to be used for imputation. Typically 'mean' or 'median'

    Returns:
        tuple: Dataframes with imputed values
    """
    for df in dfs:
        if columns is None:
            columns = df.columns
        for column in columns:
            if df[column].isnull().sum() > 0:
                df[column].fillna(df[column].agg(strategy), inplace=True)
    return dfs


def standardScaleDataframe(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """Fits and transforms train and test sets using StandardScaler

    Args:
        X_train (pd.DataFrame): Dataframe containing the training data
        X_test (pd.DataFrame): Dataframe containing the test data

    Returns:
        X_train_scaled, X_test_scaled: Dataframes with scaled values
    """
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled
