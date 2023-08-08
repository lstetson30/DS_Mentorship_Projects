import pandas as pd
from sklearn.preprocessing import StandardScaler


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
