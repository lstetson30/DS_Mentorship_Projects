import pandas as pd

def getZerosCounts(df, columns=None):
    """Returns the number of zeros in each column of the dataframe.
    
    Args:
        df (pd.DataFrame): dataframe to count zeros in
        columns (list): columns to count zeros in. If None, count zeros in all columns.
    """
    if columns:
        return (df[columns] == 0).sum()
    else:
        return (df == 0).sum()
    
def generateMissingDataCols(df, columns):
    """Generates columns for each input column that indicate whether the value is zero or not.

    Args:
        df (pd.DataFrame): dataframe to generate columns for
        columns (list): columns to generate columns for
    """
    output_df = df.copy()
    for col in columns:
        output_df[col + '_is_zero'] = (output_df[col] == 0).astype(int)
    return output_df