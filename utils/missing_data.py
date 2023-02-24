import pandas as pd

def getZerosCounts(df, columns=None):
    if columns:
        return (df[columns] == 0).sum()
    else:
        return (df == 0).sum()